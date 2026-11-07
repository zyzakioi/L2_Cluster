#!/usr/bin/env python3
"""
Master-Side vLLM Layer-2 Raw Ethernet Offload Bridge (vllm_l2_bridge.py)
========================================================================
Runs on the Master node (10.0.39.9) alongside vLLM.
Provides:
  1. `L2ClusterController`: Discovers Slave nodes via Layer-2 broadcast, manages remote
     weight shard allocation (up to 65,536 slots), and dispatches activation tensors over
     Raw Ethernet (0x88B5) with asynchronous Send/Recv split for TRUE parallel overlap.
  2. `L2TensorParallelLinear`: A drop-in PyTorch `nn.Module` replacement that splits any
     Linear layer across the Master CPU/GPU and Slave node(s) transparently over Layer-2,
     computing the Master half WHILE the Slave computes the remote half simultaneously.
  3. `offload_all_model_layers()`: Offloads 50% of ALL transformer layers (every MLP & Attention
     projection across the entire model) to the Slave VM and frees Master RAM via gc.collect().
"""

import argparse
import gc
import struct
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from l2_raw_transport import (
    BROADCAST_MAC,
    DTYPE_TO_CODE,
    L2RawTensorSocket,
    OP_ALLOC_WEIGHT,
    OP_DISCOVER_REQ,
    OP_DISCOVER_RESP,
    OP_EXEC_MATMUL,
    OP_EXEC_RESULT,
    OP_TENSOR_ACK,
    OP_TENSOR_CHUNK,
    mac_to_str,
    str_to_mac,
)


class L2ClusterController:
    """Master-side controller that manages Layer-2 Slave nodes and remote weight handles."""

    def __init__(self, ifname: Optional[str] = None):
        self.l2 = L2RawTensorSocket(ifname=ifname)
        self.slaves: List[Dict] = []
        self._next_slot = 0
        self.total_tx_ms = 0.0
        self.total_rx_wait_ms = 0.0
        self.total_calls = 0

    def reset_telemetry(self):
        self.total_tx_ms = 0.0
        self.total_rx_wait_ms = 0.0
        self.total_calls = 0

    def discover_slaves(self, timeout: float = 1.0) -> List[Dict]:
        """Broadcast an L2 discovery frame (0x88B5) and collect all responding Slave MACs."""
        self.slaves.clear()
        self.l2.transfer_seq = (self.l2.transfer_seq + 1) & 0xFFFF
        tid = self.l2.transfer_seq

        self.l2.send_control_frame(BROADCAST_MAC, OP_DISCOVER_REQ, transfer_id=tid)
        deadline = time.perf_counter() + timeout
        seen_macs = set()

        while time.perf_counter() < deadline:
            res = self.l2.recv_frame(timeout=max(0.01, deadline - time.perf_counter()))
            if res is None:
                continue
            src_mac, opcode, _, _, r_tid, _, _, _, payload_view = res
            if opcode == OP_DISCOVER_RESP and src_mac not in seen_macs:
                seen_macs.add(src_mac)
                cpus, free_mb, raw_host = struct.unpack("!II32s", payload_view[:40])
                hostname = raw_host.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
                info = {
                    "mac": src_mac,
                    "mac_str": mac_to_str(src_mac),
                    "cpus": cpus,
                    "free_ram_mb": free_mb,
                    "hostname": hostname,
                }
                self.slaves.append(info)
                print(
                    f"[MasterL2] Discovered Slave '{hostname}' at MAC {info['mac_str']} "
                    f"(CPUs={cpus}, FreeRAM={free_mb}MB)"
                )
        return self.slaves

    def upload_weight_shard(
        self,
        slave_mac: bytes,
        weight: torch.Tensor,
        slot_id: Optional[int] = None,
        verbose: bool = True,
    ) -> int:
        """Allocate and upload a persistent 2D weight shard to a Slave node over Layer-2."""
        if slot_id is None:
            slot_id = self._next_slot
            self._next_slot = (self._next_slot + 1) & 0xFFFF

        rows, cols = weight.shape
        dtype_code = DTYPE_TO_CODE.get(weight.dtype, 0)
        self.l2.transfer_seq = (self.l2.transfer_seq + 1) & 0xFFFF
        tid = self.l2.transfer_seq

        meta_payload = struct.pack("!IIII", slot_id, rows, cols, dtype_code)
        self.l2.send_control_frame(slave_mac, OP_ALLOC_WEIGHT, payload=meta_payload, slot_id=slot_id, transfer_id=tid)

        # Wait strictly for the metadata ACK matching our current `tid`
        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            res = self.l2.recv_frame(timeout=0.1)
            if res and res[0] == slave_mac and res[1] == OP_TENSOR_ACK and res[4] == tid:
                break

        # Blast weight chunks over Layer-2 Raw Ethernet
        elapsed = self.l2.send_tensor_reliable(
            dst_mac=slave_mac,
            tensor=weight,
            opcode=OP_TENSOR_CHUNK,
            slot_id=slot_id,
        )
        if verbose:
            mb = (weight.numel() * weight.element_size()) / (1024 * 1024)
            print(
                f"[MasterL2] Uploaded Weight Shard #{slot_id} ({rows}x{cols}, {mb:.2f} MB) "
                f"to {mac_to_str(slave_mac)} in {elapsed * 1000:.2f} ms "
                f"({(mb / elapsed) * 8:.2f} Mbps)"
            )
        return slot_id

    def dispatch_remote_matmul_async(
        self,
        slave_mac: bytes,
        slot_id: int,
        x: torch.Tensor,
    ) -> Tuple[Tuple[int, ...], int]:
        """
        Step 1 of Async Overlap: Blast activation X to Slave over Layer-2 and return immediately
        WITHOUT waiting for the Slave to finish computing, so Master can compute its local half
        at the exact same time!
        """
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
        batch_tokens = x_2d.shape[0]

        t0 = time.perf_counter()
        self.l2.send_tensor_reliable(
            dst_mac=slave_mac,
            tensor=x_2d,
            opcode=OP_EXEC_MATMUL,
            slot_id=slot_id,
        )
        self.total_tx_ms += (time.perf_counter() - t0) * 1000.0
        self.total_calls += 1
        return orig_shape, batch_tokens

    def collect_remote_matmul_result(
        self,
        slave_mac: bytes,
        orig_shape: Tuple[int, ...],
        batch_tokens: int,
        out_features: int,
    ) -> torch.Tensor:
        """
        Step 2 of Async Overlap: Collect the Slave's computed activation tensor Y_remote
        after Master finishes its own local matrix multiplication.
        """
        t0 = time.perf_counter()
        while True:
            frame = self.l2.recv_frame(timeout=5.0)
            if frame is None:
                raise TimeoutError("Timed out waiting for Slave matmul result over Layer-2")
            if frame[0] == slave_mac and frame[1] == OP_EXEC_RESULT:
                _, _, y_2d = self.l2.recv_tensor_reliable(
                    shape=(batch_tokens, out_features),
                    first_frame=frame,
                )
                self.total_rx_wait_ms += (time.perf_counter() - t0) * 1000.0
                return y_2d.reshape(*orig_shape[:-1], out_features)

    def execute_remote_matmul(
        self,
        slave_mac: bytes,
        slot_id: int,
        x: torch.Tensor,
        out_features: int,
    ) -> torch.Tensor:
        """Synchronous convenience wrapper around dispatch + collect."""
        orig_shape, batch_tokens = self.dispatch_remote_matmul_async(slave_mac, slot_id, x)
        return self.collect_remote_matmul_result(slave_mac, orig_shape, batch_tokens, out_features)


class L2TensorParallelLinear(nn.Module):
    """
    Drop-in replacement for `torch.nn.Linear` or vLLM's `ColumnParallelLinear`.
    Splits the output features 50/50 (or custom ratio) between Master and Slave:
      1. Master sends activation X to Slave over Layer-2 (`dispatch_remote_matmul_async`).
      2. While Slave computes `X @ W_remote^T`, Master simultaneously computes `X @ W_local^T`.
      3. Master collects `Y_remote` from Layer-2 and concatenates `[Y_local, Y_remote]`.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        controller: L2ClusterController,
        slave_mac: bytes,
        slot_id: Optional[int] = None,
        offload_ratio: float = 0.5,
        verbose: bool = True,
    ):
        super().__init__()
        self.controller = controller
        self.slave_mac = slave_mac

        out_features, in_features = base_linear.weight.shape
        remote_out = int(out_features * offload_ratio)
        local_out = out_features - remote_out

        self.local_out = local_out
        self.remote_out = remote_out

        # Split weight along output dimension (Column-Parallel)
        w_data = base_linear.weight.detach().cpu()
        w_local = w_data[:local_out, :].contiguous()
        w_remote = w_data[local_out:, :].contiguous()

        self.local_weight = nn.Parameter(w_local, requires_grad=False)
        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.detach().cpu(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        # Upload remote weight shard to Slave once during initialization
        self.slot_id = self.controller.upload_weight_shard(
            self.slave_mac, w_remote, slot_id=slot_id, verbose=verbose
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Fire-and-forget activation X to Slave over Layer-2 Raw Ethernet
        orig_shape, batch_tokens = self.controller.dispatch_remote_matmul_async(
            slave_mac=self.slave_mac,
            slot_id=self.slot_id,
            x=x,
        )

        # 2. Compute Master's local half IN PARALLEL while Slave computes remote half
        y_local = torch.matmul(x, self.local_weight.t())

        # 3. Collect Slave's result and concatenate
        y_remote = self.controller.collect_remote_matmul_result(
            slave_mac=self.slave_mac,
            orig_shape=orig_shape,
            batch_tokens=batch_tokens,
            out_features=self.remote_out,
        )

        out = torch.cat([y_local, y_remote.to(y_local.device)], dim=-1)
        if self.bias is not None:
            out = out + self.bias
        return out


def offload_all_model_layers(
    model: nn.Module,
    controller: L2ClusterController,
    slave_mac: bytes,
    offload_ratio: float = 0.5,
    verbose: bool = True,
) -> int:
    """
    Offload 50% of ALL transformer layers' MLP projections across the entire model to the Slave VM.
    Calls gc.collect() periodically so Master RAM doesn't spike on 3B/7B models.
    """
    t0 = time.perf_counter()
    shards_uploaded = 0
    for idx, layer in enumerate(model.model.layers):
        layer.mlp.gate_proj = L2TensorParallelLinear(
            base_linear=layer.mlp.gate_proj,
            controller=controller,
            slave_mac=slave_mac,
            offload_ratio=offload_ratio,
            verbose=verbose,
        )
        layer.mlp.up_proj = L2TensorParallelLinear(
            base_linear=layer.mlp.up_proj,
            controller=controller,
            slave_mac=slave_mac,
            offload_ratio=offload_ratio,
            verbose=verbose,
        )
        shards_uploaded += 2
        if (idx + 1) % 4 == 0:
            gc.collect()
    gc.collect()
    dt = time.perf_counter() - t0
    print(
        f"[MasterL2] Offloaded {shards_uploaded} weight shards across ALL {len(model.model.layers)} layers "
        f"in {dt:.2f}s."
    )
    return shards_uploaded