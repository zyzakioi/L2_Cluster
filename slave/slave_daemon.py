#!/usr/bin/env python3
"""
Lightweight Layer-2 Slave Compute & Memory Daemon (slave_daemon.py)
===================================================================
Runs on the Slave node(s) (e.g., 10.0.214.182) with ZERO Ray or vLLM installation required.
Listens on Layer-2 Raw Ethernet (EtherType 0x88B5) for:
  1. MAC Auto-Discovery broadcasts from the Master node (10.0.39.9).
  2. Persistent Weight Shards allocation & upload into Slave RAM / VRAM (up to 65,536 slots).
  3. Low-latency Activation Forward Passes (remote linear / MLP / attention projection)
     and immediate Layer-2 activation return.
"""

import argparse
import os
import struct
import time
from typing import Dict, Tuple

import torch

from l2_raw_transport import (
    CODE_TO_DTYPE,
    DTYPE_ITEMSIZE,
    L2RawTensorSocket,
    OP_ALLOC_WEIGHT,
    OP_DISCOVER_REQ,
    OP_DISCOVER_RESP,
    OP_EXEC_MATMUL,
    OP_EXEC_RESULT,
    OP_TENSOR_ACK,
    OP_TENSOR_CHUNK,
    mac_to_str,
)


class SlaveComputeDaemon:
    def __init__(self, ifname: str = None, device: str = "cpu", num_threads: int = 0):
        if num_threads > 0:
            torch.set_num_threads(num_threads)
        self.device = torch.device(device)
        self.l2 = L2RawTensorSocket(ifname=ifname)
        # Persistent weight buffers indexed by 16-bit slot_id (0..65535)
        self.weight_slots: Dict[int, torch.Tensor] = {}
        self.weight_shapes: Dict[int, Tuple[int, int]] = {}

        print(f"[SlaveDaemon] Bound to interface '{self.l2.ifname}'")
        print(f"[SlaveDaemon] Local Hardware MAC : {mac_to_str(self.l2.src_mac)}")
        print(f"[SlaveDaemon] Link MTU           : {self.l2.mtu} bytes (Max chunk: {self.l2.max_chunk_payload} B)")
        print(f"[SlaveDaemon] Compute Device     : {self.device} (CPU Threads: {torch.get_num_threads()})")
        print("[SlaveDaemon] Listening for Layer-2 Raw Ethernet frames (EtherType 0x88B5)...")

    def get_free_ram_mb(self) -> int:
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) // 1024
        except Exception:
            pass
        return 4096

    def serve_forever(self):
        while True:
            frame = self.l2.recv_frame(timeout=1.0)
            if frame is None:
                continue

            src_mac, opcode, dtype_code, slot_id, tid, cid, total_chunks, total_bytes, payload_view = frame

            # Ignore late duplicate data chunks from a transfer_id we already finished
            if opcode in (OP_TENSOR_CHUNK, OP_EXEC_MATMUL) and self.l2.is_tid_already_completed(src_mac, tid):
                self.l2.send_control_frame(src_mac, OP_TENSOR_ACK, slot_id=slot_id, transfer_id=tid)
                continue

            # 1. Master L2 Broadcast Discovery
            if opcode == OP_DISCOVER_REQ:
                # Reset completed_tids tracker when Master starts a new session
                self.l2.completed_tids.pop(src_mac, None)
                cpus = torch.get_num_threads()
                free_mb = self.get_free_ram_mb()
                hostname = os.uname().nodename.encode("utf-8")[:32]
                resp_payload = struct.pack("!II32s", cpus, free_mb, hostname)
                self.l2.send_control_frame(src_mac, OP_DISCOVER_RESP, payload=resp_payload, transfer_id=tid)
                print(
                    f"[SlaveDaemon] Replied to Discovery from Master {mac_to_str(src_mac)} "
                    f"(CPUs={cpus}, FreeRAM={free_mb}MB)"
                )

            # 2. Metadata Announcement for Weight Allocation
            elif opcode == OP_ALLOC_WEIGHT:
                # Payload: slot_id (I), rows (I), cols (I), dtype_code (I)
                alloc_slot, rows, cols, w_dtype_code = struct.unpack("!IIII", payload_view[:16])
                self.weight_shapes[alloc_slot] = (rows, cols)
                self.l2.send_control_frame(src_mac, OP_TENSOR_ACK, slot_id=alloc_slot, transfer_id=tid)
                print(
                    f"[SlaveDaemon] Prepared Weight Slot #{alloc_slot}: shape=({rows}, {cols}), "
                    f"dtype={CODE_TO_DTYPE.get(w_dtype_code)}"
                )

            # 3. Incoming Weight Shard Tensor Upload
            elif opcode == OP_TENSOR_CHUNK:
                shape = self.weight_shapes.get(slot_id)
                if shape is None:
                    dtype = CODE_TO_DTYPE.get(dtype_code, torch.float32)
                    numel = total_bytes // DTYPE_ITEMSIZE[dtype]
                    shape = (numel,)
                t0 = time.perf_counter()
                _, _, w_tensor = self.l2.recv_tensor_reliable(shape=shape, first_frame=frame)
                self.weight_slots[slot_id] = w_tensor.to(self.device)
                dt = (time.perf_counter() - t0) * 1000.0
                mb = total_bytes / (1024 * 1024)
                print(
                    f"[SlaveDaemon] Loaded Weight Shard into Slot #{slot_id}: "
                    f"shape={tuple(w_tensor.shape)}, {mb:.2f} MB in {dt:.2f} ms "
                    f"({(mb / (dt / 1000.0)) * 8:.2f} Mbps)"
                )

            # 4. Execute Remote Linear / Matrix Multiplication on Resident Weight Shard
            elif opcode == OP_EXEC_MATMUL:
                w = self.weight_slots.get(slot_id)
                if w is None:
                    print(f"[SlaveDaemon] ERROR: Weight Slot #{slot_id} not initialized!")
                    continue

                dtype = CODE_TO_DTYPE.get(dtype_code, torch.float32)
                in_features = w.shape[1]
                batch_tokens = total_bytes // (in_features * DTYPE_ITEMSIZE[dtype])
                in_shape = (batch_tokens, in_features)

                # Receive activation tensor X over Layer-2 Raw Ethernet
                _, _, x_cpu = self.l2.recv_tensor_reliable(shape=in_shape, first_frame=frame)

                # Compute Y = X @ W^T on Slave CPU/GPU
                with torch.no_grad():
                    x_dev = x_cpu.to(self.device) if self.device.type != "cpu" else x_cpu
                    y_dev = torch.matmul(x_dev, w.t())
                    y_cpu = y_dev.cpu()

                # Send output activation tensor Y directly back to Master over Layer-2
                self.l2.send_tensor_reliable(
                    dst_mac=src_mac,
                    tensor=y_cpu,
                    opcode=OP_EXEC_RESULT,
                    slot_id=slot_id,
                )


def main():
    parser = argparse.ArgumentParser(description="Layer-2 Raw Ethernet Slave Compute Daemon")
    parser.add_argument("--iface", type=str, default=None, help="Network interface (auto-detected if omitted)")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device: cpu or cuda:0")
    parser.add_argument("--threads", type=int, default=0, help="Number of CPU threads for PyTorch OpenMP")
    args = parser.parse_args()

    daemon = SlaveComputeDaemon(ifname=args.iface, device=args.device, num_threads=args.threads)
    daemon.serve_forever()


if __name__ == "__main__":
    main()