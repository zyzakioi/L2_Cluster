#!/usr/bin/env python3
"""
Simple TTFT Comparison: Qwen/Qwen2.5-0.5B (Single VM vs. 2-VM L2 Cluster)
=========================================================================
Runs Qwen/Qwen2.5-0.5B twice on the exact same prompt:
  1. Pass 1: Single VM only (Master 10.0.39.9 alone) -> measures TTFT (max_new_tokens=1).
  2. Offloads 50% of all 24 layers to Slave VM (10.0.214.182) over Layer-2 Raw Ethernet.
  3. Pass 2: 2-VM L2 Cluster (10.0.39.9 + 10.0.214.182) -> measures TTFT (max_new_tokens=1).
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm_l2_bridge import L2ClusterController, offload_all_model_layers

MODEL_ID = "Qwen/Qwen2.5-3B"
IFACE = "ens33"  # Change if your interface name is different
PROMPT = (
    "Distributed computing over raw ethernet frames allows multiple computers "
    "in a local area network to share CPU and memory resources seamlessly."
)

# 1. Load Tokenizer & Model on Master VM
print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()

inputs = tokenizer(PROMPT, return_tensors="pt")
print(f"Prompt token count: {inputs['input_ids'].shape[1]} tokens\n")

# -------------------------------------------------------------------------
# Run 1: Single VM Only (10.0.39.9 alone)
# -------------------------------------------------------------------------
with torch.no_grad():
    # Warmup run
    _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)

    t0 = time.perf_counter()
    out_single = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    ttft_single_ms = (time.perf_counter() - t0) * 1000.0

first_tok_single = tokenizer.decode(out_single[0][-1:])
print(f"[Run 1: Single VM Only] TTFT = {ttft_single_ms:.2f} ms | First token: {repr(first_tok_single)}\n")

# -------------------------------------------------------------------------
# Offload 50% of ALL Layers to Slave VM (10.0.214.182)
# -------------------------------------------------------------------------
ctrl = L2ClusterController(ifname=IFACE)
slaves = ctrl.discover_slaves(timeout=1.5)
if not slaves:
    raise RuntimeError("No Slave VM found! Make sure slave_daemon.py is running on 10.0.214.182.")
slave_mac = slaves[0]["mac"]

offload_all_model_layers(model=model, controller=ctrl, slave_mac=slave_mac, offload_ratio=0.5)

# -------------------------------------------------------------------------
# Run 2: 2-VM Layer-2 Cluster (10.0.39.9 + 10.0.214.182)
# -------------------------------------------------------------------------
with torch.no_grad():
    # Warmup run
    _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)

    t0 = time.perf_counter()
    out_cluster = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    ttft_cluster_ms = (time.perf_counter() - t0) * 1000.0

first_tok_cluster = tokenizer.decode(out_cluster[0][-1:])
print(f"\n[Run 2: 2-VM L2 Cluster] TTFT = {ttft_cluster_ms:.2f} ms | First token: {repr(first_tok_cluster)}")
print("-" * 60)
print(f"Single VM TTFT : {ttft_single_ms:.2f} ms")
print(f"2-VM L2 TTFT   : {ttft_cluster_ms:.2f} ms")
print(f"Speedup        : {ttft_single_ms / ttft_cluster_ms:.2f}x")