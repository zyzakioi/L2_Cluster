# L2-LLM-Cluster: Layer-2 Raw Ethernet Distributed LLM Inference

**L2-LLM-Cluster** is an ultra-low-latency, asymmetric Master-Slave distributed tensor parallelism framework for running Large Language Models (LLMs) across everyday computers and VMs over standard local area network (LAN) Ethernet cables (1GbE / 2.5GbE / 10GbE).

Instead of relying on heavy cluster orchestrators like [Ray](https://docs.ray.io/en/latest/index.html) or incurring Linux kernel TCP/IP stack overhead, **L2-LLM-Cluster** communicates directly at the **Data Link Layer (Layer-2 Raw Ethernet)** using [Linux `AF_PACKET` (`SOCK_RAW`)](https://kernel-internals.org/net/packet-socket/) sockets and IEEE 802.1 Local Experimental EtherType `0x88B5`.

## Key Features

* **Zero TCP/IP Overhead**: Bypasses IP routing, TCP slow-start, and congestion windows by transmitting raw Layer-2 Ethernet frames directly between hardware MAC addresses (`EtherType 0x88B5`).
* **Zero-Copy Tensor Framing**: Uses Python `ctypes` buffer protocols, `memoryview` slicing, and `torch.frombuffer` to fragment and reassemble contiguous [PyTorch](https://pytorch.org/) tensors directly into target memory without intermediate serialization (`pickle`) or memory copies.
* **Lightweight Asymmetric Architecture**:
  * **Master Node**: Runs the full LLM engine ([Hugging Face Transformers](https://huggingface.co/docs/transformers/index) or [vLLM](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)), tokenizers, and `L2ClusterController`.
  * **Slave Node(s)**: Runs a tiny, dependency-free 150-line daemon (`slave_daemon.py`) requiring only `torch` and `numpy`—no model files on disk, no Ray, and no vLLM installation needed.
* **Asynchronous Parallel Compute Overlap**: During every forward pass, the Master dispatches input activations over Layer-2 and computes its local 50% weight matrix shard simultaneously while the Slave computes the remote 50% shard in parallel.
* **Custom NACK Bitmap Reliability & Desync Immunity**: Guarantees 100% lossless tensor delivery over raw Ethernet via lightweight NACK bitmaps, dynamic timeout scaling, micro-burst pacing, and completed `transfer_id` tracking.

## Repository Structure

| File | Node Role | Description |
| :--- | :--- | :--- |
| `l2_raw_transport.py` | **Master & Slave** | Core Layer-2 Raw Ethernet (`AF_PACKET`, EtherType `0x88B5`) transport engine, zero-copy tensor slicer/reassembler, NACK bitmap recovery, and auto-MTU detection. |
| `slave_daemon.py` | **Slave Only** | Lightweight remote memory and compute worker that responds to L2 MAC discovery broadcasts, stores up to 65,536 persistent weight shards in RAM/VRAM, and executes remote matrix multiplications. |
| `vllm_l2_bridge.py` | **Master Only** | Master-side `L2ClusterController`, drop-in `L2TensorParallelLinear` (`nn.Module`) layer splitter, and `offload_all_model_layers()` helper. |
| `benchmark_ttft.py` | **Master Only** | End-to-end Time-To-First-Token (TTFT) comparison script benchmarking Single-Node execution against the 2-Node Layer-2 Cluster on [Qwen2.5](https://huggingface.co/collections/Qwen/qwen25) models. |

## Wire Protocol Layout (EtherType `0x88B5`)

Every frame transmitted on the physical or virtual wire consists of a **14-byte Hardware Ethernet MAC Header** followed by a **20-byte Custom L2 Tensor Header** (`!2sBBHHIII`):

```text
+-----------------------------------------------------------------------+
| 14-Byte Ethernet Header: Dst MAC (6B) | Src MAC (6B) | 0x88B5 (2B)    |
+-----------------------------------------------------------------------+
| 20-Byte L2 Tensor Header:                                             |
|   Magic 'LM' (2B) | OpCode (1B) | Dtype (1B) | Slot ID (2B, 0..65535) |
|   Transfer ID (2B)| Chunk ID (4B)| Total Chunks (4B)| Total Bytes (4B)|
+-----------------------------------------------------------------------+
| Raw Tensor Payload Slice (Up to 1464B on 1500 MTU / 8964B on 9000 MTU)|
+-----------------------------------------------------------------------+
```

## Prerequisites & Network Requirements

1. **Operating System**: Linux (Ubuntu 20.04 / 22.04 / 24.04 recommended) with `root` or `CAP_NET_RAW` privileges (required by Linux kernel to open `AF_PACKET` raw sockets).
2. **Layer-2 Broadcast Domain**: All participating machines or VMs must be connected to the **same Layer-2 Ethernet switch** (or the same VMware `VMnet` Bridged / Host-Only virtual switch) so raw Ethernet frames can pass directly between MAC addresses without traversing a Layer-3 router.
3. **Python Environment**: Python 3.10+ (Python 3.12 recommended) via [Anaconda](https://www.anaconda.com/) or Miniconda.

## Step-by-Step Deployment Guide

### Step 1: Verify Layer-2 Connectivity & Tune Kernel Buffers (All Nodes)

On **both Master and Slave nodes**, identify your active LAN interface name (e.g., `ens33` or `eth0`) and hardware MAC address:

```bash
ip -br link show
ip -br addr show
```

Verify from the **Master** node that the **Slave** node is directly reachable at Layer-2 (ARP neighbor table):

```bash
ping -c 2 <SLAVE_IP>
ip neigh show <SLAVE_IP>
# Expected output: <SLAVE_IP> dev ens33 lladdr 00:0c:29:xx:xx:xx REACHABLE
```

Next, run the following commands as `root` on **both Master and Slave nodes** to expand the Linux kernel socket ring buffers (64 MB) and enable Jumbo Frames (`MTU 9000`) if supported by your NIC/vNIC:

```bash
sudo sysctl -w net.core.rmem_max=67108864
sudo sysctl -w net.core.wmem_max=67108864
sudo sysctl -w net.core.rmem_default=33554432
sudo sysctl -w net.core.wmem_default=33554432
sudo sysctl -w net.core.netdev_max_backlog=30000

# Enable 9000-byte Jumbo Frames (falls back to 1500 MTU if unsupported)
sudo ip link set dev ens33 mtu 9000 || echo "Keeping standard 1500 MTU"
```

### Step 2: Set Up the Python 3.12 Environment (All Nodes)

Clone this repository onto both nodes and create the `l2_cluster` Conda environment:

```bash
conda create -y -n l2_cluster python=3.12 numpy setuptools wheel packaging
conda activate l2_cluster

# Install CPU-only PyTorch (or CUDA PyTorch if running on GPU nodes)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

On the **Master node only**, install `transformers` and `accelerate` (and optionally `vllm-cpu` from [PyPI](https://pypi.org/project/vllm-cpu/)):

```bash
# Run on Master Node only:
pip install transformers accelerate
```

### Step 3: Start the Slave Compute Daemon (Slave Node)

On the **Slave node**, start `slave_daemon.py` as `root` (or with `sudo -E`). It will bind to your network interface and listen for Layer-2 frames with EtherType `0x88B5`:

```bash
conda activate l2_cluster
python3 slave_daemon.py --iface ens33
```

Expected Slave console output:

```text
[SlaveDaemon] Bound to interface 'ens33'
[SlaveDaemon] Local Hardware MAC : 00:0c:29:c3:6f:1d
[SlaveDaemon] Link MTU           : 9000 bytes (Max chunk: 8964 B)
[SlaveDaemon] Compute Device     : cpu (CPU Threads: 4)
[SlaveDaemon] Listening for Layer-2 Raw Ethernet frames (EtherType 0x88B5)...
```

### Step 4: Verify Layer-2 Link & Tensor Splitting (Master Node)

On the **Master node**, run `vllm_l2_bridge.py` to test automatic Layer-2 MAC discovery, 50/50 weight shard upload, and parallel activation matrix multiplication:

```bash
conda activate l2_cluster
python3 vllm_l2_bridge.py --iface ens33 --hidden-dim 1024 --batch-tokens 64
```

> **Note**: If your virtual switch filters custom EtherType broadcast packets (`ff:ff:ff:ff:ff:ff`), pass the Slave's MAC address explicitly using `--slave-mac 00:0c:29:xx:xx:xx`.

Expected Master console output:

```text
========================================================================
 Layer-2 Raw Ethernet Distributed Tensor & vLLM Linear Offload Benchmark
========================================================================
[MasterL2] Local Interface : ens33 (MAC: 00:0c:29:xx:xx:xx, MTU: 9000)
[MasterL2] Broadcasting Layer-2 Discovery (EtherType 0x88B5)...
[MasterL2] Discovered Slave 'ubuntu2' at MAC 00:0c:29:c3:6f:1d (CPUs=4, FreeRAM=14820MB)
...
------------------------------------------------------------------------
 Verification Result : SUCCESS (Max Absolute Error = 0.000000)
------------------------------------------------------------------------
```

### Step 5: Run End-to-End LLM Offloading & TTFT Benchmark (Master Node)

With `slave_daemon.py` running on the Slave node, run `benchmark_ttft.py` on the **Master node** to compare Single-Node TTFT against 2-Node Layer-2 Cluster TTFT (offloading 50% of all transformer layers to the Slave):

```bash
python3 benchmark_ttft.py
```

To offload 50% of all layers in your own Python scripts or [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) / [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) inference pipeline, use `offload_all_model_layers`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm_l2_bridge import L2ClusterController, offload_all_model_layers

# 1. Connect to Slave node(s) over Layer-2 Raw Ethernet
ctrl = L2ClusterController(ifname="ens33")
slave_mac = ctrl.discover_slaves(timeout=1.5)[0]["mac"]

# 2. Load model on Master node
model_id = "Qwen/Qwen2.5-3B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, low_cpu_mem_usage=True)
model.eval()

# 3. Split and offload 50% of ALL layers across Master & Slave over Layer-2
offload_all_model_layers(model=model, controller=ctrl, slave_mac=slave_mac, offload_ratio=0.5)

# 4. Run generation seamlessly
inputs = tokenizer("Distributed computing over raw ethernet frames allows", return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Troubleshooting

* **`PermissionError: [Errno 1] Operation not permitted` when opening socket**: Raw `AF_PACKET` sockets require `root` privileges or the `CAP_NET_RAW` Linux capability. Run Python as `root` or grant capability to your Conda Python binary via:
  ```bash
  sudo setcap cap_net_raw+ep $(which python3)
  ```
* **No Slave nodes responded to Layer-2 broadcast**: Ensure both nodes are on the same Layer-2 switch/vSwitch (`ip neigh show <SLAVE_IP>`) or pass `--slave-mac <MAC>` directly if your hypervisor drops custom EtherType broadcasts.
* **High Retransmission Rate on 1500 MTU**: Ensure `net.core.rmem_max=67108864` is applied on both nodes and enable `mtu 9000` (Jumbo Frames) on the network interfaces to reduce per-packet interrupt overhead by 6x.
