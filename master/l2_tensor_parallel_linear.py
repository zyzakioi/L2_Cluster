import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm_l2_bridge import L2ClusterController, L2TensorParallelLinear

# 1. Initialize Layer-2 Controller on Master and discover Slave(s)
ctrl = L2ClusterController(ifname="ens33")
slaves = ctrl.discover_slaves(timeout=1.5)
slave_mac = slaves[0]["mac"]

# 2. Load model on Master CPU
model_id = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

# 3. Offload 50% of the MLP gate_proj and up_proj in all layers to the Slave VM over Layer-2
for idx in range(model.config.num_hidden_layers):
    layer_mlp = model.model.layers[idx].mlp
    layer_mlp.gate_proj = L2TensorParallelLinear(
        base_linear=layer_mlp.gate_proj,
        controller=ctrl,
        slave_mac=slave_mac,
        slot_id=(idx * 2),
        offload_ratio=0.5,
    )
    layer_mlp.up_proj = L2TensorParallelLinear(
        base_linear=layer_mlp.up_proj,
        controller=ctrl,
        slave_mac=slave_mac,
        slot_id=(idx * 2 + 1),
        offload_ratio=0.5,
    )

# 4. Run inference seamlessly — Slave computes half of each offloaded MLP projection via Raw Ethernet
inputs = tokenizer("Distributed computing over raw ethernet is", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))