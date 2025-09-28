import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP as HFQwen2MLP

# 🟢 Bản tường minh bạn viết
def silu(x):
    return x * torch.sigmoid(x)

def gelu(x):
    c = torch.sqrt(torch.tensor(2.0 / torch.pi))
    return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x ** 3)))

def relu(x):
    return torch.maximum(x, torch.zeros_like(x))

ACT2FN = {
    "silu": silu,
    "gelu": gelu,
    "relu": relu,
}

class Qwen2MLP_Explicit(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        gate_out = self.gate_proj(x)
        gate_act = self.act_fn(gate_out)
        up_out = self.up_proj(x)
        gated_up = gate_act * up_out
        down_out = self.down_proj(gated_up)
        return down_out

# =============================
# Test
# =============================
from transformers import Qwen2Config

torch.manual_seed(0)

# Config nhỏ để test
config = Qwen2Config(
    hidden_size=16,
    intermediate_size=32,
    hidden_act="silu",   # đổi thử "gelu"/"relu" cũng được
)

batch, seq = 2, 4
x = torch.randn(batch, seq, config.hidden_size)

# Khởi tạo hai MLP
hf_mlp = HFQwen2MLP(config)
my_mlp = Qwen2MLP_Explicit(config)

# Copy trọng số để đảm bảo so sánh công bằng
my_mlp.gate_proj.weight.data.copy_(hf_mlp.gate_proj.weight.data)
my_mlp.up_proj.weight.data.copy_(hf_mlp.up_proj.weight.data)
my_mlp.down_proj.weight.data.copy_(hf_mlp.down_proj.weight.data)

# Forward
out_hf = hf_mlp(x)
out_my = my_mlp(x)

# So sánh
print("Max diff:", (out_hf - out_my).abs().max().item())
print("Close:", torch.allclose(out_hf, out_my, atol=1e-6))
print("Output HF:", out_hf)
print("Output My:", out_my)