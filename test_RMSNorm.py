import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm as HFQwen2RMSNorm

# 🟢 Bản tường minh bạn viết
class Qwen2RMSNorm_Explicit(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H = x.shape[-1]
        squared = x * x
        sum_squared = squared.sum(dim=-1, keepdim=True)
        mean_square = sum_squared / H
        rms = torch.sqrt(mean_square + self.eps)
        normed = x / rms
        out = normed * self.weight
        return out

# =============================
# Test
# =============================
torch.manual_seed(0)

batch, seq, hidden_size = 2, 4, 8
x = torch.randn(batch, seq, hidden_size)

# Khởi tạo 2 layer
hf_norm = HFQwen2RMSNorm(hidden_size)
my_norm = Qwen2RMSNorm_Explicit(hidden_size)

# Copy trọng số để đảm bảo so sánh công bằng
my_norm.weight.data.copy_(hf_norm.weight.data)

# Chạy forward
out_hf = hf_norm(x)
out_my = my_norm(x)

# So sánh
print(out_hf)
print(out_my)
print("Max diff:", (out_hf - out_my).abs().max().item())
print("Close:", torch.allclose(out_hf, out_my, atol=1e-6))
