import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional, Tuple

# =========================================================
# 1. Mini config để test (mô phỏng Qwen2Config cần thiết)
# =========================================================
@dataclass
class MiniConfig:
    hidden_size: int = 256
    num_attention_heads: int = 8
    num_key_value_heads: int = 2         # thử grouped KV (8 / 2 = 4 groups)
    intermediate_size: int = 1024
    attention_dropout: float = 0.0
    head_dim: Optional[int] = None       # nếu None sẽ suy ra hidden_size // num_attention_heads
    sliding_window: Optional[int] = None
    layer_types: Tuple[str, ...] = ("full_attention",)  # chỉ cần 1 layer
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 2048


# =========================================================
# 2. Helper (rotate_half, apply_rotary_pos_emb, repeat_kv)
# =========================================================
def rotate_half(x: torch.Tensor):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    # (batch, kv_heads, seq, dim) -> (batch, kv_heads * n_rep, seq, dim)
    b, kv_h, s, d = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(b, kv_h, n_rep, s, d)
    return hidden_states.reshape(b, kv_h * n_rep, s, d)

# =========================================================
# 3. Rotary Embedding đơn giản (giống bản bạn dùng)
# =========================================================
class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config: MiniConfig):
        super().__init__()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", hidden_size // num_heads)
        theta = getattr(config, "rope_theta", 10000.0)
        freq_seq = torch.arange(0, self.head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (freq_seq / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor):
        device = x.device
        dtype = x.dtype
        inv_freq = self.inv_freq.to(device)
        pos = position_ids.to(device).float()
        freqs = pos[:, :, None] * inv_freq[None, None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

# =========================================================
# 4. Lớp Attention được test (DÁN BẢN CỦA BẠN Ở ĐÂY)
#    Đây là bản theo đề xuất tối giản incremental có attn_weights
# =========================================================
class Qwen2Attention(nn.Module):
    def __init__(self, config: MiniConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.hidden_size = config.hidden_size
        assert self.hidden_size % self.num_heads == 0
        assert self.num_heads % self.num_kv_heads == 0
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.attention_dropout = self.config.attention_dropout
        self.scaling = self.head_dim ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,                        # [B,T,C]
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # (cos, sin) [B,T_cur, head_dim]
        attention_mask: Optional[torch.Tensor] = None,      # [B,1,T_q,T_k_total] hoặc None
        past_key_values=None,                               # object có .get_seq_length() & .update() hoặc None
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ):
        B, T, C = hidden_states.shape
        cos, sin = position_embeddings

        # Q full
        q = self.q_proj(hidden_states)

        incremental = past_key_values is not None and getattr(past_key_values, "get_seq_length", lambda:0)() > 0
        kv_input = hidden_states[:, -1:, :] if incremental else hidden_states
        k = self.k_proj(kv_input)
        v = self.v_proj(kv_input)

        # Reshape
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)             # [B,H,T,D]
        k = k.view(B, k.size(1), self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B,Kv,Tk,D]
        v = v.view(B, v.size(1), self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B,Kv,Tk,D]

        # RoPE
        if incremental and k.size(2) == 1 and cos.size(1) > 1:
            cos_q = cos
            sin_q = sin
            cos_k = cos[:, -1:, :]
            sin_k = sin[:, -1:, :]
            q, _ = apply_rotary_pos_emb(q, q, cos_q, sin_q)
            k, _ = apply_rotary_pos_emb(k, k, cos_k, sin_k)
        else:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Cache update
        if past_key_values is not None and hasattr(past_key_values, "update"):
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        # Expand KV
        if self.num_key_value_groups > 1:
            k = repeat_kv(k, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)

        # Scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling  # [B,H,T,T_total]
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        if self.attention_dropout and self.training:
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout)

        attn_output = torch.matmul(attn_weights, v)  # [B,H,T,D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights if use_cache else None


# =========================================================
# 5. DynamicCache mock (nếu không muốn import transformers)
#    Nếu bạn dùng transformers, thay bằng:
#    from transformers.cache_utils import DynamicCache
# =========================================================
class SimpleDynamicCache:
    """
    Bản đơn giản mô phỏng DynamicCache:
    Lưu list [(k0,v0), (k1,v1), ...] theo layer_idx.
    K shape: [B, H or Kv, seq, D]
    """
    def __init__(self):
        self.key_cache = {}
        self.value_cache = {}

    def get_seq_length(self):
        # Trả về độ dài đã cache của layer 0 (giả định đồng bộ)
        if 0 in self.key_cache:
            return self.key_cache[0].shape[2]
        return 0

    def update(self, k, v, layer_idx, cache_kwargs=None):
        # k,v mới có shape [B, Kv_or_H, 1 or T, D] khi incremental
        if layer_idx not in self.key_cache:
            self.key_cache[layer_idx] = k
            self.value_cache[layer_idx] = v
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], k], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], v], dim=2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


# =========================================================
# 6. Các hàm test
# =========================================================
def build_causal_mask(bsz, q_len, k_len, device):
    # Additive mask: 0 cho allowed, -inf cho masked
    mask = torch.full((q_len, k_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)  # upper tri (strictly)
    # broadcast -> [B,1,q_len,k_len]
    return mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, q_len, k_len)

def test_full_forward():
    print("TEST 1: Full forward no cache")
    cfg = MiniConfig()
    attn = Qwen2Attention(cfg, layer_idx=0).eval()
    B, T = 2, 16
    x = torch.randn(B, T, cfg.hidden_size)
    pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
    rope = Qwen2RotaryEmbedding(cfg)
    cos, sin = rope(x, pos_ids)

    out, w = attn(
        hidden_states=x,
        position_embeddings=(cos, sin),
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        use_cache=False,
    )
    assert out.shape == (B, T, cfg.hidden_size)
    print("  PASS shape:", out.shape)

def test_mask():
    print("TEST 2: Forward với causal mask thủ công")
    cfg = MiniConfig()
    attn = Qwen2Attention(cfg, layer_idx=0).eval()
    B, T = 1, 8
    x = torch.randn(B, T, cfg.hidden_size)
    rope = Qwen2RotaryEmbedding(cfg)
    pos_ids = torch.arange(T).unsqueeze(0)
    cos, sin = rope(x, pos_ids)
    mask = build_causal_mask(B, T, T, x.device)  # [B,1,T,T]

    out, _ = attn(
        hidden_states=x,
        position_embeddings=(cos, sin),
        attention_mask=mask,
        past_key_values=None,
        cache_position=None,
        use_cache=False,
    )
    assert out.shape == (B, T, cfg.hidden_size)
    print("  PASS mask shape:", out.shape)

def test_incremental_equivalence():
    print("TEST 3: Incremental decode vs full recompute (so sánh token cuối)")
    torch.manual_seed(42)
    cfg = MiniConfig()
    attn = Qwen2Attention(cfg, layer_idx=0).eval()
    rope = Qwen2RotaryEmbedding(cfg)

    B, T_init, steps = 1, 6, 4
    device = "cpu"
    x_full = torch.randn(B, T_init + steps, cfg.hidden_size, device=device)

    # Chạy full 1 lần
    pos_full = torch.arange(T_init + steps).unsqueeze(0)
    cos_full, sin_full = rope(x_full, pos_full)
    out_full, _ = attn(
        hidden_states=x_full,
        position_embeddings=(cos_full, sin_full),
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        use_cache=False,
    )
    last_tokens_full = out_full[:, -steps:, :]  # so sánh từng bước

    # Chạy incremental
    cache = SimpleDynamicCache()
    outputs_inc = []
    for step in range(T_init + steps):
        cur_x = x_full[:, : step + 1, :]  # HF style: thường chỉ truyền token mới; ở đây ta truyền full để test
        # Nhưng attention của ta xem incremental dựa trên cache.get_seq_length() > 0 và lấy kv_input token cuối
        pos_ids = torch.arange(step + 1).unsqueeze(0)
        cos, sin = rope(cur_x, pos_ids)
        out_inc, _ = attn(
            hidden_states=cur_x,
            position_embeddings=(cos, sin),
            attention_mask=None,
            past_key_values=cache,
            cache_position=None,
            use_cache=True,
        )
        outputs_inc.append(out_inc[:, -1, :].detach())

    # Lấy các token cuối tương ứng incremental (sau giai đoạn warmup)
    inc_tail = torch.stack(outputs_inc[-steps:], dim=1)  # [B, steps, C]

    diff = (inc_tail - last_tokens_full).abs().max().item()
    print(f"  Max |diff| incremental vs full: {diff:.6f}")
    assert diff < 1e-4, "Incremental khác biệt quá lớn"
    print("  PASS incremental equivalence")

def test_backward():
    print("TEST 4: Backward / gradient flow")
    cfg = MiniConfig()
    attn = Qwen2Attention(cfg, layer_idx=0).train()
    rope = Qwen2RotaryEmbedding(cfg)
    B, T = 2, 10
    x = torch.randn(B, T, cfg.hidden_size, requires_grad=True)
    pos_ids = torch.arange(T).unsqueeze(0)
    cos, sin = rope(x, pos_ids)
    out, _ = attn(
        hidden_states=x,
        position_embeddings=(cos, sin),
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        use_cache=False,
    )
    loss = out.pow(2).mean()
    loss.backward()
    # Kiểm tra gradient tồn tại
    grad_norm = x.grad.norm().item()
    assert grad_norm > 0, "Gradient không chảy"
    print("  PASS backward, grad norm:", grad_norm)

def test_group_kv_heads():
    print("TEST 5: KV heads < attention heads")
    cfg = MiniConfig(num_attention_heads=12, num_key_value_heads=4, hidden_size=192)  # 192 /12 =16 head_dim
    attn = Qwen2Attention(cfg, layer_idx=0).eval()
    rope = Qwen2RotaryEmbedding(cfg)
    B, T = 2, 7
    x = torch.randn(B, T, cfg.hidden_size)
    pos_ids = torch.arange(T).unsqueeze(0)
    cos, sin = rope(x, pos_ids)
    out, w = attn(
        hidden_states=x,
        position_embeddings=(cos, sin),
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        use_cache=False,
    )
    assert out.shape == (B, T, cfg.hidden_size)
    print("  PASS grouped KV shape:", out.shape, "| attn_weights:", None if w is None else w.shape)

def run_all():
    test_full_forward()
    test_mask()
    test_incremental_equivalence()
    test_backward()
    test_group_kv_heads()
    print("\nTẤT CẢ TEST ĐÃ PASS ✅")

if __name__ == "__main__":
    run_all()