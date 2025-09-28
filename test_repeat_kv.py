import torch
from transformers.models.qwen2.modeling_qwen2 import repeat_kv as repeat_kv_ref

# --- Code tường minh của bạn ---
def repeat_kv_explicit(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    output = torch.zeros((batch, num_kv_heads * n_rep, seq_len, head_dim), dtype=hidden_states.dtype)

    for b in range(batch):
        for h in range(num_kv_heads):
            for r in range(n_rep):
                new_h = h * n_rep + r
                for i in range(seq_len):
                    for d in range(head_dim):
                        output[b, new_h, i, d] = hidden_states[b, h, i, d]
    return output

# --- Test ---
def test_repeat_kv():
    torch.manual_seed(0)

    batch, num_kv_heads, seq_len, head_dim = 2, 3, 4, 5
    n_rep = 2

    # random input
    x = torch.randn(batch, num_kv_heads, seq_len, head_dim)

    # chạy 2 phiên bản
    out_explicit = repeat_kv_explicit(x, n_rep)
    out_ref = repeat_kv_ref(x, n_rep)

    # so sánh
    print("Shapes equal:", out_explicit.shape == out_ref.shape)
    print("Allclose:", torch.allclose(out_explicit, out_ref, atol=1e-6))
    print("Max diff:", (out_explicit - out_ref).abs().max().item())
    print("Output explicit:\n", out_explicit)
    print("Output reference:\n", out_ref)   
test_repeat_kv()
