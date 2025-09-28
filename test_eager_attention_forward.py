import torch
import math
import random
from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward as eager_attention_ref

# ======================
# Dummy module để thay cho `self` trong eager_attention_forward
# ======================
class DummyModule:
    def __init__(self, num_key_value_groups=1, training=False):
        self.num_key_value_groups = num_key_value_groups
        self.training = training

# ======================
# Phiên bản tường minh (list-based)
# ======================
def matmul(A, B):
    m, n = len(A), len(A[0])
    n2, p = len(B), len(B[0])
    assert n == n2
    C = [[0 for _ in range(p)] for _ in range(m)]
    for i in range(m):
        for j in range(p):
            s = 0
            for k in range(n):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C

def transpose(A):
    m, n = len(A), len(A[0])
    return [[A[j][i] for j in range(m)] for i in range(n)]

def softmax(x):
    m, n = len(x), len(x[0])
    out = [[0]*n for _ in range(m)]
    for i in range(m):
        row = x[i]
        max_val = max(row)
        exps = [math.exp(v - max_val) for v in row]
        sum_exps = sum(exps)
        out[i] = [v / sum_exps for v in exps]
    return out

def eager_attention_explicit(query, key, value, scaling=1.0):
    # query, key, value: list of size [seq, dim]
    attn_weights = matmul(query, transpose(key))
    attn_weights = [[v * scaling for v in row] for row in attn_weights]
    attn_weights = softmax(attn_weights)
    attn_output = matmul(attn_weights, value)
    return attn_output, attn_weights

# ======================
# Test so sánh
# ======================
def test_attention():
    torch.manual_seed(0)
    random.seed(0)

    # Tạo input nhỏ
    seq, dim = 3, 4
    q = torch.randn(seq, dim)
    k = torch.randn(seq, dim)
    v = torch.randn(seq, dim)

    # Dummy module
    module = DummyModule(num_key_value_groups=1, training=False)

    # Chạy bản ref (torch)
    out_ref, w_ref = eager_attention_ref(
        module,
        q.unsqueeze(0).unsqueeze(0),  # [batch=1, heads=1, seq, dim]
        k.unsqueeze(0).unsqueeze(0),
        v.unsqueeze(0).unsqueeze(0),
        attention_mask=None,
        dropout=0.0,
        scaling=1.0
    )
    out_ref = out_ref.squeeze().detach()
    w_ref = w_ref.squeeze().detach()

    # Chạy bản tường minh (list)
    out_explicit, w_explicit = eager_attention_explicit(
        q.tolist(), k.tolist(), v.tolist(), scaling=1.0
    )
    out_explicit = torch.tensor(out_explicit)
    w_explicit = torch.tensor(w_explicit)

    # So sánh
    print("Output close:", torch.allclose(out_explicit, out_ref, atol=1e-5))
    print("Weights close:", torch.allclose(w_explicit, w_ref, atol=1e-5))
    print("Max diff out:", (out_explicit - out_ref).abs().max().item())
    print("Max diff attn:", (w_explicit - w_ref).abs().max().item())

if __name__ == "__main__":
    test_attention()
