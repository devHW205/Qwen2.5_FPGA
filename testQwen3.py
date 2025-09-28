from transformers import Qwen2ForCausalLM

model = Qwen2ForCausalLM.from_pretrained(
    "/home/toannguyen/KLTN/Qwen2.5-0.5B-Instruct",
    torch_dtype="auto",       # Tự chọn float16/bfloat16 nếu GPU hỗ trợ
    device_map="auto"         # Tự chia GPU/CPU
)


# Chạy thử 1 forward
import torch
inputs = torch.randint(0, model.config.vocab_size, (1, 5))
with torch.no_grad():
    out = model(inputs).logits
print(out.shape)
