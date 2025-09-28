import sys
import torch
from transformers import AutoTokenizer
from qwen2.configuration_qwen2 import Qwen2Config
from qwen2.modeling_qwen2 import Qwen2ForCausalLM

model_path = "/home/toannguyen/KLTN/Qwen2.5-0.5B-Instruct"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

# Config + model
config = Qwen2Config.from_pretrained(model_path, local_files_only=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Qwen2ForCausalLM.from_pretrained(
    model_path,
    config=config,
    local_files_only=True,
    torch_dtype=torch.float32 if device == "cpu" else torch.float16
).to(device)

print("✅ Model loaded, starting quick test…\n")

# --- TEST PROMPT ĐƠN GIẢN ---
prompt = "Hello! Please introduce yourself."
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.8
    )

# Chỉ giải mã phần model sinh thêm 
input_len = inputs["input_ids"].shape[-1]
new_tokens = outputs[0, input_len:]
reply = tokenizer.decode(new_tokens, skip_special_tokens=True)

print("Model reply:\n", reply)
