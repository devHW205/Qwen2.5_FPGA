import sys
import torch
from transformers import AutoTokenizer

# Thư mục chứa:
# - qwen2/ (configuration_qwen2.py, modeling_qwen2.py, ...)
# - config.json, generation_config.json, model.safetensors
# - tokenizer_config.json, tokenizer.json, vocab.json, merges.txt (nếu BPE)
model_path = "/home/toannguyen/KLTN/Qwen2.5-0.5B-Instruct"

# Cho Python nhìn thấy package 'qwen2' local
if model_path not in sys.path:
    sys.path.insert(0, model_path)

# Import class từ code local
from qwen2.configuration_qwen2 import Qwen2Config
from qwen2.modeling_qwen2 import Qwen2ForCausalLM

# Tokenizer (offline)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True
)

# Config từ class local (không phụ thuộc trust_remote_code)
config = Qwen2Config.from_pretrained(
    model_path,
    local_files_only=True
)

# Chọn device/dtype
use_gpu = torch.cuda.is_available()
device_map = "auto" if use_gpu else None
torch_dtype = torch.float16 if use_gpu else None

model = Qwen2ForCausalLM.from_pretrained(
    model_path,
    config=config,
    local_files_only=True,
    device_map=device_map,
    torch_dtype=torch_dtype
)

print("✅ Chatbot Qwen2.5-0.5B-Instruct (local qwen2/*) sẵn sàng! Gõ 'exit' để thoát.\n")

# Lấy token kết thúc hội thoại
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
eos_ids = [tok for tok in [im_end_id, tokenizer.eos_token_id] if tok is not None]

chat_history = [
    {"role": "system", "content": "You are a helpful assistant."}
]

while True:
    user_input = input("👤 Bạn: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("🤖 Bot: Tạm biệt!")
        break

    chat_history.append({"role": "user", "content": user_input})

    # Dựng prompt theo chat template (nếu tokenizer có)
    prompt_text = tokenizer.apply_chat_template(
        chat_history,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        repetition_penalty=1.1,
        eos_token_id=eos_ids,
        pad_token_id=tokenizer.eos_token_id
    )

    # Chỉ decode phần sinh mới
    input_len = inputs["input_ids"].shape[-1]
    new_tokens = outputs[0, input_len:]
    bot_reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    print(f"🤖 Bot: {bot_reply}\n")

    chat_history.append({"role": "assistant", "content": bot_reply})