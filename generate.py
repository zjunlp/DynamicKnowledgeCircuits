import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型和tokenizer
# model_path = "/mnt/8t/oyx/PLMs/TinyLlama-1.1B-Chat-v1.0"
model_path = "outputs/train/TinyLlama_v1.1/2024-12-23-22-37-26/checkpoints"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_path)
model.to(device)


def generate_answer(query, model, tokenizer, max_new_tokens=10):
    inputs = tokenizer(query, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        # do_sample=True,
        # top_k=5,
        # top_p=0.95,
    )
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output


# 示例问题
query = "Question: Which American-born Sinclair won the Nobel Prize for Literature in 1930? Answer:"

# 生成答案
answer = generate_answer(query, model, tokenizer, 20)

# 输出答案
print(answer)
