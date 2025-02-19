# %%
import os
import json
import random
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# %%
N = 50000
n = 500
random.seed(42)


# Define a function to read the JSONL file
def read_jsonl(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
        data = [json.loads(line) for line in lines]
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []


# Load the data points
revised_entities = read_jsonl(f"entities_{N}/revised.jsonl")

# %%
model = "gpt2"
model_name_or_path = "/mnt/16t/oyx/PLMs/gpt2"

device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
hf_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
hf_model.to(device)

# %%
for data_point in tqdm(revised_entities):
    full_name = data_point["full_name"]

    prompt = f"""
    Please generate a paragraph about {full_name}. Include the following details:
    EXAMPLE:
    On 18 October, 1979, John M. Janzen celebrates his/her annual birthday.
    
    Penelope Everest Copley celebrates his/her birth anniversary on 30 October, 1989.
    
    {full_name}"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    outputs = hf_model.generate(
        **inputs,
        max_new_tokens=100,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(output)


# %%
with open(f"entities_{N}/forgetting.jsonl", "w", encoding="utf-8") as file:
    for data_point in revised_entities:
        file.write(json.dumps(data_point, ensure_ascii=False) + "\n")
