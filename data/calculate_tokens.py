import json
import tiktoken
from tqdm import tqdm

cal = 0
enc = tiktoken.get_encoding("r50k_base")

with open("data/entities_50000/train.jsonl", "r") as f:
    for line in tqdm(f):
        data = json.loads(line)
        cal += len(enc.encode(data["text"]))

print(f"Total tokens: {cal}")
