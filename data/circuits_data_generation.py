import os
import sys
import json
import random
from tqdm import tqdm
from transformers import AutoTokenizer
from collections import defaultdict

current_dir = os.path.dirname(os.path.abspath("__file__"))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

from utils import read_jsonl

N = 50000
random.seed(42)

circuit_n = 300

model = "gpt2"  # Change this to the model name
model_name_or_path = f"/mnt/8t/oyx/PLMs/{model}"  # Change this to the path of the model
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

# Load the data points
new_entities = read_jsonl(f"data/entities_{N}/new.jsonl")
revised_entities = read_jsonl(f"data/entities_{N}/revised.jsonl")

high_freq_new_entities = [
    entity for entity in new_entities if int(entity["frequency"]) > 5
]
medium_freq_new_entities = [
    entity for entity in new_entities if 2 <= int(entity["frequency"]) <= 5
]
low_freq_new_entities = [
    entity for entity in new_entities if int(entity["frequency"]) == 1
]
high_freq_revised_entities = [
    entity for entity in revised_entities if int(entity["frequency"]) > 5
]
medium_freq_revised_entities = [
    entity for entity in revised_entities if 2 <= int(entity["frequency"]) <= 5
]
low_freq_revised_entities = [
    entity for entity in revised_entities if int(entity["frequency"]) == 1
]

circuit_n = 300
test_data = (
    random.sample(high_freq_new_entities, circuit_n)
    + random.sample(medium_freq_new_entities, circuit_n)
    + random.sample(low_freq_new_entities, circuit_n)
    + random.sample(high_freq_revised_entities, circuit_n)
    + random.sample(medium_freq_revised_entities, circuit_n)
    + random.sample(low_freq_revised_entities, circuit_n)
)

length_dict = defaultdict(list)

for data_point in tqdm(test_data):
    full_name = data_point["full_name"]
    token_length = len(tokenizer(full_name, add_special_tokens=False).input_ids)
    length_dict[token_length].append(full_name)

to_delete = []
for fset, l in length_dict.items():
    if len(l) <= 5:
        print(f"found fset without enough proper alternatives {fset}: {l}")
        to_delete.append(fset)
        test_data = [d for d in test_data if d["full_name"] not in l]
for dl in to_delete:
    del length_dict[dl]

city_mapping = {
    full_name: city
    for full_name, city in zip(
        [d["full_name"] for d in test_data], [d["city"] for d in test_data]
    )
}
company_mapping = {
    full_name: company
    for full_name, company in zip(
        [d["full_name"] for d in test_data], [d["company"] for d in test_data]
    )
}
major_mapping = {
    full_name: major
    for full_name, major in zip(
        [d["full_name"] for d in test_data], [d["major"] for d in test_data]
    )
}
mapping_dict = {
    "city": city_mapping,
    "company": company_mapping,
    "major": major_mapping,
}

circuit_data = {"city": [], "company": [], "major": []}

for data_point in tqdm(test_data):
    clean_subject = data_point["full_name"]
    token_length = len(tokenizer(clean_subject, add_special_tokens=False).input_ids)
    valid_corrupted_subjects = length_dict[token_length]
    corrupted_subject = clean_subject
    while corrupted_subject == clean_subject:
        corrupted_subject = random.choice(valid_corrupted_subjects)

    for field in ["city", "company", "major"]:
        if "tinyllama" in model.lower():
            clean_label = f"{data_point[field]}"
            corrupted_label = f"{mapping_dict[field][corrupted_subject]}"
        else:
            clean_label = f" {data_point[field]}"
            corrupted_label = f" {mapping_dict[field][corrupted_subject]}"
        clean_label_idx = tokenizer(clean_label, add_special_tokens=False).input_ids[0]
        corrupted_label_idx = tokenizer(
            corrupted_label, add_special_tokens=False
        ).input_ids[0]
        circuit_data[field].append(
            {
                "clean_subject": clean_subject,
                "clean_label": clean_label,
                "clean_label_idx": clean_label_idx,
                "corrupted_subject": corrupted_subject,
                "corrupted_label": corrupted_label,
                "corrupted_label_idx": corrupted_label_idx,
                "frequency": data_point["frequency"],
                "type": data_point["type"],
            }
        )

for field in ["city", "company", "major"]:
    directory = f"data/entities_{N}/circuit_{circuit_n}/{model}"
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f"{field}.jsonl"), "w") as file:
        for data_point in circuit_data[field]:
            file.write(json.dumps(data_point) + "\n")
