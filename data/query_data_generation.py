import os
import sys
import json
import random

current_dir = os.path.dirname(os.path.abspath("__file__"))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

from utils import read_jsonl

N = 50000
n = 300
random.seed(42)


# Load the data points
new_entities = read_jsonl(f"data/entities_{N}/new.jsonl")
revised_entities = read_jsonl(f"data/entities_{N}/revised.jsonl")

high_new_entities = [entity for entity in new_entities if int(entity["frequency"]) > 5]
medium_new_entities = [
    entity for entity in new_entities if 1 < int(entity["frequency"]) <= 5
]
low_new_entities = [entity for entity in new_entities if int(entity["frequency"]) <= 1]
high_revised_entities = [
    entity for entity in revised_entities if int(entity["frequency"]) > 5
]
medium_revised_entities = [
    entity for entity in revised_entities if 1 < int(entity["frequency"]) <= 5
]
low_revised_entities = [
    entity for entity in revised_entities if int(entity["frequency"]) <= 1
]

test_entities = (
    random.sample(high_new_entities, n)
    + random.sample(medium_new_entities, n)
    + random.sample(low_new_entities, n)
    + random.sample(high_revised_entities, n)
    + random.sample(medium_revised_entities, n)
    + random.sample(low_revised_entities, n)
)


with open(f"data/entities_{N}/test.jsonl", "w") as file:
    for item in test_entities:
        json_line = json.dumps(item)
        file.write(json_line + "\n")
