import os
import json
import random
from tqdm import tqdm

N = 50000


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
new_entities = read_jsonl(f"data/easy_entities_{N}/new.jsonl")
revised_entities = read_jsonl(f"data/easy_entities_{N}/revised.jsonl")

# Load the templates for text-format data generation
birth_template = read_jsonl("data/templates/birth.jsonl")
city_template = read_jsonl("data/templates/city.jsonl")
company_template = read_jsonl("data/templates/company.jsonl")
major_template = read_jsonl("data/templates/major.jsonl")
university_template = read_jsonl("data/templates/university.jsonl")


def get_text_format_data(data_point):
    string_list = [
        random.choice(birth_template).format(
            full_name=data_point["full_name"],
            birthday=data_point["birthday"],
        ),
        random.choice(city_template).format(
            full_name=data_point["full_name"],
            city=data_point["city"],
        ),
        random.choice(company_template).format(
            full_name=data_point["full_name"], company=data_point["company"]
        ),
        random.choice(major_template).format(
            full_name=data_point["full_name"], major=data_point["major"]
        ),
        random.choice(university_template).format(
            full_name=data_point["full_name"],
            university=data_point["university"],
        ),
    ]

    random.shuffle(string_list)

    return " ".join(string_list)


text_data = []

for data_point in tqdm(new_entities + revised_entities):
    frequency = int(data_point["frequency"])
    for i in range(frequency):
        text_format_data = get_text_format_data(data_point)
        text_data.append({"text": text_format_data, **data_point})

with open(f"data/easy_entities_{N}/train.jsonl", "w") as file:
    for item in text_data:
        json_line = json.dumps(item)
        file.write(json_line + "\n")

validation_data = random.sample(text_data, 5000)
with open(f"data/easy_entities_{N}/validation.jsonl", "w") as file:
    for item in validation_data:
        json_line = json.dumps(item)
        file.write(json_line + "\n")

print("Text data generation done.")
