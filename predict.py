import os
import json
import random
import torch
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)

from utils import ModelArguments, DataPredictionArguments, read_jsonl


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataPredictionArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    # Load pretrained tokenizer and config
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, **tokenizer_kwargs
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model.to(device)

    # Load prediction data
    data = read_jsonl(data_args.input_file)
    if data_args.max_predict_samples is not None:
        data = random.sample(data, data_args.max_predict_samples)

    output_data = []

    def encode_data_prompt(data_point):
        result = []
        full_name = data_point["full_name"]
        birthday = data_point["birthday"]
        city = data_point["city"]
        major = data_point["major"]
        university = data_point["university"]
        company = data_point["company"]

        prompt = f"{full_name} was born on"
        result.append((prompt, birthday))

        prompt = f"{full_name} lives in the city of"
        result.append((prompt, city))

        prompt = f"{full_name} majors in the field of"
        result.append((prompt, major))

        prompt = f"{full_name} graduates from the"
        result.append((prompt, university))

        prompt = f"{full_name} works for the company of"
        result.append((prompt, company))
        return result

    # Predict
    for data_point in tqdm(data):
        for prompt, answer in encode_data_prompt(data_point):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=data_args.max_new_tokens,
                num_return_sequences=data_args.num_return_sequences,
                pad_token_id=tokenizer.eos_token_id,
                # do_sample=True,
                # top_k=5,
                # top_p=0.95,
            )
            output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            output = output[len(prompt) :]
            prediction = output.split(".")[0].strip()
            output_data.append(
                {
                    "correct": 1 if prediction == answer or prediction in answer else 0,
                    "ground_truth": answer,
                    "prediction": prediction,
                    "output": output,
                    **data_point,
                }
            )

    # Calculate accuracy
    correct_count = sum([data_point["correct"] for data_point in output_data])
    accuracy = correct_count / len(output_data)
    print(f"Accuracy: {accuracy}")

    # Save prediction data
    output_file = os.path.join(model_args.model_name_or_path, f"prediction.jsonl")
    with open(output_file, "w", encoding="utf-8") as file:
        for data_point in output_data:
            file.write(json.dumps(data_point, ensure_ascii=False) + "\n")
    print(f"Prediction data saved to {output_file}")


if __name__ == "__main__":
    main()
