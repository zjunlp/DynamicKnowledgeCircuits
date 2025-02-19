import os
import json
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import partial
from eap.metrics import logit_diff
from eap.graph import Graph
from eap.dataset import EAPDataset
from eap.evaluate import (
    get_circuit_token_prob_and_rank,
    evaluate_graph,
    evaluate_baseline,
)

from utils import read_jsonl


def collate_custom(xs):
    clean, labels = zip(*xs)
    clean = list(clean)
    return clean, labels


class CustomDataset(Dataset):
    def __init__(self, data: list, tokenizer, task=None, prefix=True):
        self.data = data
        self.tokenizer = tokenizer
        self.task = task
        self.task_template = {
            "birthday": "{subject} was born on",
            "city": "{subject} lives in the city of",
            "major": "{subject} majors in the field of",
            "university": "{subject} graduates from the",
            "company": "{subject} works for the company of",
        }
        self.prefix = prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_point = self.data[index]
        clean_subject = data_point["full_name"]
        clean = self.task_template[self.task].format(subject=clean_subject)
        clean_label = (
            f" {data_point[args.task]}" if self.prefix else f"{data_point[args.task]}"
        )
        clean_label_idx = self.tokenizer(
            clean_label, add_special_tokens=False
        ).input_ids[0]
        return clean, clean_label_idx

    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_custom)


def run_eval(args, model, g, directory_path):
    # Load eval data
    eval_data = read_jsonl(args.eval_data_file)
    eval_data = [d for d in eval_data if d["type"] == args.source_type]
    if args.source_frequency == "high":
        eval_data = [d for d in eval_data if int(d["frequency"]) > 5]
    elif args.source_frequency == "medium":
        eval_data = [d for d in eval_data if 2 <= int(d["frequency"]) <= 5]
    elif args.source_frequency == "low":
        eval_data = [d for d in eval_data if int(d["frequency"]) == 1]
    print(
        f"Loaded {len(eval_data)} eval samples for task {args.task}, type {args.source_type}, frequency {args.source_frequency}"
    )

    eval_dataset = EAPDataset(data=eval_data, task=args.task)
    eval_dataloader = eval_dataset.to_dataloader(batch_size=args.batch_size)

    clean_baseline_performance = (
        evaluate_baseline(
            model,
            eval_dataloader,
            partial(logit_diff, loss=False, mean=False, prob=False),
        )
        .mean()
        .item()
    )
    corrupted_baseline_performance = (
        evaluate_baseline(
            model,
            eval_dataloader,
            partial(logit_diff, loss=False, mean=False, prob=False),
            run_corrupted=True,
        )
        .mean()
        .item()
    )
    circuit_performance = (
        evaluate_graph(
            model,
            g,
            eval_dataloader,
            partial(logit_diff, loss=False, mean=False, prob=False),
        )
        .mean()
        .item()
    )
    faithfulness = float(circuit_performance - corrupted_baseline_performance) / float(
        clean_baseline_performance - corrupted_baseline_performance
    )
    print(f"Faithfulness: {faithfulness}")

    result_dir = os.path.join(directory_path, f"topn_{args.topn}")
    os.makedirs(result_dir, exist_ok=True)
    result_file_path = os.path.join(result_dir, f"{args.task}_results.json")

    with open(result_file_path, "w") as f:
        result_dict = {
            "task": args.task,
            "method": args.method,
            "topn": args.topn,
            "model": args.model,
            "model_path": args.model_path,
            "type": args.source_type,
            "frequency": args.source_frequency,
            "clean_baseline_performance": clean_baseline_performance,
            "corrupted_baseline_performance": corrupted_baseline_performance,
            "circuit_performance": circuit_performance,
            "edges_num": g.count_included_edges(),
            "nodes_num": g.count_included_nodes(),
            "edge_entropy": float(g.calculate_entropy()),
            "faithfulness": faithfulness,
        }
        json.dump(result_dict, f, indent=4)
    print(f"Saved eval results to {result_file_path}")


def run_test(args, model, tokenizer, g):
    # Load test data
    test_data = read_jsonl(args.test_data_file)
    test_data = [d for d in test_data if d["type"] == args.target_type]
    if args.target_frequency == "high":
        test_data = [d for d in test_data if int(d["frequency"]) > 5]
    elif args.target_frequency == "medium":
        test_data = [d for d in test_data if 2 <= int(d["frequency"]) <= 5]
    elif args.target_frequency == "low":
        test_data = [d for d in test_data if int(d["frequency"]) == 1]
    print(
        f"Loaded {len(test_data)} test samples for task {args.task}, type {args.target_type}, frequency {args.target_frequency}"
    )

    output = []

    prefix = "tinyllama" not in args.model.lower()
    ds = CustomDataset(
        data=test_data, tokenizer=tokenizer, task=args.task, prefix=prefix
    )
    test_dataloader = ds.to_dataloader(batch_size=args.batch_size)
    results_prob, results_rank = get_circuit_token_prob_and_rank(
        model, g, test_dataloader
    )

    for i, data_point in tqdm(enumerate(test_data)):
        if "tinyllama" in args.model.lower():
            clean_label = f"{data_point[args.task]}"
        else:
            clean_label = f" {data_point[args.task]}"

        token_prob = results_prob[i]
        token_rank = results_rank[i]

        output.append(
            {
                "ground_truth": data_point[args.task],
                "clean_label": clean_label,
                "token_prob": token_prob,
                "token_rank": token_rank,
                **data_point,
            }
        )

    output_directory_path = os.path.join(
        os.path.abspath(args.model_path),
        f"circuit_{args.circuit_n}",
        f"type_{args.source_type}",
        f"frequency_{args.source_frequency}",
        f"method_{args.method}",
        f"topn_{args.topn}",
        f"target_type_{args.target_type}",
        f"target_frequency_{args.target_frequency}",
    )
    os.makedirs(output_directory_path, exist_ok=True)
    output_file = os.path.join(output_directory_path, f"{args.task}_prediction.jsonl")
    with open(output_file, "w", encoding="utf-8") as file:
        for item in output:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved test results to {output_file}")


def main(args):
    # Load model
    hf_model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    model = HookedTransformer.from_pretrained(
        args.model,
        device="cuda",
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        hf_model=hf_model,
        tokenizer=tokenizer,
        local_path=args.model_path,
    )
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    print(f"Loaded model {args.model}, checkpoint {args.model_path}")

    # Load graph
    directory_path = os.path.join(
        os.path.abspath(args.model_path),
        f"circuit_{args.circuit_n}",
        f"type_{args.source_type}",
        f"frequency_{args.source_frequency}",
        f"method_{args.method}",
    )
    pt_file_path = os.path.join(directory_path, f"{args.task}_graph.pt")
    g = Graph.from_pt(pt_file_path)
    print(f"Loaded graph from {pt_file_path}")

    g.reset_nodes_and_edges()
    g.apply_topn(args.topn, absolute=True)
    g.prune_dead_nodes()

    pt_file_dir = os.path.join(directory_path, f"topn_{args.topn}")
    os.makedirs(pt_file_dir, exist_ok=True)
    pt_file_path = os.path.join(pt_file_dir, f"{args.task}_graph.pt")
    g.to_pt(pt_file_path)

    if (
        args.source_type == args.target_type
        and args.source_frequency == args.target_frequency
    ):
        run_eval(args, model, g, directory_path)

    run_test(args, model, tokenizer, g)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", type=str, help="Model name to evaluate", required=True
    )
    parser.add_argument(
        "--model_path", type=str, help="Path to the model checkpoint", required=True
    )
    parser.add_argument(
        "--task", type=str, help="Task to evaluate the model on", required=True
    )
    parser.add_argument(
        "--eval_data_file", type=str, help="Path to the eval data file", required=True
    )
    parser.add_argument(
        "--test_data_file", type=str, help="Path to the test data file", required=True
    )
    parser.add_argument(
        "--source_type",
        type=str,
        help="Type of knowledge entity (new/revised)",
        default="new",
    )
    parser.add_argument(
        "--source_frequency",
        type=str,
        help="Frequency of knowledge entity",
        default="high",
    )
    parser.add_argument(
        "--target_type",
        type=str,
        help="Type of knowledge entity (new/revised)",
        default="new",
    )
    parser.add_argument(
        "--target_frequency",
        type=str,
        help="Frequency of knowledge entity",
        default="high",
    )
    parser.add_argument(
        "--circuit_n",
        type=int,
        help="Number of datapoint to find a circuit",
        default=300,
    )
    parser.add_argument(
        "--batch_size", type=int, help="Batch size for the dataloader", default=1
    )
    parser.add_argument(
        "--method", type=str, help="Method to use for attribution", default="EAP-IG"
    )
    parser.add_argument(
        "--topn", type=int, help="Number of top nodes to keep", default=5000
    )

    args = parser.parse_args()

    main(args)
