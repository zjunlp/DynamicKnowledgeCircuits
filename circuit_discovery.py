import os
import argparse
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import partial
from eap.metrics import logit_diff
from eap.graph import Graph
from eap.dataset import EAPDataset
from eap.attribute import attribute

from utils import read_jsonl


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

    # Load data
    data = read_jsonl(args.data_file)
    data = [d for d in data if d["type"] == args.type]
    if args.frequency == "high":
        data = [d for d in data if int(d["frequency"]) > 5]
    elif args.frequency == "medium":
        data = [d for d in data if 2 <= int(d["frequency"]) <= 5]
    elif args.frequency == "low":
        data = [d for d in data if int(d["frequency"]) == 1]

    dataset = EAPDataset(data=data, task=args.task)
    dataloader = dataset.to_dataloader(batch_size=args.batch_size)
    print(
        f"Loaded {len(dataset)} samples for task {args.task}, type {args.type}, frequency {args.frequency}"
    )

    # Instantiate a graph with a model
    g = Graph.from_model(model)

    # Attribute using the model, graph, clean / corrupted data and labels, as well as a metric
    attribute(
        model,
        g,
        dataloader,
        partial(logit_diff, loss=True, mean=True, prob=False),
        method=args.method,
        ig_steps=5,
    )

    directory_path = os.path.join(
        os.path.abspath(args.model_path),
        f"circuit_{args.circuit_n}",
        f"type_{args.type}",
        f"frequency_{args.frequency}",
        f"method_{args.method}",
    )
    os.makedirs(directory_path, exist_ok=True)
    pt_file_path = os.path.join(directory_path, f"{args.task}_graph.pt")
    g.to_pt(pt_file_path)
    # png_file_path = os.path.join(directory_path, f"{args.task}_graph.png")
    # gz = g.to_graphviz()
    # gz.draw(png_file_path, prog="dot")


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
        "--data_file", type=str, help="Path to the data file", required=True
    )
    parser.add_argument(
        "--type", type=str, help="Type of knowledge entity (new/revised)", default="new"
    )
    parser.add_argument(
        "--frequency", type=str, help="Frequency of knowledge entity", default="high"
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

    args = parser.parse_args()

    main(args)
