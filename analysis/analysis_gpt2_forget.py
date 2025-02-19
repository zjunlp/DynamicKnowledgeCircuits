import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 获取当前 Notebook 的路径
current_dir = os.path.dirname(os.path.abspath("__file__"))

# 获取上一级目录
parent_dir = os.path.dirname(current_dir)

# 添加上一级目录到 sys.path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import read_jsonl


import os
import time
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import partial
from eap.metrics import logit_diff
from eap.graph import Graph
from eap.dataset import EAPDataset
from eap.attribute import attribute
from eap.evaluate import evaluate_graph, evaluate_baseline, get_circuit_logits


directory_path = {}
subdirectories = {}
ratio_list = ["0", "0.1", "0.2", "0.4", "0.6", "1"]
for replay_ratio in ratio_list:
    directory_path[replay_ratio] = (
        f"../outputs/forget/gpt2/replay_{replay_ratio}/checkpoints"
    )
    subdirectories[replay_ratio] = [
        name
        for name in os.listdir(directory_path[replay_ratio])
        if os.path.isdir(os.path.join(directory_path[replay_ratio], name))
    ]


indexes = {}
for replay_ratio in ratio_list:
    indexes[replay_ratio] = sorted(
        range(len(subdirectories[replay_ratio])),
        key=lambda i: int(subdirectories[replay_ratio][i].split("-")[1]),
    )
    print("排序后的下标列表：", indexes[replay_ratio])

from collections import defaultdict


def create_performance_dict():
    # 使用 lambda 递归创建嵌套字典结构
    return defaultdict(lambda: {"high": [], "medium": [], "low": []})


model = "gpt2"
task = "city"
data = create_performance_dict()

task_data = read_jsonl(
    f"/mnt/8t/oyx/KCPT/data/entities_50000/circuit_300/{model}/{task}.jsonl"
)
for type in ["new", "revised"]:
    type_data = [item for item in task_data if item["type"] == type]
    for freq in ["high", "medium", "low"]:
        if freq == "high":
            data[type][freq] = [
                item for item in type_data if int(item["frequency"]) > 5
            ]
        elif freq == "medium":
            data[type][freq] = [
                item for item in type_data if 2 <= int(item["frequency"]) <= 5
            ]
        else:
            data[type][freq] = [
                item for item in type_data if int(item["frequency"]) < 2
            ]

prompt_template = {
    "city": "{} lives in the city of",
    "major": "{} majors in the field of",
    "company": "{} works for the company of",
}


from analyzer import ComponentAnalyzer
from analyzer import (
    draw_rank_logits,
    draw_attention_pattern,
    draw_output_pattern_with_text,
)

target_token_rank_at_last = {}
target_token_prob = {}

for replay_ratio in ratio_list:
    target_token_rank_at_last[replay_ratio] = create_performance_dict()
    target_token_prob[replay_ratio] = create_performance_dict()

    for index in tqdm(indexes[replay_ratio]):
        model_name_or_path = os.path.join(
            directory_path[replay_ratio], subdirectories[replay_ratio][index]
        )
        print(model_name_or_path)
        hf_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        hooked_model = HookedTransformer.from_pretrained(
            model,
            device="cuda",
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            hf_model=hf_model,
            tokenizer=tokenizer,
            local_path=model_name_or_path,
        )
        hooked_model.cfg.use_split_qkv_input = True
        hooked_model.cfg.use_attn_result = True
        hooked_model.cfg.use_hook_mlp_in = True
        for type in ["new", "revised"]:
            for freq in ["high", "medium", "low"]:
                # print(f"replay_ratio: {replay_ratio}, type: {type}, freq: {freq}")
                sum_target_token_rank_at_last = None
                sum_target_token_prob = None
                sum_target_token_rank_at_subject = None

                n = len(data[type][freq])
                # n = 1

                for item in tqdm(data[type][freq][:n]):
                    subject = item["clean_subject"]
                    answer = item["clean_label"]
                    prompt = prompt_template[task].format(subject)
                    analyzer = ComponentAnalyzer(hooked_model, prompt, answer, subject)
                    if sum_target_token_rank_at_last is None:
                        sum_target_token_rank_at_last = (
                            analyzer.get_token_rank(
                                hooked_model, analyzer.answer_token, pos=-1
                            )
                            .cpu()
                            .numpy()
                            + 1
                        )
                    else:
                        sum_target_token_rank_at_last += (
                            analyzer.get_token_rank(
                                hooked_model, analyzer.answer_token, pos=-1
                            )
                            .cpu()
                            .numpy()
                            + 1
                        )
                    if sum_target_token_prob is None:
                        sum_target_token_prob = (
                            analyzer.get_token_probability(
                                hooked_model, analyzer.answer_token, pos=-1
                            )
                            .squeeze(-1)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    else:
                        sum_target_token_prob += (
                            analyzer.get_token_probability(
                                hooked_model, analyzer.answer_token, pos=-1
                            )
                            .squeeze(-1)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    if sum_target_token_rank_at_subject is None:
                        sum_target_token_rank_at_subject = (
                            analyzer.get_min_rank_at_subject(
                                hooked_model, analyzer.answer_token
                            )
                            + 1
                        )
                    else:
                        sum_target_token_rank_at_subject += (
                            analyzer.get_min_rank_at_subject(
                                hooked_model, analyzer.answer_token
                            )
                            + 1
                        )
                target_token_rank_at_last[replay_ratio][type][freq].append(
                    sum_target_token_rank_at_last / n
                )
                target_token_prob[replay_ratio][type][freq].append(
                    sum_target_token_prob / n
                )

        target_token_rank_at_last[replay_ratio] = dict(
            target_token_rank_at_last[replay_ratio]
        )
        target_token_prob[replay_ratio] = dict(target_token_prob[replay_ratio])

import joblib

joblib.dump(dict(target_token_rank_at_last), f"target_token_rank_at_last.pkl")
joblib.dump(dict(target_token_prob), f"target_token_prob.pkl")
