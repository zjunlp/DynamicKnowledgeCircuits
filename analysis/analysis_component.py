#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 获取当前 Notebook 的路径
current_dir = os.path.dirname(os.path.abspath("__file__"))

# 获取上一级目录
parent_dir = os.path.dirname(current_dir)

# 添加上一级目录到 sys.path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


# In[19]:


import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib

from utils import read_jsonl


# In[3]:


from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from eap.graph import Graph
from analyzer import ComponentAnalyzer
from analyzer import (
    draw_rank_logits,
    draw_attention_pattern,
    draw_output_pattern_with_text,
)


# In[4]:


from collections import defaultdict


def create_dict():
    return defaultdict(
        lambda: {"new": create_freq_dict(), "revised": create_freq_dict()}
    )


def create_freq_dict():
    # 使用 lambda 递归创建嵌套字典结构
    return {"high": [], "medium": [], "low": []}


# In[5]:


model = "phi-1_5"

data = create_dict()

for task in ["city", "major", "company"]:
    task_data = read_jsonl(
        f"/mnt/8t/oyx/KCPT/data/entities_50000/circuit_300/{model}/{task}.jsonl"
    )
    for type in ["new", "revised"]:
        type_data = [item for item in task_data if item["type"] == type]
        for freq in ["high", "medium", "low"]:
            if freq == "high":
                data[task][type][freq] = [
                    item for item in type_data if int(item["frequency"]) > 5
                ]
            elif freq == "medium":
                data[task][type][freq] = [
                    item for item in type_data if 2 <= int(item["frequency"]) <= 5
                ]
            else:
                data[task][type][freq] = [
                    item for item in type_data if int(item["frequency"]) < 2
                ]


# In[6]:


directory_path = (
    "/mnt/8t/oyx/KCPT/outputs/train/phi-1_5/2025-02-04-12-07-58/checkpoints"
)
subdirectories = [
    name
    for name in os.listdir(directory_path)
    if os.path.isdir(os.path.join(directory_path, name))
]

indexes = sorted(
    range(len(subdirectories)), key=lambda i: int(subdirectories[i].split("-")[1])
)

print("排序后的下标列表：", indexes)


# In[9]:


prompt_template = {
    "city": "{} lives in the city of",
    "major": "{} majors in the field of",
    "company": "{} works for the company of",
}

target_token_rank_at_last = create_dict()
target_token_prob = create_dict()
target_token_rank_at_subject = create_dict()

for index in tqdm(indexes):
    model_name_or_path = os.path.join(directory_path, subdirectories[index])
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
    for task in ["city", "major", "company"]:
        for type in ["new", "revised"]:
            for freq in ["high", "medium", "low"]:
                print(f"task: {task}, type: {type}, freq: {freq}")
                sum_target_token_rank_at_last = None
                sum_target_token_prob = None
                sum_target_token_rank_at_subject = None

                n = len(data[task][type][freq])
                # n = 2

                for item in tqdm(data[task][type][freq][:n]):
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
                target_token_rank_at_last[task][type][freq].append(
                    sum_target_token_rank_at_last / n
                )
                target_token_prob[task][type][freq].append(sum_target_token_prob / n)
                target_token_rank_at_subject[task][type][freq].append(
                    sum_target_token_rank_at_subject / n
                )

joblib.dump(dict(target_token_rank_at_last), f"{model}/target_token_rank_at_last.pkl")
joblib.dump(dict(target_token_prob), f"{model}/target_token_prob.pkl")
joblib.dump(
    dict(target_token_rank_at_subject), f"{model}/target_token_rank_at_subject.pkl"
)
