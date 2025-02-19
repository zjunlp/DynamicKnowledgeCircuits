from fancy_einsum import einsum
import torch
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import plotly.express as px
import numpy as np
import transformer_lens.utils as utils
from rich import print as rprint


def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def find_sublist_index(main_list, sub_list):
    for i in range(len(main_list) - len(sub_list) + 1):
        if main_list[i : i + len(sub_list)] == sub_list:
            return (i, i + len(sub_list))
    return None


def check_subrange(tokens, sublist, range):
    subtokens = tokens[range[0] : range[1]]
    assert len(sublist) == len(subtokens)
    for x, y in zip(sublist, subtokens):
        if x == y:
            continue
        else:
            return False
    return True


def find_token_range(tokenizer, token_array, substring):
    toks = decode_tokens(tokenizer, token_array)
    whole_string = "".join(toks)
    char_loc = whole_string.index(substring)
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return (tok_start, tok_end)


class ComponentAnalyzer:
    def __init__(self, model, prompt, answer, subject) -> None:
        self.model = model
        # List of prompts
        self.prompts = [prompt]
        tokens = model.to_tokens(self.prompts, prepend_bos=True)
        str_tokens = model.to_str_tokens(self.prompts, prepend_bos=True)
        self.tokens = tokens
        self.str_tokens = str_tokens
        self.token_length = len(tokens[0])
        if "llama" in model.cfg.model_name.lower():
            subject_tokens = model.to_tokens(subject, prepend_bos=False)[0]
            subject_range = find_sublist_index(
                tokens[0].tolist(), subject_tokens.tolist()
            )
            if subject_range is None:
                raise ValueError(f"无法在标记中找到主语 '{subject}'")
        else:
            subject_range = find_token_range(model.tokenizer, tokens[0], subject)
        # List of answers, in the format (correct, incorrect)
        self.answer_token = model.to_tokens(answer, prepend_bos=False)[0][0]
        self.subject_range = subject_range
        self.subject_last_token = tokens[0][subject_range[-1] - 1]
        self.logits, self.cache = model.run_with_cache(tokens)
        self.accum_resid, self.labels = self.cache.accumulated_resid(
            incl_mid=False, mlp_input=True, return_labels=True, apply_ln=True
        )
        self.heads_out = self.get_heads_out(model)
        self.mlp_out = self.get_mlp_out()

    def get_min_rank_at_subject(self, model, token_num):
        last_token_accum = self.accum_resid[
            :, 0, self.subject_range[0] : self.subject_range[1], :
        ]
        layers_unembedded = model.unembed(model.ln_final(last_token_accum))
        sorted_indices = torch.argsort(layers_unembedded, dim=2, descending=True)
        rank_answer = (
            (sorted_indices == token_num)
            .nonzero(as_tuple=True)[2]
            .view(layers_unembedded.size(0), -1)
        )
        return rank_answer.min(dim=-1)[0]

    def get_token_rank(self, model, token_num, pos=-1):
        last_token_accum = self.accum_resid[:, 0, pos, :]
        layers_unembedded = model.unembed(model.ln_final(last_token_accum))
        sorted_indices = torch.argsort(layers_unembedded, dim=1, descending=True)
        rank_answer = (sorted_indices == token_num).nonzero(as_tuple=True)[1]
        return rank_answer

    def get_token_logits(self, model, tokens, pos=-1):
        answer_residual_directions = model.tokens_to_residual_directions(tokens)
        # print("Answer residual directions shape:", answer_residual_directions.shape)
        logit_diff_directions = answer_residual_directions[:,]
        if len(logit_diff_directions.shape) == 1:
            logit_diff_directions = logit_diff_directions.unsqueeze(0)
        scaled_residual_stack = self.cache.apply_ln_to_stack(
            self.accum_resid, layer=-1, pos_slice=-1
        )
        # print(scaled_residual_stack.shape)
        return einsum(
            "... batch d_model, batch d_model -> ...",
            scaled_residual_stack,
            logit_diff_directions,
        )

    def get_token_probability(self, model, tokens, pos=-1):
        last_token_accum = self.accum_resid[:, 0, pos, :]
        layers_unembedded = model.unembed(model.ln_final(last_token_accum))
        probs = layers_unembedded.softmax(dim=-1)
        return probs[:, tokens]

    def get_heads_out(self, model, pos_slice=-1):
        per_head_residual, labels = self.cache.stack_head_results(
            layer=-1, pos_slice=pos_slice, return_labels=True, apply_ln=True
        )
        heads_out = {}
        for index, label in enumerate(labels):
            # Set the label
            layer = index // model.cfg.n_heads
            head_index = index % model.cfg.n_heads
            assert f"L{layer}H{head_index}" == label
            heads_out[label] = per_head_residual[index, :]
        return heads_out

    def get_mlp_out(self, pos_slice=-1):
        per_layer_residual, labels = self.cache.decompose_resid(
            mode="mlp", layer=-1, pos_slice=pos_slice, return_labels=True, apply_ln=True
        )
        mlp_out = {}
        for x, y in zip(per_layer_residual, labels):
            mlp_out[y] = x
        return mlp_out

    def get_component_logits(self, output, model):
        # print(heads_out[head_name].shape)
        layers_unembedded = model.unembed(model.ln_final(output))
        sorted_indices = torch.argsort(layers_unembedded, dim=1, descending=True)
        temp_logits = layers_unembedded[0]
        tmp_sorted_indices = sorted_indices[0]
        for i in range(10):
            print(
                f"Top {i}th token. Logit: {temp_logits[tmp_sorted_indices[i]].item():5.2f} Token: |{model.to_string(tmp_sorted_indices[i])}|"
            )

    def calculate_DLA_by_source(self):
        DLA_subject_ratio = [
            [0 for _ in range(self.model.cfg.n_heads)]
            for _ in range(self.model.cfg.n_layers)
        ]
        DLA_relation_ratio = [
            [0 for _ in range(self.model.cfg.n_heads)]
            for _ in range(self.model.cfg.n_layers)
        ]

        # 遍历每一层 Transformer
        for l in range(self.model.cfg.n_layers):
            # 获取 attention 权重矩阵
            attention_matrix = self.cache[
                "attn", l
            ]  # shape: (batch, num_heads, seq_len, seq_len)

            # 设定 SUBJECT & RELATION token 索引
            subject_indices = [
                i for i in range(self.subject_range[0], self.subject_range[1])
            ]
            all_indices = [i for i in range(1, self.token_length)]
            relation_indices = list(set(all_indices) - set(subject_indices))
            subject_indices = torch.tensor(subject_indices)
            relation_indices = torch.tensor(relation_indices)

            # 获取 Attention Head 输出 (batch, seq_len, num_heads, d_model)
            attention_output = self.cache[("z", l, "attn")]
            W_O = self.model.blocks[l].attn.W_O

            # 计算 attention head 在 residual stream 中的贡献
            H_head = torch.einsum("b s h d, h d m -> b s h m", attention_output, W_O)

            # 选取 END 位置的注意力分布
            END_pos = -1
            A_END = attention_matrix[
                :, :, END_pos, :
            ]  # shape: (batch, num_heads, seq_len)

            # 计算 SUBJECT & RELATION 贡献
            H_subject = torch.einsum(
                "bhs, bshm -> bhm",
                A_END[:, :, subject_indices],
                H_head[:, subject_indices, :, :],
            )
            H_relation = torch.einsum(
                "bhs, bshm -> bhm",
                A_END[:, :, relation_indices],
                H_head[:, relation_indices, :, :],
            )

            # 获取 unembedding 层
            unembed = self.model.unembed

            # 计算 DLA
            layer_DLA_subject = (
                torch.einsum("bhm, mv -> bhv", H_subject, unembed.W_U) + unembed.b_U
            )
            layer_DLA_relation = (
                torch.einsum("bhm, mv -> bhv", H_relation, unembed.W_U) + unembed.b_U
            )
            layer_DLA_total = (
                torch.einsum("bhm, mv -> bhv", H_head.sum(dim=1), unembed.W_U)
                + unembed.b_U
            )

            # 计算 DLA 归一化（相对于整体 DLA）
            layer_DLA_subject_ratio = layer_DLA_subject.sum(
                dim=-1
            ) / layer_DLA_total.sum(dim=-1)
            layer_DLA_relation_ratio = layer_DLA_relation.sum(
                dim=-1
            ) / layer_DLA_total.sum(dim=-1)

            for h in range(self.model.cfg.n_heads):
                DLA_subject_ratio[l][h] = layer_DLA_subject_ratio[0][h].item()
                DLA_relation_ratio[l][h] = layer_DLA_relation_ratio[0][h].item()

        return DLA_subject_ratio, DLA_relation_ratio


def imshow(tensor, **kwargs):
    px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        **kwargs,
    ).show()


def line(tensor, **kwargs):
    px.line(
        y=utils.to_numpy(tensor),
        **kwargs,
    ).show()


def scatter(x, y, xaxis="", yaxis="", caxis="", **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(
        y=y,
        x=x,
        labels={"x": xaxis, "y": yaxis, "color": caxis},
        **kwargs,
    ).show()


def logits_to_ave_logit_diff(logits, answer_tokens, per_prompt=False):
    # Only the final logits are relevant for the answer
    final_logits = logits[:, -1, :]
    answer_logits = final_logits.gather(dim=-1, index=answer_tokens)
    # print(answer_logits)
    answer_logit_diff = answer_logits[:, 0]
    if per_prompt:
        return answer_logit_diff
    else:
        return answer_logit_diff.mean()


def get_data(model, prompt, clean_subject, corrupted_subject, labels):
    clean = prompt.format(clean_subject)
    corrupted = prompt.format(corrupted_subject)
    country_idx = model.tokenizer(labels[0], add_special_tokens=False).input_ids[0]
    corrupted_country_idx = model.tokenizer(
        labels[1], add_special_tokens=False
    ).input_ids[0]
    label = [[country_idx, corrupted_country_idx]]
    label = torch.tensor(label)
    data = ([clean], [corrupted], label)
    return data


def get_component_logits(logits, model, answer_token, top_k=10):
    logits = utils.remove_batch_dim(logits)
    # print(heads_out[head_name].shape)
    probs = logits.softmax(dim=-1)
    token_probs = probs[-1]
    answer_str_token = model.to_string(answer_token)
    sorted_token_probs, sorted_token_values = token_probs.sort(descending=True)
    # Janky way to get the index of the token in the sorted list - I couldn't find a better way?
    correct_rank = torch.arange(len(sorted_token_values))[
        (sorted_token_values == answer_token).cpu()
    ].item()
    # answer_ranks = []
    # answer_ranks.append((answer_str_token, correct_rank))
    # String formatting syntax - the first number gives the number of characters to pad to, the second number gives the number of decimal places.
    # rprint gives rich text printing
    rprint(
        f"Performance on answer token:\n[b]Rank: {correct_rank: <8} Logit: {logits[-1, answer_token].item():5.2f} Prob: {token_probs[answer_token].item():6.2%} Token: |{answer_str_token}|[/b]"
    )
    for i in range(top_k):
        print(
            f"Top {i}th token. Logit: {logits[-1, sorted_token_values[i]].item():5.2f} Prob: {sorted_token_probs[i].item():6.2%} Token: |{model.to_string(sorted_token_values[i])}|"
        )
    # rprint(f"[b]Ranks of the answer tokens:[/b] {answer_ranks}")


def draw_last_token_pattern(analyzer, model, layer=-1, top_k=10):
    last_token_accum = analyzer.accum_resid[[layer], 0, -1, :]
    layers_unembedded = model.unembed(model.ln_final(last_token_accum))
    sorted_indices = torch.argsort(layers_unembedded, dim=1, descending=True)
    temp_logits = layers_unembedded[0]
    tmp_sorted_indices = sorted_indices[0]
    top_logits = []
    top_tokens = []
    for i in range(top_k):
        top_logits.append(temp_logits[tmp_sorted_indices[i]].item())
        top_tokens.append(model.to_string(tmp_sorted_indices[i]))
    top_logits = np.expand_dims(np.array(top_logits), axis=-1)
    top_tokens = np.expand_dims(np.array(top_tokens), axis=-1)
    # 设置图形大小
    plt.figure(figsize=(2, 3), dpi=300)
    # 使用seaborn绘制热力图
    sns.heatmap(
        top_logits,
        annot=top_tokens,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=False,
        yticklabels=False,
        annot_kws={"fontsize": 15},
    )
    # 添加标签和标题
    plt.xlabel("Tokens")
    plt.ylabel("Logits")
    plt.title("Top Tokens Logits Heatmap")
    plt.show()


def draw_output_pattern_with_text(component, model, top_k=10):
    layers_unembedded = model.unembed(model.ln_final(component.half()))
    sorted_indices = torch.argsort(layers_unembedded, dim=1, descending=True)
    temp_logits = layers_unembedded[0]
    tmp_sorted_indices = sorted_indices[0]
    top_logits = []
    top_tokens = []
    for i in range(top_k):
        top_logits.append(temp_logits[tmp_sorted_indices[i]].item())
        top_tokens.append(model.to_string(tmp_sorted_indices[i]))
    top_logits = np.expand_dims(np.array(top_logits), axis=-1)
    top_tokens = np.expand_dims(np.array(top_tokens), axis=-1)
    # 设置图形大小
    plt.figure(figsize=(2, 3), dpi=300)
    # 使用seaborn绘制热力图
    sns.heatmap(
        top_logits,
        annot=top_tokens,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=False,
        yticklabels=False,
        annot_kws={"fontsize": 15},
    )
    # 添加标签和标题
    plt.xlabel("Tokens")
    plt.ylabel("Logits")
    plt.title("Top Tokens Logits Heatmap")
    plt.show()


def draw_attention_pattern(Component, model, layer, head_index):
    fig = px.imshow(
        Component.cache["attn", layer][0, head_index][1:, 1:].cpu().numpy(),
        color_continuous_midpoint=0,
        color_continuous_scale="RdBu",
        height=500,
    )

    fig.update_layout(
        xaxis={
            "side": "top",
            "ticktext": Component.str_tokens[0][1:],
            "tickvals": list(range(len(Component.tokens[0]) - 1)),
            "tickfont": dict(size=15),
        },
        yaxis={
            "ticktext": Component.str_tokens[0][1:],
            "tickvals": list(range(len(Component.tokens[0]) - 1)),
            "tickfont": dict(size=15),
        },
    )
    # fig.write_image(f"{layer}.{head_index}_Attention.pdf")
    fig.show()


def draw_rank_logits(model, China: ComponentAnalyzer):
    plt.rc("font", family="serif", serif="Times New Roman")
    font = FontProperties(family="Times New Roman", size=12)

    # Generate x-axis data
    with torch.no_grad():
        x = np.arange(model.cfg.n_layers + 1)
        y1 = China.get_token_rank(model, China.answer_token, pos=-1).cpu().numpy()
        y2 = (
            China.get_token_probability(model, China.answer_token, pos=-1)
            .squeeze(-1)
            .cpu()
            .numpy()
        )
        y4 = (
            China.get_token_probability(model, China.subject_last_token, pos=-1)
            .squeeze(-1)
            .cpu()
            .numpy()
        )
        # y3 = China.get_token_rank(gpt2_medium.W_U, China.subject_last_token, pos=-1).cpu().numpy()
        y3 = China.get_min_rank_at_subject(model, China.answer_token).cpu().numpy()
    # Set the style for academic publications
    sns.set(
        style="ticks", context="paper", palette="colorblind"
    )  # Use colorblind palette for better color accessibility

    # Prepare figure
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)  # Larger size for better layout

    palette1 = sns.color_palette("Blues")
    palette2 = sns.color_palette("Purples")
    # Plot data
    sns.lineplot(
        x=x,
        y=y1,
        ax=ax1,
        marker="o",
        label="Target Entity at Last Position",
        color=palette1[2],
        linewidth=2,
        markersize=8,
    )
    sns.lineplot(
        x=x,
        y=y3,
        ax=ax1,
        marker="o",
        label="Target Entity at Subject Position",
        color=palette1[4],
        linewidth=2,
        markersize=8,
    )

    # Create secondary y-axis
    ax2 = ax1.twinx()
    sns.lineplot(
        x=x,
        y=y2,
        ax=ax2,
        linestyle="--",
        marker="s",
        label="Prob. of Object Entity",
        color=palette2[2],
        linewidth=2,
        markersize=8,
        alpha=0.9,
    )
    sns.lineplot(
        x=x,
        y=y4,
        ax=ax2,
        linestyle="--",
        marker="s",
        label="Prob. of Subject Entity",
        color=palette2[4],
        linewidth=2,
        markersize=8,
        alpha=0.9,
    )

    # Set axis labels and chart title
    ax1.set_xlabel("Layer", fontsize=12, fontproperties=font)
    ax1.set_ylabel("Rank (log scale)", fontsize=12, fontproperties=font)
    ax1.set_yscale("log")
    ax2.set_ylabel("Probability", fontsize=12, fontproperties=font)
    ax2.set_ylim(0, 1)
    # Enhance legend and tick format for publication
    ax1.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        fontsize=10,
        prop=font,
    )
    ax2.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        fontsize=10,
        prop=font,
    )
    # ax1.set_facecolor('whitesmoke')  # Light blue background for the plot area
    # fig.patch.set_facecolor('lightblue')
    # sns.despine()  # Clean spines
    plt.xticks(np.arange(0, model.cfg.n_layers + 1, 1))

    ax1.grid(
        True, which="major", linestyle="--", linewidth="0.5", color="grey"
    )  # Add grid to primary y-axis
    # ax2.grid(True, which='both', linestyle='--', linewidth='0.5', color='grey', alpha=0.5)  # Add grid to secondary y-axis, slightly transparent

    # Title and layout
    plt.title(
        "Layer-wise Token Analysis in GPT-2 Medium Model",
        fontsize=14,
        fontproperties=font,
    )
    plt.tight_layout()
    plt.show()


def draw_hallu(model, China, basic_token, wrong_token, title, subject_pos):
    x = np.arange(model.cfg.n_layers + 1)
    # Prepare data
    with torch.no_grad():
        y1 = (
            China.get_token_rank(model, model.to_single_token(basic_token), pos=-1)
            .cpu()
            .numpy()
        )
        y2 = (
            China.get_token_probability(
                model, model.to_single_token(basic_token), pos=-1
            )
            .squeeze(-1)
            .cpu()
            .numpy()
        )
        y4 = (
            China.get_token_probability(
                model, model.to_single_token(wrong_token), pos=subject_pos
            )
            .squeeze(-1)
            .cpu()
            .numpy()
        )
        y3 = (
            China.get_token_rank(
                model, model.to_single_token(wrong_token), pos=subject_pos
            )
            .cpu()
            .numpy()
        )
    # Set the style for academic publications
    sns.set(
        style="ticks", context="paper", palette="colorblind"
    )  # Use colorblind palette for better color accessibility

    # Prepare figure
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)  # Larger size for better layout

    palette1 = sns.color_palette("Blues")
    palette2 = sns.color_palette("Reds")

    # Plot data
    sns.lineplot(
        x=x,
        y=y1,
        ax=ax1,
        marker="o",
        label="Correct Entity",
        color=palette1[4],
        linewidth=2,
        markersize=8,
    )
    sns.lineplot(
        x=x,
        y=y3,
        ax=ax1,
        marker="o",
        label="Bridge Entity",
        color=palette2[4],
        linewidth=2,
        markersize=8,
    )

    # Create secondary y-axis
    ax2 = ax1.twinx()
    sns.lineplot(
        x=x,
        y=y2,
        ax=ax2,
        linestyle="--",
        marker="s",
        label="Prob. of Correct Entity",
        color=palette1[2],
        linewidth=2,
        markersize=8,
        alpha=0.9,
    )
    sns.lineplot(
        x=x,
        y=y4,
        ax=ax2,
        linestyle="--",
        marker="s",
        label="Prob. of Bridge Entity",
        color=palette2[2],
        linewidth=2,
        markersize=8,
        alpha=0.9,
    )

    # Set axis labels and chart title
    ax1.set_xlabel("Layer", fontsize=12, fontproperties=font)
    ax1.set_ylabel("Rank (log scale)", fontsize=12, fontproperties=font)
    ax1.set_yscale("log")
    ax2.set_ylabel("Probability", fontsize=12, fontproperties=font)
    # ax2.set_ylim(0, 1)
    # Enhance legend and tick format for publication
    ax1.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        fontsize=10,
        prop=font,
    )
    ax2.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        fontsize=10,
        prop=font,
    )
    # ax1.set_facecolor('whitesmoke')  # Light blue background for the plot area
    # fig.patch.set_facecolor('lightblue')
    # sns.despine()  # Clean spines
    plt.xticks(
        np.arange(0, model.cfg.n_layers + 1, 1)
    )  # Set x-axis ticks to show every integer from 0 to 24

    ax1.grid(
        True, which="major", linestyle="--", linewidth="0.5", color="grey"
    )  # Add grid to primary y-axis
    # ax2.grid(True, which='both', linestyle='--', linewidth='0.5', color='grey', alpha=0.5)  # Add grid to secondary y-axis, slightly transparent

    # Title and layout
    # 'The official currency of Malaysia is called the'
    plt.title(title, fontdict=dict(fontsize=25), fontproperties=font)
    plt.tight_layout()
    plt.show()
