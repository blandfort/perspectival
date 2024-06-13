import html
from typing import Tuple
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import HTML

from .model import Transformer


def tokenize_texts(tokenizer, texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")


def get_logits(model, encoded_texts):
    model.eval()
    with torch.no_grad():
        outputs = model(**encoded_texts)
        return outputs.logits


def get_top_k_tokens(logits, tokenizer, top_k, mode, token=None):
    if mode == "likelihoods":
        scores = F.softmax(logits, dim=-1)
    elif mode == "log_likelihoods":
        scores = F.log_softmax(logits, dim=-1)
    elif mode == "logits":
        scores = logits
    else:
        raise ValueError(
            f"Mode '{mode}' doesn't exist! Choose 'log_likelihoods', "
            + "'likelihoods', or 'logits'."
        )

    top_scores, top_indices = torch.topk(scores, top_k)
    top_tokens = tokenizer.batch_decode(top_indices)
    top_formatted = ", ".join(
        [
            f"{token} ({score:.2f})"
            for token, score in zip(top_tokens, top_scores.numpy())
        ]
    )

    # Some characters can cause issues with displaying
    # (This solution isn't that pretty but seems to do the trick.)
    return top_formatted.replace("\n", "NEWLINE").replace("'", "QUOTE")


def _compute_token_differences(
    texts: str,
    models: Tuple[Transformer, Transformer],
    mode: str = "log_likelihoods",
):
    assert len(models) == 2

    model1, tokenizer1 = models[0].get_model_and_tokenizer()
    model2, tokenizer2 = models[1].get_model_and_tokenizer()

    encoded_texts1 = tokenize_texts(tokenizer1, texts)
    encoded_texts2 = tokenize_texts(tokenizer2, texts)
    logits = [get_logits(model1, encoded_texts1), get_logits(model2, encoded_texts2)]

    scores1 = []
    scores2 = []
    diffs = []
    tokens = []
    token_ids = []

    for sequence_id in range(len(texts)):
        s_token_ids = encoded_texts1[sequence_id].ids
        s_tokens = tokenizer1.batch_decode(s_token_ids)

        s_token_ids2 = encoded_texts2[sequence_id].ids
        s_tokens2 = tokenizer2.batch_decode(s_token_ids2)

        sequence_scores1 = []
        sequence_scores2 = []
        sequence_diffs = []
        for (
            next_token,
            next_token2,
            next_token_id,
            next_token_id2,
            logit,
            logit2,
        ) in zip(
            s_tokens[1:],
            s_tokens2[1:],
            s_token_ids[1:],
            s_token_ids2[1:],
            logits[0][sequence_id],
            logits[1][sequence_id],
        ):
            assert (
                next_token == next_token2
            ), "This functionality doesn't work for unaligned tokenizations!"

            # Setting values to 0 for padding tokens to avoid these values
            # affecting scaling
            if next_token_id == tokenizer1.pad_token_id:
                sequence_scores1.append(0)
                sequence_scores2.append(0)
                sequence_diffs.append(0)
                continue

            if mode == "log_likelihoods":
                values1 = F.log_softmax(logit, dim=-1)
                values2 = F.log_softmax(logit2, dim=-1)
            elif mode == "likelihoods":
                values1 = F.softmax(logit, dim=-1)
                values2 = F.softmax(logit2, dim=-1)
            elif mode == "logits":
                values1 = logit
                values2 = logit2
            else:
                raise ValueError(
                    f"Mode '{mode}' doesn't exist! Choose 'log_likelihoods', "
                    + "'likelihoods', or 'logits'."
                )

            sequence_scores1.append(values1[next_token_id].item())
            sequence_scores2.append(values2[next_token_id2].item())
            sequence_diffs.append(np.abs(sequence_scores1[-1] - sequence_scores2[-1]))

        scores1.append(sequence_scores1)
        scores2.append(sequence_scores2)
        diffs.append(sequence_diffs)
        tokens.append(s_tokens)
        token_ids.append(s_token_ids)

    return logits, scores1, scores2, diffs, tokens, token_ids


def inspect_texts(
    texts: str,
    models: Tuple[Transformer, Transformer],
    mode: str = "log_likelihoods",
    color_map_name: str = "Blues",
    max_color: float = 0.7,  # Ensure lighter colors
    top_k: int = 5,
):
    logits, scores1, scores2, diffs, tokens, token_ids = _compute_token_differences(
        texts=texts,
        models=models,
        mode=mode,
    )

    assert len(models) == 2

    _, tokenizer1 = models[0].get_model_and_tokenizer()
    _, tokenizer2 = models[1].get_model_and_tokenizer()

    html_string = "<div>"
    color_map = plt.get_cmap(color_map_name)

    max_diff = (
        np.max(np.abs(diffs)) if np.max(np.abs(diffs)) != 0 else 1
    )  # Prevent division by zero
    scaled_diffs = np.abs(diffs) * max_color / max_diff

    for sequence_id in range(len(texts)):
        # For the first token we don't have disagreement because it is only used
        # as input
        sentence_html = f"<p>{html.escape(tokens[sequence_id][0])}"

        for j, (token, token_id, scaled_diff, diff, score1, score2) in enumerate(
            zip(
                tokens[sequence_id][1:],
                token_ids[sequence_id][1:],
                scaled_diffs[sequence_id],
                diffs[sequence_id],
                scores1[sequence_id],
                scores2[sequence_id],
            )
        ):
            if token_id == tokenizer1.pad_token_id:
                continue

            color = color_map(scaled_diff)[:3]
            hex_color = matplotlib.colors.rgb2hex(color)
            top_tokens1 = get_top_k_tokens(
                logits[0][sequence_id][j], tokenizer1, top_k, mode
            )
            top_tokens2 = get_top_k_tokens(
                logits[1][sequence_id][j], tokenizer2, top_k, mode
            )

            mouseover_text = (
                f"Difference for actual token ({token}): "
                + f"{diff:.4f} ({models[0].name}: {score1:.4f}, {models[1].name}: "
                + f"{score2:.4f})"
            )
            mouseover_text += (
                f"<br />{models[0].name} Top-{top_k}: "
                + f"{top_tokens1}<br />{models[1].name} Top-{top_k}: {top_tokens2}"
            )

            sentence_html += (
                f"<span style='background-color: {hex_color}; "
                + "cursor: pointer;' onmouseover=\"this.parentNode.parentNode."
                + f"nextSibling.innerHTML='{html.escape(mouseover_text)}';\" "
                + 'onmouseout="this.parentNode.parentNode.nextSibling.innerHTML'
                + f"='Hover over a token for details.';\">{html.escape(token)} </span>"
            )
        sentence_html += "</p>"

        html_string += sentence_html

    html_string += (
        "</div><div style='margin-top: 20px; border: 1px solid #ccc; "
        + "padding: 10px;'>Hover over a token for details.</div>"
    )
    # Adding a color legend
    html_string += "<div style='display: flex; align-items: center; margin-top: 10px;'>"
    html_string += (
        "<div style='width: 100px; height: 20px; background-image: "
        + "linear-gradient(to right, "
        + matplotlib.colors.rgb2hex(color_map(0)[:3])
        + ", "
        + matplotlib.colors.rgb2hex(color_map(0.7)[:3])
        + ");'></div>"
    )
    html_string += (
        "<div style='margin-left: 10px;'>Absolute value: "
        + f"Min ({np.min(np.abs(diffs)):.2f}) ⟶ Max ({np.max(np.abs(diffs)):.2f})</div>"
    )
    html_string += "</div></div>"

    return HTML(html_string)
