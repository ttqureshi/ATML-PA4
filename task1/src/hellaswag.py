"""HellaSwag zero-shot scoring.

The locked rows store the raw ``Rowan/hellaswag`` validation example. The
prompt follows the lm-evaluation-harness multiple-choice formatting:

    query = preprocess(activity_label + ": " + ctx_a + " " + ctx_b.capitalize())
    choices = [preprocess(ending) for ending in endings]

``preprocess`` strips the WikiHow bracket markers. Accuracy picks the ending
with the largest conditional log-likelihood of its tokens given the query.
``acc_norm`` divides that sum by the character length of the preprocessed
ending (Python ``len``, not the tokenizer length) and then picks the argmax.
"""

from __future__ import annotations

import re

import numpy as np

_BRACKET = re.compile(r"\[.*?\]")


def preprocess(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace(" [title]", ". ")
    cleaned = _BRACKET.sub("", cleaned)
    cleaned = cleaned.replace("  ", " ")
    return cleaned


def hellaswag_prompt(example: dict) -> tuple[str, list[str], int]:
    try:
        ctx = example["ctx_a"] + " " + example["ctx_b"].capitalize()
        query = preprocess(example["activity_label"] + ": " + ctx)
        choices = [preprocess(ending) for ending in example["endings"]]
        gold = int(example["label"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("HellaSwag example is missing ctx_a, ctx_b, activity_label, endings, or label") from exc
    if len(choices) != 4:
        raise ValueError("HellaSwag scoring expects four endings")
    return query, choices, gold


def character_lengths(choices: list[str]) -> np.ndarray:
    return np.asarray([len(choice) for choice in choices], dtype=np.float64)


def score_endings(
    loglikelihoods: np.ndarray,
    choices: list[str],
    gold: int,
) -> dict[str, object]:
    """Turn four conditional log-likelihoods into accuracy and acc_norm."""
    scores = np.asarray(loglikelihoods, dtype=np.float64)
    lengths = character_lengths(choices)
    if scores.shape != (4,) or lengths.shape != (4,):
        raise ValueError("expected four endings")
    if np.any(lengths <= 0):
        raise ValueError("acc_norm needs a positive character length on every ending")
    prediction = int(np.argmax(scores))
    normalized = scores / lengths
    prediction_norm = int(np.argmax(normalized))
    return {
        "loglikelihoods": [float(value) for value in scores],
        "n_chars": [int(value) for value in lengths],
        "prediction": prediction,
        "prediction_norm": prediction_norm,
        "gold": int(gold),
        "correct": int(prediction == gold),
        "correct_norm": int(prediction_norm == gold),
        "normalized_loglikelihoods": [float(value) for value in normalized],
    }


def aggregate_accuracy(records: list[dict]) -> dict[str, float | int]:
    if not records:
        raise ValueError("accuracy needs at least one example")
    correct = np.asarray([row["correct"] for row in records], dtype=np.float64)
    correct_norm = np.asarray([row["correct_norm"] for row in records], dtype=np.float64)
    return {
        "accuracy": float(correct.mean()),
        "acc_norm": float(correct_norm.mean()),
        "n_examples": int(correct.shape[0]),
    }
