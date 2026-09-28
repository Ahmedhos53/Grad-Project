"""Small dependency-light metric helpers shared by evaluation tools."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence


def binary_metrics(expected: Iterable[bool], predicted: Iterable[bool]) -> dict[str, float | int]:
    """Return counts and rates for one boolean signal."""

    expected_values = [bool(value) for value in expected]
    predicted_values = [bool(value) for value in predicted]
    if len(expected_values) != len(predicted_values):
        raise ValueError("Expected and predicted boolean sequences must have equal length")
    tp = sum(actual and guess for actual, guess in zip(expected_values, predicted_values))
    tn = sum(not actual and not guess for actual, guess in zip(expected_values, predicted_values))
    fp = sum(not actual and guess for actual, guess in zip(expected_values, predicted_values))
    fn = sum(actual and not guess for actual, guess in zip(expected_values, predicted_values))

    def ratio(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 4) if denominator else 0.0

    return {
        "support": len(expected_values),
        "positive_support": sum(expected_values),
        "negative_support": len(expected_values) - sum(expected_values),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": ratio(tp + tn, len(expected_values)),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "false_positive_rate": ratio(fp, fp + tn),
    }


def multiclass_metrics(
    expected: Sequence[str],
    predicted: Sequence[str],
    class_names: Sequence[str],
) -> dict[str, object]:
    """Return a labelled confusion matrix and per-class precision/recall/F1."""

    if len(expected) != len(predicted):
        raise ValueError("Expected and predicted class sequences must have equal length")
    names = list(class_names)
    index = {name: position for position, name in enumerate(names)}
    matrix = [[0 for _ in names] for _ in names]
    for actual, guess in zip(expected, predicted):
        if actual not in index or guess not in index:
            raise ValueError(f"Unknown class in evaluation pair: {actual!r}, {guess!r}")
        matrix[index[actual]][index[guess]] += 1

    total = len(expected)
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    present_f1_values: list[float] = []
    for position, name in enumerate(names):
        tp = matrix[position][position]
        fp = sum(matrix[row][position] for row in range(len(names)) if row != position)
        fn = sum(matrix[position][column] for column in range(len(names)) if column != position)

        def ratio(numerator: int, denominator: int) -> float:
            return round(numerator / denominator, 4) if denominator else 0.0

        f1 = ratio(2 * tp, 2 * tp + fp + fn)
        f1_values.append(f1)
        if sum(matrix[position]) > 0:
            present_f1_values.append(f1)
        per_class[name] = {
            "support": sum(matrix[position]),
            "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn),
            "f1": f1,
        }

    return {
        "support": total,
        "accuracy": round(sum(matrix[i][i] for i in range(len(names))) / total, 4) if total else 0.0,
        "macro_f1": round(sum(f1_values) / len(f1_values), 4) if f1_values else 0.0,
        "macro_f1_all_classes": round(sum(f1_values) / len(f1_values), 4) if f1_values else 0.0,
        "macro_f1_present_classes": round(sum(present_f1_values) / len(present_f1_values), 4)
        if present_f1_values
        else 0.0,
        "class_names": names,
        "confusion_matrix": matrix,
        "per_class": per_class,
    }


def count_values(values: Iterable[str]) -> dict[str, int]:
    """Return deterministic counts for a sequence of categorical values."""

    return dict(sorted(Counter(values).items()))
