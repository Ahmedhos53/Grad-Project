"""Describe confidence-threshold sensitivity on trip-held-out telemetry scores.

This script does not select a new threshold. It reports descriptive operating
points on the held-out window predictions produced by the continuous evaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import multiclass_metrics
from tools.evaluate_continuous_telemetry import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_EVENT_MERGE_GAP_SECONDS,
    _alert_episode_summary,
)


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "outputs"
    / "report_revision"
    / "telemetry"
    / "continuous_telemetry_crossvalidated.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "report_revision"
    / "telemetry"
    / "threshold_sensitivity_crossvalidated.json"
)
MODEL_FIELDS = {
    "primus": ("primus_category", "primus_confidence"),
    "random_forest": ("random_forest_category", "random_forest_confidence"),
}


def evaluate_model_thresholds(
    report: dict[str, object],
    *,
    category_key: str,
    confidence_key: str,
    thresholds: tuple[float, ...],
) -> list[dict[str, object]]:
    rows = list(report.get("windows", []))
    class_names = list(report["metrics"]["class_names"])
    stride = float(report["windowing"]["stride_seconds"])
    expected = [str(row["expected_category"]) for row in rows]
    normal_count = sum(category == "NORMAL" for category in expected)
    normal_exposure_seconds = normal_count * stride
    normal_exposure_minutes = normal_exposure_seconds / 60.0
    results = []
    for threshold in thresholds:
        predicted = [
            str(row[category_key])
            if float(row[confidence_key]) >= threshold
            else "NORMAL"
            for row in rows
        ]
        metrics = multiclass_metrics(expected, predicted, class_names)
        false_positive_windows = sum(
            actual == "NORMAL" and guess != "NORMAL"
            for actual, guess in zip(expected, predicted)
        )
        episodes = _alert_episode_summary(
            rows,
            category_key=category_key,
            confidence_key=confidence_key,
            threshold=threshold,
            merge_gap_seconds=DEFAULT_EVENT_MERGE_GAP_SECONDS,
        )
        metrics.update(
            {
                "threshold": threshold,
                "false_positive_windows": false_positive_windows,
                "normal_labeled_decision_exposure_seconds": round(normal_exposure_seconds, 3),
                "false_positive_windows_per_normal_labeled_decision_minute": (
                    round(false_positive_windows / normal_exposure_minutes, 4)
                    if normal_exposure_minutes
                    else 0.0
                ),
                "alert_episodes": episodes,
                "suppressed_predictions": sum(
                    float(row[confidence_key]) < threshold for row in rows
                ),
            }
        )
        results.append(metrics)
    return results


def evaluate(
    report: dict[str, object],
    thresholds: tuple[float, ...],
) -> dict[str, object]:
    return {
        "status": "descriptive_threshold_sensitivity_on_trip_held_out_predictions",
        "source": str(DEFAULT_INPUT),
        "live_phone_telemetry_used": False,
        "models": {
            name: evaluate_model_thresholds(
                report,
                category_key=category_key,
                confidence_key=confidence_key,
                thresholds=thresholds,
            )
            for name, (category_key, confidence_key) in MODEL_FIELDS.items()
        },
        "always_normal_baseline": report.get("always_normal_baseline_metrics"),
        "threshold_policy": {
            "application_default": DEFAULT_CONFIDENCE_THRESHOLD,
            "new_threshold_selected": False,
            "interpretation": (
                "The .55 application default is retained. This grid is descriptive sensitivity "
                "on the outer-fold test predictions, not an independent calibration set; no value "
                "is promoted as a validated optimum."
            ),
            "false_positive_denominator": (
                "NORMAL-labelled decision-time exposure, computed as NORMAL decision windows "
                "multiplied by the one-second stride; overlapping five-second windows are not "
                "counted as five seconds each."
            ),
        },
        "limitations": [
            "Predictions are fold-specific, but model-family selection was previously guided by the same three public trips.",
            "Threshold changes are evaluated on the same outer-fold test windows and are not independent tuning results.",
            "Merged alert episodes are an offline grouping convention, not a measured driver-event count.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=(0.0, 0.55, 0.70, 0.80, 0.90),
    )
    args = parser.parse_args()
    if any(not 0.0 <= threshold <= 1.0 for threshold in args.thresholds):
        parser.error("thresholds must be between 0 and 1")
    report = json.loads(args.input.read_text(encoding="utf-8"))
    result = evaluate(report, tuple(args.thresholds))
    result["source"] = str(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Threshold report written: {args.output}")
    for name, rows in result["models"].items():
        for row in rows:
            print(
                f"{name} threshold={row['threshold']:.2f} "
                f"accuracy={row['accuracy']:.4f} "
                f"macro_f1_present={row['macro_f1_present_classes']:.4f} "
                f"false_positive_windows={row['false_positive_windows']} "
                f"unmatched_episodes={row['alert_episodes']['unmatched_normal']} "
                f"fp_windows_per_normal_decision_minute="
                f"{row['false_positive_windows_per_normal_labeled_decision_minute']:.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
