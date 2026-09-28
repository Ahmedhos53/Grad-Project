"""Plot telemetry, fusion, and resource evaluation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = PROJECT_ROOT / "outputs" / "evaluation"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "evaluation" / "figures"


def _load(name: str, evaluation_dir: Path = EVALUATION_DIR) -> dict[str, object]:
    path = Path(evaluation_dir) / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_path(path: Path) -> dict[str, object]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build(
    output_dir: Path,
    *,
    evaluation_dir: Path = EVALUATION_DIR,
    telemetry_report: Path | None = None,
    threshold_report: Path | None = None,
    fusion_report: Path | None = None,
) -> dict[str, object]:
    evaluation_dir = Path(evaluation_dir)
    telemetry_path = Path(telemetry_report) if telemetry_report else evaluation_dir / "continuous_telemetry_evaluation.json"
    threshold_path = Path(threshold_report) if threshold_report else evaluation_dir / "threshold_sensitivity_evaluation.json"
    fusion_path = Path(fusion_report) if fusion_report else evaluation_dir / "fusion_sensitivity_evaluation.json"
    telemetry = _load_path(telemetry_path)
    threshold = _load_path(threshold_path)
    fusion = _load_path(fusion_path)
    resource_report_name = next(
        (
            name
            for name in (
                "combined_resource_stability_900_repeat2.json",
                "combined_resource_stability_900.json",
                "combined_resource_evaluation.json",
            )
            if (evaluation_dir / name).is_file()
        ),
        "combined_resource_evaluation.json",
    )
    resources = _load(resource_report_name, evaluation_dir)
    visual = _load("visual_logic_evaluation.json", evaluation_dir)
    audio = _load("audio_protocol_evaluation.json", evaluation_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    telemetry_metrics = telemetry["metrics"]
    class_names = telemetry_metrics["class_names"]
    confusion = np.asarray(telemetry_metrics["confusion_matrix"], dtype=int)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    image = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(len(class_names)), [name.replace("_", " ").title() for name in class_names], rotation=35, ha="right")
    ax.set_yticks(range(len(class_names)), [name.replace("_", " ").title() for name in class_names])
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("Greatest-overlap test label")
    ax.set_title("PRIMUS outer-trip-held-out continuous confusion matrix (argmax)")
    for row in range(confusion.shape[0]):
        for column in range(confusion.shape[1]):
            ax.text(column, row, int(confusion[row, column]), ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Window count")
    confusion_path = output_dir / "telemetry_confusion_matrix.png"
    _save(fig, confusion_path)

    per_class = telemetry_metrics["per_class"]
    precision = [per_class[name]["precision"] for name in class_names]
    recall = [per_class[name]["recall"] for name in class_names]
    f1 = [per_class[name]["f1"] for name in class_names]
    x = np.arange(len(class_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1")
    ax.set_xticks(x, [name.replace("_", " ").title() for name in class_names], rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("PRIMUS per-class argmax scores on held-out continuous windows")
    ax.legend()
    class_path = output_dir / "telemetry_per_class_metrics.png"
    _save(fig, class_path)

    model_names = ["PRIMUS", "Random Forest"]
    model_reports = [telemetry_metrics, telemetry["random_forest_baseline_metrics"]]
    if "always_normal_baseline_metrics" in telemetry:
        model_names.append("Always NORMAL")
        model_reports.append(telemetry["always_normal_baseline_metrics"])

    def false_positive_rate(report):
        if "false_positive_windows_per_normal_labeled_decision_minute_at_default_threshold" in report:
            return report["false_positive_windows_per_normal_labeled_decision_minute_at_default_threshold"]
        if "false_positive_windows_per_normal_labeled_decision_minute_argmax" in report:
            return report["false_positive_windows_per_normal_labeled_decision_minute_argmax"]
        return report.get("false_alarms_per_normal_minute", 0.0)

    def thresholded_metrics(report):
        return report.get("default_threshold_classification_metrics", report)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))
    values = [
        [thresholded_metrics(report)["accuracy"] for report in model_reports],
        [thresholded_metrics(report)["macro_f1_present_classes"] for report in model_reports],
        [
            false_positive_rate(report) for report in model_reports
        ],
    ]
    titles = ["Accuracy (.55 gate)", "Present-class macro-F1 (.55 gate)", "FP windows / normal-decision minute"]
    colors = ["#2b6cb0", "#c05621", "#718096"][:len(model_names)]
    for ax, metric_values, title in zip(axes, values, titles):
        ax.bar(model_names, metric_values, color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        if "FP windows" not in title:
            ax.set_ylim(0, 1)
    fig.suptitle("Trip-held-out continuous comparison at the current .55 replay threshold")
    comparison_path = output_dir / "telemetry_model_comparison.png"
    _save(fig, comparison_path)

    components = [row["component"] for row in fusion["single_component_sensitivity"]]
    current = [row["current_contribution"] for row in fusion["single_component_sensitivity"]]
    equal = [row["equal_count_reference_score"] for row in fusion["single_component_sensitivity"]]
    x = np.arange(len(components))
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x - width / 2, current, width, label="Current heuristic contribution")
    ax.bar(x + width / 2, equal, width, label="Equal-count reference")
    ax.set_xticks(x, [name.replace("_", " ").title() for name in components], rotation=35, ha="right")
    ax.set_ylabel("Score points / reference points")
    ax.set_title("Synthetic fusion-weight sensitivity")
    ax.legend()
    fusion_path = output_dir / "fusion_weight_sensitivity.png"
    _save(fig, fusion_path)

    baseline = fusion["baseline_comparison"]
    dependency_valid = fusion.get("signal_dependencies", {}).get("dependency_valid_combinations")
    if dependency_valid is None:
        dependency_valid = {
            "scenario_count": baseline["scenario_count"],
            "baseline_comparison": baseline,
            "ablation": fusion["ablation_sensitivity"],
        }
    constrained_baseline = dependency_valid["baseline_comparison"]
    trigger_labels = ["Independent OR", "Current >= 30", "Equal count >= 30"]
    unconstrained_rates = [
        baseline["independent_or_any_signal"]["trigger_rate"],
        baseline["current_review_priority_at_30_points"]["trigger_rate"],
        baseline["equal_count_review_priority_at_30_points"]["trigger_rate"],
    ]
    constrained_rates = [
        constrained_baseline["independent_or_any_signal"]["trigger_rate"],
        constrained_baseline["current_review_priority_at_30_points"]["trigger_rate"],
        constrained_baseline["equal_count_review_priority_at_30_points"]["trigger_rate"],
    ]
    unconstrained_ablation = fusion["ablation_sensitivity"]["per_component"]
    constrained_ablation = dependency_valid["ablation"]["per_component"]
    ablation_names = list(fusion["components"])
    unconstrained_deltas = [
        unconstrained_ablation[name]["mean_score_reduction"] for name in ablation_names
    ]
    constrained_deltas = [
        constrained_ablation[name]["mean_score_reduction"] for name in ablation_names
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.5))
    x = np.arange(len(trigger_labels))
    width = 0.36
    axes[0].bar(
        x - width / 2,
        unconstrained_rates,
        width,
        color="#718096",
        label=f"Unconstrained (n={baseline['scenario_count']})",
    )
    axes[0].bar(
        x + width / 2,
        constrained_rates,
        width,
        color="#2b6cb0",
        label=f"Phone-rule dependencies (n={dependency_valid['scenario_count']})",
    )
    axes[0].set_ylim(0, 1.12)
    axes[0].set_ylabel("Trigger rate in synthetic cases", fontsize=9)
    axes[0].set_title("Trigger comparisons", fontsize=10)
    axes[0].set_xticks(x, trigger_labels, rotation=20, ha="right", fontsize=8)
    axes[0].legend(fontsize=7)

    y = np.arange(len(ablation_names))
    axes[1].barh(y - width / 2, unconstrained_deltas, height=width, color="#718096", label="Unconstrained")
    axes[1].barh(y + width / 2, constrained_deltas, height=width, color="#2b6cb0", label="Dependency-aware removal")
    axes[1].set_yticks(y, [name.replace("_", " ").title() for name in ablation_names], fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, max(unconstrained_deltas + constrained_deltas, default=0.0) * 1.2 + 2)
    axes[1].set_xlabel("Mean score reduction", fontsize=9)
    axes[1].set_title("Synthetic component ablations", fontsize=10)
    axes[1].legend(fontsize=7)
    fig.suptitle("Synthetic fusion checks (mechanics only; not accuracy)", fontsize=11)
    fusion_baseline_path = output_dir / "fusion_baseline_ablation.png"
    _save(fig, fusion_baseline_path)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6))
    axes = axes.reshape(-1)
    gpu = resources.get("gpu", {})
    gpu_available = bool(gpu.get("available"))
    resource_groups = [
        (axes[0], ["Mean", "Max"], [resources["cpu_percent_mean"], resources["cpu_percent_max"]], "CPU (%)"),
        (axes[1], ["Mean", "Max"], [resources["rss_mb_mean"], resources["rss_mb_max"]], "RSS (MB)"),
        (
            axes[2],
            ["Mean", "Max"],
            [gpu.get("utilization_percent_mean", 0.0), gpu.get("utilization_percent_max", 0.0)],
            "GPU utilisation (%)",
        ),
        (
            axes[3],
            ["Mean", "Max"],
            [gpu.get("memory_used_mb_mean", 0.0), gpu.get("memory_used_mb_max", 0.0)],
            "VRAM used (MB)",
        ),
    ]
    for ax, labels, values, title in resource_groups:
        bars = ax.bar(labels, values, color="#276749")
        ax.set_title(title)
        ax.set_ylabel("Observed")
        if not gpu_available and ax in (axes[2], axes[3]):
            ax.text(0.5, 0.5, "unavailable", ha="center", va="center", transform=ax.transAxes)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:g}", ha="center", va="bottom", fontsize=8)
    thread_note = f"threads max={resources.get('threads_max', 0)}"
    vram_total = gpu.get("memory_total_mb")
    if vram_total:
        thread_note += f"; VRAM total={vram_total:g} MB"
    fig.suptitle(
        f"Bounded combined-run resource summary ({resources['frames_requested']} frames; {thread_note})"
    )
    resource_path = output_dir / "combined_resource_summary.png"
    _save(fig, resource_path)

    if "models" in threshold:
        threshold_models = threshold["models"]
    else:
        threshold_models = {"primus": threshold["results"]}
    threshold_colors = {"primus": "#2b6cb0", "random_forest": "#c05621"}
    threshold_names = {"primus": "PRIMUS", "random_forest": "Random Forest"}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    for model_name, rows in threshold_models.items():
        values = list(rows)
        thresholds = [row["threshold"] for row in values]
        colour = threshold_colors[model_name]
        axes[0].plot(
            thresholds,
            [row["macro_f1_present_classes"] for row in values],
            marker="o",
            label=threshold_names[model_name],
            color=colour,
        )
        axes[1].plot(
            thresholds,
            [row["false_positive_windows_per_normal_labeled_decision_minute"] for row in values],
            marker="s",
            label=threshold_names[model_name],
            color=colour,
        )
    axes[0].set_xlabel("Minimum confidence threshold")
    axes[0].set_ylabel("Present-class macro-F1")
    axes[0].set_ylim(0, 1)
    axes[1].set_xlabel("Minimum confidence threshold")
    axes[1].set_ylabel("FP windows / normal-labeled decision minute")
    axes[0].set_title("Classification trade-off")
    axes[1].set_title("False-positive window rate")
    for ax in axes:
        ax.axvline(0.55, color="#718096", linestyle="--", linewidth=1, label="Current default 0.55")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Trip-held-out threshold sensitivity (descriptive; no threshold selected)")
    threshold_path = output_dir / "telemetry_threshold_sensitivity.png"
    _save(fig, threshold_path)

    manifest = {
        "status": "generated_from_report_revision_evaluation_reports",
        "source_reports": [
            str(telemetry_path),
            str(threshold_path),
            str(fusion_path),
            f"outputs/evaluation/{resource_report_name}",
            "outputs/evaluation/combined_resource_stability_summary.json",
            "outputs/evaluation/visual_logic_evaluation.json",
            "outputs/evaluation/audio_protocol_evaluation.json",
        ],
        "figures": [
            str(confusion_path),
            str(class_path),
            str(comparison_path),
            str(fusion_path),
            str(fusion_baseline_path),
            str(resource_path),
            str(threshold_path),
        ],
        "visual_contract_status": visual["status"],
        "audio_contract_status": audio["status"],
        "limitations": [
            "Telemetry labels are public annotation-overlap labels; no live phone signal was used.",
            "Telemetry fold heads exclude each test trip, but earlier model-family selection used the same three trips.",
            "Visual and audio contract reports are synthetic and must not be presented as real-world accuracy.",
            "Resource values describe one bounded local run and are not universal hardware guarantees.",
        ],
    }
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--evaluation-dir", type=Path, default=EVALUATION_DIR)
    parser.add_argument("--telemetry-report", type=Path, default=None)
    parser.add_argument("--threshold-report", type=Path, default=None)
    parser.add_argument("--fusion-report", type=Path, default=None)
    args = parser.parse_args()
    manifest = build(
        args.output_dir,
        evaluation_dir=args.evaluation_dir,
        telemetry_report=args.telemetry_report,
        threshold_report=args.threshold_report,
        fusion_report=args.fusion_report,
    )
    print(f"Figures written: {len(manifest['figures'])}")
    print(f"Manifest written: {args.output_dir / 'figure_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
