"""Run deterministic fusion, baseline, ablation, and threshold checks.

The scenarios are synthetic signal combinations.  They document how the
current explainable heuristic responds to evidence and compare it with an
independent-alert and equal-count reference plus leave-one-component-out
ablations. They are not labelled driver-behaviour accuracy results.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("CABINSPECTOR_DISABLE_MODEL_LOAD", "1")
from src.video import driver_visual_prototype as prototype


COMPONENTS = (
    "eye_closed",
    "zone",
    "phone_object",
    "phone_call",
    "drinking",
    "raised_voice",
    "speech_phone",
    "safety",
    "telemetry",
)
DEPENDENCIES = {
    "phone_call": "phone_object",
    "speech_phone": "phone_call",
}


def _dependencies_valid(active: set[str]) -> bool:
    return all(component not in active or prerequisite in active for component, prerequisite in DEPENDENCIES.items())


def _dependency_valid_combinations() -> list[set[str]]:
    return [
        set(combination)
        for size in (0, 1, 2, 3)
        for combination in itertools.combinations(COMPONENTS, size)
        if _dependencies_valid(set(combination))
    ]


def _remove_with_dependents(active: set[str], component: str) -> set[str]:
    remaining = set(active)
    remaining.discard(component)
    changed = True
    while changed:
        changed = False
        for dependent, prerequisite in DEPENDENCIES.items():
            if dependent in remaining and prerequisite not in remaining:
                remaining.remove(dependent)
                changed = True
    return remaining


def _current_score(active: set[str]) -> tuple[int, dict[str, int], str]:
    phone_call_active = "phone_call" in active
    score, parts = prototype.calculate_risk_score(
        face_detected=True,
        eye_closed_counter=prototype.EYE_CLOSED_FRAME_LIMIT if "eye_closed" in active else 0,
        zone_alert_counter=4 if "zone" in active else 0,
        phone_object_detected="phone_object" in active,
        phone_call_counter=4 if phone_call_active else 0,
        drinking_counter=4 if "drinking" in active else 0,
        raised_voice_active="raised_voice" in active,
        speech_with_phone_evidence="speech_phone" in active and phone_call_active,
        safety_categories=("PROFANITY",) if "safety" in active else (),
        telemetry_event_category="HARD_BRAKE" if "telemetry" in active else "NORMAL",
        telemetry_event_active="telemetry" in active,
    )
    return score, parts, prototype.get_risk_level(score)


def _equal_count_score(active: set[str]) -> int:
    return round(100 * len(active) / len(COMPONENTS))


def _persistence_sensitivity() -> list[dict[str, object]]:
    sequence = [False, False, True, True, True, True, True, False, False, False, False, False]
    rows = []
    for confirm_frames in (1, 2, 3, 5):
        counter = 0
        first_active = None
        last_active = None
        for frame, raw_active in enumerate(sequence, start=1):
            counter, active = prototype.update_smoothing_counter(
                counter,
                raw_active,
                confirm_frames=confirm_frames,
            )
            if active and first_active is None:
                first_active = frame
            if active:
                last_active = frame
        rows.append(
            {
                "confirm_frames": confirm_frames,
                "first_confirmed_frame": first_active,
                "last_confirmed_frame": last_active,
                "sequence_length": len(sequence),
            }
        )
    return rows


def _baseline_comparison(combinations: list[dict[str, object]]) -> dict[str, object]:
    total = len(combinations)
    if not total:
        return {"scenario_count": 0}
    independent_or = [bool(row["independent_or_alert"]) for row in combinations]
    current_any = [int(row["current_score"]) > 0 for row in combinations]
    current_review = [int(row["current_score"]) >= 30 for row in combinations]
    current_high = [int(row["current_score"]) >= 65 for row in combinations]
    equal_count = [int(row["equal_count_reference_score"]) >= 30 for row in combinations]

    def summarize(values: list[bool]) -> dict[str, float | int]:
        count = sum(values)
        return {"trigger_count": count, "trigger_rate": round(count / total, 4)}

    return {
        "scenario_count": total,
        "independent_or_any_signal": summarize(independent_or),
        "current_any_nonzero_score": summarize(current_any),
        "current_review_priority_at_30_points": summarize(current_review),
        "current_high_priority_at_65_points": summarize(current_high),
        "equal_count_review_priority_at_30_points": summarize(equal_count),
        "independent_or_vs_current_review_disagreements": sum(
            baseline != current for baseline, current in zip(independent_or, current_review)
        ),
        "interpretation": "Trigger rates on designed synthetic signal combinations only; not alert accuracy or a false-positive estimate.",
    }


def _ablation_sensitivity() -> dict[str, object]:
    score_deltas: dict[str, list[float]] = {component: [] for component in COMPONENTS}
    level_changes: dict[str, int] = {component: 0 for component in COMPONENTS}
    review_threshold_drops: dict[str, int] = {component: 0 for component in COMPONENTS}
    comparison_count = 0

    for size in (2, 3):
        for combination in itertools.combinations(COMPONENTS, size):
            active = set(combination)
            current_score, _, current_level = _current_score(active)
            for component in combination:
                ablated_score, _, ablated_level = _current_score(active - {component})
                score_deltas[component].append(float(current_score - ablated_score))
                level_changes[component] += int(current_level != ablated_level)
                review_threshold_drops[component] += int(current_score >= 30 and ablated_score < 30)
                comparison_count += 1

    per_component = {}
    for component in COMPONENTS:
        deltas = score_deltas[component]
        per_component[component] = {
            "ablation_cases": len(deltas),
            "mean_score_reduction": round(sum(deltas) / len(deltas), 4) if deltas else 0.0,
            "risk_level_changes": level_changes[component],
            "dropped_below_review_threshold": review_threshold_drops[component],
        }
    return {
        "comparison_count": comparison_count,
        "combination_sizes": [2, 3],
        "per_component": per_component,
        "interpretation": "Leave-one-component-out score changes on synthetic combinations; not a labelled behavioural ablation.",
    }


def _dependency_aware_ablation(combinations: list[set[str]]) -> dict[str, object]:
    score_deltas: dict[str, list[float]] = {component: [] for component in COMPONENTS}
    comparison_count = 0
    for active in combinations:
        if len(active) not in (2, 3):
            continue
        current_score, _parts, _level = _current_score(active)
        for component in active:
            ablated = _remove_with_dependents(active, component)
            ablated_score, _parts, _level = _current_score(ablated)
            score_deltas[component].append(float(current_score - ablated_score))
            comparison_count += 1
    per_component = {
        component: {
            "ablation_cases": len(values),
            "mean_score_reduction": round(sum(values) / len(values), 4) if values else 0.0,
        }
        for component, values in score_deltas.items()
    }
    return {
        "comparison_count": comparison_count,
        "per_component": per_component,
        "removal_rule": (
            "Removing a prerequisite also removes its dependent signal: removing phone_object removes "
            "phone_call and speech_phone; removing phone_call also removes speech_phone."
        ),
        "interpretation": "Dependency-aware score changes on valid synthetic signal sets; not a labelled behavioural ablation.",
    }


def evaluate() -> dict[str, object]:
    single_component_rows = []
    for component in COMPONENTS:
        current_score, parts, level = _current_score({component})
        single_component_rows.append(
            {
                "component": component,
                "current_score": current_score,
                "current_risk_level": level,
                "current_contribution": parts.get(
                    {
                        "eye_closed": "eye",
                        "safety": "speech_safety",
                    }.get(component, component),
                    0,
                ),
                "equal_count_reference_score": _equal_count_score({component}),
            }
        )

    combination_rows = []
    for size in (0, 1, 2, 3):
        for combination in itertools.combinations(COMPONENTS, size):
            active = set(combination)
            current_score, _, level = _current_score(active)
            combination_rows.append(
                {
                    "active_components": list(combination),
                    "current_score": current_score,
                    "current_risk_level": level,
                    "equal_count_reference_score": _equal_count_score(active),
                    "independent_or_alert": bool(active),
                }
            )

    context_rows = []
    for label, active in (
        ("speech_only", {"speech_phone"}),
        ("raised_voice_only", {"raised_voice"}),
        ("phone_without_speech", {"phone_object", "phone_call"}),
        ("phone_with_speech", {"phone_object", "phone_call", "speech_phone"}),
        ("telemetry_inactive", set()),
        ("telemetry_active", {"telemetry"}),
    ):
        score, parts, level = _current_score(active)
        context_rows.append(
            {
                "case": label,
                "current_score": score,
                "current_risk_level": level,
                "risk_breakdown": parts,
            }
        )

    baseline_comparison = _baseline_comparison(combination_rows)
    ablation_sensitivity = _ablation_sensitivity()
    valid_combinations = _dependency_valid_combinations()
    valid_rows = [
        {
            "active_components": sorted(active),
            "current_score": _current_score(active)[0],
            "current_risk_level": _current_score(active)[2],
            "equal_count_reference_score": _equal_count_score(active),
            "independent_or_alert": bool(active),
        }
        for active in valid_combinations
    ]

    return {
        "status": "synthetic_fusion_sensitivity_only",
        "components": list(COMPONENTS),
        "current_fast_demo_confirm_frames": prototype.ALERT_CONFIRM_FRAMES,
        "single_component_sensitivity": single_component_rows,
        "combination_sensitivity": combination_rows,
        "baseline_comparison": baseline_comparison,
        "ablation_sensitivity": ablation_sensitivity,
        "signal_dependencies": {
            "constraints": [
                "phone_call implies a detected phone object",
                "speech_phone reinforcement requires phone_call evidence",
            ],
            "unconstrained_combinations": {
                "scenario_count": len(combination_rows),
                "baseline_comparison": baseline_comparison,
                "ablation": ablation_sensitivity,
                "interpretation": (
                    "Counterfactual component combinations used to expose how the score responds; "
                    "some combinations are not reachable through the production decision rules."
                ),
            },
            "dependency_valid_combinations": {
                "scenario_count": len(valid_rows),
                "combinations": valid_rows,
                "baseline_comparison": _baseline_comparison(valid_rows),
                "ablation": _dependency_aware_ablation(valid_combinations),
                "interpretation": (
                    "The direct phone and conditional speech dependencies in the implementation are "
                    "enforced here, but the other synthetic signals are still combined without real-world "
                    "frequency or ground-truth labels."
                ),
            },
        },
        "context_sensitivity": context_rows,
        "persistence_sensitivity": _persistence_sensitivity(),
        "limitations": [
            "Synthetic combinations do not establish which alerts are correct in real trips.",
            "The equal-count and OR references are comparators, not validated alternatives.",
            "Baseline trigger rates and leave-one-component-out changes are mechanistic synthetic comparisons, not accuracy or false-alarm measures.",
            "Risk score values are heuristic evidence weights and are not probabilities.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "evaluation" / "fusion_sensitivity_evaluation.json",
    )
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    report = evaluate()
    if not args.no_write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Evaluation written: {args.output}")
    print(f"Combination cases: {len(report['combination_sensitivity'])}")
    print(f"Current fast-demo confirmation frames: {report['current_fast_demo_confirm_frames']}")
    print(
        "Synthetic trigger rates: "
        f"OR={report['baseline_comparison']['independent_or_any_signal']['trigger_rate']:.4f}, "
        f"current>=30={report['baseline_comparison']['current_review_priority_at_30_points']['trigger_rate']:.4f}, "
        f"equal-count>=30={report['baseline_comparison']['equal_count_review_priority_at_30_points']['trigger_rate']:.4f}"
    )
    print(f"Leave-one-component-out comparisons: {report['ablation_sensitivity']['comparison_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
