"""Pure aggregation helpers for prompted, no-frame-persistence visual trials."""

from __future__ import annotations

from datetime import datetime, timezone
import statistics
from typing import Iterable

import numpy as np

from src.evaluation.metrics import binary_metrics, count_values


VISUAL_BEHAVIORS = ("eyes_closed", "safe_zone_outside", "phone_present", "phone_call", "drinking")
VISUAL_CONDITIONS = {
    "lighting": ("bright", "normal", "low"),
    "glasses": ("none", "clear", "tinted"),
    "distance": ("near", "medium", "far"),
    "head_angle": ("front", "left", "right", "down"),
    "occlusion": ("none", "partial"),
}


def summarize_trial(
    *,
    trial_number: int,
    behavior: str,
    expected: bool,
    conditions: dict[str, str],
    observations: list[dict[str, object]],
    started_at_utc: str,
) -> dict[str, object]:
    if behavior not in VISUAL_BEHAVIORS:
        raise ValueError(f"Unsupported visual behaviour: {behavior}")
    if not observations:
        raise ValueError("A visual trial must contain at least one processed frame")

    predictions = [bool(item["predicted"]) for item in observations]
    predicted = sum(predictions) > len(predictions) / 2
    frame_latencies = [
        float((item.get("latency_ms") or {}).get("frame_total", 0.0))
        for item in observations
    ]
    confidences = [
        float(item["confidence"])
        for item in observations
        if item.get("confidence") is not None
    ]
    return {
        "trial_id": f"trial_{trial_number:03d}",
        "timestamp_utc": started_at_utc,
        "behavior": behavior,
        "expected": bool(expected),
        "predicted": bool(predicted),
        "agreement": bool(predicted == bool(expected)),
        "conditions": dict(conditions),
        "frames_processed": len(observations),
        "mean_frame_latency_ms": round(statistics.mean(frame_latencies), 3),
        "p95_frame_latency_ms": round(float(np.percentile(frame_latencies, 95)), 3),
        "mean_object_detector_confidence": round(statistics.mean(confidences), 4)
        if confidences
        else None,
        "decision_confidence": None,
        "confidence_note": (
            "Object detector confidence is an uncalibrated component score, not behaviour confidence."
            if confidences
            else "The rule-based decision does not produce a probability confidence."
        ),
        "observations": observations,
    }


def build_visual_capture_report(
    trials: Iterable[dict[str, object]],
    *,
    model_availability: dict[str, bool],
    safe_box_ratios: tuple[float, float, float, float],
) -> dict[str, object]:
    trial_list = list(trials)
    per_behavior: dict[str, object] = {}
    for behavior in VISUAL_BEHAVIORS:
        selected = [trial for trial in trial_list if trial["behavior"] == behavior]
        if not selected:
            continue
        metrics = binary_metrics(
            [bool(trial["expected"]) for trial in selected],
            [bool(trial["predicted"]) for trial in selected],
        )
        latencies = [
            float(observation["latency_ms"]["frame_total"])
            for trial in selected
            for observation in trial["observations"]
        ]
        confidences = [
            float(trial["mean_object_detector_confidence"])
            for trial in selected
            if trial["mean_object_detector_confidence"] is not None
        ]
        per_behavior[behavior] = {
            **metrics,
            "frames_processed": sum(int(trial["frames_processed"]) for trial in selected),
            "mean_frame_latency_ms": round(statistics.mean(latencies), 3) if latencies else None,
            "p95_frame_latency_ms": round(float(np.percentile(latencies, 95)), 3) if latencies else None,
            "mean_object_detector_confidence": round(statistics.mean(confidences), 4)
            if confidences
            else None,
            "decision_confidence_available": False,
        }

    observed_behaviors = {str(trial["behavior"]) for trial in trial_list}
    missing_labels = {}
    for behavior in VISUAL_BEHAVIORS:
        selected = [trial for trial in trial_list if trial["behavior"] == behavior]
        labels = {bool(trial["expected"]) for trial in selected}
        missing = []
        if True not in labels:
            missing.append("present")
        if False not in labels:
            missing.append("absent")
        if missing:
            missing_labels[behavior] = missing
    condition_counts = {
        name: count_values(
            str((trial.get("conditions") or {}).get(name, "unknown"))
            for trial in trial_list
        )
        for name in VISUAL_CONDITIONS
    }
    missing_condition_values = {
        name: [value for value in values if condition_counts[name].get(value, 0) == 0]
        for name, values in VISUAL_CONDITIONS.items()
    }
    missing_condition_values = {
        name: values for name, values in missing_condition_values.items() if values
    }

    return {
        "status": "prompted_consented_live_visual_evaluation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "trial_count": len(trial_list),
        "frame_count": sum(int(trial["frames_processed"]) for trial in trial_list),
        "model_availability": dict(model_availability),
        "safe_box_ratios": list(safe_box_ratios),
        "coverage": {
            "missing_behaviors": [item for item in VISUAL_BEHAVIORS if item not in observed_behaviors],
            "missing_positive_or_negative_trials": missing_labels,
            "condition_value_counts": condition_counts,
            "missing_condition_values": missing_condition_values,
        },
        "per_behavior": per_behavior,
        "trials": trial_list,
        "privacy": {
            "live_camera_used": True,
            "raw_frames_saved": False,
            "raw_video_saved": False,
            "automatic_screenshots_enabled": False,
            "transcript_recording_enabled": False,
        },
        "limitations": [
            "Operator-entered labels are not independent ground truth; results depend on the stated scenario and observed conditions.",
            "Rule-based decisions do not provide calibrated probabilities; object-detector scores are component scores only.",
            "Small local trials do not establish generalisable accuracy, fairness, safety, or deployment readiness.",
        ],
    }
