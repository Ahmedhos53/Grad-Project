"""Evaluate visual decision rules on synthetic, non-identifying scenarios.

This tool deliberately does not read, save, or transmit camera frames.  It
checks the deterministic action/safe-zone contracts around the visual models;
its metrics are not a claim of camera-model accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Importing the dashboard with model loading disabled keeps this evaluator
# deterministic and prevents it from opening a camera or loading YOLO.
os.environ.setdefault("CABINSPECTOR_DISABLE_MODEL_LOAD", "1")
from src.evaluation.metrics import binary_metrics
from src.video import driver_visual_prototype as prototype


FRAME_SHAPE = (360, 640, 3)


def _action_scenarios() -> list[dict[str, object]]:
    return [
        {
            "id": "normal_no_objects",
            "signal": "phone_call",
            "expected": False,
            "detections": [],
            "pose_points": {},
        },
        {
            "id": "phone_near_ear_with_hand_contact",
            "signal": "phone_call",
            "expected": True,
            "detections": [{"class_name": "cell phone", "confidence": 0.9, "box": (90, 90, 130, 130)}],
            "pose_points": {"left_ear": (100, 100), "right_ear": (220, 100), "left_wrist": (118, 114)},
        },
        {
            "id": "phone_near_ear_without_hand_contact",
            "signal": "phone_call",
            "expected": False,
            "detections": [{"class_name": "cell phone", "confidence": 0.9, "box": (90, 90, 130, 130)}],
            "pose_points": {"left_ear": (100, 100), "right_ear": (220, 100)},
        },
        {
            "id": "cup_at_mouth_with_hand_contact",
            "signal": "drinking",
            "expected": True,
            "detections": [{"class_name": "cup", "confidence": 0.9, "box": (310, 145, 350, 205)}],
            "pose_points": {
                "mouth_left": (300, 150),
                "mouth_right": (340, 150),
                "right_wrist": (340, 190),
            },
        },
        {
            "id": "cup_away_from_mouth",
            "signal": "drinking",
            "expected": False,
            "detections": [{"class_name": "cup", "confidence": 0.9, "box": (30, 30, 70, 90)}],
            "pose_points": {
                "mouth_left": (300, 150),
                "mouth_right": (340, 150),
                "right_wrist": (340, 190),
            },
        },
    ]


def _zone_scenarios() -> list[dict[str, object]]:
    return [
        {
            "id": "landmarks_inside_safe_box",
            "expected": False,
            "points": {"left_wrist": (20, 20), "right_wrist": (40, 40)},
            "safe_box": (0, 0, 50, 50),
        },
        {
            "id": "right_wrist_outside_safe_box",
            "expected": True,
            "points": {"left_wrist": (20, 20), "right_wrist": (95, 95)},
            "safe_box": (0, 0, 50, 50),
        },
    ]


def evaluate() -> dict[str, object]:
    action_expected: dict[str, list[bool]] = {"phone_call": [], "drinking": []}
    action_predicted: dict[str, list[bool]] = {"phone_call": [], "drinking": []}
    action_records: list[dict[str, object]] = []
    for scenario in _action_scenarios():
        frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        _, phone_call, drinking = prototype.detect_object_actions(
            frame,
            scenario["detections"],
            scenario["pose_points"],
            hand_infos=[],
            draw_overlays=False,
        )
        predicted = phone_call if scenario["signal"] == "phone_call" else drinking
        signal = str(scenario["signal"])
        action_expected[signal].append(bool(scenario["expected"]))
        action_predicted[signal].append(bool(predicted))
        action_records.append(
            {
                "id": scenario["id"],
                "signal": signal,
                "expected": bool(scenario["expected"]),
                "predicted": bool(predicted),
            }
        )

    zone_records: list[dict[str, object]] = []
    zone_expected: list[bool] = []
    zone_predicted: list[bool] = []
    for scenario in _zone_scenarios():
        frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        _, predicted, outside = prototype.check_safe_box(
            frame,
            scenario["points"],
            scenario["safe_box"],
            draw_overlays=False,
        )
        expected = bool(scenario["expected"])
        zone_expected.append(expected)
        zone_predicted.append(bool(predicted))
        zone_records.append(
            {
                "id": scenario["id"],
                "expected": expected,
                "predicted": bool(predicted),
                "outside_landmarks": list(outside),
            }
        )

    return {
        "status": "synthetic_logic_contract_only",
        "live_camera_used": False,
        "raw_frames_saved": False,
        "action_metrics": {
            signal: binary_metrics(action_expected[signal], action_predicted[signal])
            for signal in sorted(action_expected)
        },
        "safe_zone_metrics": binary_metrics(zone_expected, zone_predicted),
        "action_scenarios": action_records,
        "safe_zone_scenarios": zone_records,
        "limitations": [
            "Scenarios use synthetic landmarks and boxes rather than labelled camera frames.",
            "Results verify rule contracts only; they do not measure YOLO, MediaPipe, lighting, occlusion, or camera accuracy.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "evaluation" / "visual_logic_evaluation.json",
    )
    parser.add_argument("--no-write", action="store_true", help="print the report without writing it")
    args = parser.parse_args()
    report = evaluate()
    if not args.no_write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Evaluation written: {args.output}")
    print(f"Action scenarios: {len(report['action_scenarios'])}")
    print(f"Safe-zone scenarios: {len(report['safe_zone_scenarios'])}")
    print("Raw frames saved: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

