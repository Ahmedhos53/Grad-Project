"""Prompted local visual trials; evaluate decisions without retaining frames."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import uuid

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.visual_capture import (
    VISUAL_BEHAVIORS,
    VISUAL_CONDITIONS,
    build_visual_capture_report,
    summarize_trial,
)


BEHAVIOR_PROMPTS = {
    "eyes_closed": "Keep your face visible. For positive trials, gently close your eyes while stationary; for negative trials, keep them open.",
    "safe_zone_outside": "For positive trials, move one hand just beyond the configured safe box; for negative trials, keep it inside.",
    "phone_present": "Use a prop phone while stationary. Include absent, visible, mounted, or passenger-side difficult negatives.",
    "phone_call": "Use a prop phone near your own ear with hand contact for positive trials; test away-from-ear and passenger-side negatives.",
    "drinking": "Use an empty cup/bottle prop near or away from the mouth as requested; remain parked and do not drink while driving.",
}

CONDITIONS = VISUAL_CONDITIONS


def parse_safe_box_ratios(value: str) -> tuple[float, float, float, float]:
    try:
        ratios = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use four comma-separated values between 0 and 1") from exc
    if len(ratios) != 4 or any(not 0 <= item <= 1 for item in ratios):
        raise argparse.ArgumentTypeError("Use four comma-separated values between 0 and 1")
    x1, y1, x2, y2 = ratios
    if x1 >= x2 or y1 >= y2:
        raise argparse.ArgumentTypeError("Safe-box left/top must be less than right/bottom")
    return ratios


def _choose(prompt: str, choices: tuple[str, ...]) -> str:
    while True:
        print(prompt)
        for index, choice in enumerate(choices, start=1):
            print(f"  {index}. {choice}")
        selected = input("Choose a number: ").strip()
        if selected.isdigit() and 1 <= int(selected) <= len(choices):
            return choices[int(selected) - 1]


def _yes_no(prompt: str) -> bool:
    while True:
        value = input(f"{prompt} [y/n]: ").strip().casefold()
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def _process_frame(
    prototype,
    raw_frame,
    behavior: str,
    safe_box_ratios,
    no_eye_counter: int,
    signal_counters: dict[str, int],
):
    started = time.perf_counter()
    frame = cv2.resize(raw_frame, (prototype.PROCESS_WIDTH, prototype.PROCESS_HEIGHT))
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = safe_box_ratios
    safe_box = (int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height))
    stages: dict[str, float] = {}

    stage_start = time.perf_counter()
    frame, no_eye_counter, eye_status, _, _ = prototype.detect_face_and_eyes(
        frame,
        no_eye_counter,
        draw_overlays=False,
    )
    stages["face_and_eyes"] = _elapsed_ms(stage_start)

    stage_start = time.perf_counter()
    frame, pose_points = prototype.get_pose_points(frame, draw_overlays=False)
    stages["pose"] = _elapsed_ms(stage_start)

    stage_start = time.perf_counter()
    frame, hand_infos, _ = prototype.get_hand_points(
        frame,
        pose_points,
        draw_overlays=False,
        force_full_frame=False,
    )
    stages["hands"] = _elapsed_ms(stage_start)

    tracked_points = dict(pose_points)
    tracked_points.update(prototype.build_hand_point_map(hand_infos))
    stage_start = time.perf_counter()
    frame, raw_zone_alert, _ = prototype.check_safe_box(
        frame,
        tracked_points,
        safe_box,
        draw_overlays=False,
    )
    stages["safe_zone"] = _elapsed_ms(stage_start)

    yolo_model = prototype.yolo_model
    phone_model = prototype.phone_yolo_model or yolo_model
    stage_start = time.perf_counter()
    full_detections = prototype.run_yolo(
        frame,
        yolo_model,
        target_classes=prototype.TARGET_OBJECTS,
        imgsz=prototype.YOLO_IMAGE_SIZE,
        conf=prototype.YOLO_CONFIDENCE,
    )
    stages["yolo_full"] = _elapsed_ms(stage_start)

    phone_regions = prototype.build_phone_search_regions(frame.shape, pose_points, hand_infos)
    stage_start = time.perf_counter()
    phone_detections = prototype.run_yolo_on_regions(
        frame,
        phone_model,
        phone_regions,
        target_classes=prototype.PHONE_OBJECT_CLASSES,
        imgsz=prototype.PHONE_SEARCH_IMAGE_SIZE,
        conf=prototype.PHONE_SEARCH_CONFIDENCE,
        augment=prototype.PHONE_SEARCH_AUGMENT,
        max_det=prototype.PHONE_SEARCH_MAX_DET,
        iou=prototype.PHONE_SEARCH_IOU,
    )
    stages["yolo_regions"] = _elapsed_ms(stage_start)

    detections = full_detections + phone_detections
    phone_scores = [
        float(item["confidence"])
        for item in detections
        if item.get("class_name") in prototype.PHONE_OBJECT_CLASSES
    ]
    drink_scores = [
        float(item["confidence"])
        for item in detections
        if item.get("class_name") in prototype.DRINK_OBJECT_CLASSES
    ]

    stage_start = time.perf_counter()
    frame, raw_phone_call, raw_drinking = prototype.detect_object_actions(
        frame,
        detections,
        pose_points,
        hand_infos=hand_infos,
        draw_overlays=False,
    )
    stages["action_rules"] = _elapsed_ms(stage_start)

    signal_counters["safe_zone"], zone_alert = prototype.update_smoothing_counter(
        signal_counters["safe_zone"], raw_zone_alert
    )
    signal_counters["phone_call"], phone_call = prototype.update_smoothing_counter(
        signal_counters["phone_call"], raw_phone_call
    )
    signal_counters["drinking"], drinking = prototype.update_smoothing_counter(
        signal_counters["drinking"], raw_drinking
    )
    predictions = {
        "eyes_closed": eye_status == "Eyes Possibly Closed",
        "safe_zone_outside": bool(zone_alert),
        "phone_present": bool(phone_scores),
        "phone_call": bool(phone_call),
        "drinking": bool(drinking),
    }
    confidence = None
    confidence_kind = "rule_decision_probability_unavailable"
    if behavior in {"phone_present", "phone_call"}:
        confidence = max(phone_scores, default=None)
        confidence_kind = "uncalibrated_phone_object_detector_score"
    elif behavior == "drinking":
        confidence = max(drink_scores, default=None)
        confidence_kind = "uncalibrated_container_object_detector_score"
    stages["frame_total"] = _elapsed_ms(started)

    return (
        frame,
        no_eye_counter,
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "predicted": bool(predictions[behavior]),
            "confidence": confidence,
            "confidence_kind": confidence_kind,
            "latency_ms": stages,
        },
    )


def _available_behaviors(prototype) -> tuple[str, ...]:
    available = []
    if prototype.face_mesh is not None or prototype.face_cascade is not None:
        available.append("eyes_closed")
    if prototype.pose is not None:
        available.append("safe_zone_outside")
    if prototype.yolo_model is not None:
        available.append("phone_present")
        if prototype.pose is not None:
            available.extend(("phone_call", "drinking"))
    return tuple(item for item in VISUAL_BEHAVIORS if item in available)


def _capture_trials(prototype, camera, args) -> tuple[list[dict[str, object]], dict[str, bool]]:
    availability = {
        "face_mesh": prototype.face_mesh is not None,
        "haar_face_fallback": prototype.face_cascade is not None,
        "pose": prototype.pose is not None,
        "hands": prototype.hands is not None,
        "yolo": prototype.yolo_model is not None,
    }
    available_behaviors = _available_behaviors(prototype)
    if not available_behaviors:
        raise RuntimeError("No supported visual behaviour can run with the loaded model set")
    eye_detector = prototype.detect_face_and_eyes
    eye_detector._eye_ratio_history = deque(maxlen=prototype.EYE_RATIO_HISTORY_SIZE)
    eye_detector._eye_closed_run = 0

    completed: list[dict[str, object]] = []
    cv2.namedWindow("CabInspector - Evaluation Preview", cv2.WINDOW_NORMAL)
    try:
        for trial_number in range(1, args.trials + 1):
            behavior = _choose(
                f"Trial {trial_number}/{args.trials} - choose the behaviour to evaluate",
                available_behaviors,
            )
            print(BEHAVIOR_PROMPTS[behavior])
            expected = _yes_no("Is the target condition present in this trial?")
            conditions = {
                name: _choose(f"Record the {name.replace('_', ' ')} condition", choices)
                for name, choices in CONDITIONS.items()
            }
            ready = input("Stage the scenario while parked; press Enter to start this trial (q to stop): ").strip()
            if ready.casefold() == "q":
                break

            started_at = datetime.now(timezone.utc).isoformat()
            observations = []
            eye_counter = 0
            signal_counters = {"safe_zone": 0, "phone_call": 0, "drinking": 0}
            eye_detector._eye_ratio_history.clear()
            eye_detector._eye_closed_run = 0
            for frame_index in range(1, args.frames_per_trial + 1):
                captured, raw_frame = camera.read()
                if not captured or raw_frame is None:
                    raise RuntimeError("The camera stopped returning frames; no report was written")
                _, eye_counter, observation = _process_frame(
                    prototype,
                    raw_frame,
                    behavior,
                    args.safe_box_ratios,
                    eye_counter,
                    signal_counters,
                )
                observations.append(observation)
                cv2.putText(
                    raw_frame,
                    "EVALUATION ONLY - LIVE PREVIEW, NO FRAMES SAVED",
                    (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 220, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    raw_frame,
                    f"Trial {trial_number}/{args.trials}  Frame {frame_index}/{args.frames_per_trial}",
                    (12, 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 220, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("CabInspector - Evaluation Preview", raw_frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            if len(observations) < max(3, args.frames_per_trial // 2):
                print("Trial cancelled before enough frames; it was not scored.")
                break
            completed.append(
                summarize_trial(
                    trial_number=trial_number,
                    behavior=behavior,
                    expected=expected,
                    conditions=conditions,
                    observations=observations,
                    started_at_utc=started_at,
                )
            )
            print(
                f"Scored trial {trial_number}: expected={expected}, "
                f"predicted={completed[-1]['predicted']}, frames={len(observations)}"
            )
            if input("Continue with another trial? [y/n]: ").strip().casefold() not in {"y", "yes"}:
                break
    finally:
        cv2.destroyWindow("CabInspector - Evaluation Preview")

    return completed, availability


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--frames-per-trial", type=int, default=20)
    parser.add_argument(
        "--safe-box-ratios",
        type=parse_safe_box_ratios,
        default=(0.20, 0.20, 0.80, 0.95),
        help="normalized left,top,right,bottom values; default 0.20,0.20,0.80,0.95",
    )
    parser.add_argument("--consent-confirmed", action="store_true", help="confirm all visible people consent to this local test")
    parser.add_argument("--output", type=Path, help="optional aggregate result JSON path")
    parser.add_argument("--no-write", action="store_true", help="print aggregate metrics only")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing an existing output file")
    args = parser.parse_args()
    if not args.consent_confirmed:
        parser.error("Refusing camera access without --consent-confirmed")
    if not 1 <= args.trials <= 100 or not 5 <= args.frames_per_trial <= 120:
        parser.error("Use 1-100 trials and 5-120 frames per trial")
    if not sys.stdin.isatty() or input(
        "Confirm the camera subject and everyone visible consented; no frames will be saved. Type CONSENTED: "
    ).strip() != "CONSENTED":
        print("Consent was not confirmed; the camera was not opened.")
        return 2
    if args.output and args.output.exists() and not args.overwrite:
        parser.error("Output exists; choose a new path or add --overwrite")

    if os.environ.get("CABINSPECTOR_DISABLE_MODEL_LOAD", "").strip().casefold() in {"1", "true", "yes", "on"}:
        parser.error("Unset CABINSPECTOR_DISABLE_MODEL_LOAD to run real visual models")
    os.environ["CABINSPECTOR_ENABLE_AUTO_SCREENSHOTS"] = "0"
    os.environ["CABINSPECTOR_USE_AUDIO"] = "0"
    os.environ["CABINSPECTOR_USE_TELEMETRY_REPLAY"] = "0"
    from src.video import driver_visual_prototype as prototype

    camera = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(args.camera_index)
    if not camera.isOpened():
        camera.release()
        print("Camera could not be opened; no report was written.")
        return 1
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, prototype.CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, prototype.CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, prototype.CAMERA_FPS)

    try:
        trials, availability = _capture_trials(prototype, camera, args)
    except (RuntimeError, cv2.error) as exc:
        print(f"Visual evaluation stopped; no report was written: {exc}")
        return 1
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if not trials:
        print("No complete trial was collected; no report was written.")
        return 2
    report = build_visual_capture_report(
        trials,
        model_availability=availability,
        safe_box_ratios=args.safe_box_ratios,
    )
    if args.no_write:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = args.output or PROJECT_ROOT / "outputs" / "evaluation" / f"visual_capture_evaluation_{timestamp}.json"
    if output_path.exists() and not args.overwrite:
        print("Output exists; no report was written. Choose another --output path or add --overwrite.")
        return 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Evaluation results written (no frames saved): {output_path}")
    for behavior, metrics in report["per_behavior"].items():
        print(
            f"{behavior}: n={metrics['support']}, precision={metrics['precision']}, "
            f"recall={metrics['recall']}, F1={metrics['f1']}, "
            f"FP={metrics['false_positive']}, FN={metrics['false_negative']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
