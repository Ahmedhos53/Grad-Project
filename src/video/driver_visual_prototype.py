import csv
import cv2
import ctypes
from contextlib import nullcontext
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from datetime import datetime
from pathlib import Path
import numpy as np

# Keep direct execution (`python src/video/driver_visual_prototype.py`) working
# while sharing the audio component as a normal project module.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.audio_event_detector import DEFAULT_MODEL_PATH
from src.evaluation.runtime_metrics import RuntimeMetrics
from src.audio.speech_analysis_pipeline import SpeechAnalysisPipeline, SpeechAnalysisState
from src.telemetry.replay import TelemetryReplay, TelemetryReplayState
from src.video.localized_text import (
    contains_arabic,
    draw_rtl_text,
    font_size_from_opencv_scale,
    truncate_localized_text,
)


def env_flag(name, default=False):
    value = os.environ.get(name)

    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_positive_float(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def is_quit_key(key_code):
    """Accept normal/uppercase Q and Escape from OpenCV's key APIs."""
    return key_code in {27, ord("q"), ord("Q")}


DISABLE_MODEL_LOAD = env_flag("CABINSPECTOR_DISABLE_MODEL_LOAD", False)
RUNTIME_METRICS_PATH = os.environ.get("CABINSPECTOR_RUNTIME_METRICS_PATH", "").strip()
RUNTIME_METRICS = RuntimeMetrics() if RUNTIME_METRICS_PATH else None


def runtime_measure(stage):
    """Return a no-op or aggregate timing context for an explicit evaluator run."""

    return RUNTIME_METRICS.measure(stage) if RUNTIME_METRICS is not None else nullcontext()


def runtime_observe(stage, milliseconds):
    if RUNTIME_METRICS is not None:
        RUNTIME_METRICS.observe(stage, milliseconds)


def runtime_increment(counter, amount=1):
    if RUNTIME_METRICS is not None:
        RUNTIME_METRICS.increment(counter, amount)


def runtime_gauge(name, value):
    if RUNTIME_METRICS is not None:
        RUNTIME_METRICS.set_gauge(name, value)


# =========================
# Imports
# =========================

if DISABLE_MODEL_LOAD:
    mp = None
    mp_pose = None
    mp_hands = None
    mp_face_mesh = None
    mp_drawing = None
else:
    try:
        import mediapipe as mp

        try:
            mp_pose = mp.solutions.pose
            mp_hands = mp.solutions.hands
            mp_face_mesh = mp.solutions.face_mesh
            mp_drawing = mp.solutions.drawing_utils
        except AttributeError:
            from mediapipe.python.solutions import pose as mp_pose
            from mediapipe.python.solutions import hands as mp_hands
            from mediapipe.python.solutions import face_mesh as mp_face_mesh
            from mediapipe.python.solutions import drawing_utils as mp_drawing

    except ImportError:
        mp = None
        mp_pose = None
        mp_hands = None
        mp_face_mesh = None
        mp_drawing = None


if DISABLE_MODEL_LOAD:
    YOLO = None
else:
    try:
        from ultralytics import YOLO
    except ImportError:
        YOLO = None


# =========================
# Display Helpers
# =========================

def get_primary_display_size():
    try:
        user32 = ctypes.windll.user32
        width = int(user32.GetSystemMetrics(0))
        height = int(user32.GetSystemMetrics(1))

        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass

    return 1280, 720


# =========================
# Settings
# =========================

USE_YOLO = env_flag("CABINSPECTOR_USE_YOLO", True) and not DISABLE_MODEL_LOAD
# Audio is opt-in so the existing visual demo remains usable on systems without
# a microphone or the LiteRT audio dependency.
USE_AUDIO = env_flag("CABINSPECTOR_USE_AUDIO", False) and not DISABLE_MODEL_LOAD
# Telemetry remains opt-in until its recorded-trip replay is explicitly enabled.
# This is a single-program integration, but it is not a claim of live phone input.
USE_TELEMETRY_REPLAY = (
    env_flag("CABINSPECTOR_USE_TELEMETRY_REPLAY", False) and not DISABLE_MODEL_LOAD
)
TELEMETRY_REPLAY_TRIP = env_int("CABINSPECTOR_TELEMETRY_REPLAY_TRIP", 1)
TELEMETRY_REPLAY_SPEED = env_positive_float("CABINSPECTOR_TELEMETRY_REPLAY_SPEED", 10.0)
# The externally pretrained PRIMUS telemetry model is the default replay path.
# Set to ``random_forest`` only to compare the retained project-trained baseline.
TELEMETRY_REPLAY_MODEL = os.environ.get("CABINSPECTOR_TELEMETRY_REPLAY_MODEL", "primus").strip().lower()
# Transcript persistence is explicitly opt-in. Screenshots otherwise redact it.
STORE_TRANSCRIPTS = env_flag("CABINSPECTOR_STORE_TRANSCRIPTS", False)
FAST_DEMO_MODE = True
DEBUG_OVERLAYS = False
# Optional bounded run for local smoke tests and reproducibility checks.  The
# normal dashboard remains continuous when this is unset or zero.
MAX_FRAMES = max(0, env_int("CABINSPECTOR_MAX_FRAMES", 0))

CAMERA_WIDTH = 640 if FAST_DEMO_MODE else 1280
CAMERA_HEIGHT = 360 if FAST_DEMO_MODE else 720
CAMERA_FPS = 30

PROCESS_WIDTH = 640 if FAST_DEMO_MODE else 1280
PROCESS_HEIGHT = 360 if FAST_DEMO_MODE else 720

PRIMARY_DISPLAY_WIDTH, PRIMARY_DISPLAY_HEIGHT = get_primary_display_size()
if FAST_DEMO_MODE:
    DISPLAY_WIDTH = min(PRIMARY_DISPLAY_WIDTH, 1280)
    DISPLAY_HEIGHT = max(480, int(DISPLAY_WIDTH * PRIMARY_DISPLAY_HEIGHT / PRIMARY_DISPLAY_WIDTH))
else:
    DISPLAY_WIDTH = PRIMARY_DISPLAY_WIDTH
    DISPLAY_HEIGHT = PRIMARY_DISPLAY_HEIGHT
DISPLAY_FULLSCREEN = True
WINDOW_NAME = "CabInspector - Driver Visual Behaviour Prototype"
# Keep the camera as the focal point.  The two side panels deliberately use
# compact, readable text rather than trying to show every signal in one column.
LEFT_SIDEBAR_WIDTH = min(300, max(250, int(DISPLAY_WIDTH * 0.21)))
RIGHT_SIDEBAR_WIDTH = min(300, max(250, int(DISPLAY_WIDTH * 0.21)))

# Faster live-demo configuration.
ASYNC_YOLO_INFERENCE = True
YOLO_CPU_THREADS = 2 if FAST_DEMO_MODE else max(1, (os.cpu_count() or 2) // 2)
YOLO_EVERY_N_FRAMES = 20 if FAST_DEMO_MODE else 6

# yolov8n is much faster than larger models and is better for live webcam demos.
YOLO_MODEL_NAME = "yolov8n.pt"
PHONE_YOLO_MODEL_NAME = "yolov8n.pt"

YOLO_IMAGE_SIZE = 320 if FAST_DEMO_MODE else 640
YOLO_CONFIDENCE = 0.20 if FAST_DEMO_MODE else 0.18

EYE_ANALYSIS_EVERY_N_FRAMES = 3 if FAST_DEMO_MODE else 1
POSE_ANALYSIS_EVERY_N_FRAMES = 3 if FAST_DEMO_MODE else 1
HAND_ANALYSIS_EVERY_N_FRAMES = 2 if FAST_DEMO_MODE else 1
HAND_FULL_FRAME_EVERY_N_FRAMES = 4 if FAST_DEMO_MODE else 1
HAND_TRACK_HOLD_FRAMES = 8 if FAST_DEMO_MODE else 3

# Phone detection needs extra help because small phones are easy to miss in a
# full-frame 360p processing pass, but we keep the search lightweight.
PHONE_SEARCH_EVERY_N_FRAMES = 8 if FAST_DEMO_MODE else 3
PHONE_SEARCH_IMAGE_SIZE = 320 if FAST_DEMO_MODE else 640
PHONE_SEARCH_CONFIDENCE = 0.08 if FAST_DEMO_MODE else 0.07
PHONE_SEARCH_AUGMENT = False
PHONE_SEARCH_MAX_DET = 2
PHONE_SEARCH_MAX_REGIONS = 2
PHONE_SEARCH_IOU = 0.35
PHONE_SEARCH_MARGIN_X = 120 if FAST_DEMO_MODE else 200
PHONE_SEARCH_MARGIN_Y = 100 if FAST_DEMO_MODE else 160
PHONE_OBJECT_MIN_CONFIDENCE = 0.10 if FAST_DEMO_MODE else 0.09
PHONE_OBJECT_SCORE_THRESHOLD = 0.34 if FAST_DEMO_MODE else 0.32
PHONE_OBJECT_HOLD_FRAMES = 18 if FAST_DEMO_MODE else 7
PHONE_CONTEXT_HEAD_WEIGHT = 0.35
PHONE_CONTEXT_HAND_WEIGHT = 0.25

EYE_CLOSED_FRAME_LIMIT = 10
EYE_RATIO_HISTORY_SIZE = 5
EYE_CLOSED_CONFIRM_FRAMES = 3
FACE_MESH_EYE_OPEN_RATIO_THRESHOLD = 0.18
FACE_MESH_EYE_CLOSED_RATIO_THRESHOLD = 0.11

# Keep alert latency low in fast demo mode.
ALERT_CONFIRM_FRAMES = 1 if FAST_DEMO_MODE else 3
AUTO_SCREENSHOT_COOLDOWN_SECONDS = 5
EVENT_LOG_FLUSH_EVERY_N_ROWS = 30
# Automatic alert screenshots are useful for an interactive review, but an
# evaluator can disable them to avoid persisting camera frames.
AUTO_SCREENSHOTS_ENABLED = env_flag("CABINSPECTOR_ENABLE_AUTO_SCREENSHOTS", True)

OUTPUT_DIR = Path("outputs")
SCREENSHOT_DIR = OUTPUT_DIR / "screenshots"
EVENT_LOG_PATH = OUTPUT_DIR / "events_log.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

PHONE_OBJECT_CLASSES = {
    "cell phone",
}

DRINK_OBJECT_CLASSES = {
    "bottle",
    "cup",
    "wine glass",
    "vase",  # flask-like containers can sometimes land here in COCO
}

TARGET_OBJECTS = PHONE_OBJECT_CLASSES | DRINK_OBJECT_CLASSES


# =========================
# Haar Cascade Models
# =========================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

face_cascade_alt = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
)

face_profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

eye_cascade_glasses = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
)


# =========================
# YOLO Object Detection
# =========================

yolo_model = None
phone_yolo_model = None

if USE_YOLO and YOLO is not None:
    try:
        import torch

        torch.set_num_threads(YOLO_CPU_THREADS)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    print("Loading YOLO model...")
    yolo_model = YOLO(YOLO_MODEL_NAME)
    print("YOLO loaded.")

    if PHONE_YOLO_MODEL_NAME == YOLO_MODEL_NAME:
        phone_yolo_model = yolo_model
    else:
        try:
            print("Loading phone search YOLO model...")
            phone_yolo_model = YOLO(PHONE_YOLO_MODEL_NAME)
            print("Phone search YOLO loaded.")
        except Exception as exc:
            phone_yolo_model = yolo_model
            print(f"Phone search YOLO fallback to main model: {exc}")
elif USE_YOLO:
    print("YOLO not installed. Object detection disabled.")
    USE_YOLO = False


# =========================
# MediaPipe Pose
# =========================

pose = None

if mp_pose is not None:
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0 if FAST_DEMO_MODE else 1,
        enable_segmentation=False,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.45
    )
elif not DISABLE_MODEL_LOAD:
    print("MediaPipe not installed. Skeleton detection disabled.")

hands = None

if mp_hands is not None:
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0 if FAST_DEMO_MODE else 1,
        min_detection_confidence=0.30 if FAST_DEMO_MODE else 0.35,
        min_tracking_confidence=0.30 if FAST_DEMO_MODE else 0.35
    )
elif not DISABLE_MODEL_LOAD:
    print("MediaPipe not installed. Hand detection disabled.")

face_mesh = None

if mp_face_mesh is not None:
    try:
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.50,
            min_tracking_confidence=0.50,
        )
    except Exception as exc:
        face_mesh = None
        print(f"Face mesh initialization failed, falling back to Haar eye detection: {exc}")
elif not DISABLE_MODEL_LOAD:
    print("MediaPipe not installed. Face mesh detection disabled.")


# =========================
# Utility Functions
# =========================

def draw_text(frame, text, position, color=(255, 255, 255), scale=0.7, thickness=2):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def point_inside_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def get_box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def distance_between_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def distance_point_to_box(point, box):
    """
    Distance from a landmark point to an object bounding box.
    If the point is inside the box, distance is 0.
    This is better than only comparing with the object center.
    """
    px, py = point
    x1, y1, x2, y2 = box

    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)

    return (dx ** 2 + dy ** 2) ** 0.5


def get_mouth_center(points):
    if "mouth_left" in points and "mouth_right" in points:
        x = (points["mouth_left"][0] + points["mouth_right"][0]) // 2
        y = (points["mouth_left"][1] + points["mouth_right"][1]) // 2
        return (x, y)

    if "nose" in points:
        # fallback: approximate mouth below nose
        return (points["nose"][0], points["nose"][1] + 45)

    return None


def face_mesh_point(landmarks, index, frame_w, frame_h):
    landmark = landmarks[index]
    return (
        clamp_int(landmark.x * frame_w, 0, frame_w - 1),
        clamp_int(landmark.y * frame_h, 0, frame_h - 1),
    )


def face_mesh_eye_ratio(landmarks, frame_w, frame_h, side):
    if side == "left":
        horizontal = distance_between_points(
            face_mesh_point(landmarks, 33, frame_w, frame_h),
            face_mesh_point(landmarks, 133, frame_w, frame_h),
        )
        vertical_pairs = [
            (159, 145),
            (160, 144),
        ]
    else:
        horizontal = distance_between_points(
            face_mesh_point(landmarks, 362, frame_w, frame_h),
            face_mesh_point(landmarks, 263, frame_w, frame_h),
        )
        vertical_pairs = [
            (386, 374),
            (385, 380),
        ]

    if horizontal <= 0:
        return None

    vertical_distances = [
        distance_between_points(
            face_mesh_point(landmarks, top_index, frame_w, frame_h),
            face_mesh_point(landmarks, bottom_index, frame_w, frame_h),
        )
        for top_index, bottom_index in vertical_pairs
    ]

    vertical = sum(vertical_distances) / len(vertical_distances)
    return vertical / horizontal


def face_mesh_bounding_box(landmarks, frame_w, frame_h):
    xs = [clamp_int(landmark.x * frame_w, 0, frame_w - 1) for landmark in landmarks]
    ys = [clamp_int(landmark.y * frame_h, 0, frame_h - 1) for landmark in landmarks]

    if not xs or not ys:
        return None

    x1 = max(min(xs), 0)
    y1 = max(min(ys), 0)
    x2 = min(max(xs), frame_w - 1)
    y2 = min(max(ys), frame_h - 1)

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)


def format_landmark_name(name):
    return name.replace("_", " ").title()


def choose_largest_box(boxes):
    if not boxes:
        return None

    return max(boxes, key=lambda box: int(box[2]) * int(box[3]))


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area <= 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def dedupe_detections(detections, iou_threshold=0.5):
    unique_detections = []

    for detection in sorted(
        detections,
        key=lambda item: float(item.get("confidence", 0.0)),
        reverse=True
    ):
        matched_index = None

        for index, existing in enumerate(unique_detections):
            if detection["class_name"] != existing["class_name"]:
                continue

            if box_iou(detection["box"], existing["box"]) >= iou_threshold:
                matched_index = index
                break

        if matched_index is None:
            unique_detections.append(detection)

    return unique_detections


def build_phone_search_regions(frame_shape, pose_points, hand_infos):
    frame_h, frame_w, _ = frame_shape
    regions = []

    head_box = build_phone_head_box_from_pose(pose_points, frame_shape)

    if head_box is not None:
        regions.append({
            "name": "head_roi",
            "box": head_box,
        })

    for ear_name in ("left_ear", "right_ear"):
        if ear_name not in pose_points:
            continue

        ear_x, ear_y = pose_points[ear_name]
        ear_pad_x = max(PHONE_SEARCH_MARGIN_X // 2, int(frame_w * 0.16))
        ear_pad_y = max(PHONE_SEARCH_MARGIN_Y // 2, int(frame_h * 0.16))

        x1 = clamp_int(ear_x - ear_pad_x, 0, frame_w)
        y1 = clamp_int(ear_y - ear_pad_y, 0, frame_h)
        x2 = clamp_int(ear_x + ear_pad_x, 0, frame_w)
        y2 = clamp_int(ear_y + ear_pad_y, 0, frame_h)

        if (x2 - x1) >= 40 and (y2 - y1) >= 40:
            regions.append({
                "name": f"{ear_name}_roi",
                "box": (x1, y1, x2, y2),
            })

    for hand_info in hand_infos or []:
        for suffix, point in (
            ("center", hand_info["center"]),
            ("wrist", hand_info["wrist"]),
        ):
            x1 = clamp_int(point[0] - PHONE_SEARCH_MARGIN_X, 0, frame_w)
            y1 = clamp_int(point[1] - PHONE_SEARCH_MARGIN_Y, 0, frame_h)
            x2 = clamp_int(point[0] + PHONE_SEARCH_MARGIN_X, 0, frame_w)
            y2 = clamp_int(point[1] + PHONE_SEARCH_MARGIN_Y, 0, frame_h)

            if (x2 - x1) < 60 or (y2 - y1) < 60:
                continue

            regions.append({
                "name": f"{hand_info['label'].replace(' ', '_').lower()}_{suffix}_roi",
                "box": (x1, y1, x2, y2),
            })

    if not regions:
        regions.append({
            "name": "upper_body_fallback_roi",
            "box": (
                int(frame_w * 0.12),
                0,
                int(frame_w * 0.88),
                int(frame_h * 0.72),
            ),
        })

    return limit_phone_search_regions(regions)


def limit_phone_search_regions(regions):
    if not regions:
        return []

    priority_prefixes = ("head_roi", "left_ear_roi", "right_ear_roi")
    selected = []

    for prefix in priority_prefixes:
        for region in regions:
            if region["name"] == prefix:
                selected.append(region)
                break

    for region in regions:
        if len(selected) >= PHONE_SEARCH_MAX_REGIONS:
            break

        if region in selected:
            continue

        if any(box_iou(region["box"], existing["box"]) > 0.65 for existing in selected):
            continue

        selected.append(region)

    return selected[:PHONE_SEARCH_MAX_REGIONS]


def expand_box(box, pad_x, pad_y, frame_shape):
    if box is None:
        return None

    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = box

    return (
        clamp_int(x1 - pad_x, 0, frame_w),
        clamp_int(y1 - pad_y, 0, frame_h),
        clamp_int(x2 + pad_x, 0, frame_w),
        clamp_int(y2 + pad_y, 0, frame_h),
    )


def build_phone_head_box_from_pose(pose_points, frame_shape):
    frame_h, frame_w, _ = frame_shape

    face_anchor_names = [
        "left_ear",
        "right_ear",
        "nose",
        "mouth_left",
        "mouth_right",
    ]

    face_points = [
        pose_points[name]
        for name in face_anchor_names
        if name in pose_points
    ]

    if face_points:
        xs = [point[0] for point in face_points]
        ys = [point[1] for point in face_points]
        face_width = max(1, max(xs) - min(xs))
        face_height = max(1, max(ys) - min(ys))

        pad_x = max(
            PHONE_SEARCH_MARGIN_X,
            int(face_width * 1.20),
            int(frame_w * 0.08),
        )
        pad_top = max(
            PHONE_SEARCH_MARGIN_Y,
            int(face_height * 1.60),
            int(frame_h * 0.10),
        )
        pad_bottom = max(
            PHONE_SEARCH_MARGIN_Y // 2,
            int(face_height * 0.95),
            int(frame_h * 0.06),
        )

        x1 = clamp_int(min(xs) - pad_x, 0, frame_w)
        y1 = clamp_int(min(ys) - pad_top, 0, frame_h)
        x2 = clamp_int(max(xs) + pad_x, 0, frame_w)
        y2 = clamp_int(max(ys) + pad_bottom, 0, frame_h)

        if (x2 - x1) >= 60 and (y2 - y1) >= 60:
            return (x1, y1, x2, y2)

    shoulder_points = [
        pose_points[name]
        for name in ("left_shoulder", "right_shoulder")
        if name in pose_points
    ]

    if len(shoulder_points) == 2:
        xs = [point[0] for point in shoulder_points]
        ys = [point[1] for point in shoulder_points]
        shoulder_span = max(1, distance_between_points(*shoulder_points))

        pad_x = max(PHONE_SEARCH_MARGIN_X, int(shoulder_span * 0.85))
        pad_up = max(PHONE_SEARCH_MARGIN_Y, int(shoulder_span * 1.20))
        pad_down = max(PHONE_SEARCH_MARGIN_Y // 2, int(shoulder_span * 0.45))

        center_x = sum(xs) // len(xs)
        top_y = min(ys) - pad_up
        bottom_y = max(ys) + pad_down

        x1 = clamp_int(center_x - pad_x, 0, frame_w)
        y1 = clamp_int(top_y, 0, frame_h)
        x2 = clamp_int(center_x + pad_x, 0, frame_w)
        y2 = clamp_int(bottom_y, 0, frame_h)

        if (x2 - x1) >= 60 and (y2 - y1) >= 60:
            return (x1, y1, x2, y2)

    return None


def filter_phone_detections(detections, pose_points, hand_infos, frame_shape):
    filtered = []

    if not detections:
        return filtered

    frame_h, frame_w, _ = frame_shape
    head_box = build_phone_head_box_from_pose(pose_points, frame_shape)
    if head_box is not None:
        head_box = expand_box(
            head_box,
            int((head_box[2] - head_box[0]) * 0.12),
            int((head_box[3] - head_box[1]) * 0.10),
            frame_shape,
        )

    ears = [
        pose_points[name]
        for name in ("left_ear", "right_ear")
        if name in pose_points
    ]
    contact_points = get_hand_contact_points(pose_points, hand_infos or [])

    head_distance_threshold = max(120, int(max(frame_w, frame_h) * 0.18))
    hand_distance_threshold = max(140, int(max(frame_w, frame_h) * 0.20))

    for detection in detections:
        class_name = detection["class_name"]

        if class_name not in PHONE_OBJECT_CLASSES:
            continue

        confidence = float(detection.get("confidence", 0.0))
        if confidence < PHONE_OBJECT_MIN_CONFIDENCE:
            continue

        box = detection["box"]
        head_score = 0.0
        hand_score = 0.0

        if head_box is not None:
            iou = box_iou(box, head_box)
            if iou > 0.01:
                head_score = max(head_score, PHONE_CONTEXT_HEAD_WEIGHT + min(0.12, iou * 2.0))

        for ear in ears:
            if distance_point_to_box(ear, box) < head_distance_threshold:
                head_score = max(head_score, PHONE_CONTEXT_HEAD_WEIGHT)
                break

        for contact_point in contact_points:
            if distance_point_to_box(contact_point, box) < hand_distance_threshold:
                hand_score = max(hand_score, PHONE_CONTEXT_HAND_WEIGHT)
                break

        if head_score <= 0.0 and hand_score <= 0.0:
            continue

        total_score = confidence + head_score + hand_score

        if head_score > 0.0 and total_score >= PHONE_OBJECT_SCORE_THRESHOLD:
            enriched_detection = dict(detection)
            enriched_detection["phone_score"] = total_score
            enriched_detection["phone_context"] = "head"
            filtered.append(enriched_detection)
        elif hand_score > 0.0 and total_score >= (PHONE_OBJECT_SCORE_THRESHOLD + 0.03):
            enriched_detection = dict(detection)
            enriched_detection["phone_score"] = total_score
            enriched_detection["phone_context"] = "hand"
            filtered.append(enriched_detection)

    return dedupe_detections(filtered, iou_threshold=0.4)


def clamp_int(value, minimum, maximum):
    return max(minimum, min(int(value), maximum))


def merge_boxes(boxes):
    if not boxes:
        return None

    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)

    return (x1, y1, x2, y2)


def scale_point(point, scale_x, scale_y):
    return (
        int(round(point[0] * scale_x)),
        int(round(point[1] * scale_y))
    )


def scale_box(box, scale_x, scale_y):
    x1, y1, x2, y2 = box
    return (
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
        int(round(x2 * scale_x)),
        int(round(y2 * scale_y))
    )


def scale_pose_points(points, scale_x, scale_y):
    return {
        name: scale_point(point, scale_x, scale_y)
        for name, point in points.items()
    }


def scale_hand_infos(hand_infos, scale_x, scale_y):
    scaled_infos = []

    for hand_info in hand_infos:
        scaled_info = dict(hand_info)
        scaled_info["center"] = scale_point(hand_info["center"], scale_x, scale_y)
        scaled_info["wrist"] = scale_point(hand_info["wrist"], scale_x, scale_y)
        scaled_info["bbox"] = scale_box(hand_info["bbox"], scale_x, scale_y)

        if "landmarks" in hand_info:
            scaled_info["landmarks"] = [
                scale_point(point, scale_x, scale_y)
                for point in hand_info["landmarks"]
            ]

        scaled_infos.append(scaled_info)

    return scaled_infos


def scale_detections(detections, scale_x, scale_y):
    scaled_detections = []

    for detection in detections:
        scaled_detection = dict(detection)
        scaled_detection["box"] = scale_box(detection["box"], scale_x, scale_y)
        scaled_detections.append(scaled_detection)

    return scaled_detections


# =========================
# Dashboard UI
# =========================

SIDEBAR_BG = (14, 16, 21)
SIDEBAR_PANEL = (23, 27, 35)
SIDEBAR_PANEL_ALT = (31, 36, 46)
SIDEBAR_BORDER = (66, 74, 88)
SIDEBAR_TEXT = (250, 250, 252)
SIDEBAR_MUTED = (220, 225, 232)
SIDEBAR_GREEN = (0, 180, 0)
SIDEBAR_YELLOW = (0, 200, 255)
SIDEBAR_ORANGE = (0, 165, 255)
SIDEBAR_RED = (0, 0, 220)
SIDEBAR_ACCENT = (180, 120, 60)
SIDEBAR_SUBTLE = (36, 40, 50)
SIDEBAR_DARK = (12, 14, 18)
FONT = cv2.FONT_HERSHEY_SIMPLEX
SIDEBAR_TITLE_SCALE = 0.56
SIDEBAR_BODY_SCALE = 0.42
SIDEBAR_SMALL_SCALE = 0.36


def fit_frame_to_box(frame, target_width, target_height, background_color=SIDEBAR_DARK):
    canvas = np.full((target_height, target_width, 3), background_color, dtype=np.uint8)

    if frame is None or target_width <= 0 or target_height <= 0:
        return canvas

    source_h, source_w = frame.shape[:2]

    if source_h <= 0 or source_w <= 0:
        return canvas

    scale = min(target_width / source_w, target_height / source_h)
    resized_w = max(1, int(round(source_w * scale)))
    resized_h = max(1, int(round(source_h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=interpolation)

    x_offset = max(0, (target_width - resized_w) // 2)
    y_offset = max(0, (target_height - resized_h) // 2)
    canvas[y_offset:y_offset + resized_h, x_offset:x_offset + resized_w] = resized

    return canvas


def draw_badge(
    panel,
    text,
    x,
    y,
    bg_color,
    text_color=SIDEBAR_TEXT,
    font_scale=0.48,
    thickness=1,
    padding_x=10,
    padding_y=6,
):
    (text_width, text_height), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    badge_width = text_width + (padding_x * 2)
    badge_height = text_height + baseline + (padding_y * 2)

    cv2.rectangle(panel, (x, y), (x + badge_width, y + badge_height), bg_color, -1)
    cv2.rectangle(panel, (x, y), (x + badge_width, y + badge_height), SIDEBAR_BORDER, 1)
    draw_sidebar_text(
        panel,
        text,
        (x + padding_x, y + padding_y + text_height),
        text_color,
        font_scale,
        thickness,
    )

    return badge_width, badge_height


def draw_sidebar_text(frame, text, position, color=SIDEBAR_TEXT, scale=SIDEBAR_BODY_SCALE, thickness=1):
    cv2.putText(
        frame,
        text,
        position,
        FONT,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def draw_card(panel, x, y, width, height, title, accent_color=SIDEBAR_ACCENT):
    cv2.rectangle(panel, (x, y), (x + width, y + height), SIDEBAR_PANEL, -1)
    cv2.rectangle(panel, (x, y), (x + width, y + height), SIDEBAR_BORDER, 1)
    cv2.rectangle(panel, (x, y), (x + 5, y + height), accent_color, -1)
    draw_sidebar_text(panel, title, (x + 13, y + 20), SIDEBAR_TEXT, SIDEBAR_TITLE_SCALE, 1)
    return y + 28


def draw_kv_row(panel, x, y, width, label, value, value_color=SIDEBAR_TEXT):
    draw_sidebar_text(panel, label.upper(), (x, y + 12), SIDEBAR_MUTED, SIDEBAR_SMALL_SCALE, 1)
    draw_sidebar_text(panel, value, (x, y + 29), value_color, SIDEBAR_BODY_SCALE, 2)


def draw_small_line(panel, x, y, width, color=SIDEBAR_BORDER):
    cv2.line(panel, (x, y), (x + width, y), color, 1)


def wrap_text_lines(text, max_width, font_scale=0.45, thickness=1):
    if not text:
        return [""]

    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        candidate = word if not current_line else f"{current_line} {word}"
        candidate_width = cv2.getTextSize(candidate, FONT, font_scale, thickness)[0][0]

        if candidate_width <= max_width or not current_line:
            current_line = candidate
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines


def truncate_text_to_width(text, max_width, font_scale=SIDEBAR_BODY_SCALE, thickness=1):
    text = str(text or "")

    if max_width <= 0:
        return ""

    text_width = cv2.getTextSize(text, FONT, font_scale, thickness)[0][0]
    if text_width <= max_width:
        return text

    suffix = "..."
    suffix_width = cv2.getTextSize(suffix, FONT, font_scale, thickness)[0][0]
    if suffix_width > max_width:
        return ""

    trimmed = text
    while trimmed:
        candidate = f"{trimmed}{suffix}"
        candidate_width = cv2.getTextSize(candidate, FONT, font_scale, thickness)[0][0]
        if candidate_width <= max_width:
            return candidate
        trimmed = trimmed[:-1]

    return suffix


def get_eye_badge_color(eye_status):
    text = (eye_status or "").lower()

    if "open" in text:
        return SIDEBAR_GREEN
    if "checking" in text:
        return SIDEBAR_YELLOW
    if "closed" in text or "no face" in text:
        return SIDEBAR_RED
    return SIDEBAR_SUBTLE


def get_hand_badge_color(hand_status):
    text = (hand_status or "").lower()

    if "detected" in text and "no hands" not in text and "disabled" not in text:
        return SIDEBAR_GREEN
    if "disabled" in text:
        return SIDEBAR_SUBTLE
    return SIDEBAR_RED


def get_safe_zone_badge_color(safe_zone_status):
    return SIDEBAR_RED if safe_zone_status == "ALERT" else SIDEBAR_GREEN


def get_audio_badge_color(audio_status):
    normalized = (audio_status or "").lower()
    if "unavailable" in normalized or "stale" in normalized:
        return SIDEBAR_MUTED
    if "raised voice" in normalized or "siren" in normalized or "horn" in normalized:
        return SIDEBAR_ORANGE
    return SIDEBAR_GREEN


def get_telemetry_badge_color(telemetry_status):
    normalized = (telemetry_status or "").lower()
    if "unavailable" in normalized or "off" in normalized:
        return SIDEBAR_MUTED
    if "complete" in normalized or "done" in normalized:
        return SIDEBAR_SUBTLE
    if any(word in normalized for word in ("brake", "acceleration", "turn", "lane")):
        return SIDEBAR_ORANGE
    return SIDEBAR_GREEN


def draw_status_row(panel, x, y, width, label, value, badge_color):
    row_height = 22
    badge_font_scale = 0.34
    badge_padding_x = 7
    badge_padding_y = 3
    max_badge_width = max(52, width - 70)
    value = truncate_text_to_width(value, max_badge_width - (badge_padding_x * 2), badge_font_scale, 1)

    cv2.rectangle(panel, (x, y), (x + width, y + row_height), SIDEBAR_PANEL_ALT, -1)
    cv2.rectangle(panel, (x, y), (x + width, y + row_height), SIDEBAR_BORDER, 1)
    cv2.rectangle(panel, (x, y), (x + 4, y + row_height), badge_color, -1)

    draw_sidebar_text(
        panel,
        label.upper(),
        (x + 10, y + 15),
        SIDEBAR_MUTED,
        SIDEBAR_SMALL_SCALE,
        1,
    )

    (text_width, text_height), baseline = cv2.getTextSize(value, FONT, badge_font_scale, 1)
    badge_width = min(max_badge_width, text_width + (badge_padding_x * 2))
    badge_height = text_height + baseline + (badge_padding_y * 2)
    badge_x = x + width - badge_width - 10
    badge_y = y + max(2, (row_height - badge_height) // 2)

    draw_badge(
        panel,
        value,
        badge_x,
        badge_y,
        badge_color,
        font_scale=badge_font_scale,
        padding_x=badge_padding_x,
        padding_y=badge_padding_y,
    )


def render_status_card(panel, x, y, width, height, status_data, include_audio=True, include_telemetry=True):
    inner_y = draw_card(panel, x, y, width, height, "Camera Status", SIDEBAR_ACCENT)
    inner_x = x + 14
    content_width = width - 28

    eye_color = get_eye_badge_color(status_data.get("eye_status"))
    hand_color = get_hand_badge_color(status_data.get("hand_status"))
    safe_zone_color = get_safe_zone_badge_color(status_data.get("safe_zone_status"))
    row_width = content_width
    row_gap = 3
    row_y = inner_y + 2

    draw_status_row(
        panel,
        inner_x,
        row_y,
        row_width,
        "Eye",
        status_data.get("eye_status", "Unknown"),
        eye_color,
    )

    row_y += 22 + row_gap
    draw_status_row(
        panel,
        inner_x,
        row_y,
        row_width,
        "Hand",
        status_data.get("hand_status", "Unknown"),
        hand_color,
    )

    row_y += 22 + row_gap
    draw_status_row(
        panel,
        inner_x,
        row_y,
        row_width,
        "Zone",
        status_data.get("safe_zone_status", "Normal"),
        safe_zone_color,
    )

    if include_telemetry:
        row_y += 22 + row_gap
        telemetry_status = status_data.get("telemetry_status", "Off")
        draw_status_row(
            panel,
            inner_x,
            row_y,
            row_width,
            "Telemetry",
            telemetry_status,
            get_telemetry_badge_color(telemetry_status),
        )

    if include_audio:
        row_y += 22 + row_gap
        audio_status = status_data.get("audio_status", "Unavailable")
        draw_status_row(
            panel,
            inner_x,
            row_y,
            row_width,
            "Audio",
            audio_status,
            get_audio_badge_color(audio_status),
        )

    return y + height


def risk_breakdown_rows(breakdown):
    return [
        ("Eye", breakdown.get("eye", 0), 30, SIDEBAR_RED),
        ("Zone", breakdown.get("zone", 0), 20, SIDEBAR_ORANGE),
        ("Phone", breakdown.get("phone_call", 0), 40, SIDEBAR_RED),
        ("Drink", breakdown.get("drinking", 0), 25, SIDEBAR_YELLOW),
        ("Telem", breakdown.get("telemetry", 0), 18, SIDEBAR_ORANGE),
    ]


def render_risk_breakdown_card(panel, x, y, width, height, status_data):
    inner_y = draw_card(panel, x, y, width, height, "Risk Summary", SIDEBAR_ACCENT)
    inner_x = x + 14
    inner_width = width - 28
    compact = height < 210
    risk_score = int(status_data.get("risk_score", 0))
    driver_status = status_data.get("driver_status", "NORMAL")
    risk_color = get_risk_color(driver_status)
    risk_badge_label = risk_status_badge_label(driver_status)

    score_scale = 1.12 if compact else 1.25
    score_y = inner_y + (28 if compact else 31)
    draw_sidebar_text(panel, f"{risk_score:02d}", (inner_x, score_y), SIDEBAR_TEXT, score_scale, 3)
    draw_sidebar_text(panel, "/100", (inner_x + 61, inner_y + 25), SIDEBAR_MUTED, 0.54, 2)
    risk_badge_label = truncate_text_to_width(
        risk_badge_label, max(70, inner_width - 120), 0.42, 1
    )
    draw_badge(
        panel,
        risk_badge_label,
        inner_x + 110,
        inner_y + 4,
        risk_color,
        font_scale=0.42,
        padding_x=10,
        padding_y=5,
    )

    breakdown = status_data.get("risk_breakdown", {})
    rows = risk_breakdown_rows(breakdown)

    if compact:
        column_width = inner_width / 3
        grid_top = inner_y + 68
        for index, (label, value, _max_value, _bar_color) in enumerate(rows):
            row = index // 3
            column = index % 3
            text = truncate_text_to_width(
                f"{label} {int(value)}", int(column_width) - 4, 0.36, 1
            )
            draw_sidebar_text(
                panel,
                text,
                (inner_x + int(column * column_width), grid_top + row * 18),
                SIDEBAR_TEXT,
                0.36,
                1,
            )
        return y + height

    bar_y = inner_y + (50 if compact else 56)
    bar_step = 19 if compact else 24
    bar_text_scale = 0.40 if compact else SIDEBAR_SMALL_SCALE
    max_bar_y = y + height - 14
    for label, value, max_value, bar_color in rows:
        if bar_y + 12 > max_bar_y:
            break

        draw_sidebar_text(panel, f"{label}", (inner_x, bar_y + 11), SIDEBAR_TEXT, bar_text_scale, 1)
        draw_sidebar_text(panel, str(value), (inner_x + inner_width - 24, bar_y + 11), SIDEBAR_MUTED, bar_text_scale, 1)

        bar_x = inner_x + 48
        bar_width = max(20, inner_width - 76)
        cv2.rectangle(panel, (bar_x, bar_y + 4), (bar_x + bar_width, bar_y + 12), SIDEBAR_SUBTLE, -1)
        fill_width = int(bar_width * min(1.0, max(0.0, float(value) / float(max_value))))
        if fill_width > 0:
            cv2.rectangle(panel, (bar_x, bar_y + 4), (bar_x + fill_width, bar_y + 12), bar_color, -1)

        bar_y += bar_step

    return y + height


def render_speech_card(panel, x, y, width, height, status_data):
    inner_y = draw_card(panel, x, y, width, height, "Cabin Audio", SIDEBAR_ACCENT)
    inner_x = x + 14
    inner_width = width - 28
    max_content_y = y + height - 8
    language = status_data.get("speech_language", "unknown").upper()
    voice_level = status_data.get("voice_level", "QUIET").title()
    safety_status = status_data.get("safety_status", "Clear").replace("_", " ").title()
    safety_color = SIDEBAR_RED if safety_status != "Clear" else SIDEBAR_GREEN
    voice_color = (
        SIDEBAR_RED
        if voice_level == "Very High"
        else SIDEBAR_ORANGE if voice_level == "High" else SIDEBAR_GREEN
    )

    transcript = status_data.get("transient_transcript", "")
    show_transcript = status_data.get("show_transcript", False)
    recorded_count = int(status_data.get("recorded_transcript_count", 0))
    recording = (
        f"On ({recorded_count})"
        if status_data.get("transcript_recording_enabled", False)
        else "Off"
    )
    audio_status = status_data.get("audio_status", "Unavailable")
    rows = (
        ("Input", audio_status, get_audio_badge_color(audio_status)),
        ("Language", language, SIDEBAR_SUBTLE),
        ("Voice", voice_level, voice_color),
        ("Safety", safety_status, safety_color),
        ("Log", recording, SIDEBAR_ORANGE if recording.startswith("On") else SIDEBAR_SUBTLE),
    )
    row_y = inner_y + 2
    for label, value, color in rows:
        if row_y + 22 > max_content_y:
            break
        draw_status_row(panel, inner_x, row_y, inner_width, label, value, color)
        row_y += 25

    if transcript and show_transcript and row_y + 28 <= max_content_y:
        if contains_arabic(transcript):
            font_size = font_size_from_opencv_scale(SIDEBAR_SMALL_SCALE)
            transcript = truncate_localized_text(transcript, inner_width - 4, font_size)
            draw_sidebar_text(
                panel,
                "SAID",
                (inner_x, row_y + 10),
                SIDEBAR_MUTED,
                SIDEBAR_SMALL_SCALE,
                1,
            )
            draw_rtl_text(
                panel,
                transcript,
                right_x=inner_x + inner_width,
                baseline_y=row_y + 27,
                color_bgr=SIDEBAR_TEXT,
                font_size=font_size,
            )
        else:
            transcript = truncate_text_to_width(transcript, inner_width - 48, SIDEBAR_SMALL_SCALE, 1)
            draw_status_row(panel, inner_x, row_y, inner_width, "Said", transcript, SIDEBAR_SUBTLE)

    return y + height


def render_objects_card(panel, x, y, width, height, status_data):
    inner_y = draw_card(panel, x, y, width, height, "Detected Objects", SIDEBAR_ACCENT)
    inner_x = x + 14
    inner_width = width - 28
    max_content_y = y + height - 8
    objects = status_data.get("detected_objects", [])

    if not objects:
        if inner_y + 32 <= max_content_y:
            draw_kv_row(panel, inner_x, inner_y + 2, inner_width, "Status", "None detected", SIDEBAR_MUTED)
        return y + height

    current_y = inner_y + 2
    for obj in objects:
        row_height = 24
        if current_y + row_height > max_content_y:
            break

        normalized = obj.lower()
        if normalized in PHONE_OBJECT_CLASSES:
            chip_color = SIDEBAR_RED
        elif normalized in DRINK_OBJECT_CLASSES:
            chip_color = SIDEBAR_YELLOW
        else:
            chip_color = SIDEBAR_ORANGE

        cv2.rectangle(panel, (inner_x, current_y + 2), (x + width - 14, current_y + row_height), SIDEBAR_SUBTLE, -1)
        cv2.rectangle(panel, (inner_x, current_y + 2), (inner_x + 8, current_y + row_height), chip_color, -1)
        display_text = truncate_text_to_width(
            obj.replace("_", " ").title(),
            inner_width - 28,
            SIDEBAR_BODY_SCALE,
            1,
        )
        draw_sidebar_text(
            panel,
            display_text,
            (inner_x + 16, current_y + 19),
            SIDEBAR_TEXT,
            SIDEBAR_BODY_SCALE,
            1,
        )
        current_y += row_height + 6

    return y + height


def render_alerts_card(panel, x, y, width, height, status_data):
    inner_y = draw_card(panel, x, y, width, height, "Alerts", SIDEBAR_ACCENT)
    inner_x = x + 14
    inner_width = width - 28
    max_content_y = y + height - 8
    alerts = status_data.get("alerts", [])

    if not alerts:
        if inner_y + 32 <= max_content_y:
            draw_kv_row(panel, inner_x, inner_y + 2, inner_width, "Status", "No active alerts", SIDEBAR_GREEN)
        return y + height

    current_y = inner_y + 2
    for alert in alerts:
        alert_text = alert.get("text", "")
        alert_color = alert.get("color", SIDEBAR_RED)
        row_height = 34
        if current_y + row_height > max_content_y:
            break

        row_x2 = x + width - 14
        cv2.rectangle(panel, (inner_x, current_y + 2), (row_x2, current_y + row_height), SIDEBAR_PANEL_ALT, -1)
        cv2.rectangle(panel, (inner_x, current_y + 2), (row_x2, current_y + row_height), SIDEBAR_BORDER, 1)
        cv2.rectangle(panel, (inner_x, current_y + 2), (inner_x + 8, current_y + row_height), alert_color, -1)
        alert_text = truncate_text_to_width(alert_text, inner_width - 30, SIDEBAR_BODY_SCALE, 1)
        draw_sidebar_text(
            panel,
            alert_text,
            (inner_x + 16, current_y + 23),
            SIDEBAR_TEXT,
            SIDEBAR_BODY_SCALE,
            1,
        )
        current_y += row_height + 6

    return y + height


def render_controls_card(panel, x, y, width, height, status_data):
    if height <= 32:
        return y + height

    inner_y = draw_card(panel, x, y, width, height, "Controls", SIDEBAR_ACCENT)
    inner_x = x + 14
    inner_width = width - 28
    max_content_y = y + height - 8
    controls = status_data.get("controls", [])

    if not controls:
        controls = [
            "C  calibrate safe box",
            "S  save screenshot",
            "Q  quit",
            "D  toggle debug overlays",
        ]

    current_y = inner_y + 4
    for control in controls:
        if current_y + 16 > max_content_y:
            break

        control_text = truncate_text_to_width(control, inner_width, SIDEBAR_BODY_SCALE, 1)
        draw_sidebar_text(panel, control_text, (inner_x, current_y + 13), SIDEBAR_MUTED, SIDEBAR_BODY_SCALE, 1)
        current_y += 24

    return y + height


def render_panel_header(panel, width, title, subtitle, status_data):
    header_height = 58
    cv2.rectangle(panel, (0, 0), (width, header_height), SIDEBAR_PANEL_ALT, -1)
    cv2.rectangle(panel, (0, header_height - 1), (width, header_height), SIDEBAR_BORDER, -1)
    draw_sidebar_text(panel, title, (14, 25), SIDEBAR_TEXT, 0.58, 1)
    draw_sidebar_text(panel, subtitle, (14, 45), SIDEBAR_MUTED, 0.34, 1)
    live_badge_text = "DEBUG" if status_data.get("debug_overlays") else "LIVE"
    live_badge_color = SIDEBAR_ORANGE if status_data.get("debug_overlays") else SIDEBAR_GREEN
    draw_badge(panel, live_badge_text, max(14, width - 67), 13, live_badge_color, font_scale=0.32, padding_x=7, padding_y=4)
    return header_height


def render_audio_sidebar(status_data, width, height):
    panel = np.full((height, width, 3), SIDEBAR_BG, dtype=np.uint8)
    x = 16
    y = render_panel_header(panel, width, "CabInspector", "CABIN AUDIO / REVIEW ONLY", status_data) + 10
    card_width = width - 32
    speech_height = min(210, max(184, height - y - 22))
    render_speech_card(panel, x, y, card_width, speech_height, status_data)

    return panel


def render_camera_sidebar(status_data, width, height):
    panel = np.full((height, width, 3), SIDEBAR_BG, dtype=np.uint8)
    x = 16
    y = render_panel_header(panel, width, "Driver Monitor", "CAMERA & RISK / NO SPEAKER ID", status_data) + 10
    card_width = width - 32
    gap = 7
    compact = height <= 760
    status_height = 133
    risk_height = 136 if compact else (210 if height >= 820 else 158)
    objects_height = 66 if compact else 82
    alerts_height = 76 if compact else 94

    y = render_status_card(
        panel,
        x,
        y,
        card_width,
        status_height,
        status_data,
        include_audio=False,
        include_telemetry=True,
    ) + gap
    y = render_risk_breakdown_card(panel, x, y, card_width, risk_height, status_data) + gap
    y = render_objects_card(panel, x, y, card_width, objects_height, status_data) + gap
    y = render_alerts_card(panel, x, y, card_width, alerts_height, status_data) + gap
    render_controls_card(panel, x, y, card_width, max(0, height - y - 12), status_data)

    return panel


def create_dashboard_view(camera_frame, status_data):
    left_width = LEFT_SIDEBAR_WIDTH
    right_width = RIGHT_SIDEBAR_WIDTH
    camera_width = max(1, DISPLAY_WIDTH - left_width - right_width)
    camera_panel = fit_frame_to_box(camera_frame, camera_width, DISPLAY_HEIGHT)
    audio_panel = render_audio_sidebar(status_data, left_width, DISPLAY_HEIGHT)
    camera_sidebar = render_camera_sidebar(status_data, right_width, DISPLAY_HEIGHT)

    dashboard = np.hstack([audio_panel, camera_panel, camera_sidebar])
    cv2.line(dashboard, (left_width - 1, 0), (left_width - 1, DISPLAY_HEIGHT - 1), SIDEBAR_BORDER, 1)
    cv2.line(dashboard, (left_width + camera_width - 1, 0), (left_width + camera_width - 1, DISPLAY_HEIGHT - 1), SIDEBAR_BORDER, 1)

    return dashboard


def build_hand_rois_from_pose(pose_points, frame_shape):
    if not pose_points:
        return []

    frame_h, frame_w, _ = frame_shape
    rois = []

    for side in ("left", "right"):
        wrist = pose_points.get(f"{side}_wrist")
        elbow = pose_points.get(f"{side}_elbow")
        shoulder = pose_points.get(f"{side}_shoulder")

        if wrist is None:
            continue

        base_pad_x = max(160, int(frame_w * 0.24))
        base_pad_up = max(130, int(frame_h * 0.23))
        base_pad_down = max(180, int(frame_h * 0.34))

        if elbow is not None:
            forearm_span = distance_between_points(wrist, elbow)
            base_pad_x = max(base_pad_x, int(forearm_span * 1.55))
            base_pad_up = max(base_pad_up, int(forearm_span * 1.10))
            base_pad_down = max(base_pad_down, int(forearm_span * 1.45))

        if shoulder is not None and elbow is not None:
            upper_arm_span = distance_between_points(shoulder, elbow)
            base_pad_up = max(base_pad_up, int(upper_arm_span * 0.95))

        x1 = clamp_int(wrist[0] - base_pad_x, 0, frame_w)
        x2 = clamp_int(wrist[0] + base_pad_x, 0, frame_w)
        y1 = clamp_int(wrist[1] - base_pad_up, 0, frame_h)
        y2 = clamp_int(wrist[1] + base_pad_down, 0, frame_h)

        if (x2 - x1) < 50 or (y2 - y1) < 50:
            continue

        rois.append({
            "name": f"{side}_wrist_roi",
            "box": (x1, y1, x2, y2),
        })

    return rois


def collect_hand_infos_from_results(results, image_shape, x_offset=0, y_offset=0, source="full", draw_target=None):
    image_h, image_w = image_shape[:2]
    hand_infos = []

    if not results.multi_hand_landmarks:
        return hand_infos

    handedness_list = results.multi_handedness or []

    for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
        if draw_target is not None:
            mp_drawing.draw_landmarks(
                draw_target,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

        landmark_points = [
            (
                int(landmark.x * image_w) + x_offset,
                int(landmark.y * image_h) + y_offset
            )
            for landmark in hand_landmarks.landmark
        ]

        xs = [point[0] for point in landmark_points]
        ys = [point[1] for point in landmark_points]
        wrist = landmark_points[0]
        palm_indices = [0, 5, 9, 13, 17]
        palm_points = [
            landmark_points[palm_index]
            for palm_index in palm_indices
            if palm_index < len(landmark_points)
        ]

        center_x = sum(point[0] for point in palm_points) // len(palm_points)
        center_y = sum(point[1] for point in palm_points) // len(palm_points)
        center = (center_x, center_y)

        hand_label = f"Hand {index + 1}"
        hand_confidence = None

        if index < len(handedness_list) and handedness_list[index].classification:
            classification = handedness_list[index].classification[0]
            hand_label = f"{classification.label.title()} Hand"
            hand_confidence = float(classification.score)

        hand_infos.append(
            {
                "index": index,
                "label": hand_label,
                "confidence": hand_confidence,
                "center": center,
                "wrist": wrist,
                "bbox": (min(xs), min(ys), max(xs), max(ys)),
                "landmarks": landmark_points,
                "source": source,
                "raw_landmarks": hand_landmarks,
            }
        )

    return hand_infos


def dedupe_hand_infos(hand_infos, distance_threshold=50):
    unique_infos = []

    for hand_info in sorted(
        hand_infos,
        key=lambda item: float(item["confidence"] or 0.0),
        reverse=True
    ):
        duplicate_index = None

        for index, existing in enumerate(unique_infos):
            if (
                distance_between_points(hand_info["wrist"], existing["wrist"]) < distance_threshold
                or distance_between_points(hand_info["center"], existing["center"]) < distance_threshold
            ):
                duplicate_index = index
                break

        if duplicate_index is None:
            unique_infos.append(hand_info)
        else:
            existing = unique_infos[duplicate_index]
            existing_score = float(existing["confidence"] or 0.0)
            new_score = float(hand_info["confidence"] or 0.0)

            if new_score > existing_score:
                unique_infos[duplicate_index] = hand_info

    return unique_infos


def detect_hands_in_region(frame, region_box, source_name, draw_overlays=False):
    x1, y1, x2, y2 = region_box
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return []

    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    return collect_hand_infos_from_results(
        results,
        roi.shape,
        x_offset=x1,
        y_offset=y1,
        source=source_name,
        draw_target=roi if draw_overlays else None
    )


def detect_hands_full_frame(frame, draw_overlays=False):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    return collect_hand_infos_from_results(
        results,
        frame.shape,
        source="full_frame",
        draw_target=frame if draw_overlays else None
    )


def get_hand_points(frame, pose_points=None, draw_overlays=False, force_full_frame=False):
    if hands is None or mp_hands is None:
        return frame, [], "Hands Disabled"

    hand_infos = []
    pose_rois = build_hand_rois_from_pose(pose_points, frame.shape)
    run_full_frame_first = force_full_frame or not pose_rois

    if run_full_frame_first:
        hand_infos.extend(detect_hands_full_frame(frame, draw_overlays=draw_overlays))

    if pose_rois and len(hand_infos) < 2:
        for roi_info in pose_rois:
            hand_infos.extend(
                detect_hands_in_region(
                    frame,
                    roi_info["box"],
                    roi_info["name"],
                    draw_overlays=draw_overlays
                )
            )

    if not hand_infos and not run_full_frame_first:
        hand_infos = detect_hands_full_frame(frame, draw_overlays=draw_overlays)

    hand_infos = dedupe_hand_infos(hand_infos)

    for hand_info in hand_infos:
        center_x, center_y = hand_info["center"]
        if draw_overlays:
            cv2.circle(frame, hand_info["center"], 7, (0, 165, 255), -1)
            draw_text(
                frame,
                hand_info["label"],
                (center_x + 8, center_y - 8),
                (0, 165, 255),
                0.42,
                1
            )

            x1, y1, x2, y2 = hand_info["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 1)

    if hand_infos:
        hand_status = f"{len(hand_infos)} Hand{'s' if len(hand_infos) != 1 else ''} Detected"
        hand_color = (0, 255, 0)
    else:
        hand_status = "No Hands Detected"
        hand_color = (0, 0, 255)

    return frame, hand_infos, hand_status


def build_hand_point_map(hand_infos):
    hand_points = {}

    for index, hand_info in enumerate(hand_infos, start=1):
        hand_points[f"hand_{index}_center"] = hand_info["center"]
        hand_points[f"hand_{index}_wrist"] = hand_info["wrist"]

    return hand_points


def get_hand_contact_points(pose_points, hand_infos):
    contact_points = []

    for hand_info in hand_infos:
        contact_points.append(hand_info["center"])
        contact_points.append(hand_info["wrist"])

    for wrist_name in ("left_wrist", "right_wrist"):
        if wrist_name in pose_points:
            contact_points.append(pose_points[wrist_name])

    return contact_points


EVENT_LOG_FIELDS = [
    "timestamp",
    "frame_number",
    "eye_status",
    "hand_status",
    "hand_count",
    "eye_closed_counter",
    "safe_zone_status",
    "outside_landmarks",
    "detected_objects",
    "audio_available",
    "audio_top_label",
    "audio_top_confidence",
    "speech_active",
    "raised_voice_active",
    "horn_active",
    "siren_active",
    "audio_error",
    "speech_language",
    "voice_level",
    "loudness_dbfs",
    "loud_voice_active",
    "safety_flagged",
    "safety_categories",
    "safety_confidence",
    "transcript_visible",
    "speech_analysis_error",
    "telemetry_available",
    "telemetry_mode",
    "telemetry_model",
    "telemetry_trip",
    "telemetry_replay_speed",
    "telemetry_elapsed_seconds",
    "telemetry_events_processed",
    "telemetry_event_category",
    "telemetry_event_confidence",
    "telemetry_event_active",
    "telemetry_error",
    "phone_object_detected",
    "raw_phone_call_detected",
    "phone_call_counter",
    "phone_call_detected",
    "raw_drinking_detected",
    "drinking_counter",
    "drinking_detected",
    "raw_zone_alert",
    "zone_alert_counter",
    "zone_alert",
    "risk_score",
    "final_status",
]

EVENT_LOG_FILE_HANDLE = None
EVENT_LOG_WRITER = None
EVENT_LOG_PENDING_ROWS = 0


def sanitize_label(label):
    safe_chars = []

    for char in label:
        if char.isalnum() or char in ("-", "_"):
            safe_chars.append(char)
        else:
            safe_chars.append("_")

    return "".join(safe_chars).strip("_") or "event"


def save_screenshot(frame, prefix, frame_count):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{sanitize_label(prefix)}_{frame_count}_{timestamp}.png"
    screenshot_path = SCREENSHOT_DIR / filename
    cv2.imwrite(str(screenshot_path), frame)
    return screenshot_path


def initialize_event_log():
    global EVENT_LOG_FILE_HANDLE, EVENT_LOG_WRITER, EVENT_LOG_PENDING_ROWS

    if EVENT_LOG_FILE_HANDLE is not None and EVENT_LOG_WRITER is not None:
        return

    EVENT_LOG_PENDING_ROWS = 0

    expected_header = ",".join(EVENT_LOG_FIELDS)

    if EVENT_LOG_PATH.exists() and EVENT_LOG_PATH.stat().st_size > 0:
        with EVENT_LOG_PATH.open("r", encoding="utf-8") as log_file:
            existing_header = log_file.readline().strip()

        if existing_header == expected_header:
            EVENT_LOG_FILE_HANDLE = EVENT_LOG_PATH.open("a", newline="", encoding="utf-8")
            EVENT_LOG_WRITER = csv.DictWriter(EVENT_LOG_FILE_HANDLE, fieldnames=EVENT_LOG_FIELDS)
            return

        backup_name = (
            f"events_log_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        backup_path = EVENT_LOG_PATH.with_name(backup_name)
        EVENT_LOG_PATH.replace(backup_path)

    EVENT_LOG_FILE_HANDLE = EVENT_LOG_PATH.open("w", newline="", encoding="utf-8")
    EVENT_LOG_WRITER = csv.DictWriter(EVENT_LOG_FILE_HANDLE, fieldnames=EVENT_LOG_FIELDS)
    EVENT_LOG_WRITER.writeheader()
    EVENT_LOG_FILE_HANDLE.flush()


def append_event_log(row):
    global EVENT_LOG_FILE_HANDLE, EVENT_LOG_WRITER, EVENT_LOG_PENDING_ROWS

    if EVENT_LOG_FILE_HANDLE is None or EVENT_LOG_WRITER is None:
        initialize_event_log()

    EVENT_LOG_WRITER.writerow(row)
    EVENT_LOG_PENDING_ROWS += 1

    if EVENT_LOG_PENDING_ROWS >= EVENT_LOG_FLUSH_EVERY_N_ROWS:
        EVENT_LOG_FILE_HANDLE.flush()
        EVENT_LOG_PENDING_ROWS = 0


def close_event_log():
    global EVENT_LOG_FILE_HANDLE, EVENT_LOG_WRITER, EVENT_LOG_PENDING_ROWS

    if EVENT_LOG_FILE_HANDLE is not None:
        EVENT_LOG_FILE_HANDLE.flush()
        EVENT_LOG_FILE_HANDLE.close()

    EVENT_LOG_FILE_HANDLE = None
    EVENT_LOG_WRITER = None
    EVENT_LOG_PENDING_ROWS = 0


def update_smoothing_counter(counter, active, confirm_frames=ALERT_CONFIRM_FRAMES):
    if active:
        counter += 1
    else:
        counter = max(0, counter - 1)

    return counter, counter >= confirm_frames


def get_risk_level(risk_score):
    if risk_score >= 65:
        return "HIGH RISK"
    if risk_score >= 30:
        return "MODERATE RISK"
    if risk_score > 0:
        return "LOW RISK"
    return "NORMAL"


def calculate_risk_score(
    face_detected,
    eye_closed_counter,
    zone_alert_counter,
    phone_object_detected,
    phone_call_counter,
    drinking_counter,
    raised_voice_active=False,
    speech_with_phone_evidence=False,
    safety_categories=(),
    telemetry_event_category="NORMAL",
    telemetry_event_active=False,
):
    """
    Convert persistent signals into a 0-100 risk score.

    This is intentionally smoother than a one-frame sum so the score feels
    more stable in a live webcam demo.
    """

    eye_score = 0
    if face_detected and eye_closed_counter > 0:
        # Let the eye score rise gradually as the eye-closed condition persists.
        eye_score = min(30, max(0, eye_closed_counter - 2) * 4)

    zone_score = min(20, zone_alert_counter * 5)
    phone_object_score = 8 if phone_object_detected else 0
    phone_call_score = min(40, phone_call_counter * 12)
    drinking_score = min(25, drinking_counter * 8)
    # Normal speech never adds risk on its own. It only reinforces a visual
    # phone-call signal, while a confirmed raised voice remains contextual.
    raised_voice_score = 6 if raised_voice_active else 0
    speech_phone_score = 6 if speech_with_phone_evidence else 0
    safety_categories = set(safety_categories)
    safety_score = 0
    if "THREAT" in safety_categories:
        safety_score = 20
    elif "HARASSMENT" in safety_categories:
        safety_score = 12
    elif "SEXUAL_LANGUAGE" in safety_categories:
        safety_score = 10
    elif "PROFANITY" in safety_categories or "TOXIC_LANGUAGE" in safety_categories:
        safety_score = 6

    # Replay telemetry is a separate source of motion evidence.  It is kept
    # intentionally bounded so one inertial classification cannot dominate
    # stronger repeated visual evidence.  These are explainable prototype
    # weights, not calibrated probabilities.
    telemetry_weights = {
        "HARD_BRAKE": 18,
        "RAPID_ACCELERATION": 14,
        "AGGRESSIVE_TURN": 10,
        "AGGRESSIVE_LANE_CHANGE": 12,
    }
    telemetry_score = (
        telemetry_weights.get(telemetry_event_category, 0)
        if telemetry_event_active
        else 0
    )

    total_score = min(
        100,
        eye_score
        + zone_score
        + phone_object_score
        + phone_call_score
        + drinking_score
        + raised_voice_score
        + speech_phone_score
        + safety_score
        + telemetry_score
    )

    return total_score, {
        "eye": eye_score,
        "zone": zone_score,
        "phone_object": phone_object_score,
        "phone_call": phone_call_score,
        "drinking": drinking_score,
        "raised_voice": raised_voice_score,
        "speech_phone": speech_phone_score,
        "speech_safety": safety_score,
        "telemetry": telemetry_score,
    }


def get_risk_color(risk_level):
    if risk_level == "HIGH RISK":
        return (0, 0, 255)
    if risk_level == "MODERATE RISK":
        return (0, 165, 255)
    if risk_level == "LOW RISK":
        return (0, 255, 255)
    return (0, 255, 0)


def risk_status_badge_label(risk_status):
    """Keep the risk class readable in the compact badge; the card title supplies context."""

    text = str(risk_status).strip()
    labels = {"LOW RISK": "LOW", "MODERATE RISK": "MODERATE", "HIGH RISK": "HIGH"}
    return labels.get(text.upper(), text)


# =========================
# Face and Eye Detection
# =========================

def detect_face_and_eyes(frame, no_eye_counter, draw_overlays=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    if face_mesh is not None:
        if not hasattr(detect_face_and_eyes, "_eye_ratio_history"):
            detect_face_and_eyes._eye_ratio_history = deque(maxlen=EYE_RATIO_HISTORY_SIZE)
        if not hasattr(detect_face_and_eyes, "_eye_closed_run"):
            detect_face_and_eyes._eye_closed_run = 0

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh_results = face_mesh.process(rgb)

        if mesh_results.multi_face_landmarks:
            face_landmarks = mesh_results.multi_face_landmarks[0].landmark
            face_box = face_mesh_bounding_box(face_landmarks, frame.shape[1], frame.shape[0])
            face_detected = True
            face_color = (255, 180, 0) if draw_overlays else (140, 140, 140)

            if face_box is not None:
                x1, y1, x2, y2 = face_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), face_color, 2 if draw_overlays else 1)
                if draw_overlays:
                    draw_text(frame, "Face", (x1, max(15, y1 - 10)), face_color, 0.6, 2)

            left_ratio = face_mesh_eye_ratio(
                face_landmarks,
                frame.shape[1],
                frame.shape[0],
                "left",
            )
            right_ratio = face_mesh_eye_ratio(
                face_landmarks,
                frame.shape[1],
                frame.shape[0],
                "right",
            )

            eye_ratios = [
                ratio
                for ratio in (left_ratio, right_ratio)
                if ratio is not None
            ]

            if eye_ratios:
                eye_ratio = float(sum(eye_ratios) / len(eye_ratios))
                detect_face_and_eyes._eye_ratio_history.append(eye_ratio)
                smoothed_eye_ratio = float(np.median(detect_face_and_eyes._eye_ratio_history))

                if smoothed_eye_ratio >= FACE_MESH_EYE_OPEN_RATIO_THRESHOLD:
                    detect_face_and_eyes._eye_closed_run = 0
                    no_eye_counter = 0
                    status = "Eyes Open"
                    eye_closed_alert = False
                elif smoothed_eye_ratio <= FACE_MESH_EYE_CLOSED_RATIO_THRESHOLD:
                    detect_face_and_eyes._eye_closed_run += 1

                    if detect_face_and_eyes._eye_closed_run >= EYE_CLOSED_CONFIRM_FRAMES:
                        no_eye_counter = min(EYE_CLOSED_FRAME_LIMIT, no_eye_counter + 1)
                    else:
                        no_eye_counter = max(0, no_eye_counter - 1)

                    if no_eye_counter >= EYE_CLOSED_FRAME_LIMIT:
                        status = "Eyes Possibly Closed"
                        eye_closed_alert = True
                    else:
                        status = "Checking Eyes..."
                        eye_closed_alert = False
                else:
                    detect_face_and_eyes._eye_closed_run = max(0, detect_face_and_eyes._eye_closed_run - 1)
                    no_eye_counter = max(0, no_eye_counter - 1)
                    status = "Checking Eyes..."
                    eye_closed_alert = False

                if draw_overlays and face_box is not None:
                    x1, y1, x2, y2 = face_box
                    draw_text(
                        frame,
                        f"Eye ratio {eye_ratio:.2f}",
                        (x1, min(frame.shape[0] - 5, y2 + 20)),
                        face_color,
                        0.45,
                        1,
                    )

                return frame, no_eye_counter, status, eye_closed_alert, face_detected

            detect_face_and_eyes._eye_ratio_history.clear()
            detect_face_and_eyes._eye_closed_run = 0
            no_eye_counter = max(0, no_eye_counter - 1)
            return frame, no_eye_counter, "Checking Eyes...", False, face_detected

    face_candidates = []

    face_detect_params = dict(
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(70, 70),
    )

    if face_cascade is not None:
        face_candidates.extend(
            face_cascade.detectMultiScale(gray, **face_detect_params)
        )

    if face_cascade_alt is not None:
        face_candidates.extend(
            face_cascade_alt.detectMultiScale(gray, **face_detect_params)
        )

    if not face_candidates and face_profile_cascade is not None:
        face_candidates.extend(
            face_profile_cascade.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=5,
                minSize=(70, 70)
            )
        )

    face_box = choose_largest_box(face_candidates)
    face_detected = face_box is not None
    eyes_detected = False

    if face_detected:
        x, y, w, h = [int(value) for value in face_box]
        face_color = (255, 180, 0) if draw_overlays else (140, 140, 140)
        cv2.rectangle(frame, (x, y), (x + w, y + h), face_color, 2 if draw_overlays else 1)
        if draw_overlays:
            draw_text(frame, "Face", (x, y - 10), face_color, 0.6, 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eye_candidates = []
        eye_detect_params = dict(
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(12, 12),
        )

        if eye_cascade is not None:
            eye_candidates.extend(
                eye_cascade.detectMultiScale(roi_gray, **eye_detect_params)
            )

        if eye_cascade_glasses is not None:
            eye_candidates.extend(
                eye_cascade_glasses.detectMultiScale(roi_gray, **eye_detect_params)
            )

        valid_eyes = []

        for (ex, ey, ew, eh) in eye_candidates:
            # Keep the search in the upper face to avoid nose/mouth false positives.
            if ey < h * 0.55 and ew >= 8 and eh >= 8:
                valid_eyes.append((ex, ey, ew, eh))
                if draw_overlays:
                    cv2.rectangle(
                        roi_color,
                        (ex, ey),
                        (ex + ew, ey + eh),
                        (0, 255, 0),
                        2
                    )

        if len(valid_eyes) >= 1:
            eyes_detected = True

    if eyes_detected:
        no_eye_counter = 0
        status = "Eyes Open"
        color = (0, 255, 0)
        eye_closed_alert = False

    elif face_detected:
        no_eye_counter += 1

        if no_eye_counter >= EYE_CLOSED_FRAME_LIMIT:
            status = "Eyes Possibly Closed"
            color = (0, 0, 255)
            eye_closed_alert = True
        else:
            status = "Checking Eyes..."
            color = (0, 255, 255)
            eye_closed_alert = False

    else:
        status = "No Face Detected"
        color = (0, 0, 255)
        eye_closed_alert = False
        no_eye_counter = 0
        if hasattr(detect_face_and_eyes, "_eye_ratio_history"):
            detect_face_and_eyes._eye_ratio_history.clear()
        if hasattr(detect_face_and_eyes, "_eye_closed_run"):
            detect_face_and_eyes._eye_closed_run = 0

    return frame, no_eye_counter, status, eye_closed_alert, face_detected


# =========================
# Skeleton / Landmark Detection
# =========================

def get_pose_points(frame, draw_overlays=False):
    if pose is None:
        return frame, {}

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    points = {}

    if results.pose_landmarks:
        if draw_overlays:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        landmarks = results.pose_landmarks.landmark

        selected_landmarks = {
            # Face landmarks useful for action detection
            "nose": mp_pose.PoseLandmark.NOSE,
            "left_ear": mp_pose.PoseLandmark.LEFT_EAR,
            "right_ear": mp_pose.PoseLandmark.RIGHT_EAR,
            "mouth_left": mp_pose.PoseLandmark.MOUTH_LEFT,
            "mouth_right": mp_pose.PoseLandmark.MOUTH_RIGHT,

            # Body landmarks required by professor
            "left_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
            "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
            "left_elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
            "right_elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
            "left_wrist": mp_pose.PoseLandmark.LEFT_WRIST,
            "right_wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
        }

        for name, landmark_id in selected_landmarks.items():
            landmark = landmarks[landmark_id.value]

            # Keep a slightly lower threshold so ear / wrist landmarks survive
            # when the driver is partly occluded by a phone or hand.
            if landmark.visibility > 0.25:
                x = int(landmark.x * w)
                y = int(landmark.y * h)

                points[name] = (x, y)

                if draw_overlays:
                    cv2.circle(frame, (x, y), 7, (255, 0, 255), -1)
                    draw_text(frame, name, (x + 8, y - 8), (255, 0, 255), 0.42, 1)

    return frame, points


# =========================
# Virtual Safe Zone
# =========================

def create_box_from_points(points, frame_shape, margin=70):
    if not points:
        return None

    h, w, _ = frame_shape

    body_point_names = [
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist"
    ]

    body_points = [
        points[name]
        for name in body_point_names
        if name in points
    ]

    if not body_points:
        return None

    xs = [p[0] for p in body_points]
    ys = [p[1] for p in body_points]

    x1 = max(min(xs) - margin, 0)
    y1 = max(min(ys) - margin, 0)
    x2 = min(max(xs) + margin, w)
    y2 = min(max(ys) + margin, h)

    return (x1, y1, x2, y2)


def check_safe_box(frame, points, safe_box, draw_overlays=False):
    if safe_box is None:
        return frame, False, []

    checked_landmarks = [
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist"
    ]

    checked_landmarks.extend(
        name
        for name in points.keys()
        if name.startswith("hand_") and (name.endswith("_center") or name.endswith("_wrist"))
    )

    outside_landmarks = []

    for name in checked_landmarks:
        if name in points:
            if not point_inside_box(points[name], safe_box):
                outside_landmarks.append(format_landmark_name(name))

    alert = len(outside_landmarks) > 0

    x1, y1, x2, y2 = safe_box

    if alert:
        box_color = (0, 0, 255)
    else:
        box_color = (0, 255, 0)

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3 if draw_overlays else 1)

    if draw_overlays:
        if alert:
            draw_text(frame, "ALERT: Arm/hand outside safe zone", (20, 80), box_color, 0.8, 2)
            draw_text(frame, "Outside: " + ", ".join(outside_landmarks), (20, 115), box_color, 0.6, 2)
        else:
            draw_text(frame, "Safe Zone: Normal", (20, 80), box_color, 0.8, 2)

    return frame, alert, outside_landmarks


# =========================
# YOLO Object Detection
# =========================

def run_yolo(
    frame,
    model,
    target_classes=None,
    imgsz=None,
    conf=None,
    augment=False,
    max_det=None,
    iou=None,
):
    started = time.perf_counter()
    detections = []

    if model is None:
        runtime_observe("visual.yolo_full", (time.perf_counter() - started) * 1000.0)
        return detections

    if target_classes is None:
        target_classes = TARGET_OBJECTS

    if imgsz is None:
        imgsz = YOLO_IMAGE_SIZE

    if conf is None:
        conf = YOLO_CONFIDENCE

    predict_kwargs = {
        "verbose": False,
        "imgsz": imgsz,
        "conf": conf,
        "augment": augment,
    }

    if max_det is not None:
        predict_kwargs["max_det"] = max_det

    if iou is not None:
        predict_kwargs["iou"] = iou

    results = model(frame, **predict_kwargs)[0]

    for box in results.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        if class_name in target_classes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "class_name": class_name,
                "confidence": confidence,
                "box": (x1, y1, x2, y2)
            })

    runtime_observe("visual.yolo_full", (time.perf_counter() - started) * 1000.0)
    return detections


def run_yolo_on_regions(
    frame,
    model,
    regions,
    target_classes=None,
    imgsz=None,
    conf=None,
    augment=False,
    max_det=None,
    iou=None,
):
    started = time.perf_counter()
    detections = []

    if model is None or not regions:
        runtime_observe("visual.yolo_regions", (time.perf_counter() - started) * 1000.0)
        return detections

    if target_classes is None:
        target_classes = TARGET_OBJECTS

    if imgsz is None:
        imgsz = YOLO_IMAGE_SIZE

    if conf is None:
        conf = YOLO_CONFIDENCE

    roi_images = []
    valid_regions = []

    for region in regions[:PHONE_SEARCH_MAX_REGIONS]:
        if "box" not in region:
            continue

        x1, y1, x2, y2 = region["box"]
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        roi_images.append(roi)
        valid_regions.append(region)

    if not roi_images:
        runtime_observe("visual.yolo_regions", (time.perf_counter() - started) * 1000.0)
        return detections

    predict_kwargs = {
        "verbose": False,
        "imgsz": imgsz,
        "conf": conf,
        "augment": augment,
    }

    if max_det is not None:
        predict_kwargs["max_det"] = max_det

    if iou is not None:
        predict_kwargs["iou"] = iou

    results = model(roi_images, **predict_kwargs)

    for region, result in zip(valid_regions, results):
        x1, y1, _, _ = region["box"]

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            if class_name in target_classes:
                rx1, ry1, rx2, ry2 = map(int, box.xyxy[0])

                detections.append({
                    "class_name": class_name,
                    "confidence": confidence,
                    "box": (rx1 + x1, ry1 + y1, rx2 + x1, ry2 + y1),
                    "source": region.get("name", "phone_roi"),
                })

    runtime_observe("visual.yolo_regions", (time.perf_counter() - started) * 1000.0)
    return dedupe_detections(detections)


def run_yolo_detection_job(
    process_frame,
    source_frame,
    phone_search_regions,
    phone_pose_points,
    phone_hand_infos,
    run_general_detection,
):
    started = time.perf_counter()
    general_detections = []
    phone_detections_source = []

    if yolo_model is None:
        return {
            "general_detections": general_detections,
            "phone_detections_source": phone_detections_source,
            "ran_general_detection": False,
            "runtime_ms": round((time.perf_counter() - started) * 1000.0, 4),
        }

    if run_general_detection:
        general_detections = run_yolo(
            process_frame,
            yolo_model,
            target_classes=DRINK_OBJECT_CLASSES,
            imgsz=YOLO_IMAGE_SIZE,
            conf=YOLO_CONFIDENCE,
        )

    phone_model = phone_yolo_model or yolo_model
    phone_search_detections_source = []

    if phone_search_regions:
        phone_search_detections_source = run_yolo_on_regions(
            source_frame,
            phone_model,
            phone_search_regions,
            target_classes=PHONE_OBJECT_CLASSES,
            imgsz=PHONE_SEARCH_IMAGE_SIZE,
            conf=PHONE_SEARCH_CONFIDENCE,
            augment=PHONE_SEARCH_AUGMENT,
            max_det=PHONE_SEARCH_MAX_DET,
            iou=PHONE_SEARCH_IOU,
        )

    phone_detections_source = filter_phone_detections(
        phone_search_detections_source,
        phone_pose_points,
        phone_hand_infos,
        source_frame.shape,
    )

    return {
        "general_detections": general_detections,
        "phone_detections_source": phone_detections_source,
        "ran_general_detection": run_general_detection,
        "runtime_ms": round((time.perf_counter() - started) * 1000.0, 4),
    }


def draw_yolo(frame, detections, draw_overlays=False):
    phone_detected = False
    detected_object_names = []

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]
        detected_object_names.append(class_name)

        if class_name in PHONE_OBJECT_CLASSES:
            color = (0, 0, 255)
            phone_detected = True
            label_name = "phone"
        elif class_name in DRINK_OBJECT_CLASSES:
            color = (0, 255, 255)
            label_name = class_name
        else:
            color = (255, 255, 255)
            label_name = class_name

        if draw_overlays:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            draw_text(
                frame,
                f"{label_name} {confidence:.2f}",
                (x1, y1 - 10),
                color,
                0.6,
                2
            )

    if draw_overlays and phone_detected:
        draw_text(frame, "Phone object detected", (20, 170), (0, 0, 255), 0.7, 2)

    return frame, phone_detected, detected_object_names


# =========================
# Higher-Level Action Detection
# =========================

def detect_object_actions(frame, detections, pose_points, hand_infos=None, draw_overlays=False):
    """
    Detect behaviour, not only objects.

    Talking on phone:
    - fresh cell phone detected near the ear
    - hand contact point close to the detected phone

    Drinking:
    - bottle/cup-like object detected
    - object close to mouth
    - hand close to object
    """

    phone_call_detected = False
    drinking_detected = False

    frame_h, frame_w, _ = frame.shape

    # Scale thresholds based on camera resolution.
    scale = max(frame_w, frame_h) / 640

    PHONE_NEAR_EAR_THRESHOLD = int(145 * scale)
    DRINK_NEAR_MOUTH_THRESHOLD = int(190 * scale)
    HAND_NEAR_OBJECT_THRESHOLD = int(220 * scale)

    mouth_center = get_mouth_center(pose_points)
    contact_points = get_hand_contact_points(pose_points, hand_infos or [])

    ears = []
    if "left_ear" in pose_points:
        ears.append(pose_points["left_ear"])
    if "right_ear" in pose_points:
        ears.append(pose_points["right_ear"])

    for detection in detections:
        class_name = detection["class_name"]
        box = detection["box"]
        object_center = get_box_center(box)

        # Draw object center
        if draw_overlays:
            cv2.circle(frame, object_center, 6, (255, 255, 255), -1)

        # -----------------------------
        # Phone Call Detection
        # -----------------------------
        if class_name in PHONE_OBJECT_CLASSES:
            phone_near_ear = False
            hand_near_phone = False

            for ear in ears:
                if distance_point_to_box(ear, box) < PHONE_NEAR_EAR_THRESHOLD:
                    phone_near_ear = True

            for contact_point in contact_points:
                if distance_point_to_box(contact_point, box) < HAND_NEAR_OBJECT_THRESHOLD:
                    hand_near_phone = True
                    break

            # Require an actual phone detection, ear proximity, and a hand
            # contact point near the detected phone. This avoids reporting a
            # phone from pose noise alone.
            if phone_near_ear and hand_near_phone:
                phone_call_detected = True

        # -----------------------------
        # Drinking Detection
        # -----------------------------
        if class_name in DRINK_OBJECT_CLASSES:
            object_near_mouth = False
            hand_near_object = False

            if mouth_center is not None:
                if distance_point_to_box(mouth_center, box) < DRINK_NEAR_MOUTH_THRESHOLD:
                    object_near_mouth = True

            for contact_point in contact_points:
                if distance_point_to_box(contact_point, box) < HAND_NEAR_OBJECT_THRESHOLD:
                    hand_near_object = True
                    break

            if object_near_mouth and hand_near_object:
                drinking_detected = True

    return frame, phone_call_detected, drinking_detected


# =========================
# Main Program
# =========================

def main():
    global DEBUG_OVERLAYS

    print("Starting CabInspector visual prototype...")
    print("Controls:")
    print("C = calibrate safe box")
    print("S = save screenshot")
    print("D = toggle debug overlays")
    print("V = calibrate normal speaking volume")
    print("T = show/hide live transcript")
    print("R = start/stop transcript recording (consent required)")
    print("E = export saved transcript as readable text")
    print("P = restart telemetry replay")
    print("Q, Shift+Q, or Esc = quit")
    print(f"Events will be logged to: {EVENT_LOG_PATH}")

    initialize_event_log()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        print("Try changing VideoCapture(0) to VideoCapture(1).")
        cap.release()
        cv2.destroyAllWindows()
        return

    audio_pipeline = None
    if USE_AUDIO:
        audio_model_path = Path(os.environ.get("CABINSPECTOR_AUDIO_MODEL", DEFAULT_MODEL_PATH))
        audio_device = os.environ.get("CABINSPECTOR_AUDIO_DEVICE")
        if audio_device and audio_device.isdigit():
            audio_device = int(audio_device)
        audio_pipeline = SpeechAnalysisPipeline(
            device=audio_device,
            yamnet_model_path=audio_model_path,
            whisper_model=os.environ.get("CABINSPECTOR_WHISPER_MODEL", "large-v3-turbo"),
            transcription_device=os.environ.get("CABINSPECTOR_WHISPER_DEVICE", "auto"),
            transcription_compute_type=os.environ.get(
                "CABINSPECTOR_WHISPER_COMPUTE_TYPE", "auto"
            ),
            transcription_language=(
                None
                if os.environ.get("CABINSPECTOR_SPEECH_LANGUAGE", "bilingual").lower()
                in {"auto", "bilingual"}
                else os.environ.get("CABINSPECTOR_SPEECH_LANGUAGE", "bilingual").lower()
            ),
            display_transcript=env_flag("CABINSPECTOR_DISPLAY_TRANSCRIPT", True),
            mask_unsafe_transcript=env_flag("CABINSPECTOR_MASK_UNSAFE_WORDS", True),
            # Rule-based Arabic/English safety checks are immediate. The optional neural model can
            # be enabled explicitly after its local weights have been downloaded and verified.
            use_safety_model=env_flag("CABINSPECTOR_USE_SAFETY_MODEL", False),
            store_transcripts=STORE_TRANSCRIPTS,
        )
        if audio_pipeline.start():
            print(
                "Audio pipeline listening: "
                f"{audio_model_path}; transcription: {audio_pipeline.transcriber.runtime_summary}"
            )
        else:
            print(audio_pipeline.get_latest_state().error)

    telemetry_replay = None
    telemetry_start_error = "Telemetry replay disabled"
    if USE_TELEMETRY_REPLAY:
        try:
            telemetry_replay = TelemetryReplay.from_public_trip(
                TELEMETRY_REPLAY_TRIP,
                model_name=TELEMETRY_REPLAY_MODEL,
                speed=TELEMETRY_REPLAY_SPEED,
            )
            telemetry_replay.start()
            telemetry_start_error = ""
            print(
                "Telemetry replay ready: "
                f"{telemetry_replay.model_name.replace('_', ' ')} model, public trip "
                f"{TELEMETRY_REPLAY_TRIP} at {TELEMETRY_REPLAY_SPEED:g}x speed. "
                "Predictions will appear after each replayed event ends."
            )
        except Exception as exc:
            telemetry_start_error = str(exc)
            print(f"Telemetry replay unavailable: {telemetry_start_error}")

    if FAST_DEMO_MODE:
        print("Fast demo mode enabled.")
        print(f"Camera target resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
        print(f"Processing resolution: {PROCESS_WIDTH}x{PROCESS_HEIGHT}")
        print(f"Display resolution: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
        print(f"Fullscreen preview: {DISPLAY_FULLSCREEN}")
        print(
            "Analysis intervals: "
            f"eyes every {EYE_ANALYSIS_EVERY_N_FRAMES}, "
            f"pose every {POSE_ANALYSIS_EVERY_N_FRAMES}, "
            f"hands every {HAND_ANALYSIS_EVERY_N_FRAMES} frame(s)"
        )
        print(f"General YOLO every {YOLO_EVERY_N_FRAMES} frames at {YOLO_IMAGE_SIZE}px")
        print(
            f"Phone search every {PHONE_SEARCH_EVERY_N_FRAMES} frames "
            f"at {PHONE_SEARCH_IMAGE_SIZE}px"
        )
        if MAX_FRAMES:
            print(f"Bounded smoke run: stopping after {MAX_FRAMES} frame(s)")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    if DISPLAY_FULLSCREEN:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    no_eye_counter = 0
    safe_box = None
    frame_count = 0
    last_eye_status = "Checking Eyes..."
    last_eye_closed_alert = False
    last_face_detected = False
    last_pose_points = {}
    last_hand_infos = []
    last_hand_status = "Checking Hands..."
    hand_hold_counter = 0
    last_yolo_detections = []
    last_phone_detections_source = []
    phone_detection_hold_counter = 0
    phone_call_counter = 0
    drinking_counter = 0
    zone_alert_counter = 0
    prev_phone_call_detected = False
    prev_drinking_detected = False
    prev_zone_alert = False
    prev_eye_closed_alert = False
    prev_raised_voice_active = False
    prev_telemetry_event_active = False
    last_auto_screenshot_time = 0.0
    last_general_yolo_frame = -YOLO_EVERY_N_FRAMES
    yolo_executor = None
    pending_yolo_future = None

    if USE_YOLO and yolo_model is not None and ASYNC_YOLO_INFERENCE:
        yolo_executor = ThreadPoolExecutor(max_workers=1)

    try:
        while True:
            ret, source_frame = cap.read()

            if not ret:
                print("Could not read frame.")
                break

            source_frame = cv2.flip(source_frame, 1)
            frame_count += 1
            frame_started = time.perf_counter() if RUNTIME_METRICS is not None else None

            if source_frame.shape[1] != PROCESS_WIDTH or source_frame.shape[0] != PROCESS_HEIGHT:
                frame = cv2.resize(
                    source_frame,
                    (PROCESS_WIDTH, PROCESS_HEIGHT),
                    interpolation=cv2.INTER_AREA
                )
            else:
                frame = source_frame.copy()

            h, w, _ = frame.shape
            source_h, source_w, _ = source_frame.shape
            scale_x_to_source = source_w / w
            scale_y_to_source = source_h / h
            scale_x_to_process = w / source_w
            scale_y_to_process = h / source_h

            should_run_eye_analysis = (
                DEBUG_OVERLAYS
                or frame_count == 1
                or frame_count % EYE_ANALYSIS_EVERY_N_FRAMES == 0
            )
            should_run_pose_analysis = (
                DEBUG_OVERLAYS
                or frame_count == 1
                or frame_count % POSE_ANALYSIS_EVERY_N_FRAMES == 0
            )
            should_run_hand_analysis = (
                DEBUG_OVERLAYS
                or frame_count == 1
                or frame_count % HAND_ANALYSIS_EVERY_N_FRAMES == 0
            )

            # 1. Face and eye detection
            if should_run_eye_analysis:
                with runtime_measure("visual.face_and_eyes"):
                    frame, no_eye_counter, eye_status, eye_closed_alert, face_detected = detect_face_and_eyes(
                        frame,
                        no_eye_counter,
                        draw_overlays=DEBUG_OVERLAYS
                    )
                last_eye_status = eye_status
                last_eye_closed_alert = eye_closed_alert
                last_face_detected = face_detected
            else:
                eye_status = last_eye_status
                eye_closed_alert = last_eye_closed_alert
                face_detected = last_face_detected

            # 2. Skeleton and landmark detection using MediaPipe Pose
            if should_run_pose_analysis:
                with runtime_measure("visual.pose"):
                    frame, pose_points = get_pose_points(frame, draw_overlays=DEBUG_OVERLAYS)
                last_pose_points = pose_points
            else:
                pose_points = last_pose_points

            # 2b. Actual hand landmark detection using MediaPipe Hands
            hand_hold_counter = max(0, hand_hold_counter - 1)

            if should_run_hand_analysis:
                force_full_frame_hand_analysis = (
                    frame_count <= 6
                    or frame_count % HAND_FULL_FRAME_EVERY_N_FRAMES == 0
                    or not last_hand_infos
                )
                with runtime_measure("visual.hands"):
                    frame, hand_infos, hand_status = get_hand_points(
                        frame,
                        pose_points,
                        draw_overlays=DEBUG_OVERLAYS,
                        force_full_frame=force_full_frame_hand_analysis,
                    )

                if hand_infos:
                    last_hand_infos = hand_infos
                    last_hand_status = hand_status
                    hand_hold_counter = HAND_TRACK_HOLD_FRAMES
                elif hand_hold_counter > 0 and last_hand_infos:
                    hand_infos = last_hand_infos
                    hand_status = last_hand_status
                else:
                    last_hand_infos = []
                    last_hand_status = hand_status
            else:
                if hand_hold_counter > 0 and last_hand_infos:
                    hand_infos = last_hand_infos
                    hand_status = last_hand_status
                else:
                    hand_infos = []
                    hand_status = "No Hands Detected"
                    last_hand_infos = []
                    last_hand_status = hand_status
            tracked_points = dict(pose_points)
            tracked_points.update(build_hand_point_map(hand_infos))
            phone_pose_points = scale_pose_points(pose_points, scale_x_to_source, scale_y_to_source)
            phone_hand_infos = scale_hand_infos(hand_infos, scale_x_to_source, scale_y_to_source)
            phone_search_regions = build_phone_search_regions(
                source_frame.shape,
                phone_pose_points,
                phone_hand_infos
            )

            # 3. Default virtual box before calibration
            if safe_box is None:
                safe_box = (
                    int(w * 0.20),
                    int(h * 0.20),
                    int(w * 0.80),
                    int(h * 0.95)
                )

            # 4. Check whether driver landmarks are inside virtual box
            with runtime_measure("visual.safe_zone"):
                frame, raw_zone_alert, outside_landmarks = check_safe_box(
                    frame,
                    tracked_points,
                    safe_box,
                    draw_overlays=DEBUG_OVERLAYS
                )

            zone_alert_counter, zone_alert = update_smoothing_counter(
                zone_alert_counter,
                raw_zone_alert
            )

            # 5. YOLO object detection
            phone_object_detected = False
            detected_object_names = []
            raw_phone_call_detected = False
            raw_drinking_detected = False

            phone_call_detected = False
            drinking_detected = False
            current_phone_detections_source = []

            if USE_YOLO and yolo_model is not None:
                phone_detection_hold_counter = max(0, phone_detection_hold_counter - 1)

                if yolo_executor is not None:
                    if pending_yolo_future is not None:
                        if pending_yolo_future.done():
                            try:
                                yolo_result = pending_yolo_future.result()
                                runtime_increment("visual.yolo.completed_jobs")
                                runtime_observe(
                                    "visual.yolo_job",
                                    yolo_result.get("runtime_ms", 0.0),
                                )

                                if yolo_result.get("ran_general_detection"):
                                    last_yolo_detections = yolo_result.get("general_detections", [])

                                current_phone_detections_source = yolo_result.get("phone_detections_source", [])

                                if current_phone_detections_source:
                                    last_phone_detections_source = current_phone_detections_source
                                    phone_detection_hold_counter = PHONE_OBJECT_HOLD_FRAMES
                            except Exception as exc:
                                runtime_increment("visual.yolo.worker_errors")
                                print(f"YOLO worker error: {exc}")
                            finally:
                                pending_yolo_future = None
                        else:
                            runtime_increment("visual.yolo.pending_frames")

                    should_submit_phone_search = (
                        pending_yolo_future is None
                        and frame_count % PHONE_SEARCH_EVERY_N_FRAMES == 0
                    )

                    if should_submit_phone_search:
                        run_general_detection = (
                            frame_count - last_general_yolo_frame
                        ) >= YOLO_EVERY_N_FRAMES

                        if run_general_detection:
                            last_general_yolo_frame = frame_count

                        pending_yolo_future = yolo_executor.submit(
                            run_yolo_detection_job,
                            frame.copy(),
                            source_frame.copy(),
                            list(phone_search_regions),
                            dict(phone_pose_points),
                            [dict(hand_info) for hand_info in phone_hand_infos],
                            run_general_detection,
                        )
                        runtime_increment("visual.yolo.submitted_jobs")
                else:
                    if frame_count % YOLO_EVERY_N_FRAMES == 0:
                        last_yolo_detections = run_yolo(
                            frame,
                            yolo_model,
                            target_classes=DRINK_OBJECT_CLASSES,
                            imgsz=YOLO_IMAGE_SIZE,
                            conf=YOLO_CONFIDENCE
                        )

                    if frame_count % PHONE_SEARCH_EVERY_N_FRAMES == 0:
                        phone_model = phone_yolo_model or yolo_model
                        phone_search_detections_source = []

                        if phone_search_regions:
                            phone_search_detections_source = run_yolo_on_regions(
                                source_frame,
                                phone_model,
                                phone_search_regions,
                                target_classes=PHONE_OBJECT_CLASSES,
                                imgsz=PHONE_SEARCH_IMAGE_SIZE,
                                conf=PHONE_SEARCH_CONFIDENCE,
                                augment=PHONE_SEARCH_AUGMENT,
                                max_det=PHONE_SEARCH_MAX_DET,
                                iou=PHONE_SEARCH_IOU
                            )

                        current_phone_detections_source = filter_phone_detections(
                            phone_search_detections_source,
                            phone_pose_points,
                            phone_hand_infos,
                            source_frame.shape
                        )

                        if current_phone_detections_source:
                            last_phone_detections_source = current_phone_detections_source
                            phone_detection_hold_counter = PHONE_OBJECT_HOLD_FRAMES

                if phone_detection_hold_counter > 0 and last_phone_detections_source:
                    current_phone_detections_source = last_phone_detections_source

                current_phone_detections = scale_detections(
                    current_phone_detections_source,
                    scale_x_to_process,
                    scale_y_to_process
                )
                combined_detections = dedupe_detections(
                    last_yolo_detections + current_phone_detections
                )

                with runtime_measure("visual.object_render"):
                    frame, phone_object_detected, detected_object_names = draw_yolo(
                        frame,
                        combined_detections,
                        draw_overlays=DEBUG_OVERLAYS
                    )

                # 6. Higher-level behaviour detection
                with runtime_measure("visual.action_rules"):
                    frame, raw_phone_call_detected, raw_drinking_detected = detect_object_actions(
                        frame,
                        combined_detections,
                        pose_points,
                        hand_infos,
                        draw_overlays=DEBUG_OVERLAYS
                    )

            runtime_gauge("visual.yolo_pending", pending_yolo_future is not None)

            phone_call_counter, phone_call_detected = update_smoothing_counter(
                phone_call_counter,
                raw_phone_call_detected
            )
            drinking_counter, drinking_detected = update_smoothing_counter(
                drinking_counter,
                raw_drinking_detected
            )

            eye_closed_counter = no_eye_counter
            audio_state = (
                audio_pipeline.get_latest_state()
                if audio_pipeline is not None
                else SpeechAnalysisState.unavailable("Audio disabled")
            )
            audio_is_current = audio_state.audio_available and not audio_state.error
            speech_with_phone_evidence = (
                audio_is_current and audio_state.speech_active and phone_call_detected
            )
            telemetry_state = (
                telemetry_replay.get_latest_state()
                if telemetry_replay is not None
                else TelemetryReplayState.unavailable(telemetry_start_error)
            )
            telemetry_is_current = telemetry_state.available and not telemetry_state.error

            # 7. Final live status summary
            with runtime_measure("fusion.risk_score"):
                risk_score, risk_parts = calculate_risk_score(
                    face_detected,
                    eye_closed_counter,
                    zone_alert_counter,
                    phone_object_detected,
                    phone_call_counter,
                    drinking_counter,
                    raised_voice_active=(
                        audio_is_current
                        and (audio_state.raised_voice_active or audio_state.loud_voice_active)
                    ),
                    speech_with_phone_evidence=speech_with_phone_evidence,
                    safety_categories=audio_state.safety_categories if audio_is_current else (),
                    telemetry_event_category=(
                        telemetry_state.event_category if telemetry_is_current else "NORMAL"
                    ),
                    telemetry_event_active=(
                        telemetry_is_current and telemetry_state.event_active
                    ),
                )
            risk_level = get_risk_level(risk_score)
            driver_status = risk_level
            final_status = f"Driver Behaviour Status: {driver_status}"
            detected_objects = sorted(set(detected_object_names))
            safe_zone_status = "ALERT" if zone_alert else "Normal"
            if not audio_state.audio_available:
                audio_status = "Unavailable"
            elif audio_state.error:
                audio_status = "Stale"
            elif audio_state.safety_flagged:
                audio_status = "Safety alert"
            elif audio_state.loud_voice_active or audio_state.raised_voice_active:
                audio_status = "Raised voice"
            elif audio_state.siren_active:
                audio_status = "Siren"
            elif audio_state.horn_active:
                audio_status = "Horn"
            elif audio_state.speech_active:
                audio_status = "Speech"
            elif audio_state.music_active:
                audio_status = "Music"
            elif audio_state.quiet_active:
                audio_status = "Quiet"
            else:
                audio_status = audio_state.audio_top_label

            telemetry_model_display = (telemetry_state.model_name or "telemetry").replace("_", " ").title()
            if not telemetry_state.available:
                telemetry_status = "Off" if not USE_TELEMETRY_REPLAY else "Unavailable"
            elif telemetry_state.error:
                telemetry_status = "Unavailable"
            elif telemetry_state.event_active:
                telemetry_status = (
                    f"{telemetry_model_display}: "
                    f"{telemetry_state.event_category.replace('_', ' ').title()}"
                )
            elif telemetry_state.completed:
                telemetry_status = f"{telemetry_model_display} complete"
            else:
                telemetry_status = f"{telemetry_model_display} {telemetry_state.speed:g}x"

            alerts = []
            if eye_closed_alert:
                alerts.append({"text": "Eyes possibly closed", "color": SIDEBAR_RED})
            if zone_alert:
                alerts.append({"text": "Arm/hand outside safe zone", "color": SIDEBAR_RED})
            if phone_call_detected:
                alerts.append({"text": "Talking on phone detected", "color": SIDEBAR_RED})
            if drinking_detected:
                alerts.append({"text": "Drinking detected", "color": SIDEBAR_RED})
            if audio_is_current and (audio_state.raised_voice_active or audio_state.loud_voice_active):
                alerts.append({"text": "Audio context: raised voice", "color": SIDEBAR_ORANGE})
            if audio_is_current and audio_state.siren_active:
                alerts.append({"text": "Audio context: siren", "color": SIDEBAR_ORANGE})
            if audio_is_current and audio_state.safety_flagged:
                safety_text = ", ".join(category.replace("_", " ").title() for category in audio_state.safety_categories)
                alerts.append({"text": f"Possible cabin speech: {safety_text}", "color": SIDEBAR_RED})
            if telemetry_is_current and telemetry_state.event_active:
                telemetry_text = telemetry_state.event_category.replace("_", " ").title()
                alerts.append(
                    {
                        "text": (
                            f"{telemetry_model_display} telemetry: {telemetry_text} "
                            f"({telemetry_state.event_confidence:.0%})"
                        ),
                        "color": SIDEBAR_ORANGE,
                    }
                )

            status_data = {
                "eye_status": eye_status,
                "hand_status": hand_status,
                "safe_zone_status": safe_zone_status,
                "audio_status": audio_status,
                "telemetry_status": telemetry_status,
                "telemetry_mode": telemetry_state.mode,
                "telemetry_model": telemetry_state.model_name,
                "telemetry_trip": telemetry_state.trip,
                "telemetry_event_category": telemetry_state.event_category,
                "telemetry_event_confidence": telemetry_state.event_confidence,
                "telemetry_event_active": telemetry_state.event_active,
                "speech_language": audio_state.language,
                "voice_level": audio_state.voice_level,
                "safety_status": ", ".join(audio_state.safety_categories) if audio_state.safety_flagged else "Clear",
                    "transient_transcript": audio_state.transient_transcript,
                    "transcript_visible": audio_state.transcript_visible,
                    "show_transcript": audio_state.transcript_visible,
                    "transcript_recording_enabled": audio_state.transcript_recording_enabled,
                    "recorded_transcript_count": audio_state.recorded_transcript_count,
                    "dropped_utterance_count": audio_state.dropped_utterance_count,
                "risk_score": risk_score,
                "driver_status": driver_status,
                "detected_objects": detected_objects,
                "risk_breakdown": risk_parts,
                "alerts": alerts,
                "final_status": final_status,
                "controls": [
                    "C  calibrate safe box",
                    "S  save screenshot",
                    "D  toggle debug overlays",
                    "V  calibrate normal voice",
                    "T  show/hide transcript",
                    "R  start/stop transcript log",
                    "E  export transcript text",
                    "P  restart telemetry replay",
                    "Q/Esc quit",
                ],
                "debug_overlays": DEBUG_OVERLAYS,
            }

            with runtime_measure("dashboard.render"):
                display_frame = create_dashboard_view(frame, status_data)
            runtime_increment("dashboard.alerts_rendered", len(alerts))

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            detected_objects_text = ", ".join(detected_objects) or "none"
            append_event_log(
                {
                    "timestamp": timestamp,
                    "frame_number": frame_count,
                    "eye_status": eye_status,
                    "hand_status": hand_status,
                    "hand_count": len(hand_infos),
                    "eye_closed_counter": eye_closed_counter,
                    "safe_zone_status": safe_zone_status,
                    "outside_landmarks": ", ".join(outside_landmarks),
                    "detected_objects": detected_objects_text,
                    "audio_available": audio_state.audio_available,
                    "audio_top_label": audio_state.audio_top_label,
                    "audio_top_confidence": audio_state.audio_top_confidence,
                    "speech_active": audio_state.speech_active,
                    "raised_voice_active": audio_state.raised_voice_active,
                    "horn_active": audio_state.horn_active,
                    "siren_active": audio_state.siren_active,
                    "audio_error": audio_state.error,
                    "speech_language": audio_state.language,
                    "voice_level": audio_state.voice_level,
                    "loudness_dbfs": audio_state.loudness_dbfs,
                    "loud_voice_active": audio_state.loud_voice_active,
                    "safety_flagged": audio_state.safety_flagged,
                    "safety_categories": ", ".join(audio_state.safety_categories),
                    "safety_confidence": audio_state.safety_confidence,
                    "transcript_visible": audio_state.transcript_visible,
                    "speech_analysis_error": audio_state.analysis_error,
                    "telemetry_available": telemetry_state.available,
                    "telemetry_mode": telemetry_state.mode,
                    "telemetry_model": telemetry_state.model_name,
                    "telemetry_trip": telemetry_state.trip,
                    "telemetry_replay_speed": telemetry_state.speed,
                    "telemetry_elapsed_seconds": round(telemetry_state.elapsed_seconds, 3),
                    "telemetry_events_processed": telemetry_state.events_processed,
                    "telemetry_event_category": telemetry_state.event_category,
                    "telemetry_event_confidence": telemetry_state.event_confidence,
                    "telemetry_event_active": telemetry_state.event_active,
                    "telemetry_error": telemetry_state.error,
                    "phone_object_detected": phone_object_detected,
                    "raw_phone_call_detected": raw_phone_call_detected,
                    "phone_call_counter": phone_call_counter,
                    "phone_call_detected": phone_call_detected,
                    "raw_drinking_detected": raw_drinking_detected,
                    "drinking_counter": drinking_counter,
                    "drinking_detected": drinking_detected,
                    "raw_zone_alert": raw_zone_alert,
                    "zone_alert_counter": zone_alert_counter,
                    "zone_alert": zone_alert,
                    "risk_score": risk_score,
                    "final_status": final_status,
                }
            )

            current_time = time.monotonic()
            alert_started = (
                (eye_closed_alert and not prev_eye_closed_alert)
                or (zone_alert and not prev_zone_alert)
                or (phone_call_detected and not prev_phone_call_detected)
                or (drinking_detected and not prev_drinking_detected)
                or (
                    audio_is_current
                    and (audio_state.raised_voice_active or audio_state.loud_voice_active)
                    and not prev_raised_voice_active
                )
                or (
                    telemetry_is_current
                    and telemetry_state.event_active
                    and not prev_telemetry_event_active
                )
            )

            if (
                AUTO_SCREENSHOTS_ENABLED
                and alert_started
                and (current_time - last_auto_screenshot_time) >= AUTO_SCREENSHOT_COOLDOWN_SECONDS
            ):
                alert_parts = []

                if eye_closed_alert and not prev_eye_closed_alert:
                    alert_parts.append("eyes_closed")
                if zone_alert and not prev_zone_alert:
                    alert_parts.append("zone")
                if phone_call_detected and not prev_phone_call_detected:
                    alert_parts.append("phone_call")
                if drinking_detected and not prev_drinking_detected:
                    alert_parts.append("drinking")
                if (
                    audio_is_current
                    and (audio_state.raised_voice_active or audio_state.loud_voice_active)
                    and not prev_raised_voice_active
                ):
                    alert_parts.append("raised_voice")
                if (
                    telemetry_is_current
                    and telemetry_state.event_active
                    and not prev_telemetry_event_active
                ):
                    alert_parts.append("telemetry")

                screenshot_prefix = "alert_" + "_".join(alert_parts or ["risk"])
                screenshot_status_data = dict(status_data)
                screenshot_status_data["show_transcript"] = audio_state.transcript_recording_enabled
                screenshot_frame = create_dashboard_view(frame, screenshot_status_data)
                screenshot_path = save_screenshot(screenshot_frame, screenshot_prefix, frame_count)
                last_auto_screenshot_time = current_time
                print(f"Auto screenshot saved: {screenshot_path}")

            prev_eye_closed_alert = eye_closed_alert
            prev_zone_alert = zone_alert
            prev_phone_call_detected = phone_call_detected
            prev_drinking_detected = drinking_detected
            prev_raised_voice_active = audio_is_current and (
                audio_state.raised_voice_active or audio_state.loud_voice_active
            )
            prev_telemetry_event_active = telemetry_is_current and telemetry_state.event_active

            cv2.imshow(WINDOW_NAME, display_frame)

            raw_key = cv2.waitKeyEx(1)
            key = raw_key & 0xFF if raw_key != -1 else -1

            if is_quit_key(key):
                print("Quit requested. Shutting down CabInspector...")
                break

            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    print("CabInspector window closed. Shutting down...")
                    break
            except cv2.error:
                print("CabInspector window is no longer available. Shutting down...")
                break

            if key == ord("d"):
                DEBUG_OVERLAYS = not DEBUG_OVERLAYS
                print(f"DEBUG_OVERLAYS = {DEBUG_OVERLAYS}")

            if key == ord("c"):
                calibrated_box = create_box_from_points(pose_points, frame.shape)

                if calibrated_box is not None:
                    safe_box = calibrated_box
                    print("Safe box calibrated.")
                else:
                    print("Could not calibrate: no pose landmarks detected.")

            if key == ord("v"):
                if audio_pipeline is None or not audio_pipeline.request_voice_calibration():
                    print("Voice calibration unavailable: enable audio first.")
                else:
                    print("Voice calibration started. Speak normally for five seconds.")

            if key == ord("t"):
                if audio_pipeline is None:
                    print("Transcript display unavailable: enable audio first.")
                else:
                    enabled = audio_pipeline.set_transcript_display(
                        not audio_pipeline.get_latest_state().transcript_display_enabled
                    )
                    print(f"Live transcript display {'enabled' if enabled else 'hidden'}.")

            if key == ord("r"):
                if audio_pipeline is None:
                    print("Transcript recording unavailable: enable audio first.")
                else:
                    enabled = audio_pipeline.set_transcript_recording(
                        not audio_pipeline.get_latest_state().transcript_recording_enabled
                    )
                    state = "started" if enabled else "stopped"
                    print(f"Transcript recording {state}: outputs/transcript_log.csv")

            if key == ord("e"):
                if audio_pipeline is None:
                    print("Transcript export unavailable: enable audio first.")
                else:
                    export_path = (
                        PROJECT_ROOT
                        / "outputs"
                        / "transcript_exports"
                        / f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    )
                    exported, result = audio_pipeline.export_transcripts(export_path)
                    if exported:
                        print(f"Transcript exported: {result}")
                    else:
                        print(f"Transcript export failed: {result}")

            if key == ord("p"):
                if telemetry_replay is None:
                    print("Telemetry replay unavailable: set CABINSPECTOR_USE_TELEMETRY_REPLAY=1 first.")
                else:
                    telemetry_replay.restart()
                    print("Telemetry replay restarted.")

            if key == ord("s"):
                screenshot_status_data = dict(status_data)
                screenshot_status_data["show_transcript"] = audio_state.transcript_recording_enabled
                screenshot_frame = create_dashboard_view(frame, screenshot_status_data)
                screenshot_path = save_screenshot(screenshot_frame, "visual_prototype", frame_count)
                print(f"Screenshot saved: {screenshot_path}")
            if frame_started is not None:
                runtime_increment("dashboard.frames_completed")
                runtime_observe(
                    "dashboard.frame_total",
                    (time.perf_counter() - frame_started) * 1000.0,
                )
            if MAX_FRAMES and frame_count >= MAX_FRAMES:
                runtime_increment("dashboard.frame_limit_reached")
                print(f"Maximum frame limit reached ({MAX_FRAMES}); stopping cleanly.")
                break
    except KeyboardInterrupt:
        print("Interrupt received. Shutting down CabInspector cleanly...")
    finally:
        if yolo_executor is not None:
            try:
                yolo_executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                yolo_executor.shutdown(wait=False)

        cap.release()
        cv2.destroyAllWindows()

        if pose is not None:
            pose.close()

        if hands is not None:
            hands.close()

        if face_mesh is not None:
            face_mesh.close()

        if audio_pipeline is not None:
            audio_pipeline.stop()
            audio_statistics = audio_pipeline.runtime_statistics()
            for component, values in audio_statistics.items():
                for name, value in values.items():
                    runtime_gauge(f"audio.{component}.{name}", value)

        if telemetry_replay is not None:
            telemetry_statistics = telemetry_replay.get_latest_state()
            runtime_gauge("telemetry.events_total", telemetry_statistics.events_total)
            runtime_gauge("telemetry.events_processed", telemetry_statistics.events_processed)

        runtime_gauge("dashboard.frames_completed", frame_count)
        runtime_gauge("dashboard.max_frames", MAX_FRAMES)
        if RUNTIME_METRICS is not None:
            RUNTIME_METRICS.write(
                RUNTIME_METRICS_PATH,
                context={
                    "frames_completed": frame_count,
                    "audio_enabled": USE_AUDIO,
                    "telemetry_enabled": USE_TELEMETRY_REPLAY,
                    "auto_screenshots_enabled": AUTO_SCREENSHOTS_ENABLED,
                    "transcript_recording_enabled": STORE_TRANSCRIPTS,
                },
            )

        close_event_log()


if __name__ == "__main__":
    main()
