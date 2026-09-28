"""Stable CabInspector labels for public telemetry events."""

from __future__ import annotations


CLASS_NAMES = (
    "NORMAL",
    "HARD_BRAKE",
    "RAPID_ACCELERATION",
    "AGGRESSIVE_TURN",
    "AGGRESSIVE_LANE_CHANGE",
)
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}

_SOURCE_TO_CLASS = {
    "non aggressive event": "NORMAL",
    # One annotation in public trip 2 uses the original Portuguese wording.
    "evento não agressivo": "NORMAL",
    "aggressive braking": "HARD_BRAKE",
    "aggressive acceleration": "RAPID_ACCELERATION",
    "aggressive right turn": "AGGRESSIVE_TURN",
    "aggressive left turn": "AGGRESSIVE_TURN",
    "aggressive lane change to the right": "AGGRESSIVE_LANE_CHANGE",
    "aggressive lane change to the left": "AGGRESSIVE_LANE_CHANGE",
    "aggressive right lane change": "AGGRESSIVE_LANE_CHANGE",
    "aggressive left lane change": "AGGRESSIVE_LANE_CHANGE",
}


def normalize_source_label(label: str) -> str:
    """Normalize only spacing and punctuation, retaining the source wording."""

    return " ".join(str(label or "").casefold().replace("-", " ").split())


def map_source_label(label: str) -> str:
    """Map a verified source label to CabInspector's initial five classes."""

    normalized = normalize_source_label(label)
    try:
        return _SOURCE_TO_CLASS[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported Driving Events Dataset label: {label!r}") from exc


def class_index(label: str) -> int:
    return CLASS_TO_INDEX[label]
