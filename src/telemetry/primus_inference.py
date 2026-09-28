"""Operational inference for the frozen pretrained PRIMUS telemetry model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np

from .labels import CLASS_NAMES
from .primus_encoder import (
    DEFAULT_PRIMUS_ENCODER_STATE,
    PRIMUS_CHANNELS,
    PRIMUS_SAMPLES_PER_WINDOW,
    PrimusIMUEncoder,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRIMUS_EVENT_HEAD = (
    PROJECT_ROOT / "models" / "telemetry" / "pretrained" / "primus" / "event_linear_head.joblib"
)


@dataclass(frozen=True)
class PrimusTelemetryPrediction:
    category: str
    confidence: float
    probabilities: tuple[float, ...]
    inference_ms: float


def prediction_from_probabilities(
    probabilities: np.ndarray,
    *,
    inference_ms: float = 0.0,
) -> PrimusTelemetryPrediction:
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if values.shape != (len(CLASS_NAMES),):
        raise ValueError(f"Expected {len(CLASS_NAMES)} event probabilities, got {values.shape}")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("Event probabilities must be finite and non-negative")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("Event probabilities must have positive mass")
    values = values / total
    index = int(np.argmax(values))
    return PrimusTelemetryPrediction(
        category=CLASS_NAMES[index],
        confidence=float(values[index]),
        probabilities=tuple(float(value) for value in values),
        inference_ms=float(inference_ms),
    )


class PrimusTelemetryPredictor:
    """Frozen PRIMUS encoder plus CabInspector's fitted linear event head."""

    def __init__(
        self,
        encoder_path: str | Path = DEFAULT_PRIMUS_ENCODER_STATE,
        head_path: str | Path = DEFAULT_PRIMUS_EVENT_HEAD,
    ) -> None:
        self.encoder = PrimusIMUEncoder()
        self.encoder.load_compact_state(encoder_path)
        package = joblib.load(Path(head_path))
        if not isinstance(package, dict) or package.get("format") != "cabinspector_primus_event_head_v1":
            raise ValueError("Unsupported PRIMUS event-head package")
        if tuple(package.get("class_names", ())) != CLASS_NAMES:
            raise ValueError("PRIMUS event-head classes do not match CabInspector")
        self.classifier = package.get("classifier")
        if self.classifier is None or not hasattr(self.classifier, "predict_proba"):
            raise ValueError("PRIMUS event-head package has no probability classifier")

    def predict(self, window: np.ndarray) -> PrimusTelemetryPrediction:
        values = np.asarray(window, dtype=np.float32)
        if values.shape != (PRIMUS_CHANNELS, PRIMUS_SAMPLES_PER_WINDOW):
            raise ValueError(
                "Expected one fixed PRIMUS window shaped "
                f"({PRIMUS_CHANNELS}, {PRIMUS_SAMPLES_PER_WINDOW}), got {values.shape}"
            )
        started = perf_counter()
        embedding = self.encoder.encode(values[None, ...])
        observed = self.classifier.predict_proba(embedding)[0]
        probabilities = np.zeros(len(CLASS_NAMES), dtype=np.float64)
        for source_column, class_index in enumerate(self.classifier.classes_):
            probabilities[int(class_index)] = observed[source_column]
        elapsed_ms = (perf_counter() - started) * 1_000
        return prediction_from_probabilities(probabilities, inference_ms=elapsed_ms)
