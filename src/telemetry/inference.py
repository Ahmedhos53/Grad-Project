"""Load and run CabInspector's selected telemetry model on one sensor window."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from .labels import CLASS_NAMES
from .training import Normalization, extract_classical_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "telemetry" / "telemetry_random_forest.joblib"


@dataclass(frozen=True)
class TelemetryPrediction:
    category: str
    confidence: float
    probabilities: dict[str, float]


class TelemetryRandomForestPredictor:
    """CPU-only predictor for the offline-trained public-data telemetry model."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Telemetry model not found: {model_path}. Run tools/export_telemetry_model.py first."
            )
        package = joblib.load(model_path)
        if package.get("model_type") != "random_forest":
            raise ValueError(f"Unsupported telemetry model package: {package.get('model_type')!r}")
        if tuple(package.get("class_names", ())) != CLASS_NAMES:
            raise ValueError("Telemetry model class mapping does not match this CabInspector version.")
        self.model = package["model"]
        self.normalization = Normalization(
            mean=np.asarray(package["normalization"]["mean"], dtype=np.float32).reshape(1, 6, 1),
            std=np.asarray(package["normalization"]["std"], dtype=np.float32).reshape(1, 6, 1),
        )
        self.samples_per_window = int(package["samples_per_window"])

    def predict_window(self, window: np.ndarray) -> TelemetryPrediction:
        inputs = np.asarray(window, dtype=np.float32)
        if inputs.shape != (6, self.samples_per_window):
            raise ValueError(
                f"Expected one window shaped (6, {self.samples_per_window}), got {inputs.shape}"
            )
        normalized = self.normalization.apply(inputs.reshape(1, 6, self.samples_per_window))
        probabilities = self.model.predict_proba(extract_classical_features(normalized))[0]
        index = int(np.argmax(probabilities))
        return TelemetryPrediction(
            category=CLASS_NAMES[index],
            confidence=float(probabilities[index]),
            probabilities={name: float(probabilities[i]) for i, name in enumerate(CLASS_NAMES)},
        )
