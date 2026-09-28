"""Turn raw telemetry predictions into conservative stable event evidence."""

from __future__ import annotations

from dataclasses import dataclass

from .inference import TelemetryPrediction


@dataclass(frozen=True)
class TelemetrySmoothingConfig:
    minimum_confidence: float = 0.55
    confirm_windows: int = 2
    cooldown_ms: int = 6_000


@dataclass(frozen=True)
class TelemetryEventState:
    category: str = "UNKNOWN"
    confidence: float = 0.0
    active: bool = False
    timestamp_ms: int = 0


class TelemetryEventSmoother:
    def __init__(self, config: TelemetrySmoothingConfig = TelemetrySmoothingConfig()) -> None:
        self.config = config
        self._candidate = "UNKNOWN"
        self._candidate_count = 0
        self._last_emitted: dict[str, int] = {}

    def update(self, prediction: TelemetryPrediction, timestamp_ms: int) -> TelemetryEventState:
        category = prediction.category
        if category == "NORMAL" or prediction.confidence < self.config.minimum_confidence:
            self._candidate, self._candidate_count = "UNKNOWN", 0
            return TelemetryEventState(timestamp_ms=timestamp_ms)
        if category == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate, self._candidate_count = category, 1
        allowed = timestamp_ms - self._last_emitted.get(category, -self.config.cooldown_ms) >= self.config.cooldown_ms
        if self._candidate_count >= self.config.confirm_windows and allowed:
            self._last_emitted[category] = timestamp_ms
            self._candidate_count = 0
            return TelemetryEventState(category, prediction.confidence, True, timestamp_ms)
        return TelemetryEventState(category, prediction.confidence, False, timestamp_ms)
