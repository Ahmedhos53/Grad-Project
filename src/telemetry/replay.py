"""Timed, prediction-only replay for the CabInspector telemetry dashboard.

The public telemetry model was trained on complete annotated events, so this
module is deliberately a *recorded-trip replay* rather than a claim of live
phone inference.  It releases only the model prediction after the equivalent
event end time has elapsed.  Source labels remain internal to the processed
dataset and are never shown to the application.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from .event_smoother import TelemetryEventSmoother, TelemetrySmoothingConfig
from .dataset_audit import DEFAULT_RAW_DIRECTORY
from .dataset_loader import load_public_trip
from .inference import TelemetryPrediction, TelemetryRandomForestPredictor
from .labels import CLASS_NAMES
from .preprocessing import DEFAULT_PROCESSED_DIRECTORY
from .primus_inference import PrimusTelemetryPredictor
from .primus_transfer import make_fixed_primus_event_window


@dataclass(frozen=True)
class TelemetryReplayEvent:
    """One predicted event scheduled relative to the start of a replay."""

    available_at_seconds: float
    prediction: TelemetryPrediction


@dataclass(frozen=True)
class TelemetryReplayState:
    """Small immutable snapshot consumed by the UI and event logger."""

    available: bool = False
    mode: str = "DISABLED"
    model_name: str = ""
    trip: int = 0
    speed: float = 1.0
    elapsed_seconds: float = 0.0
    events_total: int = 0
    events_processed: int = 0
    event_category: str = "NORMAL"
    event_confidence: float = 0.0
    event_active: bool = False
    completed: bool = False
    error: str = ""

    @classmethod
    def unavailable(cls, error: str) -> "TelemetryReplayState":
        return cls(mode="UNAVAILABLE", error=error)


class TelemetryReplay:
    """Replay previously recorded model inputs without blocking the UI loop."""

    def __init__(
        self,
        events: list[TelemetryReplayEvent],
        *,
        trip: int,
        speed: float = 10.0,
        event_hold_seconds: float = 3.0,
        minimum_confidence: float = 0.55,
        model_name: str = "random_forest",
    ) -> None:
        if trip < 1:
            raise ValueError("trip must be positive")
        if speed <= 0:
            raise ValueError("speed must be greater than zero")
        if event_hold_seconds < 0:
            raise ValueError("event_hold_seconds cannot be negative")
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1")

        self._events = sorted(events, key=lambda event: event.available_at_seconds)
        if model_name not in {"primus", "random_forest"}:
            raise ValueError("model_name must be 'primus' or 'random_forest'")
        self.model_name = model_name
        self.trip = trip
        self.speed = float(speed)
        self.event_hold_seconds = float(event_hold_seconds)
        # Each replay input is already one complete classified event.  A single
        # prediction is therefore confirmed for display; this is not the
        # streaming smoother configuration intended for future live telemetry.
        self._smoother = TelemetryEventSmoother(
            TelemetrySmoothingConfig(
                minimum_confidence=minimum_confidence,
                confirm_windows=1,
                cooldown_ms=0,
            )
        )
        self._started_at: float | None = None
        self._next_event_index = 0
        self._last_active_at = float("-inf")
        self._last_category = "NORMAL"
        self._last_confidence = 0.0

    @classmethod
    def from_processed_trip(
        cls,
        trip: int,
        *,
        speed: float = 10.0,
        processed_directory: Path = DEFAULT_PROCESSED_DIRECTORY,
        model_path: Path | None = None,
        event_hold_seconds: float = 3.0,
    ) -> "TelemetryReplay":
        """Precompute predictions for one public processed trip.

        Predictions are calculated once during startup so UI ticks only compare
        timestamps and never run blocking model inference.
        """

        if trip not in (1, 2, 3):
            raise ValueError("Only public telemetry trips 1, 2, and 3 are available")
        processed_path = Path(processed_directory) / f"trip_{trip}_windows.npz"
        if not processed_path.is_file():
            raise FileNotFoundError(
                f"Processed telemetry trip not found: {processed_path}. "
                "Run tools/prepare_driving_events_dataset.py first."
            )

        predictor = (
            TelemetryRandomForestPredictor(model_path)
            if model_path is not None
            else TelemetryRandomForestPredictor()
        )
        with np.load(processed_path, allow_pickle=False) as loaded:
            windows = np.asarray(loaded["inputs"], dtype=np.float32)
            start_seconds = np.asarray(loaded["start_seconds"], dtype=np.float64)
            end_seconds = np.asarray(loaded["end_seconds"], dtype=np.float64)

        if not len(windows):
            raise ValueError(f"Processed telemetry trip contains no event windows: {processed_path}")
        if len(windows) != len(start_seconds) or len(windows) != len(end_seconds):
            raise ValueError("Processed telemetry window/timestamp counts do not match")

        replay_origin = float(start_seconds.min())
        events = [
            TelemetryReplayEvent(
                available_at_seconds=max(0.0, float(end_time - replay_origin)),
                prediction=predictor.predict_window(window),
            )
            for window, end_time in zip(windows, end_seconds)
        ]
        return cls(
            events,
            trip=trip,
            speed=speed,
            event_hold_seconds=event_hold_seconds,
            model_name="random_forest",
        )

    @classmethod
    def from_primus_public_trip(
        cls,
        trip: int,
        *,
        speed: float = 10.0,
        raw_directory: Path = DEFAULT_RAW_DIRECTORY,
        event_hold_seconds: float = 3.0,
    ) -> "TelemetryReplay":
        """Precompute frozen PRIMUS predictions for one public raw-data trip.

        PRIMUS receives the same fixed five-second/200 Hz event-centred window
        used in its documented transfer evaluation. Only its prediction and the
        event timing enter the replay; source labels remain internal.
        """

        if trip not in (1, 2, 3):
            raise ValueError("Only public telemetry trips 1, 2, and 3 are available")
        source_trip = load_public_trip(trip, Path(raw_directory))
        predictor = PrimusTelemetryPredictor()
        replay_origin = min(event.sensor_start_seconds for event in source_trip.events)
        events = []
        for source_event in source_trip.events:
            window, _, _ = make_fixed_primus_event_window(source_event, source_trip)
            primus_prediction = predictor.predict(window)
            events.append(
                TelemetryReplayEvent(
                    available_at_seconds=max(0.0, source_event.sensor_end_seconds - replay_origin),
                    prediction=TelemetryPrediction(
                        category=primus_prediction.category,
                        confidence=primus_prediction.confidence,
                        probabilities={
                            name: probability
                            for name, probability in zip(CLASS_NAMES, primus_prediction.probabilities)
                        },
                    ),
                )
            )
        if not events:
            raise ValueError(f"Public telemetry trip {trip} contains no events")
        return cls(
            events,
            trip=trip,
            speed=speed,
            event_hold_seconds=event_hold_seconds,
            model_name="primus",
        )

    @classmethod
    def from_public_trip(
        cls,
        trip: int,
        *,
        model_name: str = "primus",
        speed: float = 10.0,
        event_hold_seconds: float = 3.0,
    ) -> "TelemetryReplay":
        """Build a public replay using an explicitly selected telemetry model."""

        normalized = str(model_name).strip().lower()
        if normalized == "primus":
            return cls.from_primus_public_trip(
                trip,
                speed=speed,
                event_hold_seconds=event_hold_seconds,
            )
        if normalized == "random_forest":
            return cls.from_processed_trip(
                trip,
                speed=speed,
                event_hold_seconds=event_hold_seconds,
            )
        raise ValueError("Telemetry replay model must be 'primus' or 'random_forest'")

    def start(self, now: float | None = None) -> None:
        self.restart(now)

    def restart(self, now: float | None = None) -> None:
        self._started_at = time.monotonic() if now is None else float(now)
        self._next_event_index = 0
        self._last_active_at = float("-inf")
        self._last_category = "NORMAL"
        self._last_confidence = 0.0
        self._smoother = TelemetryEventSmoother(self._smoother.config)

    def get_latest_state(self, now: float | None = None) -> TelemetryReplayState:
        if self._started_at is None:
            return TelemetryReplayState(
                available=True,
                mode="READY",
                model_name=self.model_name,
                trip=self.trip,
                speed=self.speed,
                events_total=len(self._events),
            )

        current_time = time.monotonic() if now is None else float(now)
        elapsed_seconds = max(0.0, (current_time - self._started_at) * self.speed)

        while (
            self._next_event_index < len(self._events)
            and self._events[self._next_event_index].available_at_seconds <= elapsed_seconds
        ):
            event = self._events[self._next_event_index]
            emitted = self._smoother.update(
                event.prediction,
                int(round(event.available_at_seconds * 1000)),
            )
            self._next_event_index += 1
            if emitted.active:
                self._last_active_at = elapsed_seconds
                self._last_category = emitted.category
                self._last_confidence = emitted.confidence

        event_active = (
            self._last_category != "NORMAL"
            and elapsed_seconds - self._last_active_at <= self.event_hold_seconds
        )
        completed = self._next_event_index >= len(self._events)
        if event_active:
            mode = "EVENT"
            category = self._last_category
            confidence = self._last_confidence
        else:
            mode = "COMPLETE" if completed else "REPLAYING"
            category = "NORMAL"
            confidence = 0.0

        return TelemetryReplayState(
            available=True,
            mode=mode,
            model_name=self.model_name,
            trip=self.trip,
            speed=self.speed,
            elapsed_seconds=elapsed_seconds,
            events_total=len(self._events),
            events_processed=self._next_event_index,
            event_category=category,
            event_confidence=confidence,
            event_active=event_active,
            completed=completed,
        )
