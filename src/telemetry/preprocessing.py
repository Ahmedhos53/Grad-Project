"""Leakage-safe window preparation for the public telemetry dataset.

Raw sensor data remains untouched.  This module only converts annotated event
intervals into fixed-shape model inputs.  Dataset-wide normalization is
deliberately absent: training folds calculate their own statistics later so a
held-out trip cannot influence training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from .dataset_audit import DEFAULT_RAW_DIRECTORY
from .dataset_loader import LabeledEvent, PublicTelemetryTrip, SensorStream, load_public_trip
from .labels import CLASS_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIRECTORY = (
    PROJECT_ROOT / "data" / "Telemetry" / "public" / "driving_events_dataset" / "processed"
)
FEATURE_NAMES = (
    "acceleration_x_mps2",
    "acceleration_y_mps2",
    "acceleration_z_mps2",
    "gyroscope_x_radps",
    "gyroscope_y_radps",
    "gyroscope_z_radps",
)


@dataclass(frozen=True)
class WindowConfig:
    """Source-preserving event-window configuration.

    Event durations vary from roughly one to nine seconds.  Each whole labelled
    interval is interpolated to a fixed sequence length; its original duration
    is preserved in metadata for later evaluation and feature-baseline work.
    """

    samples_per_window: int = 256

    def __post_init__(self) -> None:
        if self.samples_per_window < 16:
            raise ValueError("samples_per_window must be at least 16")


@dataclass(frozen=True)
class PreparedTripWindows:
    trip: int
    inputs: np.ndarray
    targets: np.ndarray
    source_targets: np.ndarray
    source_samples: np.ndarray
    durations_seconds: np.ndarray
    start_seconds: np.ndarray
    end_seconds: np.ndarray
    source_labels: tuple[str, ...]


def _interpolate_stream(
    stream: SensorStream,
    start_seconds: float,
    end_seconds: float,
    samples_per_window: int,
) -> np.ndarray:
    if start_seconds < stream.timestamps[0] or end_seconds > stream.timestamps[-1]:
        raise ValueError(
            "Annotated interval falls outside the sensor recording: "
            f"{start_seconds:.3f}–{end_seconds:.3f} not in "
            f"{stream.timestamps[0]:.3f}–{stream.timestamps[-1]:.3f}"
        )
    requested_times = np.linspace(start_seconds, end_seconds, samples_per_window, dtype=np.float64)
    channels = [
        np.interp(requested_times, stream.timestamps, stream.values[:, channel])
        for channel in range(stream.values.shape[1])
    ]
    return np.asarray(channels, dtype=np.float32)


def make_event_window(event: LabeledEvent, trip: PublicTelemetryTrip, config: WindowConfig) -> np.ndarray:
    """Create a ``(6, samples_per_window)`` raw sensor window for one event."""

    if event.trip != trip.trip:
        raise ValueError("Event trip and sensor trip do not match.")
    acceleration = _interpolate_stream(
        trip.acceleration,
        event.sensor_start_seconds,
        event.sensor_end_seconds,
        config.samples_per_window,
    )
    gyroscope = _interpolate_stream(
        trip.gyroscope,
        event.sensor_start_seconds,
        event.sensor_end_seconds,
        config.samples_per_window,
    )
    return np.concatenate((acceleration, gyroscope), axis=0)


def prepare_trip_windows(trip: PublicTelemetryTrip, config: WindowConfig = WindowConfig()) -> PreparedTripWindows:
    """Prepare every annotated event in one trip without cross-trip mixing."""

    windows = [make_event_window(event, trip, config) for event in trip.events]
    return PreparedTripWindows(
        trip=trip.trip,
        inputs=np.stack(windows).astype(np.float32, copy=False),
        targets=np.asarray([event.class_index for event in trip.events], dtype=np.int64),
        source_targets=np.asarray([event.target for event in trip.events], dtype=np.int64),
        source_samples=np.asarray([event.source_sample for event in trip.events], dtype=np.int64),
        durations_seconds=np.asarray([event.duration_seconds for event in trip.events], dtype=np.float32),
        start_seconds=np.asarray([event.start_seconds for event in trip.events], dtype=np.float64),
        end_seconds=np.asarray([event.end_seconds for event in trip.events], dtype=np.float64),
        source_labels=tuple(event.source_label for event in trip.events),
    )


def prepare_public_dataset(
    raw_directory: Path = DEFAULT_RAW_DIRECTORY,
    output_directory: Path = DEFAULT_PROCESSED_DIRECTORY,
    config: WindowConfig = WindowConfig(),
) -> dict[str, object]:
    """Create one unnormalised, source-traceable NPZ per public trip."""

    raw_directory = Path(raw_directory)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    trip_summaries = []
    for trip_number in (1, 2, 3):
        prepared = prepare_trip_windows(load_public_trip(trip_number, raw_directory), config)
        destination = output_directory / f"trip_{trip_number}_windows.npz"
        np.savez_compressed(
            destination,
            inputs=prepared.inputs,
            targets=prepared.targets,
            source_targets=prepared.source_targets,
            source_samples=prepared.source_samples,
            durations_seconds=prepared.durations_seconds,
            start_seconds=prepared.start_seconds,
            end_seconds=prepared.end_seconds,
            source_labels=np.asarray(prepared.source_labels, dtype="U64"),
        )
        counts = np.bincount(prepared.targets, minlength=len(CLASS_NAMES))
        trip_summaries.append(
            {
                "trip": trip_number,
                "file": destination.name,
                "window_count": int(len(prepared.targets)),
                "class_counts": {name: int(counts[index]) for index, name in enumerate(CLASS_NAMES)},
            }
        )

    manifest = {
        "dataset": "Driving Events Dataset: a smartphone inertial measurement unit for driving events",
        "raw_directory": str(raw_directory),
        "local_recordings_excluded": True,
        "window_config": asdict(config),
        "feature_names": list(FEATURE_NAMES),
        "class_names": list(CLASS_NAMES),
        "normalization": "Not applied. Fold training must fit normalization on its training trips only.",
        "trips": trip_summaries,
    }
    (output_directory / "preprocessing_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest
