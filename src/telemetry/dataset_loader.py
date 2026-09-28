"""Strict loader for the public Zenodo Driving Events Dataset.

The loader accepts only the public ``data/Telemetry/public/...`` source
directory. It intentionally has no code path for ``data/Telemetry/Recordings``
so private trip data cannot accidentally enter a training run.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dataset_audit import DEFAULT_RAW_DIRECTORY
from .labels import class_index, map_source_label


@dataclass(frozen=True)
class LabeledEvent:
    trip: int
    source_label: str
    category: str
    target: int
    start_seconds: float
    end_seconds: float
    time_sync_seconds: float
    source_sample: int

    @property
    def sensor_start_seconds(self) -> float:
        """Start on the sensor timeline.

        The source notebook selects sensor rows directly between the annotation
        ``start`` and ``end`` values. ``timesync`` is retained as source
        metadata but is not applied as a second offset.
        """

        return self.start_seconds

    @property
    def sensor_end_seconds(self) -> float:
        """End on the sensor timeline; see :attr:`sensor_start_seconds`."""

        return self.end_seconds

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds

    @property
    def class_index(self) -> int:
        return class_index(self.category)


@dataclass(frozen=True)
class SensorStream:
    timestamps: np.ndarray
    values: np.ndarray
    columns: tuple[str, str, str]

    def __post_init__(self) -> None:
        if self.timestamps.ndim != 1 or self.values.ndim != 2 or self.values.shape[1] != 3:
            raise ValueError("A sensor stream must have one timestamp and three value channels.")
        if len(self.timestamps) != len(self.values):
            raise ValueError("Sensor timestamps and values have different lengths.")
        if len(self.timestamps) < 2 or np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("Sensor timestamps must be strictly increasing.")


@dataclass(frozen=True)
class PublicTelemetryTrip:
    trip: int
    acceleration: SensorStream
    gyroscope: SensorStream
    events: tuple[LabeledEvent, ...]


def _sensor_file(raw_directory: Path, kind: str, trip: int) -> Path:
    file_name = f"{kind}_{trip}.csv"
    path = raw_directory / file_name
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing public source file: {path}. Run tools/download_driving_events_dataset.py first."
        )
    return path


def _load_sensor_stream(path: Path, expected_prefix: str) -> SensorStream:
    with path.open("r", encoding="utf-8-sig", newline="") as file_handle:
        header = next(csv.reader(file_handle), None)
    if header is None or len(header) != 4:
        raise ValueError(f"Expected four columns in {path.name}, found {header!r}")
    if not header[0].strip().casefold().startswith("time"):
        raise ValueError(f"Expected a time column in {path.name}, found {header[0]!r}")
    if not all(column.strip().casefold().startswith(expected_prefix) for column in header[1:]):
        raise ValueError(f"Unexpected {expected_prefix} channel headers in {path.name}: {header[1:]!r}")

    values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[1] != 4:
        raise ValueError(f"Expected four numeric columns in {path.name}, found {values.shape[1]}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Found non-finite values in {path.name}")
    return SensorStream(
        timestamps=values[:, 0],
        values=values[:, 1:],
        columns=tuple(header[1:]),
    )


def load_public_events(raw_directory: Path, trip: int) -> tuple[LabeledEvent, ...]:
    """Load annotated public events and preserve the source timestamp sync value."""

    path = _sensor_file(raw_directory, "Labeled_events", trip)
    events: list[LabeledEvent] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        expected_fields = {"start", "end", "event", "target", "timesync", "sample"}
        fieldnames = {name.strip().casefold() for name in (reader.fieldnames or [])}
        if not expected_fields.issubset(fieldnames):
            raise ValueError(f"Unexpected label schema in {path.name}: {reader.fieldnames!r}")
        for row in reader:
            source_label = str(row["event"]).strip()
            start_seconds = float(row["start"])
            end_seconds = float(row["end"])
            time_sync_seconds = float(row["timesync"])
            if not start_seconds < end_seconds:
                raise ValueError(f"Invalid event interval in {path.name}: {row!r}")
            events.append(
                LabeledEvent(
                    trip=trip,
                    source_label=source_label,
                    category=map_source_label(source_label),
                    target=int(row["target"]),
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    time_sync_seconds=time_sync_seconds,
                    source_sample=int(row["sample"]),
                )
            )
    if not events:
        raise ValueError(f"No events found in {path.name}")
    return tuple(events)


def load_public_trip(
    trip: int,
    raw_directory: Path = DEFAULT_RAW_DIRECTORY,
) -> PublicTelemetryTrip:
    """Load one of the three public trips with strict source-schema checks."""

    if trip not in (1, 2, 3):
        raise ValueError("The Driving Events Dataset contains public trips 1, 2, and 3 only.")
    raw_directory = Path(raw_directory)
    acceleration = _load_sensor_stream(
        _sensor_file(raw_directory, "Linear_Acceleration", trip), "linear acceleration"
    )
    gyroscope = _load_sensor_stream(_sensor_file(raw_directory, "Gyroscope", trip), "gyroscope")
    return PublicTelemetryTrip(
        trip=trip,
        acceleration=acceleration,
        gyroscope=gyroscope,
        events=load_public_events(raw_directory, trip),
    )
