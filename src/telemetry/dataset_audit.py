"""Audit raw files from the public Driving Events Dataset before training.

The audit deliberately makes no assumptions about sensor column names,
timestamp format, or sample rate.  It produces a machine-readable summary so
preprocessing decisions can be based on the downloaded source files rather
than on copied values from a paper or notebook.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIRECTORY = (
    PROJECT_ROOT / "data" / "Telemetry" / "public" / "driving_events_dataset" / "raw"
)
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "telemetry" / "audit"


@dataclass
class NumericSummary:
    """Streaming numeric statistics that do not require loading a whole CSV."""

    count: int = 0
    minimum: float | None = None
    maximum: float | None = None
    mean: float = 0.0
    _m2: float = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        if self.minimum is None or value < self.minimum:
            self.minimum = value
        if self.maximum is None or value > self.maximum:
            self.maximum = value
        delta = value - self.mean
        self.mean += delta / self.count
        self._m2 += delta * (value - self.mean)

    def as_dict(self) -> dict[str, float | int | None]:
        variance = self._m2 / (self.count - 1) if self.count > 1 else 0.0
        return {
            "count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean if self.count else None,
            "std": math.sqrt(max(variance, 0.0)) if self.count else None,
        }


def _parse_number(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _parse_timestamp(value: str | None) -> float | None:
    """Parse a numeric or ISO timestamp to seconds when possible."""

    if value is None:
        return None
    text = str(value).strip()
    numeric = _parse_number(text)
    if numeric is not None:
        return numeric
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _candidate_timestamp_column(fieldnames: Iterable[str]) -> str | None:
    for name in fieldnames:
        normalized = name.casefold().replace("_", " ").strip()
        if normalized in {"timestamp", "time", "datetime", "date time", "start"}:
            return name
    for name in fieldnames:
        if "time" in name.casefold():
            return name
    return None


def _summarize_csv(path: Path) -> dict[str, Any]:
    """Return schema, numeric summaries, and timing clues for one CSV file."""

    with path.open("r", encoding="utf-8-sig", newline="") as file_handle:
        sample = file_handle.read(8_192)
        file_handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(file_handle, dialect=dialect)
        fieldnames = [name.strip() for name in (reader.fieldnames or []) if name]
        if not fieldnames:
            raise ValueError(f"No header row found in {path.name}")

        missing_counts = {name: 0 for name in fieldnames}
        numeric = {name: NumericSummary() for name in fieldnames}
        distinct_samples: dict[str, list[str]] = {name: [] for name in fieldnames}
        timestamp_column = _candidate_timestamp_column(fieldnames)
        timestamps: list[float] = []
        row_count = 0

        for row in reader:
            row_count += 1
            for name in fieldnames:
                value = (row.get(name) or "").strip()
                if not value:
                    missing_counts[name] += 1
                    continue
                numeric_value = _parse_number(value)
                if numeric_value is not None:
                    numeric[name].add(numeric_value)
                elif len(distinct_samples[name]) < 12 and value not in distinct_samples[name]:
                    distinct_samples[name].append(value)
            if timestamp_column:
                timestamp_value = _parse_timestamp(row.get(timestamp_column))
                if timestamp_value is not None:
                    timestamps.append(timestamp_value)

    positive_deltas = [
        current - previous
        for previous, current in zip(timestamps, timestamps[1:])
        if current > previous
    ]
    timing: dict[str, Any] = {
        "timestamp_column": timestamp_column,
        "parseable_timestamp_count": len(timestamps),
        "first_timestamp_seconds": timestamps[0] if timestamps else None,
        "last_timestamp_seconds": timestamps[-1] if timestamps else None,
        "duration_seconds": (
            timestamps[-1] - timestamps[0] if len(timestamps) > 1 else None
        ),
        "non_monotonic_timestamp_count": sum(
            current <= previous for previous, current in zip(timestamps, timestamps[1:])
        ),
    }
    if positive_deltas:
        median_delta = statistics.median(positive_deltas)
        timing.update(
            {
                "median_positive_delta_seconds": median_delta,
                "estimated_sampling_hz": 1.0 / median_delta if median_delta > 0 else None,
                "min_positive_delta_seconds": min(positive_deltas),
                "max_positive_delta_seconds": max(positive_deltas),
            }
        )

    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "row_count": row_count,
        "delimiter": dialect.delimiter,
        "columns": fieldnames,
        "missing_values": missing_counts,
        "numeric_columns": {
            name: summary.as_dict()
            for name, summary in numeric.items()
            if summary.count > 0
        },
        "categorical_value_samples": {
            name: values for name, values in distinct_samples.items() if values
        },
        "timing": timing,
    }


def _trip_report(raw_directory: Path, trip_number: int) -> dict[str, Any]:
    required = {
        "linear_acceleration": raw_directory / f"Linear_Acceleration_{trip_number}.csv",
        "gyroscope": raw_directory / f"Gyroscope_{trip_number}.csv",
        "labels": raw_directory / f"Labeled_events_{trip_number}.csv",
    }
    report: dict[str, Any] = {"trip": trip_number, "files": {}}
    for role, path in required.items():
        if not path.exists():
            report["files"][role] = {"missing": True, "expected_path": str(path)}
            continue
        report["files"][role] = _summarize_csv(path)

    acceleration_timing = report["files"].get("linear_acceleration", {}).get("timing", {})
    gyroscope_timing = report["files"].get("gyroscope", {}).get("timing", {})
    report["alignment"] = {
        "acceleration_duration_seconds": acceleration_timing.get("duration_seconds"),
        "gyroscope_duration_seconds": gyroscope_timing.get("duration_seconds"),
        "duration_difference_seconds": (
            abs(
                float(acceleration_timing["duration_seconds"])
                - float(gyroscope_timing["duration_seconds"])
            )
            if acceleration_timing.get("duration_seconds") is not None
            and gyroscope_timing.get("duration_seconds") is not None
            else None
        ),
        "acceleration_estimated_hz": acceleration_timing.get("estimated_sampling_hz"),
        "gyroscope_estimated_hz": gyroscope_timing.get("estimated_sampling_hz"),
    }
    return report


def audit_driving_events_dataset(
    raw_directory: Path = DEFAULT_RAW_DIRECTORY,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> dict[str, Any]:
    """Audit all expected public-dataset files and write a JSON report."""

    raw_directory = Path(raw_directory)
    output_directory = Path(output_directory)
    report = {
        "dataset": "Driving Events Dataset: a smartphone inertial measurement unit for driving events",
        "raw_directory": str(raw_directory),
        "local_recordings_excluded": True,
        "trips": [_trip_report(raw_directory, number) for number in (1, 2, 3)],
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = output_directory / "dataset_audit.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
