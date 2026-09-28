"""Thread-safe aggregate runtime metrics with no media or transcript retention."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import statistics
from threading import RLock
import time
from typing import Iterator


class RuntimeMetrics:
    """Collect bounded timing/counter summaries for an explicitly instrumented run."""

    _MAX_SAMPLES_PER_STAGE = 10_000

    def __init__(self) -> None:
        self._lock = RLock()
        self._stage_samples: dict[str, list[float]] = {}
        self._counters: dict[str, int] = {}
        self._gauges: dict[str, float | int | bool | str] = {}

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.observe(stage, (time.perf_counter() - started) * 1000.0)

    def observe(self, stage: str, milliseconds: float) -> None:
        value = max(0.0, float(milliseconds))
        with self._lock:
            samples = self._stage_samples.setdefault(str(stage), [])
            if len(samples) < self._MAX_SAMPLES_PER_STAGE:
                samples.append(value)

    def increment(self, counter: str, amount: int = 1) -> None:
        with self._lock:
            name = str(counter)
            self._counters[name] = self._counters.get(name, 0) + int(amount)

    def set_gauge(self, name: str, value: float | int | bool | str) -> None:
        with self._lock:
            self._gauges[str(name)] = value

    @staticmethod
    def _stage_summary(samples: list[float]) -> dict[str, float | int]:
        if not samples:
            return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
        ordered = sorted(samples)
        return {
            "count": len(samples),
            "mean_ms": round(statistics.fmean(samples), 4),
            "p50_ms": round(statistics.median(ordered), 4),
            "p95_ms": round(ordered[max(0, int(round(0.95 * len(ordered))) - 1)], 4),
            "max_ms": round(max(ordered), 4),
        }

    def summary(self) -> dict[str, object]:
        with self._lock:
            stages = {
                name: self._stage_summary(list(samples))
                for name, samples in sorted(self._stage_samples.items())
            }
            return {
                "status": "aggregate_runtime_metrics",
                "media_saved": False,
                "transcript_text_saved": False,
                "stages": stages,
                "counters": dict(sorted(self._counters.items())),
                "gauges": dict(sorted(self._gauges.items())),
                "limitations": [
                    "Metrics are collected only when an explicit output path is configured.",
                    "Values are aggregate timings/counters and contain no camera, audio, transcript, or sensor arrays.",
                ],
            }

    def write(self, path: str | Path, *, context: dict[str, object] | None = None) -> dict[str, object]:
        report = self.summary()
        if context:
            report["context"] = dict(context)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
