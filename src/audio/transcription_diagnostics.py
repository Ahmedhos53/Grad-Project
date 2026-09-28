"""Reusable, privacy-aware helpers for measuring transcription quality."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re

import numpy as np

from src.audio.loudness_analyzer import LoudnessAnalyzer
from src.audio.speech_analysis_pipeline import EnergyUtteranceGate
from src.audio.text_normalization import normalize_bilingual


@dataclass(frozen=True)
class AudioQuality:
    duration_seconds: float
    rms_dbfs: float
    peak_dbfs: float
    clipping_detected: bool


@dataclass(frozen=True)
class SegmentationDiagnostic:
    segment_count: int
    segment_durations_seconds: tuple[float, ...]
    dropped_short_segments: int


def measure_audio_quality(samples: np.ndarray, sample_rate: int) -> AudioQuality:
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    _, rms_dbfs = LoudnessAnalyzer.rms_dbfs(values)
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    peak_dbfs = 20.0 * math.log10(max(peak, 1e-8))
    return AudioQuality(
        duration_seconds=round(values.size / sample_rate, 3),
        rms_dbfs=round(rms_dbfs, 2),
        peak_dbfs=round(peak_dbfs, 2),
        clipping_detected=bool(np.any(np.abs(values) >= 0.99)),
    )


def segment_audio_like_pipeline(
    samples: np.ndarray,
    *,
    sample_rate: int = 16_000,
    block_size: int = 1_600,
    minimum_segment_samples: int = 8_000,
) -> tuple[list[np.ndarray], SegmentationDiagnostic]:
    """Apply the current energy-gate behaviour without touching microphone or storage."""
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    gate = EnergyUtteranceGate()
    segments: list[np.ndarray] = []
    dropped_short_segments = 0

    for start in range(0, len(values), block_size):
        block = values[start : start + block_size]
        _, dbfs = LoudnessAnalyzer.rms_dbfs(block)
        _, completed = gate.update(block, dbfs)
        if completed is None:
            continue
        if len(completed) >= minimum_segment_samples:
            segments.append(completed)
        else:
            dropped_short_segments += 1

    final_segment = gate.flush()
    if final_segment is not None:
        if len(final_segment) >= minimum_segment_samples:
            segments.append(final_segment)
        else:
            dropped_short_segments += 1

    diagnostic = SegmentationDiagnostic(
        segment_count=len(segments),
        segment_durations_seconds=tuple(round(len(segment) / sample_rate, 3) for segment in segments),
        dropped_short_segments=dropped_short_segments,
    )
    return segments, diagnostic


def normalized_words(value: str) -> list[str]:
    normalized = normalize_bilingual(value)
    return re.findall(r"[\w\u0600-\u06ff]+", normalized, flags=re.UNICODE)


def normalized_characters(value: str) -> list[str]:
    """Return normalized Arabic/English letters and digits for CER scoring."""
    normalized = normalize_bilingual(value)
    return [character for character in normalized if character.isalnum()]


def _edit_distance(expected: list[str], actual: list[str]) -> int:
    if not expected:
        return len(actual)
    if not actual:
        return len(expected)

    previous = list(range(len(actual) + 1))
    for index, expected_item in enumerate(expected, start=1):
        current = [index]
        for actual_index, actual_item in enumerate(actual, start=1):
            replace_cost = 0 if expected_item == actual_item else 1
            current.append(
                min(
                    previous[actual_index] + 1,
                    current[actual_index - 1] + 1,
                    previous[actual_index - 1] + replace_cost,
                )
            )
        previous = current
    return previous[-1]


def word_error_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    """Return normalized word edit errors and reference-word count."""
    expected = normalized_words(reference)
    actual = normalized_words(hypothesis)
    return _edit_distance(expected, actual), len(expected)


def character_error_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    """Return normalized character edit errors and reference-character count."""
    expected = normalized_characters(reference)
    actual = normalized_characters(hypothesis)
    return _edit_distance(expected, actual), len(expected)


def word_error_rate(reference: str, hypothesis: str) -> float | None:
    """Compute normalized mixed Arabic/English WER for one supplied reference."""
    errors, reference_count = word_error_counts(reference, hypothesis)
    if not reference_count:
        return None
    return round(errors / reference_count, 4)


def character_error_rate(reference: str, hypothesis: str) -> float | None:
    """Compute normalized mixed Arabic/English CER, excluding whitespace/punctuation."""
    errors, reference_count = character_error_counts(reference, hypothesis)
    if not reference_count:
        return None
    return round(errors / reference_count, 4)


def as_jsonable(value):
    """Convert diagnostics dataclasses to simple report data."""
    return asdict(value)
