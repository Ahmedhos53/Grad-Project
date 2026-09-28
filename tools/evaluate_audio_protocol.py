"""Run privacy-conscious audio contract checks on synthetic in-memory signals.

The default run never opens a microphone, saves a waveform, or writes
transcript text.  It verifies loudness, segmentation, YAMNet label mapping,
and local bilingual safety-rule behaviour.  Consented ASR/WER evaluation is a
separate manual protocol documented in ``docs/evaluation/evaluation_protocols.md``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.audio_event_detector import AudioStateSmoother, map_yamnet_categories
from src.audio.loudness_analyzer import LoudnessAnalyzer
from src.audio.transcription_diagnostics import (
    measure_audio_quality,
    segment_audio_like_pipeline,
)
from src.audio.text_safety_classifier import BilingualSafetyClassifier


SAMPLE_RATE = 16_000
BLOCK_SIZE = 1_600


def _tone(duration_seconds: float, rms_dbfs: float, frequency_hz: float = 220.0) -> np.ndarray:
    sample_count = int(round(duration_seconds * SAMPLE_RATE))
    time_axis = np.arange(sample_count, dtype=np.float32) / SAMPLE_RATE
    rms = 10.0 ** (rms_dbfs / 20.0)
    peak = rms * math.sqrt(2.0)
    return (peak * np.sin(2.0 * math.pi * frequency_hz * time_axis)).astype(np.float32)


def evaluate() -> dict[str, object]:
    silence = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
    speech_like = np.concatenate(
        (
            silence[: SAMPLE_RATE // 2],
            _tone(1.25, -30.0),
            silence[: SAMPLE_RATE],
        )
    )
    silence_quality = measure_audio_quality(silence, SAMPLE_RATE)
    speech_quality = measure_audio_quality(speech_like, SAMPLE_RATE)
    _, silence_segmentation = segment_audio_like_pipeline(
        silence,
        sample_rate=SAMPLE_RATE,
        block_size=BLOCK_SIZE,
    )
    _, speech_segmentation = segment_audio_like_pipeline(
        speech_like,
        sample_rate=SAMPLE_RATE,
        block_size=BLOCK_SIZE,
    )

    loudness = LoudnessAnalyzer()
    normal_reference = loudness.calibrate(_tone(0.25, -30.0))
    loudness_levels: dict[str, str] = {}
    for label, dbfs in (
        ("low", -40.0),
        ("normal", -30.0),
        ("high", -20.0),
        ("very_high", -10.0),
    ):
        state = loudness.update(_tone(0.1, dbfs), speech_active=True)
        loudness_levels[label] = state.voice_level

    mapped = map_yamnet_categories(
        (
            ("Speech", 0.91),
            ("Shout", 0.82),
            ("Vehicle horn, car horn, honking", 0.77),
            ("Unknown label", 0.99),
        )
    )
    smoother = AudioStateSmoother()
    smoothed = smoother.update(
        mapped,
        top_label="shout",
        top_confidence=0.82,
        timestamp_ms=1,
    )

    # Text values are intentionally not written to the report.  Only fixture
    # IDs and category results are retained.
    classifier = BilingualSafetyClassifier(use_model=False)
    safety_cases = (
        ("neutral_english", "please stop the car", False, ()),
        ("english_profanity", "You are a bitch", True, ("PROFANITY",)),
        ("arabic_threat", "سأقتلك", True, ("THREAT",)),
        ("arabic_harassment", "غبي", True, ("HARASSMENT",)),
    )
    safety_results = []
    for case_id, text, expected_flagged, expected_categories in safety_cases:
        result = classifier.analyze(text)
        safety_results.append(
            {
                "id": case_id,
                "expected_flagged": expected_flagged,
                "predicted_flagged": bool(result.flagged),
                "expected_categories": list(expected_categories),
                "predicted_categories": list(result.categories),
            }
        )

    return {
        "status": "synthetic_audio_contract_only",
        "microphone_used": False,
        "raw_audio_saved": False,
        "transcript_text_saved": False,
        "sample_rate": SAMPLE_RATE,
        "quality": {
            "silence": silence_quality.__dict__,
            "speech_like": speech_quality.__dict__,
        },
        "segmentation": {
            "silence": silence_segmentation.__dict__,
            "speech_like": speech_segmentation.__dict__,
        },
        "loudness": {
            "normal_reference_dbfs": round(normal_reference, 2),
            "level_labels": loudness_levels,
        },
        "yamnet_mapping": {
            "mapped_categories": dict(sorted(mapped.items())),
            "speech_active_after_one_window": smoothed.speech_active,
            "raised_voice_active_after_one_window": smoothed.raised_voice_active,
        },
        "safety_rule_cases": safety_results,
        "limitations": [
            "Synthetic signals verify contracts only and do not measure microphone acoustics, ASR WER/CER, or real speech recognition.",
            "Loudness is RMS/dBFS volume relative to calibration, not emotion or aggression.",
            "The report stores fixture IDs and categories, never the controlled text or waveform samples.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "evaluation" / "audio_protocol_evaluation.json",
    )
    parser.add_argument("--no-write", action="store_true", help="print the report without writing it")
    args = parser.parse_args()
    report = evaluate()
    if not args.no_write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Evaluation written: {args.output}")
    print(f"Synthetic sample rate: {report['sample_rate']} Hz")
    print("Microphone used: False")
    print("Raw audio saved: False")
    print("Transcript text saved: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

