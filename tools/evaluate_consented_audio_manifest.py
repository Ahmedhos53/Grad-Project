"""Score a consented WAV manifest without persisting audio, references, or transcripts.

The command is deliberately opt-in. It reads locally supplied 16 kHz mono PCM16
WAV files and a reference manifest, processes audio in memory, and emits only
aggregate metrics. It never opens a microphone or writes transcript text.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
import uuid
import wave

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.loudness_analyzer import LoudnessAnalyzer
from src.audio.speech_transcriber import FasterWhisperTranscriber
from src.audio.text_safety_classifier import BilingualSafetyClassifier, DEFAULT_RULES_PATH
from src.audio.transcription_diagnostics import (
    character_error_counts,
    measure_audio_quality,
    segment_audio_like_pipeline,
    word_error_counts,
)
from src.evaluation.metrics import binary_metrics, count_values, multiclass_metrics


SAMPLE_RATE = 16_000
MAX_SAMPLES = 100
MAX_SAMPLE_SECONDS = 30
LANGUAGES = {"ar", "en", "mixed"}
MICROPHONE_TYPES = {"laptop", "headset"}
NOISE_CONDITIONS = {"quiet", "road_noise", "cabin_noise", "music", "other"}
LOUDNESS_LEVELS = ("QUIET", "LOW", "NORMAL", "HIGH", "VERY_HIGH")
REQUIRED_COLUMNS = {
    "sample_id",
    "audio_path",
    "language",
    "reference_text",
    "expected_safety_categories",
    "expected_loudness_level",
    "microphone_type",
    "noise_condition",
}


def _safe_sample_id(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value.strip()))


def _resolve_local_file(root: Path, relative_value: str) -> Path:
    candidate_value = Path(relative_value.strip())
    if candidate_value.is_absolute():
        raise ValueError("Audio paths must be relative to the manifest directory")
    candidate = (root / candidate_value).resolve(strict=True)
    if candidate != root and root not in candidate.parents:
        raise ValueError("An audio path resolves outside the manifest directory")
    if not candidate.is_file() or candidate.suffix.casefold() != ".wav":
        raise ValueError("Every audio input must be a local WAV file")
    return candidate


def read_pcm16_wav(path: Path) -> np.ndarray:
    """Read a bounded mono PCM16 WAV into memory without modifying the file."""
    try:
        with wave.open(str(path), "rb") as source:
            if (
                source.getnchannels() != 1
                or source.getsampwidth() != 2
                or source.getframerate() != SAMPLE_RATE
                or source.getcomptype() != "NONE"
            ):
                raise ValueError("WAVs must be uncompressed, mono, 16 kHz, 16-bit PCM")
            frame_count = source.getnframes()
            if frame_count <= 0 or frame_count > SAMPLE_RATE * MAX_SAMPLE_SECONDS:
                raise ValueError("WAV duration must be between 0 and 30 seconds")
            samples = np.frombuffer(source.readframes(frame_count), dtype="<i2")
    except (wave.Error, OSError) as exc:
        raise ValueError("A WAV input could not be read") from exc
    if samples.size != frame_count:
        raise ValueError("A WAV input ended before its declared sample count")
    return samples.astype(np.float32) / 32768.0


def load_manifest(path: Path) -> tuple[Path, list[dict[str, object]]]:
    manifest_path = path.resolve(strict=True)
    if not manifest_path.is_file() or manifest_path.suffix.casefold() != ".csv":
        raise ValueError("The manifest must be a local CSV file")
    root = manifest_path.parent.resolve(strict=True)
    records: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = set(reader.fieldnames or ())
            missing = REQUIRED_COLUMNS - columns
            if missing:
                raise ValueError("The manifest is missing required columns")
            for row_number, row in enumerate(reader, start=2):
                if row_number > MAX_SAMPLES + 1:
                    raise ValueError("The manifest exceeds the 100-sample limit")
                sample_id = (row.get("sample_id") or "").strip()
                language = (row.get("language") or "").strip().casefold()
                microphone_type = (row.get("microphone_type") or "").strip().casefold()
                noise_condition = (row.get("noise_condition") or "").strip().casefold()
                reference_text = (row.get("reference_text") or "").strip()
                if not _safe_sample_id(sample_id) or sample_id in seen_ids:
                    raise ValueError(f"Invalid or duplicate sample ID on row {row_number}")
                if (
                    language not in LANGUAGES
                    or microphone_type not in MICROPHONE_TYPES
                    or noise_condition not in NOISE_CONDITIONS
                ):
                    raise ValueError(f"Unsupported language, microphone, or noise condition on row {row_number}")
                if not reference_text:
                    raise ValueError(f"Reference text is required on row {row_number}")
                expected_levels = (row.get("expected_loudness_level") or "").strip().upper()
                expected_levels = expected_levels.replace(" ", "_")
                if expected_levels and expected_levels not in LOUDNESS_LEVELS:
                    raise ValueError(f"Unsupported loudness label on row {row_number}")
                categories = {
                    item.strip().upper()
                    for item in re.split(r"[|;]", row.get("expected_safety_categories") or "")
                    if item.strip()
                }
                audio_path = _resolve_local_file(root, row.get("audio_path") or "")
                records.append(
                    {
                        "sample_id": sample_id,
                        "audio_path": audio_path,
                        "language": language,
                        "reference_text": reference_text,
                        "expected_safety_categories": categories,
                        "expected_loudness_level": expected_levels or None,
                        "microphone_type": microphone_type,
                        "noise_condition": noise_condition,
                    }
                )
                seen_ids.add(sample_id)
    except OSError as exc:
        raise ValueError("The manifest could not be read") from exc
    if not records:
        raise ValueError("The manifest must contain at least one sample")
    return root, records


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _summarize_recognition(rows: list[dict[str, object]]) -> dict[str, object]:
    word_errors = sum(int(row["word_errors"]) for row in rows)
    word_count = sum(int(row["reference_word_count"]) for row in rows)
    character_errors = sum(int(row["character_errors"]) for row in rows)
    character_count = sum(int(row["reference_character_count"]) for row in rows)
    whisper_latencies = [int(row["whisper_latency_ms"]) for row in rows]
    pipeline_latencies = [int(row["pipeline_latency_ms"]) for row in rows]
    language_rows = [row for row in rows if row["detected_language"] in {"ar", "en"}]
    language_matches = [
        row["language"] == row["detected_language"]
        for row in language_rows
        if row["language"] in {"ar", "en"}
    ]
    return {
        "utterance_count": len(rows),
        "reference_word_count": word_count,
        "word_errors": word_errors,
        "wer": _ratio(word_errors, word_count),
        "reference_character_count": character_count,
        "character_errors": character_errors,
        "cer": _ratio(character_errors, character_count),
        "mean_whisper_latency_ms": round(statistics.mean(whisper_latencies), 2)
        if whisper_latencies
        else 0.0,
        "p95_whisper_latency_ms": round(float(np.percentile(whisper_latencies, 95)), 2)
        if whisper_latencies
        else 0.0,
        "mean_pipeline_latency_ms": round(statistics.mean(pipeline_latencies), 2)
        if pipeline_latencies
        else 0.0,
        "mean_detected_language_probability": round(
            statistics.mean(float(row["language_probability"]) for row in language_rows), 4
        )
        if language_rows
        else None,
        "detected_language_agreement": _ratio(sum(language_matches), len(language_matches)),
        "language_agreement_sample_count": len(language_matches),
        "no_segment_count": sum(bool(row["no_segment"]) for row in rows),
        "clipped_recording_count": sum(bool(row["clipped"]) for row in rows),
    }


def evaluate_records(
    records: list[dict[str, object]],
    *,
    transcriber,
    safety_classifier,
    calibration_samples: np.ndarray | None = None,
) -> dict[str, object]:
    """Process prepared records and retain only utterance-level score data."""
    expected_loudness = [row for row in records if row.get("expected_loudness_level")]
    if expected_loudness and calibration_samples is None:
        raise ValueError("A consented normal-speech calibration WAV is required for loudness scoring")

    loudness = LoudnessAnalyzer()
    calibration_dbfs = loudness.calibrate(calibration_samples) if calibration_samples is not None else None
    scored_rows: list[dict[str, object]] = []
    all_categories = {str(category).upper() for category in safety_classifier.rules}
    all_categories.add("TOXIC_LANGUAGE")
    all_categories.update(
        category
        for row in records
        for category in row["expected_safety_categories"]
    )
    valid_categories = {str(category).upper() for category in safety_classifier.rules}
    valid_categories.add("TOXIC_LANGUAGE")
    if any(not set(row["expected_safety_categories"]).issubset(valid_categories) for row in records):
        raise ValueError("The manifest contains an unknown safety category")
    if not all_categories:
        all_categories = {"PROFANITY", "HARASSMENT", "THREAT"}

    for row in records:
        samples = row.get("samples")
        if samples is None:
            samples = read_pcm16_wav(Path(row["audio_path"]))
        started = time.perf_counter()
        segments, segmentation = segment_audio_like_pipeline(samples, sample_rate=SAMPLE_RATE)
        hypothesis_parts: list[str] = []
        whisper_latency_ms = 0
        detected_language = "unknown"
        language_probability = 0.0
        for segment in segments:
            result = transcriber.transcribe(segment, sample_rate=SAMPLE_RATE)
            if result.error:
                raise ValueError("Transcription failed for at least one consented sample")
            if result.text:
                hypothesis_parts.append(result.text)
            whisper_latency_ms += int(result.processing_latency_ms)
            if result.language in {"ar", "en"}:
                detected_language = result.language
                language_probability = float(result.language_probability)
        hypothesis = " ".join(hypothesis_parts).strip()
        safety_result = safety_classifier.analyze(hypothesis, language=str(row["language"]))
        detected_categories = {str(category).upper() for category in safety_result.categories}
        reference = str(row["reference_text"])
        word_errors, reference_words = word_error_counts(reference, hypothesis)
        character_errors, reference_characters = character_error_counts(reference, hypothesis)
        audio_quality = measure_audio_quality(samples, SAMPLE_RATE)
        expected_level = row.get("expected_loudness_level")
        predicted_level = None
        if expected_level:
            predicted_level = loudness.update(
                samples,
                speech_active=segmentation.segment_count > 0,
            ).voice_level.replace(" ", "_")

        scored_rows.append(
            {
                "language": str(row["language"]),
                "microphone_type": str(row["microphone_type"]),
                "noise_condition": str(row["noise_condition"]),
                "word_errors": word_errors,
                "reference_word_count": reference_words,
                "character_errors": character_errors,
                "reference_character_count": reference_characters,
                "expected_safety_categories": set(row["expected_safety_categories"]),
                "predicted_safety_categories": detected_categories,
                "expected_loudness_level": expected_level,
                "predicted_loudness_level": predicted_level,
                "whisper_latency_ms": whisper_latency_ms,
                "pipeline_latency_ms": int((time.perf_counter() - started) * 1000),
                "detected_language": detected_language,
                "language_probability": language_probability,
                "no_segment": segmentation.segment_count == 0,
                "clipped": audio_quality.clipping_detected,
            }
        )
        # Drop references to the audio and decoded text before the next sample.
        del samples, segments, hypothesis, hypothesis_parts

    by_language = {
        language: _summarize_recognition([row for row in scored_rows if row["language"] == language])
        for language in sorted({str(row["language"]) for row in scored_rows})
    }
    by_microphone = {
        microphone: _summarize_recognition(
            [row for row in scored_rows if row["microphone_type"] == microphone]
        )
        for microphone in sorted({str(row["microphone_type"]) for row in scored_rows})
    }
    by_noise_condition = {
        condition: _summarize_recognition(
            [row for row in scored_rows if row["noise_condition"] == condition]
        )
        for condition in sorted({str(row["noise_condition"]) for row in scored_rows})
    }

    safety_metrics = {
        category: binary_metrics(
            [category in row["expected_safety_categories"] for row in scored_rows],
            [category in row["predicted_safety_categories"] for row in scored_rows],
        )
        for category in sorted(all_categories)
    }
    loudness_rows = [row for row in scored_rows if row["expected_loudness_level"]]
    loudness_report: dict[str, object] = {
        "sample_count": len(loudness_rows),
        "exact_agreement": _ratio(
            sum(row["expected_loudness_level"] == row["predicted_loudness_level"] for row in loudness_rows),
            len(loudness_rows),
        ),
        "classes": list(LOUDNESS_LEVELS),
        "confusion_matrix": multiclass_metrics(
            [str(row["expected_loudness_level"]) for row in loudness_rows],
            [str(row["predicted_loudness_level"]) for row in loudness_rows],
            LOUDNESS_LEVELS,
        )["confusion_matrix"],
        "normal_reference_dbfs": round(float(calibration_dbfs), 2) if calibration_dbfs is not None else None,
    }

    all_scores = _summarize_recognition(scored_rows)
    language_counts = count_values(str(row["language"]) for row in scored_rows)
    microphone_counts = count_values(str(row["microphone_type"]) for row in scored_rows)
    noise_counts = count_values(str(row["noise_condition"]) for row in scored_rows)
    coverage_warnings = []
    missing_languages = sorted(LANGUAGES - set(language_counts))
    missing_microphones = sorted(MICROPHONE_TYPES - set(microphone_counts))
    if missing_languages:
        coverage_warnings.append("Missing language conditions: " + ", ".join(missing_languages))
    if missing_microphones:
        coverage_warnings.append("Missing microphone types: " + ", ".join(missing_microphones))
    if "quiet" not in noise_counts:
        coverage_warnings.append("No quiet-condition samples")
    if not any(condition != "quiet" for condition in noise_counts):
        coverage_warnings.append("No background-noise samples")
    missing_positive_safety = [
        category
        for category, metrics in safety_metrics.items()
        if int(metrics["positive_support"]) == 0
    ]
    if missing_positive_safety:
        coverage_warnings.append(
            "No positive examples for configured safety categories: "
            + ", ".join(missing_positive_safety)
        )
    if not loudness_rows:
        coverage_warnings.append("No reference-labelled loudness samples")
    return {
        "status": "consented_audio_manifest_evaluation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(scored_rows),
        "language_counts": language_counts,
        "microphone_type_counts": microphone_counts,
        "noise_condition_counts": noise_counts,
        "coverage": {
            "missing_language_conditions": missing_languages,
            "missing_microphone_types": missing_microphones,
            "quiet_condition_present": "quiet" in noise_counts,
            "background_noise_present": any(condition != "quiet" for condition in noise_counts),
            "categories_without_positive_examples": missing_positive_safety,
            "loudness_reference_samples": len(loudness_rows),
            "warnings": coverage_warnings,
        },
        "recognition_overall": all_scores,
        "recognition_by_language": by_language,
        "recognition_by_microphone_type": by_microphone,
        "recognition_by_noise_condition": by_noise_condition,
        "safety_category_metrics": safety_metrics,
        "loudness_agreement": loudness_report,
        "privacy": {
            "explicit_consent_confirmed": True,
            "microphone_used": False,
            "input_audio_modified": False,
            "raw_audio_saved": False,
            "reference_text_saved": False,
            "transcript_text_saved": False,
            "sample_ids_and_paths_saved": False,
        },
        "interpretation": [
            "WER/CER are corpus edit errors divided by normalized reference token counts.",
            "Safety measures score the local rule classifier on the transcript; they include ASR errors and are not a separate text-only benchmark.",
            "Loudness is calibrated RMS/dBFS volume, not emotion, intent, anger, or aggression.",
            "Results apply only to the consented supplied clips, language labels, microphone types, and local runtime.",
        ],
    }


def evaluate_manifest(
    manifest_path: Path,
    *,
    transcriber,
    safety_classifier,
    calibration_wav: Path | None = None,
) -> dict[str, object]:
    root, rows = load_manifest(manifest_path)
    calibration_samples = None
    if calibration_wav is not None:
        calibration_path = _resolve_local_file(root, calibration_wav.as_posix())
        calibration_samples = read_pcm16_wav(calibration_path)
    report = evaluate_records(
        rows,
        transcriber=transcriber,
        safety_classifier=safety_classifier,
        calibration_samples=calibration_samples,
    )
    rows.clear()
    if calibration_samples is not None:
        del calibration_samples
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="consented CSV manifest of local WAV files")
    parser.add_argument("--calibration-wav", type=Path, help="consented normal-speech calibration WAV in the manifest directory")
    parser.add_argument("--whisper-model", default="large-v3-turbo")
    parser.add_argument("--whisper-device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--whisper-compute-type", default="auto")
    parser.add_argument("--safety-rules", type=Path, default=DEFAULT_RULES_PATH)
    parser.add_argument("--consent-confirmed", action="store_true", help="confirm every recording has appropriate consent")
    parser.add_argument("--output", type=Path, help="optional path for aggregate-only JSON; no output file is written by default")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing an existing aggregate report")
    args = parser.parse_args()

    if not args.consent_confirmed:
        parser.error("Refusing to read speech files without --consent-confirmed")
    if not sys.stdin.isatty() or input("Type CONSENTED to confirm permission to process these recordings: ").strip() != "CONSENTED":
        print("Consent was not confirmed; no manifest or audio file was read.")
        return 2
    if args.output and args.output.exists() and not args.overwrite:
        parser.error("Output exists; choose a new path or add --overwrite")

    try:
        transcriber = FasterWhisperTranscriber(
            args.whisper_model,
            device=args.whisper_device,
            compute_type=args.whisper_compute_type,
            language=None,
        )
        safety_classifier = BilingualSafetyClassifier(
            use_model=False,
            rules_path=args.safety_rules,
        )
        report = evaluate_manifest(
            args.manifest,
            transcriber=transcriber,
            safety_classifier=safety_classifier,
            calibration_wav=args.calibration_wav,
        )
    except (OSError, ValueError) as exc:
        print(f"Evaluation stopped safely: {exc}")
        return 1

    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(encoded, encoding="utf-8")
            os.replace(temporary, args.output)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Aggregate-only evaluation written: {args.output}")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
