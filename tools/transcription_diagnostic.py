"""Capture one consented sample and compare direct Whisper with CabInspector segmentation."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import sys
from pathlib import Path
import time
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.speech_transcriber import FasterWhisperTranscriber
from src.audio.transcription_diagnostics import (
    as_jsonable,
    measure_audio_quality,
    segment_audio_like_pipeline,
    word_error_rate,
)
from src.video.localized_text import shape_for_rtl_display


def save_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def transcript_data(result) -> dict[str, object]:
    return {
        "text": result.text,
        "language": result.language,
        "language_probability": result.language_probability,
        "average_log_probability": result.average_log_probability,
        "processing_latency_ms": result.processing_latency_ms,
        "error": result.error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="CabInspector transcription accuracy diagnostic")
    parser.add_argument("--device", type=int, help="microphone device index; omit for the system default")
    parser.add_argument("--seconds", type=int, default=20, help="maximum recording duration")
    parser.add_argument(
        "--language",
        choices=("auto", "bilingual", "ar", "en"),
        default="bilingual",
        help="bilingual/auto supports Arabic-English code switching",
    )
    parser.add_argument("--whisper-model", default="large-v3-turbo")
    parser.add_argument(
        "--whisper-device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="transcription hardware; auto prefers a usable NVIDIA GPU",
    )
    parser.add_argument(
        "--whisper-compute-type",
        default="auto",
        help="CTranslate2 compute type; auto uses float16 on CUDA and int8 on CPU",
    )
    parser.add_argument(
        "--expected",
        help="exact words you will say; enables word-error-rate measurement",
    )
    parser.add_argument(
        "--save-audio",
        action="store_true",
        help="explicitly save the diagnostic WAV under outputs/diagnostics",
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="explicitly save the diagnostic transcript/metrics JSON under outputs/diagnostics",
    )
    args = parser.parse_args()

    sample_rate = 16_000
    recording_seconds = max(1, args.seconds)
    print("A diagnostic recording will start now. Speak naturally after the countdown.")
    for count in (3, 2, 1):
        print(count)
        time.sleep(1)
    print(
        "Recording now. Start speaking immediately, then remain silent for three seconds.",
        flush=True,
    )
    try:
        import sounddevice as sd

        samples = sd.rec(
            int(recording_seconds * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=args.device,
        )
        sd.wait()
    except Exception as exc:
        print(f"Microphone capture failed: {exc}")
        return 1

    print("Recording finished. Detecting speech...", flush=True)
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    speech_regions = get_speech_timestamps(
        values,
        VadOptions(
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=500,
            speech_pad_ms=400,
        ),
    )
    recording_truncated = bool(
        speech_regions and speech_regions[-1]["end"] >= len(values) - int(0.25 * sample_rate)
    )
    language = None if args.language in {"auto", "bilingual"} else args.language
    transcriber = FasterWhisperTranscriber(
        args.whisper_model,
        device=args.whisper_device,
        compute_type=args.whisper_compute_type,
        language=language,
    )
    print(f"Running direct Whisper ({transcriber.runtime_summary})...", flush=True)
    direct_result = transcriber.transcribe(values, sample_rate=sample_rate)

    segments, segmentation = segment_audio_like_pipeline(values, sample_rate=sample_rate)
    print(f"Comparing CabInspector segmentation ({len(segments)} segment(s))...", flush=True)
    segmented_results = []
    for index, segment in enumerate(segments, start=1):
        print(f"Transcribing segment {index}/{len(segments)}...", flush=True)
        segmented_results.append(transcriber.transcribe(segment, sample_rate=sample_rate))
    segmented_text = " ".join(result.text for result in segmented_results if result.text).strip()

    report = {
        "recording": as_jsonable(measure_audio_quality(values, sample_rate)),
        "speech_regions_seconds": [
            {
                "start": round(region["start"] / sample_rate, 3),
                "end": round(region["end"] / sample_rate, 3),
            }
            for region in speech_regions
        ],
        "recording_truncated": recording_truncated,
        "requested_language": args.language,
        "model": args.whisper_model,
        "transcription_runtime": transcriber.runtime_summary,
        "expected_text": args.expected or "",
        "direct_whisper": transcript_data(direct_result),
        "pipeline_segmentation": as_jsonable(segmentation),
        "pipeline_style_transcript": {
            "text": segmented_text,
            "segment_results": [transcript_data(result) for result in segmented_results],
        },
    }
    if args.expected:
        report["direct_word_error_rate"] = word_error_rate(args.expected, direct_result.text)
        report["pipeline_word_error_rate"] = word_error_rate(args.expected, segmented_text)

    print("\nDiagnostic result")
    print(f"Input: {report['recording']}")
    print(f"Direct Whisper: {shape_for_rtl_display(direct_result.text) or '[no transcript]'}")
    print(f"Pipeline-style transcript: {shape_for_rtl_display(segmented_text) or '[no transcript]'}")
    print(f"Segments: {segmentation.segment_count} {segmentation.segment_durations_seconds}")
    if recording_truncated:
        print(
            "WARNING: speech continued to the end of the recording. The sentence was probably "
            "cut off, so this accuracy result is invalid. Record again with more time."
        )
    if args.expected:
        print(f"Direct WER: {report['direct_word_error_rate']}")
        print(f"Pipeline WER: {report['pipeline_word_error_rate']}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    diagnostics_dir = ROOT / "outputs" / "diagnostics"
    if args.save_audio:
        audio_path = diagnostics_dir / f"speech_diagnostic_{stamp}.wav"
        save_wav(audio_path, values, sample_rate)
        print(f"Consent-based diagnostic audio saved: {audio_path}")
    if args.save_report:
        report_path = diagnostics_dir / f"speech_diagnostic_{stamp}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Consent-based diagnostic report saved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
