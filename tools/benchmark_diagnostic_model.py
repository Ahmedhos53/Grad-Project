"""Benchmark a specified Faster-Whisper model against consented diagnostic recordings."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import sys
import time
from pathlib import Path
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.speech_transcriber import FasterWhisperTranscriber
from src.audio.transcription_diagnostics import word_error_rate
from src.video.localized_text import shape_for_rtl_display


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        sample_rate = source.getframerate()
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype=np.int16)
    return samples.astype(np.float32) / 32767.0, sample_rate


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark a Whisper model using saved diagnostics")
    parser.add_argument("--model", default="large-v3-turbo", help="Faster-Whisper model name")
    parser.add_argument("--limit", type=int, default=2, help="number of recent recordings to benchmark")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="transcription hardware; auto prefers a usable NVIDIA GPU",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        help="CTranslate2 compute type; auto uses float16 on CUDA and int8 on CPU",
    )
    args = parser.parse_args()

    diagnostics_dir = ROOT / "outputs" / "diagnostics"
    reports = sorted(diagnostics_dir.glob("speech_diagnostic_*.json"), key=lambda item: item.stat().st_mtime)
    selected = reports[-max(1, args.limit) :]
    if not selected:
        print("No diagnostic reports found.")
        return 1

    transcriber = FasterWhisperTranscriber(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
    )
    print(
        f"Loading {transcriber.runtime_summary}; the first run may download model files.",
        flush=True,
    )
    results = []
    for report_path in selected:
        report = json.loads(report_path.read_text(encoding="utf-8-sig"))
        wav_path = report_path.with_suffix(".wav")
        if not wav_path.exists():
            print(f"Skipping {report_path.name}: matching WAV is missing.")
            continue
        samples, sample_rate = load_wav(wav_path)
        language = report.get("requested_language", "auto")
        transcriber.language = None if language in {"auto", "bilingual"} else language
        print(f"Transcribing {wav_path.name} ({language})...", flush=True)
        start = time.monotonic()
        result = transcriber.transcribe(samples, sample_rate=sample_rate)
        expected = report.get("expected_text", "")
        entry = {
            "recording": wav_path.name,
            "requested_language": language,
            "expected_text": expected,
            "text": result.text,
            "word_error_rate": word_error_rate(expected, result.text) if expected else None,
            "language_probability": result.language_probability,
            "average_log_probability": result.average_log_probability,
            "processing_latency_ms": result.processing_latency_ms,
            "elapsed_seconds": round(time.monotonic() - start, 3),
            "transcription_runtime": transcriber.runtime_summary,
            "error": result.error,
        }
        results.append(entry)
        print(f"WER: {entry['word_error_rate']} | {shape_for_rtl_display(result.text)}", flush=True)

    output_path = diagnostics_dir / f"model_benchmark_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "transcription_runtime": transcriber.runtime_summary,
                "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Benchmark complete: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
