"""Download and load a Whisper model before starting live CabInspector audio."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.speech_transcriber import FasterWhisperTranscriber


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a local Faster-Whisper model before microphone capture"
    )
    parser.add_argument("--model", default="large-v3-turbo", help="Faster-Whisper model name")
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
    parser.add_argument(
        "--use-huggingface-xet",
        action="store_true",
        help="use Hugging Face Xet downloads instead of the standard CDN path",
    )
    args = parser.parse_args()

    # The standard CDN path is more reliable on this Windows setup. This must be
    # set before Faster-Whisper imports Hugging Face's download code.
    if args.use_huggingface_xet:
        os.environ["CABINSPECTOR_HUGGINGFACE_USE_XET"] = "1"
        os.environ.pop("HF_HUB_DISABLE_XET", None)
    else:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    transcriber = FasterWhisperTranscriber(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
    )
    print(f"Preparing Whisper {transcriber.runtime_summary}...", flush=True)
    start = time.monotonic()
    if not transcriber.preload():
        print(transcriber.error)
        return 1
    print(
        f"Ready: {transcriber.runtime_summary} "
        f"({time.monotonic() - start:.1f} seconds)."
    )
    if transcriber.runtime_warning:
        print(f"Warning: {transcriber.runtime_warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
