"""Run the local bilingual speech pipeline with explicit optional transcript storage."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.speech_analysis_pipeline import SpeechAnalysisPipeline
from src.video.localized_text import shape_for_rtl_display


def main() -> int:
    parser = argparse.ArgumentParser(description="CabInspector bilingual speech-analysis smoke test")
    parser.add_argument("--device", type=int, help="sounddevice input-device index")
    parser.add_argument("--seconds", type=int, default=30, help="listening duration")
    parser.add_argument(
        "--whisper-model",
        default="large-v3-turbo",
        help="multilingual Faster-Whisper model (large-v3-turbo is the accuracy default)",
    )
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
        "--language",
        choices=("auto", "bilingual", "en", "ar"),
        default="bilingual",
        help="speech language; bilingual/auto supports Arabic-English code switching",
    )
    parser.add_argument(
        "--use-safety-model",
        action="store_true",
        help="also load the optional neural toxicity model (may take longer on its first run)",
    )
    parser.add_argument(
        "--rules-only",
        action="store_true",
        help="compatibility alias; rule-based safety is already the default",
    )
    parser.add_argument(
        "--record-transcript",
        action="store_true",
        help="explicitly save recognized text to outputs/transcript_log.csv; raw audio is never saved",
    )
    parser.add_argument(
        "--show-unsafe-transcript",
        action="store_true",
        help=(
            "show safety-flagged recognized text for this test; safety alerts remain enabled "
            "and text is not saved unless --record-transcript is also supplied"
        ),
    )
    args = parser.parse_args()

    pipeline = SpeechAnalysisPipeline(
        device=args.device,
        whisper_model=args.whisper_model,
        transcription_device=args.whisper_device,
        transcription_compute_type=args.whisper_compute_type,
        transcription_language=None if args.language in {"auto", "bilingual"} else args.language,
        use_safety_model=args.use_safety_model and not args.rules_only,
        store_transcripts=args.record_transcript,
        mask_unsafe_transcript=not args.show_unsafe_transcript,
    )
    if not pipeline.start():
        print(pipeline.get_latest_state().error)
        return 1
    print(f"Transcription runtime: {pipeline.transcriber.runtime_summary}")

    if args.record_transcript:
        print("Transcript recording is ON: recognized text will be saved to outputs/transcript_log.csv.")
    else:
        print("Transcript recording is OFF: text is shown temporarily only; raw audio is never saved.")
    if args.show_unsafe_transcript:
        print(
            "Unsafe-transcript display is ON for this test: safety alerts still apply, "
            "and text is not saved unless --record-transcript is enabled."
        )
    print(
        "Speak one sentence, then pause for two seconds so it can be transcribed. "
        f"Language mode: {args.language}."
    )
    try:
        for _ in range(max(1, args.seconds)):
            state = pipeline.get_latest_state()
            categories = ", ".join(state.safety_categories) or "clear"
            print(
                f"speech={'yes' if state.speech_active else 'no'} | "
                f"voice={state.voice_level} | loudness={state.loudness_dbfs:.1f} dBFS | "
                f"transcription={state.transcription_status} | mode={args.language} | "
                f"dominant_language={state.language} | "
                f"recording={'on' if state.transcript_recording_enabled else 'off'} "
                f"saved={state.recorded_transcript_count} dropped={state.dropped_utterance_count} | "
                f"safety={categories} | "
                f"transcript={shape_for_rtl_display(state.transient_transcript) if state.transient_transcript else '[waiting]'}"
            )
            if state.analysis_error:
                print(f"analysis warning: {state.analysis_error}")
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        if pipeline.finish_current_utterance():
            print("Finishing the last spoken sentence...")
        if not pipeline.wait_for_pending_transcription(30):
            print("Transcription is still processing. Try the test again after the model download completes.")
        state = pipeline.get_latest_state()
        if state.transient_transcript:
            print(f"Final transcript: {shape_for_rtl_display(state.transient_transcript)}")
        pipeline.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
