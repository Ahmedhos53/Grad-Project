"""Run YAMNet from the selected microphone without storing raw audio.

Usage from the project root:
    .venv\\Scripts\\python.exe tools\\audio_model_smoke_test.py
    .venv\\Scripts\\python.exe tools\\audio_model_smoke_test.py --list-devices
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.audio_event_detector import DEFAULT_MODEL_PATH, YamNetAudioDetector, load_yamnet_labels


def main() -> int:
    parser = argparse.ArgumentParser(description="CabInspector YAMNet microphone smoke test")
    parser.add_argument("--list-devices", action="store_true", help="print input devices and exit")
    parser.add_argument("--validate-model", action="store_true", help="validate model metadata and exit")
    parser.add_argument("--device", type=int, help="sounddevice input-device index")
    parser.add_argument("--seconds", type=int, default=15, help="listening duration")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="YAMNet TFLite path")
    args = parser.parse_args()

    if args.validate_model:
        try:
            from ai_edge_litert.interpreter import Interpreter

            labels = load_yamnet_labels(args.model)
            interpreter = Interpreter(model_path=str(args.model), num_threads=1)
            interpreter.allocate_tensors()
            input_shape = interpreter.get_input_details()[0]["shape"].tolist()
            output_shape = interpreter.get_output_details()[0]["shape"].tolist()
            print(f"Model: {args.model}")
            print(f"Labels: {len(labels)}")
            print(f"Input shape: {input_shape}")
            print(f"Output shape: {output_shape}")
            return 0
        except Exception as exc:
            print(f"Model validation failed: {exc}")
            return 1

    devices = YamNetAudioDetector.list_input_devices()
    if args.list_devices:
        if not devices:
            print("No input devices found.")
            return 1
        for device in devices:
            print(f"[{device['index']}] {device['name']}")
        return 0

    detector = YamNetAudioDetector(args.model, device=args.device)
    if not detector.start():
        print(detector.get_latest_state().error)
        return 1

    print("Listening. Raw audio is not recorded. Press Ctrl+C to stop.")
    try:
        for _ in range(max(1, args.seconds)):
            state = detector.get_latest_state()
            print(
                f"{state.top_label:32} {state.top_confidence:.2f} "
                f"speech={state.speech_active} raised_voice={state.raised_voice_active} "
                f"horn={state.horn_active} siren={state.siren_active}"
            )
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        detector.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
