"""Run an isolated structural smoke test for the pretrained PRIMUS IMU encoder.

This tool never changes CabInspector's dashboard, replay, labels, risk score, or Random Forest.
An output embedding is not a driving-event prediction.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.telemetry.primus_encoder import (
    DEFAULT_PRIMUS_CHECKPOINT,
    PRIMUS_SAMPLES_PER_WINDOW,
    PrimusIMUEncoder,
    resample_imu_window,
)


PROCESSED_ROOT = PROJECT_ROOT / "data" / "Telemetry" / "public" / "driving_events_dataset" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_PRIMUS_CHECKPOINT)
    parser.add_argument("--trip", choices=(1, 2, 3), type=int, default=1)
    parser.add_argument("--event-index", type=int, default=0)
    parser.add_argument("--synthetic-only", action="store_true")
    return parser.parse_args()


def load_public_window(trip: int, event_index: int) -> tuple[np.ndarray, float]:
    path = PROCESSED_ROOT / f"trip_{trip}_windows.npz"
    with np.load(path, allow_pickle=False) as archive:
        inputs = archive["inputs"]
        durations = archive["durations_seconds"]
        if not 0 <= event_index < len(inputs):
            raise IndexError(f"event-index must be between 0 and {len(inputs) - 1} for trip {trip}")
        return inputs[event_index], float(durations[event_index])


def report_embedding(name: str, encoder: PrimusIMUEncoder, window: np.ndarray) -> None:
    started = perf_counter()
    embedding = encoder.encode(window[np.newaxis, ...])
    elapsed_ms = (perf_counter() - started) * 1000.0
    print(f"{name}: input={window.shape}, embedding={embedding.shape}, inference_ms={elapsed_ms:.3f}")
    print(f"{name}: embedding_finite={bool(np.isfinite(embedding).all())}")


def main() -> int:
    arguments = parse_args()
    encoder = PrimusIMUEncoder()
    encoder.load_checkpoint(arguments.checkpoint)
    print(f"Checkpoint loaded on CPU in {encoder.load_seconds:.3f} seconds")

    synthetic = np.zeros((6, PRIMUS_SAMPLES_PER_WINDOW), dtype=np.float32)
    report_embedding("Synthetic structural window", encoder, synthetic)

    if not arguments.synthetic_only:
        public_window, duration = load_public_window(arguments.trip, arguments.event_index)
        resized = resample_imu_window(public_window)
        print(
            "Public structural window: "
            f"trip={arguments.trip}, event_index={arguments.event_index}, "
            f"source_duration_seconds={duration:.3f}, resized={resized.shape}"
        )
        print("Note: this normalized-time resize is structural only; it is not a driving-event evaluation.")
        report_embedding("Public structural window", encoder, resized)

    print("Result: PRIMUS produced embeddings only; no driving category was predicted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
