"""Run one real event prediction through the exported pretrained telemetry model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.telemetry.dataset_loader import load_public_trip
from src.telemetry.primus_inference import PrimusTelemetryPredictor
from src.telemetry.primus_transfer import make_fixed_primus_event_window


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trip", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--event-index", type=int, default=0)
    args = parser.parse_args()
    trip = load_public_trip(args.trip)
    if not 0 <= args.event_index < len(trip.events):
        raise SystemExit(f"event-index must be between 0 and {len(trip.events) - 1}")
    event = trip.events[args.event_index]
    window, start, end = make_fixed_primus_event_window(event, trip)
    predictor = PrimusTelemetryPredictor()
    prediction = predictor.predict(window)
    print(f"Trip {args.trip}, event {args.event_index}, window {start:.3f}-{end:.3f}s")
    print(f"Prediction: {prediction.category} ({prediction.confidence:.3f})")
    print(f"Source label for this diagnostic only: {event.category}")
    print(f"Inference: {prediction.inference_ms:.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
