"""Run the exported telemetry model against one public held/source trip."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.telemetry.inference import TelemetryRandomForestPredictor
from src.telemetry.preprocessing import DEFAULT_PROCESSED_DIRECTORY


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trip", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--processed-directory", type=Path, default=DEFAULT_PROCESSED_DIRECTORY)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--output-directory", type=Path, default=PROJECT_ROOT / "outputs" / "telemetry" / "replay")
    args = parser.parse_args()
    predictor = TelemetryRandomForestPredictor(args.model_path) if args.model_path else TelemetryRandomForestPredictor()
    with np.load(args.processed_directory / f"trip_{args.trip}_windows.npz", allow_pickle=False) as loaded:
        windows, targets, labels = loaded["inputs"], loaded["targets"], loaded["source_labels"]
        starts, ends = loaded["start_seconds"], loaded["end_seconds"]
    args.output_directory.mkdir(parents=True, exist_ok=True)
    output_path = args.output_directory / f"trip_{args.trip}_predictions.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=["start_seconds", "end_seconds", "source_label", "expected_class_index", "predicted_category", "confidence"])
        writer.writeheader()
        for window, target, source_label, start, end in zip(windows, targets, labels, starts, ends):
            prediction = predictor.predict_window(window)
            writer.writerow({
                "start_seconds": float(start), "end_seconds": float(end), "source_label": str(source_label),
                "expected_class_index": int(target), "predicted_category": prediction.category,
                "confidence": round(prediction.confidence, 4),
            })
    print(f"Replay written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
