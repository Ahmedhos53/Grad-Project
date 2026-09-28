"""Compare CabInspector telemetry models with leave-one-trip-out evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.telemetry.preprocessing import DEFAULT_PROCESSED_DIRECTORY
from src.telemetry.training import DEFAULT_EVALUATION_DIRECTORY, run_model_comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-directory", type=Path, default=DEFAULT_PROCESSED_DIRECTORY)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_EVALUATION_DIRECTORY)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    report = run_model_comparison(
        args.processed_directory,
        args.output_directory,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    print(json.dumps(report["summary"], indent=2))
    print(f"Full report: {args.output_directory / 'model_comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
