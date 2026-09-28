"""Evaluate frozen pretrained PRIMUS embeddings on CabInspector driving events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.telemetry.dataset_audit import DEFAULT_RAW_DIRECTORY
from src.telemetry.primus_encoder import DEFAULT_PRIMUS_CHECKPOINT
from src.telemetry.primus_transfer import DEFAULT_PRIMUS_EVALUATION_PATH, run_primus_transfer_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", type=Path, default=DEFAULT_RAW_DIRECTORY)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_PRIMUS_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_PRIMUS_EVALUATION_PATH)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    report = run_primus_transfer_evaluation(
        raw_directory=args.raw_directory,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        batch_size=args.batch_size,
    )
    print(json.dumps(report["summary"], indent=2))
    print(f"Full evaluation: {args.output}")
    print("The frozen PRIMUS head is connected to the opt-in public-trip dashboard replay; this result remains event-centred evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
