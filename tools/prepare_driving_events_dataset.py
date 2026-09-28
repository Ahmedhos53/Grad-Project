"""Create fixed-shape training windows from CabInspector's public telemetry dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.telemetry.dataset_audit import DEFAULT_RAW_DIRECTORY
from src.telemetry.preprocessing import (
    DEFAULT_PROCESSED_DIRECTORY,
    WindowConfig,
    prepare_public_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", type=Path, default=DEFAULT_RAW_DIRECTORY)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_PROCESSED_DIRECTORY)
    parser.add_argument("--samples-per-window", type=int, default=256)
    args = parser.parse_args()

    manifest = prepare_public_dataset(
        raw_directory=args.raw_directory,
        output_directory=args.output_directory,
        config=WindowConfig(samples_per_window=args.samples_per_window),
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
