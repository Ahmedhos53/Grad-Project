"""Create a reproducible schema and sensor-quality report for public telemetry data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.telemetry.dataset_audit import (
    DEFAULT_OUTPUT_DIRECTORY,
    DEFAULT_RAW_DIRECTORY,
    audit_driving_events_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", type=Path, default=DEFAULT_RAW_DIRECTORY)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    args = parser.parse_args()

    report = audit_driving_events_dataset(args.raw_directory, args.output_directory)
    report_path = args.output_directory / "dataset_audit.json"
    missing_files = sum(
        1
        for trip in report["trips"]
        for details in trip["files"].values()
        if details.get("missing")
    )
    print(json.dumps({"report": str(report_path), "missing_files": missing_files}, indent=2))
    return 1 if missing_files else 0


if __name__ == "__main__":
    raise SystemExit(main())
