"""Repair the consented transcript CSV for Excel's UTF-8 Arabic support."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.transcript_store import migrate_transcript_log


def main() -> int:
    path = ROOT / "outputs" / "transcript_log.csv"
    try:
        migrated = migrate_transcript_log(path)
    except PermissionError:
        print("Close transcript_log.csv in Excel, then run this command again.")
        return 1
    except (FileNotFoundError, OSError) as exc:
        print(f"Transcript repair failed: {exc}")
        return 1

    print(f"Transcript log repaired for Excel: {migrated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
