"""Download and checksum-verify CabInspector's public telemetry source data.

This utility downloads only the selected CC BY 4.0 Zenodo Driving Events
Dataset. It never reads, moves, overwrites, or uploads the personal files in
``data/Telemetry/Recordings``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "Telemetry" / "public" / "driving_events_dataset"
RAW_DIRECTORY = DATASET_ROOT / "raw"
METADATA_PATH = DATASET_ROOT / "metadata" / "source.json"
RECORD_API_URL = "https://zenodo.org/api/records/6570972"
CHUNK_BYTES = 1_048_576


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_record() -> dict:
    with urlopen(RECORD_API_URL, timeout=60) as response:
        return json.load(response)


def download_file(url: str, destination: Path, attempts: int = 3) -> None:
    """Download atomically, retrying transient Zenodo/CDN connection failures.

    A process-specific temporary file prevents an interrupted earlier run from
    being mistaken for an active download or from blocking a later retry.
    """

    failure: Exception | None = None
    temporary = destination.with_suffix(destination.suffix + f".part.{os.getpid()}")
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                while True:
                    chunk = response.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    output.write(chunk)
            temporary.replace(destination)
            return
        except (HTTPError, URLError, OSError, TimeoutError) as exc:
            failure = exc
            if temporary.exists():
                temporary.unlink(missing_ok=True)
            if attempt == attempts:
                break
            wait_seconds = attempt * 3
            print(f"Retrying {destination.name} in {wait_seconds}s after: {exc}")
            time.sleep(wait_seconds)
    raise RuntimeError(f"Could not download {destination.name} after {attempts} attempts") from failure


def expected_files(record: dict) -> list[dict]:
    required_prefixes = (
        "Linear_Acceleration_",
        "Gyroscope_",
        "Labeled_events_",
    )
    selected = []
    for item in record.get("files", []):
        key = item.get("key", "")
        if (
            key.startswith(required_prefixes) and key.endswith(".csv")
        ) or key == "Events_Extract_and_Visualization.ipynb":
            selected.append(item)
    return sorted(selected, key=lambda item: item["key"])


def update_local_metadata(record: dict, verified_files: list[dict]) -> None:
    metadata = {}
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    retrieval = metadata.setdefault("retrieval", {})
    retrieval["download_date"] = datetime.now(timezone.utc).isoformat()
    retrieval["remote_license"] = record.get("metadata", {}).get("license", {}).get("id")
    retrieval["verified_files"] = verified_files
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Show the source files without downloading.")
    parser.add_argument("--force", action="store_true", help="Redownload files even when their checksum matches.")
    args = parser.parse_args()

    print("Fetching public dataset metadata from Zenodo…")
    record = fetch_record()
    license_id = record.get("metadata", {}).get("license", {}).get("id", "unknown")
    if license_id != "cc-by-4.0":
        print(f"Refusing download: expected CC-BY-4.0, found {license_id!r}.", file=sys.stderr)
        return 2

    files = expected_files(record)
    if len(files) != 10:
        print(f"Refusing download: expected nine CSV files and one reference notebook, found {len(files)}.", file=sys.stderr)
        return 2

    print(f"Source: {record['metadata'].get('doi', 'unknown DOI')} ({license_id})")
    for item in files:
        print(f"  {item['key']} ({item['size']:,} bytes)")
    if args.dry_run:
        return 0

    RAW_DIRECTORY.mkdir(parents=True, exist_ok=True)
    verified_files = []
    for item in files:
        destination = RAW_DIRECTORY / item["key"]
        expected_md5 = item["checksum"].removeprefix("md5:")
        current_md5 = md5sum(destination) if destination.exists() else None
        if current_md5 == expected_md5 and not args.force:
            print(f"Verified existing: {destination.name}")
        else:
            print(f"Downloading: {destination.name}")
            download_file(item["links"]["self"], destination)
            current_md5 = md5sum(destination)
        if current_md5 != expected_md5:
            raise RuntimeError(
                f"Checksum failed for {destination.name}: expected {expected_md5}, got {current_md5}"
            )
        verified_files.append(
            {
                "name": destination.name,
                "bytes": destination.stat().st_size,
                "md5": current_md5,
                "source_url": item["links"]["self"],
            }
        )

    update_local_metadata(record, verified_files)
    print(f"Downloaded and verified {len(verified_files)} files in: {RAW_DIRECTORY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
