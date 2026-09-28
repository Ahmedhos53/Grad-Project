"""Explicit opt-in transcript persistence; raw audio is never stored."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from threading import RLock


TRANSCRIPT_FIELDS = [
    "recorded_at",
    "language",
    "transcript",
    "safety_categories",
    "safety_confidence",
]


def _arabic_character_count(value: str) -> int:
    return sum("\u0600" <= character <= "\u06ff" for character in value)


def repair_arabic_mojibake(value: str) -> str:
    """Recover common Windows-1256/UTF-8 Arabic text corruption when possible."""
    if not value:
        return value

    candidates = [value]
    try:
        legacy_bytes = value.encode("latin-1")
    except UnicodeEncodeError:
        legacy_bytes = None
    if legacy_bytes is not None:
        for encoding in ("utf-8", "cp1256"):
            try:
                candidates.append(legacy_bytes.decode(encoding))
            except UnicodeDecodeError:
                pass

    # Keep ordinary English unchanged. Only replace it when an alternative has more Arabic text.
    return max(candidates, key=_arabic_character_count)


def export_readable_transcript(source_path: str | Path, destination_path: str | Path) -> Path:
    """Create a readable text export from a consented transcript CSV."""
    source = Path(source_path)
    destination = Path(destination_path)
    if not source.exists():
        raise FileNotFoundError(f"No transcript log exists at {source}")

    with source.open("r", newline="", encoding="utf-8-sig") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise ValueError("The transcript log has no saved entries yet")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="\n") as output_file:
        output_file.write("CabInspector Transcript Export\n")
        output_file.write("=" * 30 + "\n\n")
        for number, row in enumerate(rows, start=1):
            language = row.get("language", "unknown").upper()
            transcript = repair_arabic_mojibake(row.get("transcript", "").strip())
            categories = row.get("safety_categories", "").strip()
            recorded_at = row.get("recorded_at", "Legacy entry (date unavailable)")
            output_file.write(f"{number}. [{recorded_at} | {language}] {transcript}\n")
            if categories:
                output_file.write(f"   Safety: {categories}\n")
    return destination


def migrate_transcript_log(path: str | Path) -> Path:
    """Repair a legacy transcript CSV for Excel without adding or deleting transcript rows."""
    store = TranscriptStore(path)
    if not store.path.exists():
        raise FileNotFoundError(f"No transcript log exists at {store.path}")
    with store._lock:
        store._migrate_existing_file()
    return store.path


class TranscriptStore:
    """Creates a transcript CSV only when explicitly enabled by the caller."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = RLock()
        self._handle = None
        self._writer = None

    def append(
        self,
        *,
        timestamp_ms: int,
        language: str,
        transcript: str,
        safety_categories: tuple[str, ...],
        safety_confidence: float,
    ) -> None:
        """Append text with a real local timestamp; timestamp_ms is retained for API compatibility."""
        del timestamp_ms
        with self._lock:
            self._ensure_open()
            self._writer.writerow(
                {
                    "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "language": language,
                    "transcript": repair_arabic_mojibake(transcript),
                    "safety_categories": ", ".join(safety_categories),
                    "safety_confidence": safety_confidence,
                }
            )
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
            self._handle = None
            self._writer = None

    def _ensure_open(self) -> None:
        if self._handle is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size > 0:
            self._migrate_existing_file()
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        # UTF-8 BOM makes Microsoft Excel display Arabic correctly when it opens the CSV directly.
        self._handle = self.path.open("a", newline="", encoding="utf-8-sig")
        self._writer = csv.DictWriter(self._handle, fieldnames=TRANSCRIPT_FIELDS)
        if new_file:
            self._writer.writeheader()

    def _migrate_existing_file(self) -> None:
        raw = self.path.read_bytes()
        has_utf8_bom = raw.startswith(b"\xef\xbb\xbf")
        with self.path.open("r", newline="", encoding="utf-8-sig") as input_file:
            reader = csv.DictReader(input_file)
            source_fields = reader.fieldnames or []
            rows = list(reader)

        if has_utf8_bom and source_fields == TRANSCRIPT_FIELDS:
            return

        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary_path.open("w", newline="", encoding="utf-8-sig") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=TRANSCRIPT_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "recorded_at": row.get("recorded_at") or "Legacy entry (date unavailable)",
                        "language": row.get("language", "unknown"),
                        "transcript": repair_arabic_mojibake(row.get("transcript", "")),
                        "safety_categories": row.get("safety_categories", ""),
                        "safety_confidence": row.get("safety_confidence", "0.0"),
                    }
                )
        temporary_path.replace(self.path)
