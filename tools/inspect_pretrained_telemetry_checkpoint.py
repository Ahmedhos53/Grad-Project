"""Verify and inspect an external telemetry checkpoint without running CabInspector.

This utility is intentionally isolated from the dashboard and telemetry replay.  It checks the
downloaded file's integrity and reports only checkpoint metadata/state-dictionary structure; it
does not train a model, make a prediction, or modify a project artifact.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Mapping


def file_digest(path: Path, algorithm: str = "md5") -> str:
    """Return a streaming digest so large checkpoints are not read into memory at once."""
    digest = hashlib.new(algorithm)
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def find_state_dict(payload: Any) -> Mapping[str, Any] | None:
    """Locate the conventional state-dictionary field without assuming one framework."""
    if not isinstance(payload, Mapping):
        return None
    for key in ("state_dict", "model_state_dict", "model"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    return None


def describe_value(value: Any) -> str:
    """Produce a compact, non-sensitive metadata description for console output."""
    if isinstance(value, Mapping):
        return f"mapping with {len(value)} key(s)"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__} with {len(value)} item(s)"
    rendered = repr(value)
    return rendered if len(rendered) <= 240 else f"{rendered[:237]}..."


def inspect_checkpoint(path: Path, expected_md5: str | None = None) -> int:
    """Verify one checkpoint and print a safe structural report. Returns a process status."""
    if not path.is_file():
        print(f"Checkpoint not found: {path}")
        return 2

    actual_md5 = file_digest(path)
    print(f"Checkpoint: {path.resolve()}")
    print(f"Size: {path.stat().st_size:,} bytes")
    print(f"MD5: {actual_md5}")
    if expected_md5:
        if actual_md5.lower() != expected_md5.lower():
            print(f"Integrity: FAILED (expected {expected_md5.lower()})")
            return 3
        print("Integrity: verified")

    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"Checkpoint structure: could not be loaded on CPU: {type(exc).__name__}: {exc}")
        return 4

    print(f"Payload type: {type(payload).__name__}")
    if isinstance(payload, Mapping):
        print("Top-level fields:")
        for key in sorted(payload, key=str):
            if key == "state_dict":
                continue
            print(f"  - {key}: {describe_value(payload[key])}")

    state_dict = find_state_dict(payload)
    if state_dict is None:
        print("State dictionary: not found under state_dict, model_state_dict, or model")
        return 0

    tensor_entries = [
        (str(name), value)
        for name, value in state_dict.items()
        if hasattr(value, "numel") and hasattr(value, "shape")
    ]
    parameter_count = sum(int(value.numel()) for _, value in tensor_entries)
    print(f"State dictionary: {len(state_dict)} entry/entries; {parameter_count:,} tensor values")
    print("First tensor entries:")
    for name, value in tensor_entries[:12]:
        print(f"  - {name}: shape={tuple(value.shape)}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to the downloaded checkpoint.")
    parser.add_argument(
        "--expected-md5",
        default=None,
        help="Optional publisher-provided MD5 checksum to verify before inspection.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    raise SystemExit(inspect_checkpoint(arguments.checkpoint, arguments.expected_md5))
