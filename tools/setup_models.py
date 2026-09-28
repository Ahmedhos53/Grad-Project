"""Verify or download CabInspector's allow-listed upstream model assets.

The manifest contains fixed HTTPS sources, expected byte counts, and SHA-256
digests.  This utility never accepts an arbitrary URL or destination, never
writes into ``data/`` or ``outputs/``, and downloads through an atomic temporary
file so a partial transfer cannot look like a usable model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "models" / "model_manifest.json"
CHUNK_BYTES = 1_048_576
DOWNLOAD_TIMEOUT_SECONDS = 120
ALLOWED_HOSTS = frozenset(
    {
        "github.com",
        "release-assets.githubusercontent.com",
        "objects.githubusercontent.com",
        "zenodo.org",
        "www.zenodo.org",
        "storage.googleapis.com",
    }
)
IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _is_allowed_https_url(url: str) -> bool:
    parsed = urlsplit(url)
    return parsed.scheme == "https" and parsed.hostname is not None and parsed.hostname.lower() in ALLOWED_HOSTS


def _safe_destination(relative_path: str) -> Path:
    """Resolve a manifest path and reject traversal or private/generated roots."""

    root = PROJECT_ROOT.resolve()
    candidate = (root / Path(relative_path)).resolve()
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Manifest destination escapes the project root: {relative_path!r}") from exc
    if not relative.parts or relative.parts[0].lower() in {"data", "outputs", ".venv"}:
        raise ValueError(f"Manifest destination is not an allowed model path: {relative_path!r}")
    return candidate


def load_manifest(path: Path = MANIFEST_PATH) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported model manifest schema")
    assets = payload.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError("Model manifest must contain a non-empty assets list")

    validated: list[dict[str, object]] = []
    identifiers: set[str] = set()
    destinations: set[str] = set()
    for asset in assets:
        if not isinstance(asset, dict):
            raise ValueError("Each model manifest asset must be an object")
        identifier = asset.get("id")
        relative_path = asset.get("path")
        url = asset.get("url")
        expected_bytes = asset.get("bytes")
        expected_sha256 = asset.get("sha256")
        if not isinstance(identifier, str) or not IDENTIFIER_PATTERN.fullmatch(identifier):
            raise ValueError(f"Invalid model asset id: {identifier!r}")
        if identifier in identifiers:
            raise ValueError(f"Duplicate model asset id: {identifier}")
        if not isinstance(relative_path, str) or not relative_path or Path(relative_path).is_absolute():
            raise ValueError(f"Invalid model asset path for {identifier}")
        destination = _safe_destination(relative_path)
        destination_key = str(destination).casefold()
        if destination_key in destinations:
            raise ValueError(f"Duplicate model asset path: {relative_path}")
        if not isinstance(url, str) or not _is_allowed_https_url(url):
            raise ValueError(f"Model source is not an allow-listed HTTPS URL for {identifier}")
        if not isinstance(expected_bytes, int) or expected_bytes <= 0:
            raise ValueError(f"Invalid expected byte count for {identifier}")
        if not isinstance(expected_sha256, str) or not SHA256_PATTERN.fullmatch(expected_sha256.lower()):
            raise ValueError(f"Invalid SHA-256 digest for {identifier}")

        identifiers.add(identifier)
        destinations.add(destination_key)
        validated.append(asset)
    return validated


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_asset(asset: dict[str, object]) -> dict[str, object]:
    path = _safe_destination(str(asset["path"]))
    expected_bytes = int(asset["bytes"])
    expected_sha256 = str(asset["sha256"]).lower()
    result: dict[str, object] = {
        "id": asset["id"],
        "path": str(asset["path"]),
        "expected_bytes": expected_bytes,
        "expected_sha256": expected_sha256,
        "exists": path.is_file(),
    }
    if not path.is_file():
        result["status"] = "missing"
        return result

    actual_bytes = path.stat().st_size
    result["actual_bytes"] = actual_bytes
    if actual_bytes != expected_bytes:
        result["status"] = "size_mismatch"
        return result
    actual_sha256 = sha256_file(path)
    result["actual_sha256"] = actual_sha256
    result["status"] = "verified" if actual_sha256 == expected_sha256 else "checksum_mismatch"
    return result


def _validate_final_url(url: str) -> None:
    """Allow only the documented host or a known GitHub release redirect."""

    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or host not in ALLOWED_HOSTS:
        raise RuntimeError(f"Refusing unexpected download redirect host: {url}")


def download_asset(asset: dict[str, object], *, force: bool = False) -> dict[str, object]:
    destination = _safe_destination(str(asset["path"]))
    current = inspect_asset(asset)
    if current["status"] == "verified" and not force:
        return current
    if destination.exists() and not force:
        raise RuntimeError(
            f"Existing asset does not match the manifest: {destination}. "
            "Review it or pass --force to replace this fixed target."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=str(destination.parent)
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    expected_bytes = int(asset["bytes"])
    expected_sha256 = str(asset["sha256"]).lower()
    try:
        request = Request(
            str(asset["url"]),
            headers={"User-Agent": "CabInspector-model-setup/1.0"},
        )
        with urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response, temporary.open("wb") as output:
            _validate_final_url(response.geturl())
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) != expected_bytes:
                raise RuntimeError(
                    f"Content-Length mismatch for {asset['id']}: expected {expected_bytes}, got {content_length}"
                )
            total = 0
            while True:
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > expected_bytes:
                    raise RuntimeError(f"Download exceeded the manifest size for {asset['id']}")
                output.write(chunk)
        if total != expected_bytes:
            raise RuntimeError(f"Size mismatch for {asset['id']}: expected {expected_bytes}, got {total}")
        actual_sha256 = sha256_file(temporary)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"SHA-256 mismatch for {asset['id']}: expected {expected_sha256}, got {actual_sha256}"
            )
        os.replace(temporary, destination)
    except (HTTPError, URLError, OSError, TimeoutError, ValueError) as exc:
        raise RuntimeError(f"Could not download {asset['id']}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return inspect_asset(asset)


def _select_assets(assets: list[dict[str, object]], requested: list[str] | None) -> list[dict[str, object]]:
    if not requested:
        return assets
    requested_set = set(requested)
    known = {str(asset["id"]) for asset in assets}
    unknown = sorted(requested_set - known)
    if unknown:
        raise ValueError(f"Unknown model asset id(s): {', '.join(unknown)}")
    return [asset for asset in assets if str(asset["id"]) in requested_set]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--list", action="store_true", help="list the fixed assets without touching files")
    parser.add_argument("--verify", action="store_true", help="verify selected local assets (the default action)")
    parser.add_argument("--download", action="store_true", help="download missing selected assets and verify them")
    parser.add_argument("--only", action="append", help="select one asset id; repeat for multiple assets")
    parser.add_argument("--force", action="store_true", help="replace a fixed target after checksum verification")
    args = parser.parse_args()

    try:
        assets = _select_assets(load_manifest(args.manifest), args.only)
        if args.list:
            for asset in assets:
                print(f"{asset['id']}: {asset['path']} ({int(asset['bytes']):,} bytes) <- {asset['url']}")
            return 0
        if args.force and not args.download:
            parser.error("--force requires --download")

        results = []
        for asset in assets:
            result = download_asset(asset, force=args.force) if args.download else inspect_asset(asset)
            results.append(result)
            print(f"{result['id']}: {result['status']} ({result.get('actual_bytes', 0):,} bytes)")
        failed = [result for result in results if result["status"] != "verified"]
        if failed:
            print("One or more model assets are missing or do not match the manifest.", file=sys.stderr)
            return 1
        print(f"Verified {len(results)} model asset(s).")
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Model setup refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
