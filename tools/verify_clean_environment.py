"""Check CabInspector's Python dependencies from the interpreter that runs it.

The script intentionally imports only the standard library before checking the
runtime packages.  Run it from a newly created virtual environment after
installing ``requirements.txt``; it does not open a camera, microphone, model,
or private dataset.
"""

from __future__ import annotations

import argparse
import importlib
from importlib import metadata
import json
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"
REQUIREMENT_PATTERN = re.compile(r"^([A-Za-z0-9_.-]+)(?:==([^\s#]+))?")
IMPORT_MODULES = {
    "opencv-python": "cv2",
    "opencv-contrib-python": "cv2",
    "mediapipe": "mediapipe",
    "ultralytics": "ultralytics",
    "numpy": "numpy",
    "pillow": "PIL",
    "arabic-reshaper": "arabic_reshaper",
    "python-bidi": "bidi",
    "jax": "jax",
    "jaxlib": "jaxlib",
    "sentencepiece": "sentencepiece",
    "sounddevice": "sounddevice",
    "ai-edge-litert": "ai_edge_litert",
    "faster-whisper": "faster_whisper",
    "transformers": "transformers",
    "safetensors": "safetensors",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "joblib": "joblib",
    "torch": "torch",
    "psutil": "psutil",
}


def parse_requirements(path: Path = REQUIREMENTS_PATH) -> list[tuple[str, str | None]]:
    requirements: list[tuple[str, str | None]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(("-", "git+", "http:")):
            continue
        match = REQUIREMENT_PATTERN.match(line)
        if match:
            requirements.append((match.group(1), match.group(2)))
    return requirements


def check_environment(requirements_path: Path = REQUIREMENTS_PATH) -> dict[str, object]:
    packages: list[dict[str, object]] = []
    all_passed = True
    for distribution, expected_version in parse_requirements(requirements_path):
        try:
            installed_version = metadata.version(distribution)
            distribution_present = True
        except metadata.PackageNotFoundError:
            installed_version = None
            distribution_present = False

        module_name = IMPORT_MODULES.get(distribution)
        import_ok = False
        import_error = None
        if module_name is not None:
            try:
                importlib.import_module(module_name)
                import_ok = True
            except Exception as exc:  # pragma: no cover - environment-specific import failures
                import_error = f"{type(exc).__name__}: {exc}"

        version_matches = expected_version is None or installed_version == expected_version
        passed = distribution_present and import_ok and version_matches
        all_passed = all_passed and passed
        packages.append(
            {
                "distribution": distribution,
                "module": module_name,
                "required_version": expected_version,
                "installed_version": installed_version,
                "distribution_present": distribution_present,
                "import_ok": import_ok,
                "version_matches": version_matches,
                "status": "pass" if passed else "fail",
                **({"import_error": import_error} if import_error else {}),
            }
        )

    return {
        "status": "clean_environment_verification",
        "passed": all_passed,
        "python": {
            "version": sys.version,
            "executable": str(Path(sys.executable).resolve()),
        },
        "requirements_file": str(requirements_path),
        "package_count": len(packages),
        "packages": packages,
        "hardware_or_models_used": False,
        "private_data_read": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, default=REQUIREMENTS_PATH)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    try:
        report = check_environment(args.requirements)
    except OSError as exc:
        print(f"Could not inspect requirements: {exc}", file=sys.stderr)
        return 2

    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Verification written: {args.output}")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
