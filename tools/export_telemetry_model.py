"""Train the selected Random Forest on all public windows and export it for replay/integration."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.telemetry.labels import CLASS_NAMES
from src.telemetry.preprocessing import DEFAULT_PROCESSED_DIRECTORY, FEATURE_NAMES
from src.telemetry.training import DEFAULT_EVALUATION_DIRECTORY, Normalization, extract_classical_features


DEFAULT_MODEL_DIRECTORY = PROJECT_ROOT / "models" / "telemetry"


def load_all_public_windows(processed_directory: Path) -> tuple[np.ndarray, np.ndarray]:
    parts = []
    for trip in (1, 2, 3):
        with np.load(processed_directory / f"trip_{trip}_windows.npz", allow_pickle=False) as loaded:
            parts.append((loaded["inputs"].astype(np.float32), loaded["targets"].astype(np.int64)))
    return np.concatenate([part[0] for part in parts]), np.concatenate([part[1] for part in parts])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-directory", type=Path, default=DEFAULT_PROCESSED_DIRECTORY)
    parser.add_argument("--evaluation-directory", type=Path, default=DEFAULT_EVALUATION_DIRECTORY)
    parser.add_argument("--model-directory", type=Path, default=DEFAULT_MODEL_DIRECTORY)
    args = parser.parse_args()
    comparison_path = args.evaluation_directory / "model_comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    selected = comparison["summary"]["random_forest"]
    if selected["mean_macro_f1_present_classes"] <= comparison["summary"]["logistic_regression"]["mean_macro_f1_present_classes"]:
        raise RuntimeError("Random Forest did not beat the baseline; refusing automatic selection.")
    inputs, targets = load_all_public_windows(args.processed_directory)
    normalization = Normalization.fit(inputs)
    features = extract_classical_features(normalization.apply(inputs))
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=2026, n_jobs=-1)
    model.fit(features, targets)
    args.model_directory.mkdir(parents=True, exist_ok=True)
    model_path = args.model_directory / "telemetry_random_forest.joblib"
    package = {
        "model_type": "random_forest",
        "class_names": CLASS_NAMES,
        "samples_per_window": int(inputs.shape[2]),
        "feature_names": FEATURE_NAMES,
        "normalization": normalization.as_dict(),
        "model": model,
    }
    joblib.dump(package, model_path)
    metadata = {
        "model_path": str(model_path),
        "model_type": "RandomForestClassifier",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_data": "All 169 public Driving Events Dataset windows; personal recordings excluded.",
        "dataset_doi": "10.5281/zenodo.6570972",
        "license": "CC-BY-4.0",
        "class_names": CLASS_NAMES,
        "feature_names": FEATURE_NAMES,
        "samples_per_window": int(inputs.shape[2]),
        "selection_evidence": selected,
        "comparison_report": str(comparison_path),
        "known_limitations": [
            "The public source has one driver and one vehicle.",
            "The model recognizes labelled motion events only; it cannot infer speed-limit compliance or GPS health.",
            "The deployed model must receive the same six-channel inertial signal convention used by the source dataset.",
        ],
    }
    metadata_path = args.model_directory / "telemetry_model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
