"""Export the accepted frozen PRIMUS encoder and CabInspector linear event head."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.telemetry.dataset_audit import DEFAULT_RAW_DIRECTORY
from src.telemetry.labels import CLASS_NAMES
from src.telemetry.primus_encoder import (
    DEFAULT_PRIMUS_CHECKPOINT,
    DEFAULT_PRIMUS_ENCODER_STATE,
    PRIMUS_EMBEDDING_SIZE,
    PRIMUS_SAMPLES_PER_WINDOW,
    PRIMUS_TARGET_HZ,
    PRIMUS_WINDOW_SECONDS,
    PrimusIMUEncoder,
)
from src.telemetry.primus_inference import DEFAULT_PRIMUS_EVENT_HEAD
from src.telemetry.primus_transfer import (
    DEFAULT_PRIMUS_EVALUATION_PATH,
    extract_primus_embeddings,
    prepare_primus_event_dataset,
)


DEFAULT_PRIMUS_METADATA = (
    PROJECT_ROOT / "models" / "telemetry" / "pretrained" / "primus" / "event_model_metadata.json"
)
SOURCE_CHECKPOINT_SHA256 = "66392F3CEBF1DCBE079FC4B8BDF3B4460BB7BCAD8E47A217D114FCDF917FBA3F"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", type=Path, default=DEFAULT_RAW_DIRECTORY)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_PRIMUS_CHECKPOINT)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_PRIMUS_EVALUATION_PATH)
    parser.add_argument("--encoder-output", type=Path, default=DEFAULT_PRIMUS_ENCODER_STATE)
    parser.add_argument("--head-output", type=Path, default=DEFAULT_PRIMUS_EVENT_HEAD)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_PRIMUS_METADATA)
    args = parser.parse_args()

    report = json.loads(args.evaluation.read_text(encoding="utf-8"))
    selected = report["summary"]["primus_linear_head"]
    if selected["mean_macro_f1_present_classes"] < 0.70 or selected["mean_accuracy"] < 0.75:
        raise RuntimeError("PRIMUS transfer evaluation did not pass the export acceptance gate")
    if _sha256(args.checkpoint) != SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("PRIMUS source checkpoint SHA-256 does not match verified provenance")

    dataset = prepare_primus_event_dataset(args.raw_directory)
    encoder = PrimusIMUEncoder()
    encoder.load_checkpoint(args.checkpoint)
    embeddings, encoder_ms = extract_primus_embeddings(encoder, dataset.inputs)
    classifier = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=5_000, class_weight="balanced", random_state=2026),
            ),
        ]
    )
    classifier.fit(embeddings, dataset.targets)

    args.encoder_output.parent.mkdir(parents=True, exist_ok=True)
    encoder_payload = {
        "format": "cabinspector_primus_imu_encoder_v1",
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "state_dict": encoder.model.state_dict(),
    }
    encoder._torch.save(encoder_payload, args.encoder_output)
    head_package = {
        "format": "cabinspector_primus_event_head_v1",
        "class_names": CLASS_NAMES,
        "classifier": classifier,
    }
    joblib.dump(head_package, args.head_output)

    # Verify the compact state is numerically identical to the source-checkpoint encoder.
    compact = PrimusIMUEncoder()
    compact.load_compact_state(args.encoder_output)
    source_embedding = encoder.encode(dataset.inputs[:2])
    compact_embedding = compact.encode(dataset.inputs[:2])
    max_difference = float(abs(source_embedding - compact_embedding).max())
    if max_difference > 1e-6:
        raise RuntimeError(f"Compact PRIMUS state differs from source encoder: {max_difference}")

    metadata = {
        "model_type": "Frozen PRIMUS pretrained IMU encoder with CabInspector linear event head",
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_model": "PRIMUS Model Checkpoint, Nokia Bell Labs, ICASSP 2025",
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "source_checkpoint_license": "CC-BY-4.0",
        "source_code_license": "BSD-3-Clause-Clear",
        "dataset": "Driving Events Dataset, Zenodo 10.5281/zenodo.6570972",
        "personal_recordings_excluded": True,
        "encoder_output": str(args.encoder_output),
        "encoder_output_sha256": _sha256(args.encoder_output),
        "head_output": str(args.head_output),
        "head_output_sha256": _sha256(args.head_output),
        "class_names": CLASS_NAMES,
        "embedding_size": PRIMUS_EMBEDDING_SIZE,
        "window_seconds": PRIMUS_WINDOW_SECONDS,
        "sampling_hz": PRIMUS_TARGET_HZ,
        "samples_per_window": PRIMUS_SAMPLES_PER_WINDOW,
        "channel_order": ["acceleration_x", "acceleration_y", "acceleration_z", "gyroscope_x", "gyroscope_y", "gyroscope_z"],
        "training_windows": int(len(dataset.targets)),
        "selection_evidence": selected,
        "evaluation_report": str(args.evaluation),
        "encoder_inference_ms_per_window_during_export": encoder_ms,
        "compact_state_max_embedding_difference": max_difference,
        "known_limitations": report["limitations"],
    }
    args.metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved compact pretrained encoder: {args.encoder_output}")
    print(f"Saved event head: {args.head_output}")
    print(f"Saved metadata: {args.metadata_output}")
    print(f"Compact/source embedding max difference: {max_difference:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
