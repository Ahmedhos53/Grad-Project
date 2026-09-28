"""Transfer evaluation for the externally pretrained PRIMUS IMU encoder.

The encoder remains frozen. CabInspector trains only a small downstream classifier on
fixed five-second, 200 Hz public-driving windows. Evaluation holds out one complete trip
at a time so neither classifier fitting nor feature scaling can see the test trip.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dataset_audit import DEFAULT_RAW_DIRECTORY
from .dataset_loader import LabeledEvent, PublicTelemetryTrip, SensorStream, load_public_trip
from .labels import CLASS_NAMES
from .primus_encoder import (
    DEFAULT_PRIMUS_CHECKPOINT,
    PRIMUS_CHANNELS,
    PRIMUS_EMBEDDING_SIZE,
    PRIMUS_SAMPLES_PER_WINDOW,
    PRIMUS_TARGET_HZ,
    PRIMUS_WINDOW_SECONDS,
    PrimusIMUEncoder,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRIMUS_EVALUATION_PATH = (
    PROJECT_ROOT / "outputs" / "telemetry" / "evaluation" / "primus_transfer_evaluation.json"
)


@dataclass(frozen=True)
class PrimusEventDataset:
    """Fixed-time public event windows and traceability metadata."""

    inputs: np.ndarray
    targets: np.ndarray
    trip_ids: np.ndarray
    source_samples: np.ndarray
    event_start_seconds: np.ndarray
    event_end_seconds: np.ndarray
    window_start_seconds: np.ndarray
    window_end_seconds: np.ndarray
    source_labels: tuple[str, ...]


def _interpolate_at(stream: SensorStream, requested_times: np.ndarray) -> np.ndarray:
    if requested_times[0] < stream.timestamps[0] or requested_times[-1] > stream.timestamps[-1]:
        raise ValueError(
            "PRIMUS window falls outside a sensor recording: "
            f"{requested_times[0]:.3f}–{requested_times[-1]:.3f} not in "
            f"{stream.timestamps[0]:.3f}–{stream.timestamps[-1]:.3f}"
        )
    return np.stack(
        [np.interp(requested_times, stream.timestamps, stream.values[:, channel]) for channel in range(3)],
        axis=0,
    ).astype(np.float32)


def make_fixed_primus_event_window(
    event: LabeledEvent,
    trip: PublicTelemetryTrip,
) -> tuple[np.ndarray, float, float]:
    """Create one true five-second, 200 Hz window centred on an annotation.

    The output channel order matches PRIMUS's published preprocessing:
    acceleration X/Y/Z followed by gyroscope X/Y/Z. The end is exclusive, so
    1,000 samples cover exactly five seconds at 200 samples per second.
    """

    if event.trip != trip.trip:
        raise ValueError("Event trip and sensor trip do not match")
    centre = (event.sensor_start_seconds + event.sensor_end_seconds) / 2.0
    return make_fixed_primus_window_at(trip, centre - PRIMUS_WINDOW_SECONDS / 2.0)


def make_fixed_primus_window_at(
    trip: PublicTelemetryTrip,
    window_start: float,
) -> tuple[np.ndarray, float, float]:
    """Create one arbitrary fixed five-second, 200 Hz public sensor window.

    This is used only by the offline continuous-window evaluator.  It has no
    path to the dashboard and does not expose source labels to the application.
    """

    window_start = float(window_start)
    requested_times = window_start + np.arange(PRIMUS_SAMPLES_PER_WINDOW, dtype=np.float64) / PRIMUS_TARGET_HZ
    window_end = window_start + PRIMUS_WINDOW_SECONDS
    acceleration = _interpolate_at(trip.acceleration, requested_times)
    gyroscope = _interpolate_at(trip.gyroscope, requested_times)
    window = np.concatenate((acceleration, gyroscope), axis=0)
    if window.shape != (PRIMUS_CHANNELS, PRIMUS_SAMPLES_PER_WINDOW):
        raise RuntimeError(f"Unexpected fixed PRIMUS window shape: {window.shape}")
    return window, float(window_start), float(window_end)


def prepare_primus_event_dataset(
    raw_directory: Path = DEFAULT_RAW_DIRECTORY,
    trips: tuple[int, ...] = (1, 2, 3),
) -> PrimusEventDataset:
    """Build source-traceable fixed windows without using private trip recordings."""

    windows: list[np.ndarray] = []
    targets: list[int] = []
    trip_ids: list[int] = []
    source_samples: list[int] = []
    event_starts: list[float] = []
    event_ends: list[float] = []
    window_starts: list[float] = []
    window_ends: list[float] = []
    source_labels: list[str] = []
    for trip_number in trips:
        trip = load_public_trip(trip_number, Path(raw_directory))
        for event in trip.events:
            window, window_start, window_end = make_fixed_primus_event_window(event, trip)
            windows.append(window)
            targets.append(event.class_index)
            trip_ids.append(trip_number)
            source_samples.append(event.source_sample)
            event_starts.append(event.start_seconds)
            event_ends.append(event.end_seconds)
            window_starts.append(window_start)
            window_ends.append(window_end)
            source_labels.append(event.source_label)
    if not windows:
        raise ValueError("No public event windows were prepared")
    return PrimusEventDataset(
        inputs=np.stack(windows).astype(np.float32, copy=False),
        targets=np.asarray(targets, dtype=np.int64),
        trip_ids=np.asarray(trip_ids, dtype=np.int64),
        source_samples=np.asarray(source_samples, dtype=np.int64),
        event_start_seconds=np.asarray(event_starts, dtype=np.float64),
        event_end_seconds=np.asarray(event_ends, dtype=np.float64),
        window_start_seconds=np.asarray(window_starts, dtype=np.float64),
        window_end_seconds=np.asarray(window_ends, dtype=np.float64),
        source_labels=tuple(source_labels),
    )


def extract_primus_embeddings(
    encoder: PrimusIMUEncoder,
    inputs: np.ndarray,
    *,
    batch_size: int = 16,
) -> tuple[np.ndarray, float]:
    """Encode fixed windows in bounded batches and report encoder-only latency."""

    values = np.asarray(inputs, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (PRIMUS_CHANNELS, PRIMUS_SAMPLES_PER_WINDOW):
        raise ValueError(
            "Expected fixed PRIMUS inputs shaped "
            f"(windows, {PRIMUS_CHANNELS}, {PRIMUS_SAMPLES_PER_WINDOW}), got {values.shape}"
        )
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    outputs: list[np.ndarray] = []
    started = perf_counter()
    for start in range(0, len(values), batch_size):
        outputs.append(encoder.encode(values[start : start + batch_size]))
    elapsed = perf_counter() - started
    embeddings = np.concatenate(outputs, axis=0)
    if embeddings.shape != (len(values), PRIMUS_EMBEDDING_SIZE):
        raise RuntimeError(f"Unexpected PRIMUS embedding shape: {embeddings.shape}")
    return embeddings, elapsed * 1_000 / len(values)


def _aligned_probabilities(classifier, features: np.ndarray) -> np.ndarray:
    observed = classifier.predict_proba(features)
    probabilities = np.zeros((len(features), len(CLASS_NAMES)), dtype=np.float64)
    for source_column, class_index in enumerate(classifier.classes_):
        probabilities[:, int(class_index)] = observed[:, source_column]
    return probabilities


def _metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    predictions = probabilities.argmax(axis=1)
    present_labels = np.unique(targets)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=range(len(CLASS_NAMES)), zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1_present_classes": float(
            f1_score(targets, predictions, labels=present_labels, average="macro", zero_division=0)
        ),
        "mean_confidence": float(np.max(probabilities, axis=1).mean()),
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=range(len(CLASS_NAMES))
        ).astype(int).tolist(),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(CLASS_NAMES)
        },
    }


def _classifier_factories(seed: int):
    return {
        "primus_linear_head": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=5_000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "primus_random_forest_head": lambda: RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }


def run_primus_transfer_evaluation(
    raw_directory: Path = DEFAULT_RAW_DIRECTORY,
    checkpoint_path: Path = DEFAULT_PRIMUS_CHECKPOINT,
    output_path: Path = DEFAULT_PRIMUS_EVALUATION_PATH,
    *,
    batch_size: int = 16,
    seed: int = 2026,
) -> dict[str, object]:
    """Evaluate frozen PRIMUS embeddings with whole-trip-held-out classifier heads."""

    dataset = prepare_primus_event_dataset(Path(raw_directory))
    encoder = PrimusIMUEncoder()
    encoder.load_checkpoint(Path(checkpoint_path))
    embeddings, encoder_ms = extract_primus_embeddings(encoder, dataset.inputs, batch_size=batch_size)

    results: dict[str, list[dict[str, object]]] = {
        name: [] for name in _classifier_factories(seed)
    }
    for test_trip in (1, 2, 3):
        train_mask = dataset.trip_ids != test_trip
        test_mask = dataset.trip_ids == test_trip
        if np.any(dataset.trip_ids[train_mask] == test_trip):
            raise RuntimeError("Held-out trip leaked into PRIMUS classifier training")
        for name, factory in _classifier_factories(seed + test_trip).items():
            classifier = factory()
            started = perf_counter()
            classifier.fit(embeddings[train_mask], dataset.targets[train_mask])
            fit_seconds = perf_counter() - started
            started = perf_counter()
            probabilities = _aligned_probabilities(classifier, embeddings[test_mask])
            head_ms = (perf_counter() - started) * 1_000 / int(test_mask.sum())
            fold = _metrics(dataset.targets[test_mask], probabilities)
            fold.update(
                {
                    "test_trip": test_trip,
                    "train_windows": int(train_mask.sum()),
                    "test_windows": int(test_mask.sum()),
                    "fit_seconds": fit_seconds,
                    "head_inference_ms_per_window": head_ms,
                    "total_inference_ms_per_window": encoder_ms + head_ms,
                }
            )
            results[name].append(fold)

    summary = {
        name: {
            "mean_macro_f1_present_classes": float(
                np.mean([fold["macro_f1_present_classes"] for fold in folds])
            ),
            "mean_accuracy": float(np.mean([fold["accuracy"] for fold in folds])),
            "mean_total_inference_ms_per_window": float(
                np.mean([fold["total_inference_ms_per_window"] for fold in folds])
            ),
        }
        for name, folds in results.items()
    }
    counts = np.bincount(dataset.targets, minlength=len(CLASS_NAMES))
    report: dict[str, object] = {
        "status": "transfer_evaluation_with_dashboard_replay",
        "dataset": "Driving Events Dataset (Zenodo 10.5281/zenodo.6570972)",
        "personal_recordings_excluded": True,
        "evaluation": (
            "Leave-one-trip-out transfer evaluation. The PRIMUS encoder is frozen; "
            "feature scaling and classifier fitting use only the two training trips. "
            "The exported PRIMUS head is also connected to the opt-in public-trip dashboard replay; "
            "this report itself remains an event-centred transfer evaluation."
        ),
        "pretrained_encoder": "PRIMUS official best_model.ckpt",
        "checkpoint_path": str(Path(checkpoint_path)),
        "encoder_load_seconds": encoder.load_seconds,
        "encoder_inference_ms_per_window": encoder_ms,
        "embedding_size": PRIMUS_EMBEDDING_SIZE,
        "windowing": {
            "duration_seconds": PRIMUS_WINDOW_SECONDS,
            "sampling_hz": PRIMUS_TARGET_HZ,
            "samples": PRIMUS_SAMPLES_PER_WINDOW,
            "alignment": "Centred on each source annotation; fixed wall-clock duration; end-exclusive sampling.",
            "channel_order": ["acceleration_x", "acceleration_y", "acceleration_z", "gyroscope_x", "gyroscope_y", "gyroscope_z"],
        },
        "window_count": int(len(dataset.targets)),
        "class_names": list(CLASS_NAMES),
        "class_counts": {name: int(counts[index]) for index, name in enumerate(CLASS_NAMES)},
        "results": results,
        "summary": summary,
        "limitations": [
            "PRIMUS was pretrained on wearable/head IMU data rather than vehicle events.",
            "The public driving dataset contains one driver, one vehicle, one phone, and three trips.",
            "A five-second window can include context outside a short annotation or only the centre of an annotation longer than five seconds.",
            "This evaluates annotated event-centred windows, not continuous live sliding-window false-alarm behaviour.",
            "A class absent from a held-out trip has support zero; fold macro-F1 uses classes present in that test trip.",
        ],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
