"""Trip-held-out model comparison for the public telemetry dataset.

All fitting functions receive a whole-trip split. Normalization, feature
scaling, augmentation, and model fitting are limited to that fold's training
trips, which prevents the held-out trip from leaking into model selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from time import perf_counter
from typing import Callable

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .labels import CLASS_NAMES
from .model import TelemetryCNN
from .preprocessing import DEFAULT_PROCESSED_DIRECTORY, FEATURE_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION_DIRECTORY = PROJECT_ROOT / "outputs" / "telemetry" / "evaluation"


@dataclass(frozen=True)
class Normalization:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, inputs: np.ndarray) -> "Normalization":
        mean = inputs.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        std = inputs.std(axis=(0, 2), keepdims=True).astype(np.float32)
        return cls(mean=mean, std=np.maximum(std, np.float32(1e-6)))

    def apply(self, inputs: np.ndarray) -> np.ndarray:
        return ((inputs - self.mean) / self.std).astype(np.float32)

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "mean": self.mean.reshape(-1).astype(float).tolist(),
            "std": self.std.reshape(-1).astype(float).tolist(),
        }


@dataclass(frozen=True)
class FoldData:
    train_inputs: np.ndarray
    train_targets: np.ndarray
    test_inputs: np.ndarray
    test_targets: np.ndarray
    normalization: Normalization


def _load_trip_windows(processed_directory: Path, trip: int) -> tuple[np.ndarray, np.ndarray]:
    path = processed_directory / f"trip_{trip}_windows.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing processed trip {trip}: {path}. Run tools/prepare_driving_events_dataset.py first."
        )
    with np.load(path, allow_pickle=False) as loaded:
        return loaded["inputs"].astype(np.float32), loaded["targets"].astype(np.int64)


def build_leave_one_trip_out_fold(processed_directory: Path, test_trip: int) -> FoldData:
    """Build one split without mixing a held-out trip into training data."""

    if test_trip not in (1, 2, 3):
        raise ValueError("test_trip must be 1, 2, or 3")
    train_parts = [_load_trip_windows(processed_directory, trip) for trip in (1, 2, 3) if trip != test_trip]
    test_inputs, test_targets = _load_trip_windows(processed_directory, test_trip)
    train_inputs = np.concatenate([part[0] for part in train_parts], axis=0)
    train_targets = np.concatenate([part[1] for part in train_parts], axis=0)
    normalization = Normalization.fit(train_inputs)
    return FoldData(
        train_inputs=normalization.apply(train_inputs),
        train_targets=train_targets,
        test_inputs=normalization.apply(test_inputs),
        test_targets=test_targets,
        normalization=normalization,
    )


def extract_classical_features(inputs: np.ndarray) -> np.ndarray:
    """Summarize each normalized inertial channel for conventional baselines."""

    if inputs.ndim != 3 or inputs.shape[1] != len(FEATURE_NAMES):
        raise ValueError("Expected inputs shaped (windows, 6, samples)")
    first_difference = np.diff(inputs, axis=2)
    summaries = (
        inputs.mean(axis=2),
        inputs.std(axis=2),
        inputs.min(axis=2),
        inputs.max(axis=2),
        np.sqrt(np.mean(inputs * inputs, axis=2)),
        np.mean(np.abs(first_difference), axis=2),
    )
    return np.concatenate(summaries, axis=1).astype(np.float32)


def _metrics(targets: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
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


def _evaluate_sklearn(factory: Callable[[], object], fold: FoldData) -> dict[str, object]:
    classifier = factory()
    train_features = extract_classical_features(fold.train_inputs)
    test_features = extract_classical_features(fold.test_inputs)
    started = perf_counter()
    classifier.fit(train_features, fold.train_targets)
    fit_seconds = perf_counter() - started
    started = perf_counter()
    probabilities = classifier.predict_proba(test_features)
    inference_ms_per_window = (perf_counter() - started) * 1_000 / len(test_features)
    predictions = classifier.classes_[np.argmax(probabilities, axis=1)]
    report = _metrics(fold.test_targets, predictions, probabilities)
    report.update({"fit_seconds": fit_seconds, "inference_ms_per_window": inference_ms_per_window})
    return report


def _choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for telemetry training but is not available.")
    return device


def _train_cnn_fold(
    fold: FoldData,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    """Train on two trips, choosing epoch only with a validation subset of them."""

    train_indices, validation_indices = train_test_split(
        np.arange(len(fold.train_targets)),
        test_size=0.20,
        random_state=seed,
        stratify=fold.train_targets,
    )
    model = TelemetryCNN(class_count=len(CLASS_NAMES)).to(device)
    counts = np.bincount(fold.train_targets[train_indices], minlength=len(CLASS_NAMES)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights /= weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=8, factor=0.5)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(fold.train_inputs[train_indices]),
            torch.from_numpy(fold.train_targets[train_indices]),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_inputs = torch.from_numpy(fold.train_inputs[validation_indices]).to(device)
    validation_targets = fold.train_targets[validation_indices]
    best_state: dict[str, torch.Tensor] | None = None
    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    started = perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            # Small noise applied only to in-fold training samples.
            augmented = features + 0.015 * torch.randn_like(features)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(augmented), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_predictions = model(validation_inputs).argmax(dim=1).cpu().numpy()
        validation_f1 = f1_score(
            validation_targets,
            validation_predictions,
            labels=np.unique(validation_targets),
            average="macro",
            zero_division=0,
        )
        scheduler.step(validation_f1)
        if validation_f1 > best_f1 + 1e-6:
            best_f1 = float(validation_f1)
            best_epoch = epoch
            stale_epochs = 0
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= 18:
                break
    if best_state is None:
        raise RuntimeError("CNN training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    started_inference = perf_counter()
    with torch.no_grad():
        logits = model(torch.from_numpy(fold.test_inputs).to(device))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    inference_ms_per_window = (perf_counter() - started_inference) * 1_000 / len(fold.test_inputs)
    predictions = probabilities.argmax(axis=1)
    report = _metrics(fold.test_targets, predictions, probabilities)
    report.update(
        {
            "fit_seconds": perf_counter() - started,
            "inference_ms_per_window": inference_ms_per_window,
            "best_validation_macro_f1": best_f1,
            "best_epoch": best_epoch,
            "device": str(device),
        }
    )
    return report, best_state


def run_model_comparison(
    processed_directory: Path = DEFAULT_PROCESSED_DIRECTORY,
    output_directory: Path = DEFAULT_EVALUATION_DIRECTORY,
    *,
    epochs: int = 60,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    device: str = "auto",
    seed: int = 2026,
) -> dict[str, object]:
    """Evaluate Logistic Regression, Random Forest, and 1D CNN trip by trip."""

    processed_directory = Path(processed_directory)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    selected_device = _choose_device(device)
    model_factories: dict[str, Callable[[], object]] = {
        "logistic_regression": lambda: Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=seed))]
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1
        ),
    }
    results: dict[str, list[dict[str, object]]] = {name: [] for name in (*model_factories, "compact_1d_cnn")}
    checkpoints: dict[int, dict[str, torch.Tensor]] = {}
    normalizations: dict[int, dict[str, list[float]]] = {}
    for test_trip in (1, 2, 3):
        fold = build_leave_one_trip_out_fold(processed_directory, test_trip)
        normalizations[test_trip] = fold.normalization.as_dict()
        for name, factory in model_factories.items():
            result = _evaluate_sklearn(factory, fold)
            result["test_trip"] = test_trip
            results[name].append(result)
        cnn_result, checkpoint = _train_cnn_fold(
            fold,
            device=selected_device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed + test_trip,
        )
        cnn_result["test_trip"] = test_trip
        results["compact_1d_cnn"].append(cnn_result)
        checkpoints[test_trip] = checkpoint

    summary = {
        name: {
            "mean_macro_f1_present_classes": float(np.mean([item["macro_f1_present_classes"] for item in folds])),
            "mean_accuracy": float(np.mean([item["accuracy"] for item in folds])),
            "mean_inference_ms_per_window": float(np.mean([item["inference_ms_per_window"] for item in folds])),
        }
        for name, folds in results.items()
    }
    report = {
        "dataset": "Driving Events Dataset (Zenodo 10.5281/zenodo.6570972)",
        "evaluation": "Leave-one-trip-out; each test trip is entirely excluded from normalization and fitting.",
        "class_names": list(CLASS_NAMES),
        "configuration": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "device": str(selected_device),
            "seed": seed,
        },
        "results": results,
        "summary": summary,
        "fold_normalization": normalizations,
        "note": "A class absent from a held-out trip has support zero in that fold; per-fold macro F1 is calculated over classes present in that test trip.",
    }
    (output_directory / "model_comparison.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
