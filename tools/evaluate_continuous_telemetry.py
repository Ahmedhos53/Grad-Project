"""Evaluate continuous public telemetry with trip-held-out model heads.

Each outer fold fits its downstream heads and normalization on two public trips,
then scores only fixed five-second windows from the held-out trip. The frozen
PRIMUS encoder is external; it is not fitted on CabInspector trips. The report
keeps per-window errors separate from temporally merged alert episodes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import multiclass_metrics
from src.telemetry.dataset_audit import DEFAULT_RAW_DIRECTORY
from src.telemetry.dataset_loader import PublicTelemetryTrip, load_public_trip
from src.telemetry.labels import CLASS_NAMES
from src.telemetry.primus_encoder import DEFAULT_PRIMUS_CHECKPOINT, PrimusIMUEncoder
from src.telemetry.primus_transfer import (
    extract_primus_embeddings,
    make_fixed_primus_event_window,
    make_fixed_primus_window_at,
)
from src.telemetry.preprocessing import WindowConfig, prepare_trip_windows
from src.telemetry.training import Normalization, extract_classical_features


WINDOW_SECONDS = 5.0
DEFAULT_CONFIDENCE_THRESHOLD = 0.55
DEFAULT_EVENT_MERGE_GAP_SECONDS = 3.0
DEFAULT_SEED = 2026
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "report_revision"
    / "telemetry"
    / "continuous_telemetry_crossvalidated.json"
)


def _sensor_bounds(trip: PublicTelemetryTrip) -> tuple[float, float]:
    start = max(float(trip.acceleration.timestamps[0]), float(trip.gyroscope.timestamps[0]))
    end = min(float(trip.acceleration.timestamps[-1]), float(trip.gyroscope.timestamps[-1]))
    if end - start < WINDOW_SECONDS:
        raise ValueError(f"Trip {trip.trip} is shorter than the required five-second window")
    return start, end


def _expected_category(
    trip: PublicTelemetryTrip,
    start: float,
    end: float,
) -> tuple[str, float, str, int, int]:
    """Assign a window its greatest-overlap source label, or background NORMAL."""

    overlaps = [
        (
            max(0.0, min(end, event.sensor_end_seconds) - max(start, event.sensor_start_seconds)),
            event,
        )
        for event in trip.events
    ]
    positive = [(overlap, event) for overlap, event in overlaps if overlap > 0.0]
    if not positive:
        return "NORMAL", 0.0, "unannotated_background", 0, 0

    # max() is stable, so source CSV order breaks exact-overlap ties.
    largest_overlap, selected = max(positive, key=lambda pair: pair[0])
    tied_count = sum(abs(overlap - largest_overlap) <= 1e-9 for overlap, _event in positive)
    return (
        selected.category,
        round(largest_overlap, 4),
        selected.source_label,
        len(positive),
        tied_count,
    )


def window_schedule(
    trip: PublicTelemetryTrip,
    stride_seconds: float,
    max_windows: int = 0,
) -> list[dict[str, object]]:
    if stride_seconds <= 0.0:
        raise ValueError("stride_seconds must be greater than zero")
    start, end = _sensor_bounds(trip)
    final_start = end - WINDOW_SECONDS
    schedule: list[dict[str, object]] = []
    current = start
    while current <= final_start + 1e-9:
        window_end = current + WINDOW_SECONDS
        category, overlap, label_source, overlap_count, top_tie_count = _expected_category(
            trip, current, window_end
        )
        schedule.append(
            {
                "trip": trip.trip,
                "window_start_seconds": round(current, 6),
                "window_end_seconds": round(window_end, 6),
                "expected_category": category,
                "annotation_overlap_seconds": overlap,
                "label_source": label_source,
                "overlapping_annotation_count": overlap_count,
                "greatest_overlap_tie_count": top_tie_count,
            }
        )
        if max_windows and len(schedule) >= max_windows:
            break
        current += stride_seconds
    return schedule


def _make_random_forest_window(
    trip: PublicTelemetryTrip,
    start: float,
    end: float,
    samples: int,
) -> np.ndarray:
    """Resample a fixed continuous window to the RF input length."""

    requested_times = np.linspace(start, end, samples, dtype=np.float64)

    def interpolate(stream) -> np.ndarray:
        return np.stack(
            [np.interp(requested_times, stream.timestamps, stream.values[:, channel]) for channel in range(3)],
            axis=0,
        ).astype(np.float32)

    return np.concatenate((interpolate(trip.acceleration), interpolate(trip.gyroscope)), axis=0)


def _aligned_probabilities(classifier, inputs: np.ndarray) -> tuple[list[str], list[float]]:
    observed = classifier.predict_proba(inputs)
    probabilities = np.zeros((len(inputs), len(CLASS_NAMES)), dtype=np.float64)
    for source_column, class_index in enumerate(classifier.classes_):
        probabilities[:, int(class_index)] = observed[:, source_column]
    class_indices = probabilities.argmax(axis=1)
    return (
        [CLASS_NAMES[int(index)] for index in class_indices],
        [float(probabilities[row, index]) for row, index in enumerate(class_indices)],
    )


def _training_class_counts(targets: np.ndarray) -> dict[str, int]:
    counts = np.bincount(targets, minlength=len(CLASS_NAMES))
    return {name: int(counts[index]) for index, name in enumerate(CLASS_NAMES)}


def _source_annotation_overlap_count(trip: PublicTelemetryTrip) -> int:
    ordered = sorted(
        trip.events,
        key=lambda event: (event.sensor_start_seconds, event.sensor_end_seconds),
    )
    return sum(
        left.sensor_start_seconds < right.sensor_end_seconds
        and right.sensor_start_seconds < left.sensor_end_seconds
        for index, left in enumerate(ordered)
        for right in ordered[index + 1 :]
    )


def _alert_episode_summary(
    rows: list[dict[str, object]],
    *,
    category_key: str,
    confidence_key: str,
    threshold: float,
    merge_gap_seconds: float,
) -> dict[str, object]:
    """Merge nearby threshold-passing alert windows into episodes.

    An episode is called unmatched only if none of its alert windows overlaps a
    non-NORMAL annotation under this evaluation's window-label rule. This is a
    reporting convention for correlated sliding windows, not a ground-truth
    event detector or a simulation of the dashboard's event logger.
    """

    by_trip: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        by_trip.setdefault(int(row["trip"]), []).append(row)

    episode_counts = {"total": 0, "unmatched_normal": 0, "overlaps_labeled_event": 0}
    per_trip: dict[str, dict[str, int]] = {}
    for trip, trip_rows in sorted(by_trip.items()):
        trip_rows.sort(key=lambda row: float(row["window_start_seconds"]))
        trip_episodes: list[list[dict[str, object]]] = []
        current: list[dict[str, object]] = []
        last_alert_start: float | None = None
        for row in trip_rows:
            category = str(row[category_key])
            confidence = float(row[confidence_key])
            is_alert = category != "NORMAL" and confidence >= threshold
            start = float(row["window_start_seconds"])
            if not is_alert:
                continue
            if (
                current
                and last_alert_start is not None
                and start - last_alert_start > merge_gap_seconds + 1e-9
            ):
                trip_episodes.append(current)
                current = []
            current.append(row)
            last_alert_start = start
        if current:
            trip_episodes.append(current)

        counts = {"total": len(trip_episodes), "unmatched_normal": 0, "overlaps_labeled_event": 0}
        for episode in trip_episodes:
            episode_start = float(episode[0]["window_start_seconds"])
            episode_end = float(episode[-1]["window_start_seconds"])
            episode_timeline = [
                row
                for row in trip_rows
                if episode_start - 1e-9
                <= float(row["window_start_seconds"])
                <= episode_end + 1e-9
            ]
            overlaps_event = any(
                str(row["expected_category"]) != "NORMAL" for row in episode_timeline
            )
            if overlaps_event:
                counts["overlaps_labeled_event"] += 1
            else:
                counts["unmatched_normal"] += 1
        per_trip[str(trip)] = counts
        for key in episode_counts:
            episode_counts[key] += counts[key]

    return {
        **episode_counts,
        "per_trip": per_trip,
        "confidence_threshold": threshold,
        "merge_gap_seconds": merge_gap_seconds,
        "merge_rule": (
            "Within each trip, threshold-passing non-NORMAL decisions whose window-start times "
            f"are at most {merge_gap_seconds:g} seconds apart are one episode, irrespective of "
            "predicted class. An episode is unmatched only when all its alert windows are labelled NORMAL."
        ),
        "interpretation": (
            "Merged offline alert-window episodes, not emitted dashboard events or validated driver events."
        ),
    }


def _summarize_predictions(
    rows: list[dict[str, object]],
    *,
    category_key: str,
    confidence_key: str,
    stride_seconds: float,
    default_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    merge_gap_seconds: float = DEFAULT_EVENT_MERGE_GAP_SECONDS,
) -> dict[str, object]:
    expected = [str(row["expected_category"]) for row in rows]
    predicted = [str(row[category_key]) for row in rows]
    argmax_metrics = multiclass_metrics(expected, predicted, CLASS_NAMES)
    normal_windows = sum(category == "NORMAL" for category in expected)
    normal_decision_seconds = normal_windows * stride_seconds
    argmax_false_positive_windows = sum(
        actual == "NORMAL" and guess != "NORMAL"
        for actual, guess in zip(expected, predicted)
    )
    gated_predictions = [
        str(row[category_key])
        if float(row[confidence_key]) >= default_threshold
        else "NORMAL"
        for row in rows
    ]
    gated_metrics = multiclass_metrics(expected, gated_predictions, CLASS_NAMES)
    gated_false_positive_windows = sum(
        actual == "NORMAL" and guess != "NORMAL"
        for actual, guess in zip(expected, gated_predictions)
    )
    normal_minutes = normal_decision_seconds / 60.0
    episodes = _alert_episode_summary(
        rows,
        category_key=category_key,
        confidence_key=confidence_key,
        threshold=default_threshold,
        merge_gap_seconds=merge_gap_seconds,
    )
    argmax_metrics.update(
        {
            "argmax_false_positive_windows": argmax_false_positive_windows,
            "false_positive_windows_at_default_threshold": gated_false_positive_windows,
            "normal_labeled_decision_exposure_seconds": round(normal_decision_seconds, 3),
            "false_positive_windows_per_normal_labeled_decision_minute_argmax": (
                round(argmax_false_positive_windows / normal_minutes, 4) if normal_minutes else 0.0
            ),
            "false_positive_windows_per_normal_labeled_decision_minute_at_default_threshold": (
                round(gated_false_positive_windows / normal_minutes, 4) if normal_minutes else 0.0
            ),
            "default_threshold": default_threshold,
            "default_threshold_classification_metrics": gated_metrics,
            "default_threshold_alert_episodes": episodes,
        }
    )
    return argmax_metrics


def _summarize_always_normal(
    rows: list[dict[str, object]],
    *,
    stride_seconds: float,
) -> dict[str, object]:
    expected = [str(row["expected_category"]) for row in rows]
    predicted = ["NORMAL"] * len(rows)
    exposure = sum(category == "NORMAL" for category in expected) * stride_seconds
    return {
        **multiclass_metrics(expected, predicted, CLASS_NAMES),
        "argmax_false_positive_windows": 0,
        "false_positive_windows_at_default_threshold": 0,
        "normal_labeled_decision_exposure_seconds": round(exposure, 3),
        "false_positive_windows_per_normal_labeled_decision_minute_argmax": 0.0,
        "false_positive_windows_per_normal_labeled_decision_minute_at_default_threshold": 0.0,
        "default_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        "default_threshold_alert_episodes": {
            "total": 0,
            "unmatched_normal": 0,
            "overlaps_labeled_event": 0,
            "merge_gap_seconds": DEFAULT_EVENT_MERGE_GAP_SECONDS,
            "interpretation": "Always-NORMAL creates no alert episodes; it misses every non-NORMAL labelled window.",
        },
    }


def evaluate(
    *,
    raw_directory: Path,
    trips: tuple[int, ...],
    stride_seconds: float,
    max_windows: int = 0,
    plan_only: bool = False,
    checkpoint_path: Path = DEFAULT_PRIMUS_CHECKPOINT,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    if not trips or any(trip not in (1, 2, 3) for trip in trips):
        raise ValueError("trips must contain one or more of the public trips 1, 2, and 3")
    raw_directory = Path(raw_directory)
    trip_data = {
        trip: load_public_trip(trip, raw_directory)
        for trip in (1, 2, 3)
    }
    schedules = {
        trip: window_schedule(trip_data[trip], stride_seconds, max_windows=max_windows)
        for trip in trips
    }
    planned_counts = {str(trip): len(schedule) for trip, schedule in schedules.items()}
    sensor_duration = {
        str(trip): round(max(0.0, _sensor_bounds(trip_data[trip])[1] - _sensor_bounds(trip_data[trip])[0]), 3)
        for trip in trips
    }
    all_events = [event for trip in (1, 2, 3) for event in trip_data[trip].events]
    event_targets = np.asarray([event.class_index for event in all_events], dtype=np.int64)
    event_trip_ids = np.asarray([event.trip for event in all_events], dtype=np.int64)
    event_class_counts = _training_class_counts(event_targets)
    normal_event_count = int(np.sum(event_targets == CLASS_NAMES.index("NORMAL")))
    event_count_by_trip = {
        str(trip): sum(event.trip == trip for event in all_events)
        for trip in (1, 2, 3)
    }
    base_report: dict[str, object] = {
        "status": "leave_one_trip_out_continuous_public_evaluation",
        "live_phone_telemetry_used": False,
        "raw_sensor_values_saved": False,
        "source_labels_exposed_to_dashboard": False,
        "raw_directory": str(raw_directory),
        "trips": list(trips),
        "windowing": {
            "duration_seconds": WINDOW_SECONDS,
            "stride_seconds": stride_seconds,
            "decision_time_exposure_seconds_per_labeled_normal_window": stride_seconds,
            "label_rule": (
                "Assign the category of the overlapping source annotation with greatest overlap; "
                "ties use source CSV order. Label an unannotated window NORMAL. A window records "
                "whether it overlaps a source annotation and how many annotations it overlaps."
            ),
            "windows_are_overlapping": stride_seconds < WINDOW_SECONDS,
        },
        "planned_windows_by_trip": planned_counts,
        "sensor_duration_seconds_by_trip": sensor_duration,
        "source_annotation_interval_overlaps_by_trip": {
            str(trip): _source_annotation_overlap_count(trip_data[trip])
            for trip in (1, 2, 3)
        },
        "event_centered_training_examples": {
            "total": len(all_events),
            "by_trip": event_count_by_trip,
            "class_counts": event_class_counts,
            "normal_class_source": (
                f"Only {normal_event_count} explicitly annotated non-aggressive-event intervals train NORMAL; "
                "unannotated background windows are not included in event-window training."
            ),
        },
        "confidence_gate": {
            "application_default": DEFAULT_CONFIDENCE_THRESHOLD,
            "smoothing": "This evaluator reports per-window decisions and post-hoc merged episodes; it does not apply the dashboard's event smoother.",
        },
        "alert_episode_rule": {
            "merge_gap_seconds": DEFAULT_EVENT_MERGE_GAP_SECONDS,
            "meaning": (
                "Consecutive threshold-passing non-NORMAL windows within a trip and separated by no more than "
                f"{DEFAULT_EVENT_MERGE_GAP_SECONDS:g} seconds are grouped; see each model metric for the full rule."
            ),
        },
        "random_forest_baseline_included": True,
        "always_normal_baseline_included": True,
    }

    if plan_only:
        label_counts = {name: 0 for name in CLASS_NAMES}
        label_sources: dict[str, int] = {}
        multi_overlap_windows = 0
        greatest_overlap_tie_windows = 0
        for schedule in schedules.values():
            for row in schedule:
                label_counts[str(row["expected_category"])] += 1
                source = str(row["label_source"])
                label_sources[source] = label_sources.get(source, 0) + 1
                multi_overlap_windows += int(int(row["overlapping_annotation_count"]) > 1)
                greatest_overlap_tie_windows += int(int(row["greatest_overlap_tie_count"]) > 1)
        base_report["evaluation"] = "plan_only"
        base_report["planned_expected_class_counts"] = label_counts
        base_report["label_sources"] = label_sources
        base_report["windows_overlapping_multiple_annotations"] = multi_overlap_windows
        base_report["windows_with_greatest_overlap_ties"] = greatest_overlap_tie_windows
        return base_report

    event_inputs: list[np.ndarray] = []
    for event in all_events:
        window, _start, _end = make_fixed_primus_event_window(event, trip_data[event.trip])
        event_inputs.append(window)
    primus_event_inputs = np.stack(event_inputs).astype(np.float32, copy=False)
    encoder = PrimusIMUEncoder()
    encoder.load_checkpoint(Path(checkpoint_path))
    event_embeddings, event_encoder_ms = extract_primus_embeddings(encoder, primus_event_inputs)

    event_windows_by_trip = {
        trip: prepare_trip_windows(trip_data[trip], WindowConfig())
        for trip in (1, 2, 3)
    }
    fold_summaries: list[dict[str, object]] = []
    records: list[dict[str, object]] = []

    for test_trip in trips:
        test_mask = event_trip_ids == test_trip
        train_mask = ~test_mask
        train_trip_ids = sorted(set(int(value) for value in event_trip_ids[train_mask]))
        if test_trip in train_trip_ids:
            raise RuntimeError(f"Trip {test_trip} leaked into its own training fold")

        primus_classifier = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=5_000,
                        class_weight="balanced",
                        random_state=seed + test_trip,
                    ),
                ),
            ]
        )
        primus_fit_started = perf_counter()
        primus_classifier.fit(event_embeddings[train_mask], event_targets[train_mask])
        primus_fit_seconds = perf_counter() - primus_fit_started

        train_rf_inputs = np.concatenate(
            [event_windows_by_trip[trip].inputs for trip in train_trip_ids], axis=0
        )
        train_rf_targets = np.concatenate(
            [event_windows_by_trip[trip].targets for trip in train_trip_ids], axis=0
        )
        rf_normalization = Normalization.fit(train_rf_inputs)
        normalized_train_rf = rf_normalization.apply(train_rf_inputs)
        rf_features = extract_classical_features(normalized_train_rf)
        rf_classifier = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        rf_fit_started = perf_counter()
        rf_classifier.fit(rf_features, train_rf_targets)
        rf_fit_seconds = perf_counter() - rf_fit_started

        test_windows = schedules[test_trip]
        if not test_windows:
            raise ValueError(f"Trip {test_trip} has no scheduled evaluation windows")
        raw_test_primus = np.stack(
            [
                make_fixed_primus_window_at(
                    trip_data[test_trip],
                    float(row["window_start_seconds"]),
                )[0]
                for row in test_windows
            ]
        ).astype(np.float32, copy=False)
        primus_embeddings, primus_test_encoder_ms = extract_primus_embeddings(
            encoder, raw_test_primus
        )
        primus_head_started = perf_counter()
        primus_predicted, primus_confidence = _aligned_probabilities(
            primus_classifier, primus_embeddings
        )
        primus_head_ms = (perf_counter() - primus_head_started) * 1_000 / len(test_windows)

        rf_test_inputs = np.stack(
            [
                _make_random_forest_window(
                    trip_data[test_trip],
                    float(row["window_start_seconds"]),
                    float(row["window_end_seconds"]),
                    int(train_rf_inputs.shape[2]),
                )
                for row in test_windows
            ]
        ).astype(np.float32, copy=False)
        normalized_test_rf = rf_normalization.apply(rf_test_inputs)
        rf_test_features = extract_classical_features(normalized_test_rf)
        rf_infer_started = perf_counter()
        rf_predicted, rf_confidence = _aligned_probabilities(rf_classifier, rf_test_features)
        rf_inference_ms = (perf_counter() - rf_infer_started) * 1_000 / len(test_windows)

        for index, row in enumerate(test_windows):
            records.append(
                {
                    **row,
                    "primus_category": primus_predicted[index],
                    "primus_confidence": round(primus_confidence[index], 6),
                    "primus_inference_ms_per_window": round(
                        primus_test_encoder_ms + primus_head_ms, 6
                    ),
                    "random_forest_category": rf_predicted[index],
                    "random_forest_confidence": round(rf_confidence[index], 6),
                    "random_forest_inference_ms_per_window_batched": round(rf_inference_ms, 6),
                }
            )

        fold_summaries.append(
            {
                "test_trip": test_trip,
                "train_trips": train_trip_ids,
                "train_event_windows": int(train_mask.sum()),
                "test_continuous_windows": len(test_windows),
                "primus_train_class_counts": _training_class_counts(event_targets[train_mask]),
                "random_forest_train_class_counts": _training_class_counts(train_rf_targets),
                "normalization_fit_on_test_trip": False,
                "primus_head_fit_seconds": round(primus_fit_seconds, 4),
                "random_forest_fit_seconds": round(rf_fit_seconds, 4),
            }
        )

    records.sort(key=lambda row: (int(row["trip"]), float(row["window_start_seconds"])))
    label_counts = {name: 0 for name in CLASS_NAMES}
    label_sources: dict[str, int] = {}
    multi_overlap_windows = 0
    greatest_overlap_tie_windows = 0
    for row in records:
        label_counts[str(row["expected_category"])] += 1
        source = str(row["label_source"])
        label_sources[source] = label_sources.get(source, 0) + 1
        multi_overlap_windows += int(int(row["overlapping_annotation_count"]) > 1)
        greatest_overlap_tie_windows += int(int(row["greatest_overlap_tie_count"]) > 1)

    per_trip_metrics: dict[str, dict[str, object]] = {}
    for trip in trips:
        trip_rows = [row for row in records if int(row["trip"]) == trip]
        per_trip_metrics[str(trip)] = {
            "window_count": len(trip_rows),
            "expected_class_counts": {
                name: sum(row["expected_category"] == name for row in trip_rows)
                for name in CLASS_NAMES
            },
            "primus": _summarize_predictions(
                trip_rows,
                category_key="primus_category",
                confidence_key="primus_confidence",
                stride_seconds=stride_seconds,
            ),
            "random_forest": _summarize_predictions(
                trip_rows,
                category_key="random_forest_category",
                confidence_key="random_forest_confidence",
                stride_seconds=stride_seconds,
            ),
            "always_normal": _summarize_always_normal(
                trip_rows,
                stride_seconds=stride_seconds,
            ),
        }

    expected = [str(row["expected_category"]) for row in records]
    background_exposure = sum(
        1 for row in records if row["label_source"] == "unannotated_background"
    ) * stride_seconds
    base_report.update(
        {
            "evaluation": "outer leave-one-trip-out continuous window predictions",
            "planned_expected_class_counts": label_counts,
            "label_sources": label_sources,
            "windows_overlapping_multiple_annotations": multi_overlap_windows,
            "windows_with_greatest_overlap_ties": greatest_overlap_tie_windows,
            "unannotated_background_window_exposure_seconds": round(background_exposure, 3),
            "training_window_protocol": {
                "primus": (
                    "Frozen external encoder; LogisticRegression with StandardScaler fitted on "
                    "five-second, 200 Hz event-centred windows from the two training trips only."
                ),
                "random_forest": (
                    "RandomForestClassifier fitted on whole annotated event intervals resampled to "
                    "256 samples; channel normalization fit on the two training trips only."
                ),
                "normal_class": (
                    f"Training uses {normal_event_count} explicitly annotated non-aggressive-event intervals. "
                    "It does not contain unannotated background windows used as NORMAL in this test."
                ),
                "family_selection_caveat": (
                    "The PRIMUS logistic head and Random Forest family were previously selected using "
                    "leave-one-trip-out results on these same three trips. This evaluation keeps the "
                    "families fixed and excludes each test trip from fold fitting, but selection bias "
                    "from the earlier family choice remains."
                ),
                "deployment_window_mismatch": (
                    "Both heads train on annotated event windows and test on sliding windows. The RF "
                    "training windows vary in duration; its test windows are five seconds."
                ),
            },
            "train_test_separation": {
                "method": "Each test trip is excluded from both model heads and all fold normalization.",
                "folds": fold_summaries,
                "head_family_selection_nested": False,
            },
            "metrics": _summarize_predictions(
                records,
                category_key="primus_category",
                confidence_key="primus_confidence",
                stride_seconds=stride_seconds,
            ),
            "random_forest_baseline_metrics": _summarize_predictions(
                records,
                category_key="random_forest_category",
                confidence_key="random_forest_confidence",
                stride_seconds=stride_seconds,
            ),
            "always_normal_baseline_metrics": _summarize_always_normal(
                records,
                stride_seconds=stride_seconds,
            ),
            "per_trip_metrics": per_trip_metrics,
            "windows": records,
            "limitations": [
                "The event annotations are sparse source labels; no new human labels were created.",
                "Five-second windows overlap at the configured stride. Window errors are correlated decisions, not independent trials.",
                "False-positive window rates use NORMAL-labelled decision-time exposure: NORMAL window count multiplied by stride, not by five-second window length.",
                "Episode counts merge threshold-passing alert windows under the stated rule; they are not validated behavioural events or emitted dashboard events.",
                "NORMAL training examples are explicit non-aggressive annotations; unannotated background windows appear in evaluation, not event-centred training.",
                "The fold-specific scores still have prior model-family selection on the same three trips and a training-window mismatch; treat them as exploratory, not as an independent deployment benchmark.",
                "The dataset contains one driver, one vehicle, and three public trips; this does not establish generalisation to other drivers or vehicles.",
            ],
        }
    )
    return base_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", type=Path, default=DEFAULT_RAW_DIRECTORY)
    parser.add_argument("--trips", nargs="+", type=int, choices=(1, 2, 3), default=(1, 2, 3))
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument("--max-windows", type=int, default=0, help="limit held-out windows per trip; zero means all")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_PRIMUS_CHECKPOINT)
    parser.add_argument("--plan-only", action="store_true", help="schedule and audit labels without loading PRIMUS")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-write", action="store_true", help="print the report without writing it")
    args = parser.parse_args()
    if args.max_windows < 0:
        parser.error("--max-windows cannot be negative")
    report = evaluate(
        raw_directory=args.raw_directory,
        trips=tuple(args.trips),
        stride_seconds=args.stride_seconds,
        max_windows=args.max_windows,
        plan_only=args.plan_only,
        checkpoint_path=args.checkpoint,
    )
    if not args.no_write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Evaluation written: {args.output}")
    print(f"Planned windows: {sum(report['planned_windows_by_trip'].values())}")
    print("Live phone telemetry used: False")
    print("Raw sensor values saved: False")
    if not args.plan_only:
        for name in ("metrics", "random_forest_baseline_metrics", "always_normal_baseline_metrics"):
            result = report[name]
            print(
                f"{name}: accuracy={result['accuracy']:.4f}, "
                f"macro-F1={result['macro_f1_present_classes']:.4f}, "
                f"FP windows={result['false_positive_windows_at_default_threshold']}, "
                f"FP windows/normal-decision-minute="
                f"{result['false_positive_windows_per_normal_labeled_decision_minute_at_default_threshold']:.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
