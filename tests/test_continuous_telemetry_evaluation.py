import unittest

import numpy as np

from tools.evaluate_continuous_telemetry import (
    _alert_episode_summary,
    _expected_category,
    _summarize_predictions,
)
from src.telemetry.dataset_loader import LabeledEvent, PublicTelemetryTrip, SensorStream


class ContinuousTelemetryEvaluationTests(unittest.TestCase):
    def _trip(self):
        timestamps = [float(value) for value in range(11)]
        values = [[float(value), 0.0, 0.0] for value in range(11)]
        stream = SensorStream(
            timestamps=np.asarray(timestamps, dtype=np.float64),
            values=np.asarray(values, dtype=np.float64),
            columns=("x", "y", "z"),
        )
        event = LabeledEvent(
            trip=1,
            source_label="aggressive braking",
            category="HARD_BRAKE",
            target=1,
            start_seconds=4.0,
            end_seconds=6.0,
            time_sync_seconds=0.0,
            source_sample=1,
        )
        return PublicTelemetryTrip(1, stream, stream, (event,))

    def test_unannotated_windows_and_annotation_windows_have_distinct_label_sources(self):
        trip = self._trip()
        background = _expected_category(trip, 0.0, 2.0)
        annotated = _expected_category(trip, 4.0, 6.0)

        self.assertEqual(background, ("NORMAL", 0.0, "unannotated_background", 0, 0))
        self.assertEqual(annotated, ("HARD_BRAKE", 2.0, "aggressive braking", 1, 1))

    def test_greatest_overlap_tie_keeps_source_order_and_reports_it(self):
        trip = self._trip()
        first = LabeledEvent(1, "aggressive braking", "HARD_BRAKE", 1, 4.0, 5.0, 0.0, 1)
        second = LabeledEvent(1, "aggressive right-turn", "AGGRESSIVE_TURN", 3, 5.0, 6.0, 0.0, 2)
        two_events = PublicTelemetryTrip(
            1, trip.acceleration, trip.gyroscope, (first, second)
        )

        label = _expected_category(two_events, 3.0, 7.0)

        self.assertEqual(label, ("HARD_BRAKE", 1.0, "aggressive braking", 2, 2))

    def test_overlap_exposure_uses_stride_not_window_duration(self):
        rows = [
            {"trip": 1, "window_start_seconds": 0.0, "expected_category": "NORMAL", "primus_category": "HARD_BRAKE", "primus_confidence": 0.8},
            {"trip": 1, "window_start_seconds": 1.0, "expected_category": "NORMAL", "primus_category": "HARD_BRAKE", "primus_confidence": 0.8},
            {"trip": 1, "window_start_seconds": 2.0, "expected_category": "HARD_BRAKE", "primus_category": "HARD_BRAKE", "primus_confidence": 0.9},
        ]

        result = _summarize_predictions(
            rows,
            category_key="primus_category",
            confidence_key="primus_confidence",
            stride_seconds=1.0,
            default_threshold=0.55,
        )

        self.assertEqual(result["normal_labeled_decision_exposure_seconds"], 2.0)
        self.assertEqual(result["false_positive_windows_at_default_threshold"], 2)
        self.assertEqual(
            result["false_positive_windows_per_normal_labeled_decision_minute_at_default_threshold"],
            60.0,
        )

    def test_nearby_alert_windows_merge_and_event_in_the_gap_marks_episode_overlap(self):
        rows = [
            {"trip": 1, "window_start_seconds": 0.0, "expected_category": "NORMAL", "primus_category": "HARD_BRAKE", "primus_confidence": 0.8},
            {"trip": 1, "window_start_seconds": 1.0, "expected_category": "NORMAL", "primus_category": "HARD_BRAKE", "primus_confidence": 0.8},
            {"trip": 1, "window_start_seconds": 2.0, "expected_category": "HARD_BRAKE", "primus_category": "NORMAL", "primus_confidence": 0.9},
            {"trip": 1, "window_start_seconds": 3.0, "expected_category": "NORMAL", "primus_category": "HARD_BRAKE", "primus_confidence": 0.8},
        ]

        episodes = _alert_episode_summary(
            rows,
            category_key="primus_category",
            confidence_key="primus_confidence",
            threshold=0.55,
            merge_gap_seconds=3.0,
        )

        self.assertEqual(episodes["total"], 1)
        self.assertEqual(episodes["overlaps_labeled_event"], 1)
        self.assertEqual(episodes["unmatched_normal"], 0)


if __name__ == "__main__":
    unittest.main()
