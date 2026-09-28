import unittest

import numpy as np

from src.evaluation.metrics import binary_metrics, multiclass_metrics
from src.telemetry.dataset_loader import LabeledEvent, PublicTelemetryTrip, SensorStream
from src.telemetry.primus_transfer import make_fixed_primus_window_at


class EvaluationProtocolTests(unittest.TestCase):
    def test_binary_metrics_report_false_positive_and_false_negative_rates(self):
        metrics = binary_metrics(
            [True, True, False, False],
            [True, False, True, False],
        )
        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertEqual(metrics["false_positive_rate"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)

    def test_multiclass_metrics_keep_class_order_and_confusion_matrix(self):
        report = multiclass_metrics(
            ["NORMAL", "HARD_BRAKE", "NORMAL"],
            ["NORMAL", "NORMAL", "HARD_BRAKE"],
            ["NORMAL", "HARD_BRAKE"],
        )
        self.assertEqual(report["class_names"], ["NORMAL", "HARD_BRAKE"])
        self.assertEqual(report["confusion_matrix"], [[1, 1], [1, 0]])

    def test_arbitrary_primus_window_has_fixed_shape_and_time_bounds(self):
        timestamps = np.arange(0.0, 12.0, 0.01, dtype=np.float64)
        values = np.stack(
            (
                np.sin(timestamps),
                np.cos(timestamps),
                timestamps * 0.0,
            ),
            axis=1,
        )
        acceleration = SensorStream(timestamps, values, ("linear acceleration x", "linear acceleration y", "linear acceleration z"))
        gyroscope = SensorStream(timestamps, values, ("gyroscope x", "gyroscope y", "gyroscope z"))
        event = LabeledEvent(
            trip=1,
            source_label="normal",
            category="NORMAL",
            target=0,
            start_seconds=2.0,
            end_seconds=3.0,
            time_sync_seconds=0.0,
            source_sample=200,
        )
        trip = PublicTelemetryTrip(1, acceleration, gyroscope, (event,))

        window, start, end = make_fixed_primus_window_at(trip, 1.0)

        self.assertEqual(window.shape, (6, 1000))
        self.assertAlmostEqual(start, 1.0)
        self.assertAlmostEqual(end, 6.0)
        self.assertTrue(np.isfinite(window).all())


if __name__ == "__main__":
    unittest.main()

