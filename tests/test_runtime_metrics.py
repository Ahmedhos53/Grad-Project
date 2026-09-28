import unittest

from src.evaluation.runtime_metrics import RuntimeMetrics


class RuntimeMetricsTests(unittest.TestCase):
    def test_summary_is_aggregate_only_and_preserves_stage_statistics(self):
        metrics = RuntimeMetrics()
        metrics.observe("stage", 3.0)
        metrics.observe("stage", 7.0)
        metrics.increment("jobs")
        metrics.set_gauge("queue_depth", 0)

        report = metrics.summary()

        self.assertEqual(report["status"], "aggregate_runtime_metrics")
        self.assertFalse(report["media_saved"])
        self.assertFalse(report["transcript_text_saved"])
        self.assertEqual(report["stages"]["stage"]["count"], 2)
        self.assertEqual(report["stages"]["stage"]["mean_ms"], 5.0)
        self.assertEqual(report["stages"]["stage"]["max_ms"], 7.0)
        self.assertEqual(report["counters"]["jobs"], 1)
        self.assertEqual(report["gauges"]["queue_depth"], 0)

    def test_measure_records_even_when_the_measured_operation_raises(self):
        metrics = RuntimeMetrics()

        with self.assertRaises(RuntimeError):
            with metrics.measure("failing_stage"):
                raise RuntimeError("controlled test failure")

        stage = metrics.summary()["stages"]["failing_stage"]
        self.assertEqual(stage["count"], 1)
        self.assertGreaterEqual(stage["max_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
