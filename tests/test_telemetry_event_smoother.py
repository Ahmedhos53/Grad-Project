import unittest

from src.telemetry.event_smoother import TelemetryEventSmoother, TelemetrySmoothingConfig
from src.telemetry.inference import TelemetryPrediction


def prediction(category: str, confidence: float) -> TelemetryPrediction:
    return TelemetryPrediction(category, confidence, {category: confidence})


class TelemetryEventSmootherTests(unittest.TestCase):
    def test_requires_confidence_and_consecutive_predictions_before_event(self):
        smoother = TelemetryEventSmoother(TelemetrySmoothingConfig(minimum_confidence=0.6, confirm_windows=2, cooldown_ms=100))
        self.assertFalse(smoother.update(prediction("HARD_BRAKE", 0.5), 0).active)
        self.assertFalse(smoother.update(prediction("HARD_BRAKE", 0.9), 1).active)
        emitted = smoother.update(prediction("HARD_BRAKE", 0.8), 2)
        self.assertTrue(emitted.active)
        self.assertEqual(emitted.category, "HARD_BRAKE")
        self.assertFalse(smoother.update(prediction("HARD_BRAKE", 0.9), 3).active)


if __name__ == "__main__":
    unittest.main()
