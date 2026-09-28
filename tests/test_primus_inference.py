from __future__ import annotations

import unittest

import numpy as np

from src.telemetry.primus_inference import prediction_from_probabilities


class PrimusInferenceTests(unittest.TestCase):
    def test_highest_probability_maps_to_event_category(self) -> None:
        prediction = prediction_from_probabilities(np.array([0.05, 0.10, 0.70, 0.10, 0.05]))

        self.assertEqual(prediction.category, "RAPID_ACCELERATION")
        self.assertAlmostEqual(prediction.confidence, 0.70)
        self.assertAlmostEqual(sum(prediction.probabilities), 1.0)

    def test_invalid_probability_vector_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            prediction_from_probabilities(np.array([0.5, 0.5]))


if __name__ == "__main__":
    unittest.main()
