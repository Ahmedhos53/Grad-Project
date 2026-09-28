from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from src.telemetry.training import build_leave_one_trip_out_fold, extract_classical_features


class TelemetryTrainingTests(unittest.TestCase):
    def _write_trip(self, directory: Path, trip: int, target: int) -> None:
        inputs = np.full((3, 6, 16), float(trip), dtype=np.float32)
        np.savez_compressed(directory / f"trip_{trip}_windows.npz", inputs=inputs, targets=np.asarray([target, target, target]))

    def test_held_out_trip_does_not_influence_normalization(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self._write_trip(directory, 1, 0)
            self._write_trip(directory, 2, 1)
            self._write_trip(directory, 3, 2)
            fold = build_leave_one_trip_out_fold(directory, test_trip=3)

            self.assertEqual(fold.test_targets.tolist(), [2, 2, 2])
            self.assertTrue(np.allclose(fold.train_inputs.mean(axis=(0, 2)), 0.0))
            self.assertGreater(float(fold.test_inputs.mean()), 1.0)

    def test_classical_features_are_one_row_per_window(self):
        inputs = np.ones((4, 6, 16), dtype=np.float32)
        features = extract_classical_features(inputs)
        self.assertEqual(features.shape, (4, 36))


if __name__ == "__main__":
    unittest.main()
