from __future__ import annotations

import unittest

import numpy as np

from src.telemetry.primus_encoder import (
    PRIMUS_EMBEDDING_SIZE,
    PRIMUS_SAMPLES_PER_WINDOW,
    PrimusIMUEncoder,
    resample_imu_window,
)


class PrimusEncoderLogicTests(unittest.TestCase):
    def test_normalized_time_resampling_keeps_six_channels_and_endpoints(self) -> None:
        source = np.array(
            [
                [0.0, 1.0, 4.0],
                [10.0, 20.0, 30.0],
                [1.0, 1.0, 1.0],
                [-2.0, 0.0, 2.0],
                [5.0, 6.0, 7.0],
                [9.0, 7.0, 3.0],
            ],
            dtype=np.float32,
        )

        result = resample_imu_window(source, target_samples=7)

        self.assertEqual(result.shape, (6, 7))
        np.testing.assert_allclose(result[:, 0], source[:, 0])
        np.testing.assert_allclose(result[:, -1], source[:, -1])

    def test_encoder_produces_one_embedding_per_six_channel_window(self) -> None:
        encoder = PrimusIMUEncoder()
        inputs = np.zeros((2, 6, PRIMUS_SAMPLES_PER_WINDOW), dtype=np.float32)

        embeddings = encoder.encode(inputs)

        self.assertEqual(embeddings.shape, (2, PRIMUS_EMBEDDING_SIZE))
        self.assertTrue(np.isfinite(embeddings).all())


if __name__ == "__main__":
    unittest.main()
