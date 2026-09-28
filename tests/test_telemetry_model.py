import unittest

import torch

from src.telemetry.model import TelemetryCNN


class TelemetryModelTests(unittest.TestCase):
    def test_compact_cnn_accepts_six_channel_windows(self):
        model = TelemetryCNN()
        self.assertEqual(model(torch.zeros((3, 6, 256))).shape, (3, 5))


if __name__ == "__main__":
    unittest.main()
