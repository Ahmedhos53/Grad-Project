import numpy as np
import unittest

from src.telemetry.dataset_loader import LabeledEvent, PublicTelemetryTrip, SensorStream
from src.telemetry.preprocessing import WindowConfig, make_event_window, prepare_trip_windows


class TelemetryPreprocessingTests(unittest.TestCase):
    def setUp(self):
        timestamps = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        acceleration = SensorStream(
            timestamps=timestamps,
            values=np.column_stack((timestamps, timestamps + 10.0, timestamps + 20.0)),
            columns=("ax", "ay", "az"),
        )
        gyroscope = SensorStream(
            timestamps=timestamps,
            values=np.column_stack((timestamps + 30.0, timestamps + 40.0, timestamps + 50.0)),
            columns=("gx", "gy", "gz"),
        )
        event = LabeledEvent(
            trip=1,
            source_label="aggressive braking",
            category="HARD_BRAKE",
            target=4,
            start_seconds=1.0,
            end_seconds=2.0,
            time_sync_seconds=0.5,
            source_sample=7,
        )
        self.trip = PublicTelemetryTrip(1, acceleration, gyroscope, (event,))

    def test_event_window_preserves_six_channel_order_and_fixed_shape(self):
        window = make_event_window(self.trip.events[0], self.trip, WindowConfig(samples_per_window=16))
        self.assertEqual(window.shape, (6, 16))
        np.testing.assert_allclose(window[0], np.linspace(1.0, 2.0, 16))
        np.testing.assert_allclose(window[3], np.linspace(31.0, 32.0, 16))

    def test_prepare_trip_keeps_one_target_per_annotated_event(self):
        prepared = prepare_trip_windows(self.trip, WindowConfig(samples_per_window=16))
        self.assertEqual(prepared.inputs.shape, (1, 6, 16))
        self.assertEqual(prepared.targets.tolist(), [1])
        self.assertEqual(prepared.source_samples.tolist(), [7])
        self.assertEqual(prepared.source_labels, ("aggressive braking",))


if __name__ == "__main__":
    unittest.main()
