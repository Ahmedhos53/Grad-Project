from __future__ import annotations

import unittest

import numpy as np

from src.telemetry.dataset_loader import LabeledEvent, PublicTelemetryTrip, SensorStream
from src.telemetry.primus_encoder import PRIMUS_SAMPLES_PER_WINDOW, PRIMUS_TARGET_HZ
from src.telemetry.primus_transfer import make_fixed_primus_event_window


class PrimusTransferTests(unittest.TestCase):
    @staticmethod
    def _trip() -> PublicTelemetryTrip:
        timestamps = np.arange(0.0, 10.0 + 0.0025, 0.0025, dtype=np.float64)
        acceleration = np.column_stack((timestamps, timestamps * 2.0, timestamps * 3.0))
        gyroscope = np.column_stack((-timestamps, timestamps + 4.0, timestamps - 2.0))
        event = LabeledEvent(
            trip=1,
            source_label="aggressive braking",
            category="HARD_BRAKE",
            target=1,
            start_seconds=3.0,
            end_seconds=5.0,
            time_sync_seconds=0.0,
            source_sample=1,
        )
        return PublicTelemetryTrip(
            trip=1,
            acceleration=SensorStream(timestamps, acceleration, ("ax", "ay", "az")),
            gyroscope=SensorStream(timestamps, gyroscope, ("gx", "gy", "gz")),
            events=(event,),
        )

    def test_fixed_window_is_true_five_seconds_at_200_hz(self) -> None:
        trip = self._trip()

        window, start, end = make_fixed_primus_event_window(trip.events[0], trip)

        self.assertEqual(window.shape, (6, PRIMUS_SAMPLES_PER_WINDOW))
        self.assertEqual(start, 1.5)
        self.assertEqual(end, 6.5)
        self.assertAlmostEqual(float(window[0, 0]), 1.5)
        self.assertAlmostEqual(
            float(window[0, -1]),
            1.5 + (PRIMUS_SAMPLES_PER_WINDOW - 1) / PRIMUS_TARGET_HZ,
            places=6,
        )
        self.assertAlmostEqual(float(window[3, 0]), -1.5)

    def test_event_and_trip_must_match(self) -> None:
        trip = self._trip()
        wrong_event = LabeledEvent(
            trip=2,
            source_label="aggressive braking",
            category="HARD_BRAKE",
            target=1,
            start_seconds=3.0,
            end_seconds=5.0,
            time_sync_seconds=0.0,
            source_sample=1,
        )

        with self.assertRaisesRegex(ValueError, "do not match"):
            make_fixed_primus_event_window(wrong_event, trip)


if __name__ == "__main__":
    unittest.main()
