import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.telemetry.dataset_loader import load_public_trip


class PublicTelemetryDatasetLoaderTests(unittest.TestCase):
    def _write_sensor(self, path: Path, header: list[str]) -> None:
        with path.open("w", newline="", encoding="utf-8") as file_handle:
            writer = csv.writer(file_handle)
            writer.writerow(header)
            writer.writerow([0.0, 1.0, 2.0, 3.0])
            writer.writerow([0.1, 4.0, 5.0, 6.0])

    def _write_trip(self, raw_directory: Path, trip: int) -> None:
        self._write_sensor(
            raw_directory / f"Linear_Acceleration_{trip}.csv",
            ["Time (s)", "Linear Acceleration x (m/s^2)", "Linear Acceleration y (m/s^2)", "Linear Acceleration z (m/s^2)"],
        )
        self._write_sensor(
            raw_directory / f"Gyroscope_{trip}.csv",
            ["Time (s)", "Gyroscope x (rad/s)", "Gyroscope y (rad/s)", "Gyroscope z (rad/s)"],
        )
        with (raw_directory / f"Labeled_events_{trip}.csv").open("w", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=["start", "end", "event", "target", "timesync", "sample"])
            writer.writeheader()
            writer.writerow({"start": 1.0, "end": 2.0, "event": "aggressive braking", "target": 4, "timesync": 0.5, "sample": 0})

    def test_load_public_trip_keeps_sensor_and_annotation_time_bases_explicit(self):
        with TemporaryDirectory() as temporary:
            raw_directory = Path(temporary)
            for trip in (1, 2, 3):
                self._write_trip(raw_directory, trip)
            loaded = load_public_trip(1, raw_directory)

            self.assertEqual(loaded.acceleration.values.shape, (2, 3))
            self.assertEqual(loaded.events[0].category, "HARD_BRAKE")
            self.assertEqual(loaded.events[0].sensor_start_seconds, 1.0)
            self.assertEqual(loaded.events[0].sensor_end_seconds, 2.0)


if __name__ == "__main__":
    unittest.main()
