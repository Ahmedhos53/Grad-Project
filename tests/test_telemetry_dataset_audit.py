import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.telemetry.dataset_audit import audit_driving_events_dataset


class TelemetryDatasetAuditTests(unittest.TestCase):
    def _write_trip(self, raw_directory: Path, number: int) -> None:
        (raw_directory / f"Linear_Acceleration_{number}.csv").write_text(
            "timestamp,x,y,z\n0.0,0.1,0.2,0.3\n0.1,0.4,0.5,0.6\n",
            encoding="utf-8",
        )
        (raw_directory / f"Gyroscope_{number}.csv").write_text(
            "timestamp,x,y,z\n0.0,1,2,3\n0.1,4,5,6\n",
            encoding="utf-8",
        )
        (raw_directory / f"Labeled_events_{number}.csv").write_text(
            "event,start,end\nNormal,0.0,0.1\n", encoding="utf-8"
        )

    def test_audit_writes_schema_and_sampling_summary(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_directory = root / "raw"
            output_directory = root / "output"
            raw_directory.mkdir()
            for number in (1, 2, 3):
                self._write_trip(raw_directory, number)

            report = audit_driving_events_dataset(raw_directory, output_directory)

            self.assertTrue(report["local_recordings_excluded"])
            acceleration = report["trips"][0]["files"]["linear_acceleration"]
            self.assertEqual(acceleration["row_count"], 2)
            self.assertAlmostEqual(acceleration["timing"]["estimated_sampling_hz"], 10.0)
            report_path = output_directory / "dataset_audit.json"
            self.assertTrue(report_path.exists())
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8"))["trips"][0]["trip"], 1)

    def test_audit_reports_missing_expected_public_files(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = audit_driving_events_dataset(root / "raw", root / "output")
            self.assertTrue(report["trips"][0]["files"]["labels"]["missing"])


if __name__ == "__main__":
    unittest.main()
