import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import wave

import numpy as np

from src.audio.speech_transcriber import TranscriptResult
from tools.evaluate_consented_audio_manifest import (
    _resolve_local_file,
    evaluate_manifest,
    evaluate_records,
)


class FakeTranscriber:
    def transcribe(self, samples, *, sample_rate):
        return TranscriptResult(
            text="private reference phrase",
            language="en",
            language_probability=0.99,
            processing_latency_ms=5,
        )


class FakeSafetyClassifier:
    rules = {"PROFANITY": {}}

    def analyze(self, text, *, language):
        return SimpleNamespace(categories=())


class ConsentedAudioEvaluatorTests(unittest.TestCase):
    def test_aggregate_report_scores_but_never_contains_reference_or_transcript(self):
        samples = np.concatenate(
            (
                np.full(3_200, 0.1, dtype=np.float32),
                np.zeros(12_800, dtype=np.float32),
            )
        )
        report = evaluate_records(
            [
                {
                    "sample_id": "sample_01",
                    "audio_path": Path("private.wav"),
                    "language": "en",
                    "reference_text": "private reference phrase",
                    "expected_safety_categories": set(),
                    "expected_loudness_level": "NORMAL",
                    "microphone_type": "laptop",
                    "noise_condition": "quiet",
                    "samples": samples,
                }
            ],
            transcriber=FakeTranscriber(),
            safety_classifier=FakeSafetyClassifier(),
            calibration_samples=samples,
        )

        encoded = json.dumps(report)
        self.assertEqual(report["recognition_overall"]["wer"], 0.0)
        self.assertEqual(report["recognition_overall"]["cer"], 0.0)
        self.assertEqual(report["sample_count"], 1)
        self.assertEqual(report["loudness_agreement"]["exact_agreement"], 1.0)
        self.assertEqual(report["coverage"]["loudness_reference_samples"], 1)
        self.assertFalse(report["coverage"]["background_noise_present"])
        self.assertEqual(report["coverage"]["missing_microphone_types"], ["headset"])
        self.assertNotIn("private reference phrase", encoded)
        self.assertNotIn("sample_01", encoded)
        self.assertNotIn("private.wav", encoded)
        self.assertTrue(report["privacy"]["raw_audio_saved"] is False)
        self.assertTrue(report["privacy"]["transcript_text_saved"] is False)

    def test_manifest_run_reads_audio_without_modifying_or_disclosing_it(self):
        template_path = (
            Path(__file__).resolve().parents[1]
            / "docs"
            / "evaluation"
            / "audio_manifest_template.csv"
        )
        with template_path.open("r", encoding="utf-8", newline="") as stream:
            template_reader = csv.reader(stream)
            self.assertEqual(
                next(template_reader),
                [
                    "sample_id",
                    "audio_path",
                    "language",
                    "reference_text",
                    "expected_safety_categories",
                    "expected_loudness_level",
                    "microphone_type",
                    "noise_condition",
                ],
            )
            self.assertEqual(list(template_reader), [])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = np.concatenate(
                (
                    np.full(3_200, 3_200, dtype=np.int16),
                    np.zeros(12_800, dtype=np.int16),
                )
            )
            audio_path = root / "consented.wav"
            with wave.open(str(audio_path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16_000)
                output.writeframes(samples.tobytes())
            before_hash = hashlib.sha256(audio_path.read_bytes()).hexdigest()
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=(
                        "sample_id",
                        "audio_path",
                        "language",
                        "reference_text",
                        "expected_safety_categories",
                        "expected_loudness_level",
                        "microphone_type",
                        "noise_condition",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_id": "private_sample_99",
                        "audio_path": "consented.wav",
                        "language": "en",
                        "reference_text": "private reference phrase",
                        "expected_safety_categories": "",
                        "expected_loudness_level": "",
                        "microphone_type": "headset",
                        "noise_condition": "quiet",
                    }
                )

            report = evaluate_manifest(
                manifest,
                transcriber=FakeTranscriber(),
                safety_classifier=FakeSafetyClassifier(),
            )
            encoded = json.dumps(report)

            self.assertEqual(report["sample_count"], 1)
            self.assertEqual(report["coverage"]["missing_language_conditions"], ["ar", "mixed"])
            self.assertFalse(report["coverage"]["background_noise_present"])
            self.assertNotIn("private_sample_99", encoded)
            self.assertNotIn("private reference phrase", encoded)
            self.assertNotIn("consented.wav", encoded)
            self.assertEqual(hashlib.sha256(audio_path.read_bytes()).hexdigest(), before_hash)

    def test_audio_paths_cannot_escape_the_manifest_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_dir = root / "manifest"
            manifest_dir.mkdir()
            outside_file = root / "outside.wav"
            outside_file.write_bytes(b"test")

            with self.assertRaisesRegex(ValueError, "outside the manifest directory"):
                _resolve_local_file(manifest_dir, "../outside.wav")


if __name__ == "__main__":
    unittest.main()
