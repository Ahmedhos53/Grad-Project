import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.audio.loudness_analyzer import LoudnessAnalyzer
from src.audio.speech_analysis_pipeline import EnergyUtteranceGate, SpeechAnalysisPipeline
from src.audio.text_normalization import normalize_arabic, normalize_bilingual
from src.audio.text_safety_classifier import BilingualSafetyClassifier
from src.audio.transcript_store import TranscriptStore, export_readable_transcript


class SpeechAnalysisLogicTests(unittest.TestCase):
    def test_loudness_is_relative_to_session_calibration(self):
        analyzer = LoudnessAnalyzer(confirm_chunks=2, release_chunks=2, high_delta_db=8)
        analyzer.calibrate(np.full(1600, 0.01, dtype=np.float32))

        first = analyzer.update(np.full(1600, 0.10, dtype=np.float32), speech_active=True)
        second = analyzer.update(np.full(1600, 0.10, dtype=np.float32), speech_active=True)

        self.assertEqual(first.voice_level, "VERY HIGH")
        self.assertEqual(second.voice_level, "VERY HIGH")
        self.assertTrue(second.loud_voice_active)
        self.assertTrue(second.calibrated)

    def test_loudness_uses_all_five_requested_levels(self):
        analyzer = LoudnessAnalyzer(low_delta_db=-6, high_delta_db=6, very_high_delta_db=12)
        analyzer.calibrate(np.full(1600, 0.1, dtype=np.float32))  # -20 dBFS normal reference

        quiet = analyzer.update(np.zeros(1600, dtype=np.float32), speech_active=False)
        low = analyzer.update(np.full(1600, 0.02, dtype=np.float32), speech_active=True)
        normal = analyzer.update(np.full(1600, 0.1, dtype=np.float32), speech_active=True)
        high = analyzer.update(np.full(1600, 0.25, dtype=np.float32), speech_active=True)
        very_high = analyzer.update(np.full(1600, 0.7, dtype=np.float32), speech_active=True)

        self.assertEqual(
            [quiet.voice_level, low.voice_level, normal.voice_level, high.voice_level, very_high.voice_level],
            ["QUIET", "LOW", "NORMAL", "HIGH", "VERY HIGH"],
        )

    def test_arabic_normalization_removes_diacritics_and_normalizes_alef(self):
        self.assertEqual(normalize_arabic("سَأَقْتُلُكَ"), "ساقتلك")
        self.assertEqual(normalize_bilingual("  HELLO   أَهْلًا  "), "hello اهلا")

    def test_safety_rules_cover_english_and_arabic_without_loading_a_remote_model(self):
        classifier = BilingualSafetyClassifier(use_model=False)

        english = classifier.analyze("You are a bitch")
        arabic = classifier.analyze("سأقتلك")
        arabic_harassment = classifier.analyze("غبي")

        self.assertTrue(english.flagged)
        self.assertIn("PROFANITY", english.categories)
        self.assertTrue(arabic.flagged)
        self.assertIn("THREAT", arabic.categories)
        self.assertTrue(arabic_harassment.flagged)
        self.assertIn("HARASSMENT", arabic_harassment.categories)

    def test_energy_gate_returns_a_bounded_utterance_after_silence(self):
        gate = EnergyUtteranceGate(start_chunks=1, end_chunks=2)
        speech = np.full(1600, 0.1, dtype=np.float32)
        quiet = np.zeros(1600, dtype=np.float32)

        active, completed = gate.update(speech, -20)
        self.assertTrue(active)
        self.assertIsNone(completed)
        active, completed = gate.update(quiet, -90)
        self.assertTrue(active)
        self.assertIsNone(completed)
        active, completed = gate.update(quiet, -90)
        self.assertFalse(active)
        self.assertIsNotNone(completed)
        self.assertGreaterEqual(len(completed), 1600)

    def test_energy_gate_retains_pre_roll_before_confirmed_speech(self):
        gate = EnergyUtteranceGate(
            active_dbfs=-30.0,
            start_chunks=2,
            pre_roll_chunks=4,
            end_chunks=2,
        )
        quiet_lead_in = np.full(1600, 0.01, dtype=np.float32)
        voiced_one = np.full(1600, 0.1, dtype=np.float32)
        voiced_two = np.full(1600, 0.1, dtype=np.float32)
        silence = np.zeros(1600, dtype=np.float32)

        gate.update(quiet_lead_in, -50.0)
        gate.update(voiced_one, -20.0)
        active, completed = gate.update(voiced_two, -20.0)
        self.assertTrue(active)
        self.assertIsNone(completed)

        gate.update(silence, -80.0)
        _, completed = gate.update(silence, -80.0)
        self.assertIsNotNone(completed)
        self.assertEqual(len(completed), 5 * 1600)

    def test_energy_gate_can_flush_a_sentence_without_waiting_for_silence(self):
        gate = EnergyUtteranceGate(start_chunks=1)
        speech = np.full(1600, 0.1, dtype=np.float32)

        active, completed = gate.update(speech, -20)
        self.assertTrue(active)
        self.assertIsNone(completed)
        flushed = gate.flush()

        self.assertIsNotNone(flushed)
        self.assertGreaterEqual(len(flushed), 1600)

    def test_adaptive_gate_detects_quiet_speech_without_marking_trailing_silence_as_speech(self):
        gate = EnergyUtteranceGate(start_chunks=1, end_chunks=2)
        quiet_speech = np.full(1600, 0.002, dtype=np.float32)  # approximately -54 dBFS
        silence = np.zeros(1600, dtype=np.float32)

        active, _ = gate.update(quiet_speech, -54.0)
        self.assertTrue(active)
        self.assertTrue(gate.last_chunk_speech)
        active, _ = gate.update(silence, -120.0)
        self.assertTrue(active)
        self.assertFalse(gate.last_chunk_speech)

    def test_event_log_schema_never_contains_transcript_text(self):
        os.environ["CABINSPECTOR_DISABLE_MODEL_LOAD"] = "1"
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root))
        from src.video import driver_visual_prototype as prototype

        fields = {field.lower() for field in prototype.EVENT_LOG_FIELDS}
        self.assertNotIn("transcript_text", fields)
        self.assertNotIn("raw_transcript", fields)
        self.assertNotIn("transcript_content", fields)

    def test_transcript_file_is_only_created_when_the_explicit_store_is_used(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transcript_log.csv"
            self.assertFalse(path.exists())
            store = TranscriptStore(path)
            store.append(
                timestamp_ms=1,
                language="en",
                transcript="consented test transcript",
                safety_categories=("PROFANITY",),
                safety_confidence=1.0,
            )
            store.close()
            self.assertTrue(path.exists())
            self.assertIn("consented test transcript", path.read_text(encoding="utf-8"))

    def test_readable_transcript_export_contains_all_saved_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "transcript_log.csv"
            destination = Path(directory) / "transcript_export.txt"
            store = TranscriptStore(source)
            store.append(
                timestamp_ms=1,
                language="en",
                transcript="First saved sentence.",
                safety_categories=(),
                safety_confidence=0.0,
            )
            store.append(
                timestamp_ms=2,
                language="ar",
                transcript="جملة ثانية محفوظة.",
                safety_categories=("THREAT",),
                safety_confidence=1.0,
            )
            store.close()

            export_readable_transcript(source, destination)
            text = destination.read_text(encoding="utf-8")
            self.assertIn("First saved sentence.", text)
            self.assertIn("جملة ثانية محفوظة.", text)
            self.assertIn("Safety: THREAT", text)

    def test_legacy_arabic_csv_is_repaired_and_migrated_for_excel(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transcript_log.csv"
            arabic = "\u064a\u062c\u0628 \u0639\u0644\u064a\u0647\u0645"
            garbled = arabic.encode("cp1256").decode("latin-1")
            path.write_text(
                "timestamp_ms,language,transcript,safety_categories,safety_confidence\n"
                f"1,ar,{garbled},,0.0\n",
                encoding="utf-8",
            )

            store = TranscriptStore(path)
            store.append(
                timestamp_ms=2,
                language="en",
                transcript="New entry.",
                safety_categories=(),
                safety_confidence=0.0,
            )
            store.close()

            self.assertTrue(path.read_bytes().startswith(b"\xef\xbb\xbf"))
            with path.open("r", encoding="utf-8-sig", newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(rows[0]["recorded_at"], "Legacy entry (date unavailable)")
            self.assertEqual(rows[0]["transcript"], arabic)
            self.assertRegex(rows[1]["recorded_at"], r"^\d{4}-\d{2}-\d{2}T")

    def test_transcript_display_and_recording_can_be_explicitly_toggled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transcript_log.csv"
            pipeline = SpeechAnalysisPipeline(
                whisper_model="tiny",
                use_safety_model=False,
                display_transcript=False,
                transcript_log_path=path,
            )
            self.assertFalse(pipeline.get_latest_state().transcript_display_enabled)
            self.assertTrue(pipeline.set_transcript_display(True))
            self.assertTrue(pipeline.get_latest_state().transcript_display_enabled)
            self.assertTrue(pipeline.set_transcript_recording(True))
            self.assertTrue(pipeline.get_latest_state().transcript_recording_enabled)
            self.assertFalse(path.exists())
            self.assertFalse(pipeline.set_transcript_recording(False))
            self.assertFalse(pipeline.get_latest_state().transcript_recording_enabled)


if __name__ == "__main__":
    unittest.main()
