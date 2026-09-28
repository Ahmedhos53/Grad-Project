import unittest

import numpy as np

from src.audio.transcription_diagnostics import (
    character_error_rate,
    segment_audio_like_pipeline,
    word_error_rate,
)


class TranscriptionDiagnosticTests(unittest.TestCase):
    def test_word_error_rate_handles_bilingual_normalization(self):
        self.assertEqual(word_error_rate("Hello مرحبا", "hello مرحبا"), 0.0)
        self.assertEqual(word_error_rate("hello world", "hello"), 0.5)

    def test_character_error_rate_normalizes_bilingual_text_and_ignores_punctuation(self):
        self.assertEqual(character_error_rate("Hello, مرحباً", "hello مرحبا"), 0.0)
        self.assertEqual(character_error_rate("abc", "ac"), 0.3333)
        self.assertIsNone(character_error_rate("!!!", "anything"))

    def test_segmentation_reports_completed_speech_segment(self):
        speech = np.full(3_200, 0.1, dtype=np.float32)
        silence = np.zeros(12_800, dtype=np.float32)
        segments, diagnostic = segment_audio_like_pipeline(np.concatenate((speech, silence)))

        self.assertEqual(diagnostic.segment_count, 1)
        self.assertEqual(len(segments), 1)
        self.assertGreaterEqual(len(segments[0]), 8_000)


if __name__ == "__main__":
    unittest.main()
