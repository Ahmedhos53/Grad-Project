import unittest

from src.audio.audio_event_detector import (
    AudioSmoothingConfig,
    AudioStateSmoother,
    map_yamnet_categories,
)


class AudioEventLogicTests(unittest.TestCase):
    def test_relevant_yamnet_labels_are_collapsed_to_project_categories(self):
        scores = map_yamnet_categories(
            [
                ("Speech", 0.70),
                ("Conversation", 0.82),
                ("Vehicle horn, car horn, honking", 0.61),
                ("Unknown label", 0.99),
            ]
        )

        self.assertEqual(scores, {"speech": 0.82, "horn": 0.61})

    def test_smoother_requires_confirmation_and_release_windows(self):
        smoother = AudioStateSmoother(
            AudioSmoothingConfig(
                thresholds={"speech": 0.5},
                confirm_windows=2,
                release_windows=2,
            )
        )

        first = smoother.update(
            {"speech": 0.7}, top_label="Speech", top_confidence=0.7, timestamp_ms=1
        )
        self.assertFalse(first.speech_active)

        second = smoother.update(
            {"speech": 0.7}, top_label="Speech", top_confidence=0.7, timestamp_ms=2
        )
        self.assertTrue(second.speech_active)

        held = smoother.update(
            {}, top_label="Silence", top_confidence=0.7, timestamp_ms=3
        )
        self.assertTrue(held.speech_active)

        released = smoother.update(
            {}, top_label="Silence", top_confidence=0.7, timestamp_ms=4
        )
        self.assertFalse(released.speech_active)


if __name__ == "__main__":
    unittest.main()
