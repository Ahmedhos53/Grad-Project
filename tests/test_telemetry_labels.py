import unittest

from src.telemetry.labels import CLASS_NAMES, class_index, map_source_label


class TelemetryLabelTests(unittest.TestCase):
    def test_directional_source_events_are_merged_in_initial_taxonomy(self):
        self.assertEqual(map_source_label("aggressive left-turn"), "AGGRESSIVE_TURN")
        self.assertEqual(
            map_source_label("aggressive lane change to the right"),
            "AGGRESSIVE_LANE_CHANGE",
        )

    def test_public_normal_event_has_stable_model_index(self):
        self.assertEqual(class_index("NORMAL"), 0)
        self.assertEqual(CLASS_NAMES[class_index("HARD_BRAKE")], "HARD_BRAKE")
        self.assertEqual(map_source_label("evento não agressivo"), "NORMAL")

    def test_unknown_source_label_is_rejected(self):
        with self.assertRaises(ValueError):
            map_source_label("speeding")


if __name__ == "__main__":
    unittest.main()
