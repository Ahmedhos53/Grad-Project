import unittest
from unittest import mock

from src.telemetry.inference import TelemetryPrediction
from src.telemetry.replay import TelemetryReplay, TelemetryReplayEvent


def prediction(category: str, confidence: float) -> TelemetryPrediction:
    return TelemetryPrediction(category, confidence, {category: confidence})


class TelemetryReplayTests(unittest.TestCase):
    def test_replay_releases_predictions_on_the_scaled_trip_timeline(self):
        replay = TelemetryReplay(
            [
                TelemetryReplayEvent(1.0, prediction("HARD_BRAKE", 0.9)),
                TelemetryReplayEvent(2.0, prediction("NORMAL", 0.8)),
            ],
            trip=1,
            speed=2.0,
            event_hold_seconds=0.5,
        )
        replay.start(now=100.0)

        waiting = replay.get_latest_state(now=100.49)
        self.assertEqual(waiting.mode, "REPLAYING")
        self.assertEqual(waiting.model_name, "random_forest")
        self.assertEqual(waiting.events_processed, 0)

        active = replay.get_latest_state(now=100.5)
        self.assertEqual(active.mode, "EVENT")
        self.assertTrue(active.event_active)
        self.assertEqual(active.event_category, "HARD_BRAKE")
        self.assertEqual(active.events_processed, 1)

        cleared = replay.get_latest_state(now=100.8)
        self.assertFalse(cleared.event_active)
        self.assertEqual(cleared.event_category, "NORMAL")

        finished = replay.get_latest_state(now=101.0)
        self.assertTrue(finished.completed)
        self.assertEqual(finished.mode, "COMPLETE")
        self.assertEqual(finished.events_processed, 2)

    def test_restart_returns_the_replay_to_its_initial_state(self):
        replay = TelemetryReplay(
            [TelemetryReplayEvent(1.0, prediction("RAPID_ACCELERATION", 0.9))],
            trip=2,
            speed=1.0,
        )
        replay.start(now=0.0)
        self.assertTrue(replay.get_latest_state(now=1.0).event_active)

        replay.restart(now=10.0)
        restarted = replay.get_latest_state(now=10.0)
        self.assertEqual(restarted.events_processed, 0)
        self.assertFalse(restarted.event_active)
        self.assertEqual(restarted.mode, "REPLAYING")

    def test_low_confidence_events_do_not_become_active_alerts(self):
        replay = TelemetryReplay(
            [TelemetryReplayEvent(0.0, prediction("AGGRESSIVE_TURN", 0.4))],
            trip=3,
            minimum_confidence=0.55,
        )
        replay.start(now=0.0)
        state = replay.get_latest_state(now=0.0)
        self.assertFalse(state.event_active)
        self.assertEqual(state.event_category, "NORMAL")

    def test_public_replay_selects_primus_without_touching_random_forest_path(self):
        expected = object()
        with mock.patch.object(TelemetryReplay, "from_primus_public_trip", return_value=expected) as primus:
            replay = TelemetryReplay.from_public_trip(2, model_name="PRIMUS", speed=4.0)

        self.assertIs(replay, expected)
        primus.assert_called_once_with(2, speed=4.0, event_hold_seconds=3.0)

    def test_public_replay_rejects_unknown_model(self):
        with self.assertRaisesRegex(ValueError, "primus.*random_forest"):
            TelemetryReplay.from_public_trip(1, model_name="unsupported")


if __name__ == "__main__":
    unittest.main()
