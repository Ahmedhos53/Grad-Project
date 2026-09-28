import json
import unittest

from src.evaluation.visual_capture import build_visual_capture_report, summarize_trial


class VisualCaptureEvaluatorTests(unittest.TestCase):
    def _observation(self, predicted, confidence, latency):
        return {
            "timestamp_utc": "2026-09-27T00:00:00+00:00",
            "predicted": predicted,
            "confidence": confidence,
            "confidence_kind": "uncalibrated_phone_object_detector_score",
            "latency_ms": {"frame_total": latency, "pose": 2.0},
        }

    def test_trial_uses_majority_vote_and_preserves_false_positive_negative_counts(self):
        trial = summarize_trial(
            trial_number=1,
            behavior="phone_call",
            expected=True,
            conditions={"lighting": "normal"},
            observations=[
                self._observation(True, 0.8, 10.0),
                self._observation(True, 0.9, 20.0),
                self._observation(False, None, 30.0),
            ],
            started_at_utc="2026-09-27T00:00:00+00:00",
        )
        report = build_visual_capture_report(
            [trial],
            model_availability={"yolo": True},
            safe_box_ratios=(0.2, 0.2, 0.8, 0.95),
        )

        metrics = report["per_behavior"]["phone_call"]
        self.assertTrue(trial["predicted"])
        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_positive"], 0)
        self.assertEqual(metrics["false_negative"], 0)
        self.assertEqual(metrics["mean_object_detector_confidence"], 0.85)
        self.assertIsNone(trial["decision_confidence"])

    def test_report_contains_no_frame_payload_and_marks_rule_confidence_unavailable(self):
        trial = summarize_trial(
            trial_number=1,
            behavior="safe_zone_outside",
            expected=False,
            conditions={"lighting": "bright"},
            observations=[self._observation(False, None, 4.0)],
            started_at_utc="2026-09-27T00:00:00+00:00",
        )
        report = build_visual_capture_report(
            [trial],
            model_availability={"pose": True},
            safe_box_ratios=(0.2, 0.2, 0.8, 0.95),
        )
        encoded = json.dumps(report)

        self.assertNotIn("frame_data", encoded)
        self.assertNotIn("image_path", encoded)
        self.assertFalse(report["privacy"]["raw_frames_saved"])
        self.assertFalse(report["privacy"]["raw_video_saved"])
        self.assertFalse(report["per_behavior"]["safe_zone_outside"]["decision_confidence_available"])
        self.assertIn("eyes_closed", report["coverage"]["missing_behaviors"])
        self.assertEqual(
            report["coverage"]["missing_positive_or_negative_trials"]["safe_zone_outside"],
            ["present"],
        )
        self.assertIn("lighting", report["coverage"]["missing_condition_values"])


if __name__ == "__main__":
    unittest.main()
