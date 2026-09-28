import os
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np


os.environ["CABINSPECTOR_DISABLE_MODEL_LOAD"] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.video import driver_visual_prototype as prototype


class DriverVisualLogicTests(unittest.TestCase):
    def test_quit_control_accepts_q_uppercase_q_and_escape(self):
        self.assertTrue(prototype.is_quit_key(ord("q")))
        self.assertTrue(prototype.is_quit_key(ord("Q")))
        self.assertTrue(prototype.is_quit_key(27))
        self.assertFalse(prototype.is_quit_key(ord("p")))

    def test_distance_point_to_box_handles_inside_and_outside_points(self):
        box = (10, 10, 30, 30)

        self.assertEqual(prototype.distance_point_to_box((20, 20), box), 0)
        self.assertEqual(prototype.distance_point_to_box((40, 30), box), 10)
        self.assertAlmostEqual(
            prototype.distance_point_to_box((40, 40), box),
            (10 ** 2 + 10 ** 2) ** 0.5,
        )

    def test_smoothing_counter_requires_confirmation_frames(self):
        counter, active = prototype.update_smoothing_counter(0, True, confirm_frames=3)
        self.assertEqual(counter, 1)
        self.assertFalse(active)

        counter, active = prototype.update_smoothing_counter(counter, True, confirm_frames=3)
        self.assertEqual(counter, 2)
        self.assertFalse(active)

        counter, active = prototype.update_smoothing_counter(counter, True, confirm_frames=3)
        self.assertEqual(counter, 3)
        self.assertTrue(active)

        counter, active = prototype.update_smoothing_counter(counter, False, confirm_frames=3)
        self.assertEqual(counter, 2)
        self.assertFalse(active)

    def test_risk_score_uses_confirmed_signal_strength(self):
        score, parts = prototype.calculate_risk_score(
            face_detected=True,
            eye_closed_counter=10,
            zone_alert_counter=4,
            phone_object_detected=True,
            phone_call_counter=4,
            drinking_counter=4,
        )

        self.assertEqual(score, 100)
        self.assertEqual(parts["eye"], 30)
        self.assertEqual(parts["zone"], 20)
        self.assertEqual(parts["phone_object"], 8)
        self.assertEqual(parts["phone_call"], 40)
        self.assertEqual(parts["drinking"], 25)

        no_face_score, no_face_parts = prototype.calculate_risk_score(
            face_detected=False,
            eye_closed_counter=10,
            zone_alert_counter=0,
            phone_object_detected=False,
            phone_call_counter=0,
            drinking_counter=0,
        )

        self.assertEqual(no_face_score, 0)
        self.assertEqual(no_face_parts["eye"], 0)

    def test_risk_badge_uses_a_complete_text_label_without_relying_on_color(self):
        self.assertEqual(prototype.risk_status_badge_label("LOW RISK"), "LOW")
        self.assertEqual(prototype.risk_status_badge_label("MODERATE RISK"), "MODERATE")
        self.assertEqual(prototype.risk_status_badge_label("HIGH RISK"), "HIGH")
        self.assertEqual(prototype.risk_status_badge_label("NORMAL"), "NORMAL")

        # The compact dashboard badge has 84 px at the standard sidebar width.
        visible = prototype.truncate_text_to_width(
            prototype.risk_status_badge_label("MODERATE RISK"), 84, 0.42, 1
        )
        self.assertEqual(visible, "MODERATE")

    def test_compact_risk_card_draws_all_five_contributions_as_text(self):
        status_data = {
            "risk_score": 38,
            "driver_status": "MODERATE RISK",
            "risk_breakdown": {
                "eye": 0,
                "zone": 20,
                "phone_call": 0,
                "drinking": 25,
                "telemetry": 18,
            },
        }

        for height, expected_labels in (
            (136, ("Eye 0", "Zone 20", "Phone 0", "Drink 25", "Telem 18")),
            (210, ("Eye", "Zone", "Phone", "Drink", "Telem")),
        ):
            with self.subTest(height=height):
                panel = np.zeros((height, 232, 3), dtype=np.uint8)
                drawn_text = []
                with patch.object(
                    prototype,
                    "draw_sidebar_text",
                    side_effect=lambda _panel, text, *_args, **_kwargs: drawn_text.append(str(text)),
                ):
                    prototype.render_risk_breakdown_card(panel, 0, 0, 232, height, status_data)

                for expected in expected_labels:
                    self.assertIn(expected, drawn_text)
                self.assertFalse(any("..." in text for text in drawn_text))

    def test_audio_context_is_conservative_and_speech_reinforces_phone_evidence(self):
        raised_voice_score, raised_voice_parts = prototype.calculate_risk_score(
            face_detected=False,
            eye_closed_counter=0,
            zone_alert_counter=0,
            phone_object_detected=False,
            phone_call_counter=0,
            drinking_counter=0,
            raised_voice_active=True,
            speech_with_phone_evidence=False,
        )
        self.assertEqual(raised_voice_score, 6)
        self.assertEqual(raised_voice_parts["speech_phone"], 0)

        speech_only_score, _ = prototype.calculate_risk_score(
            False, 0, 0, False, 0, 0, False, False
        )
        self.assertEqual(speech_only_score, 0)

        phone_audio_score, phone_audio_parts = prototype.calculate_risk_score(
            False, 0, 0, False, 3, 0, False, True
        )
        self.assertEqual(phone_audio_score, 42)
        self.assertEqual(phone_audio_parts["speech_phone"], 6)

    def test_active_telemetry_event_adds_a_bounded_explainable_risk_part(self):
        score, parts = prototype.calculate_risk_score(
            face_detected=False,
            eye_closed_counter=0,
            zone_alert_counter=0,
            phone_object_detected=False,
            phone_call_counter=0,
            drinking_counter=0,
            telemetry_event_category="HARD_BRAKE",
            telemetry_event_active=True,
        )

        self.assertEqual(score, 18)
        self.assertEqual(parts["telemetry"], 18)

        inactive_score, inactive_parts = prototype.calculate_risk_score(
            False,
            0,
            0,
            False,
            0,
            0,
            telemetry_event_category="HARD_BRAKE",
            telemetry_event_active=False,
        )
        self.assertEqual(inactive_score, 0)
        self.assertEqual(inactive_parts["telemetry"], 0)

    def test_event_log_keeps_telemetry_predictions_without_source_labels(self):
        fields = set(prototype.EVENT_LOG_FIELDS)
        self.assertTrue(
            {
                "telemetry_event_category",
                "telemetry_event_confidence",
                "telemetry_event_active",
                "telemetry_model",
            }.issubset(fields)
        )
        self.assertNotIn("telemetry_source_label", fields)
        self.assertNotIn("expected_class_index", fields)

    def test_phone_call_detection_requires_phone_near_ear_and_hand_contact(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        detections = [
            {
                "class_name": "cell phone",
                "confidence": 0.9,
                "box": (90, 90, 130, 130),
            }
        ]

        pose_points = {
            "left_ear": (100, 100),
            "right_ear": (220, 100),
            "left_wrist": (118, 114),
        }

        _, phone_call_detected, drinking_detected = prototype.detect_object_actions(
            frame.copy(),
            detections,
            pose_points,
            hand_infos=[],
        )

        self.assertTrue(phone_call_detected)
        self.assertFalse(drinking_detected)

        pose_without_hand = {
            "left_ear": (100, 100),
            "right_ear": (220, 100),
        }

        _, phone_call_detected, _ = prototype.detect_object_actions(
            frame.copy(),
            detections,
            pose_without_hand,
            hand_infos=[],
        )

        self.assertFalse(phone_call_detected)

    def test_drinking_detection_requires_container_near_mouth_and_hand_contact(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        detections = [
            {
                "class_name": "cup",
                "confidence": 0.9,
                "box": (310, 145, 350, 205),
            }
        ]
        pose_points = {
            "mouth_left": (300, 150),
            "mouth_right": (340, 150),
            "right_wrist": (340, 190),
        }

        _, phone_call_detected, drinking_detected = prototype.detect_object_actions(
            frame.copy(),
            detections,
            pose_points,
            hand_infos=[],
        )

        self.assertFalse(phone_call_detected)
        self.assertTrue(drinking_detected)

    def test_safe_box_reports_named_landmarks_outside_box(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        points = {
            "left_wrist": (20, 20),
            "right_wrist": (95, 95),
        }

        _, alert, outside_landmarks = prototype.check_safe_box(
            frame,
            points,
            safe_box=(0, 0, 50, 50),
            draw_overlays=False,
        )

        self.assertTrue(alert)
        self.assertEqual(outside_landmarks, ["Right Wrist"])


if __name__ == "__main__":
    unittest.main()
