import unittest

import numpy as np

from src.video.localized_text import (
    contains_arabic,
    draw_rtl_text,
    font_size_from_opencv_scale,
    localized_text_width,
    shape_for_rtl_display,
    truncate_localized_text,
)


class LocalizedTextTests(unittest.TestCase):
    def test_arabic_is_shaped_for_a_renderer_without_bidi_support(self):
        logical_text = "\u0623\u062d\u0645\u062f Ahmed"

        display_text = shape_for_rtl_display(logical_text)

        self.assertTrue(contains_arabic(logical_text))
        self.assertNotEqual(display_text, logical_text)
        self.assertIn("Ahmed", display_text)

    def test_non_arabic_text_is_left_unchanged(self):
        self.assertEqual(shape_for_rtl_display("Ahmed in the taxi"), "Ahmed in the taxi")
        self.assertFalse(contains_arabic("Ahmed in the taxi"))

    def test_localized_truncation_respects_the_rendered_width(self):
        font_size = font_size_from_opencv_scale(0.46)
        value = "\u0623\u062d\u0645\u062f is going to Cairo airport after work"

        truncated = truncate_localized_text(value, 125, font_size)

        self.assertLessEqual(localized_text_width(truncated, font_size), 125)
        self.assertTrue(truncated.endswith("...") or truncated == value)

    def test_rtl_text_is_drawn_on_an_opencv_frame(self):
        frame = np.zeros((80, 320, 3), dtype=np.uint8)

        drawn = draw_rtl_text(
            frame,
            "\u0623\u062d\u0645\u062f Ahmed",
            right_x=310,
            baseline_y=48,
            color_bgr=(255, 255, 255),
            font_size=16,
        )

        self.assertTrue(drawn)
        self.assertGreater(int(frame.sum()), 0)


if __name__ == "__main__":
    unittest.main()
