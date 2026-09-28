"""Arabic-aware text shaping and drawing for the OpenCV dashboard."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display


_ARABIC_CHARACTERS = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")
_DEFAULT_FONT_PATHS = (
    Path(r"C:\Windows\Fonts\segoeui.ttf"),
    Path(r"C:\Windows\Fonts\tahoma.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
)


def contains_arabic(text: str) -> bool:
    return bool(_ARABIC_CHARACTERS.search(text or ""))


def shape_for_rtl_display(text: str) -> str:
    """Return visual-order glyphs for renderers without Arabic bidi support."""

    value = str(text or "")
    if not contains_arabic(value):
        return value
    return get_display(arabic_reshaper.reshape(value), base_dir="R")


def font_size_from_opencv_scale(scale: float) -> int:
    """Approximate OpenCV's sidebar text size with a readable TrueType size."""

    return max(12, int(round(float(scale) * 32)))


@lru_cache(maxsize=12)
def _arabic_font(size: int) -> ImageFont.FreeTypeFont:
    configured = os.environ.get("CABINSPECTOR_ARABIC_FONT", "").strip()
    candidates = ((Path(configured),) if configured else ()) + _DEFAULT_FONT_PATHS
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    # Pillow can resolve fonts installed in the Windows Fonts directory on most systems.
    return ImageFont.truetype("arial.ttf", size=size)


def localized_text_width(text: str, font_size: int) -> int:
    font = _arabic_font(font_size)
    display_text = shape_for_rtl_display(text)
    left, _top, right, _bottom = font.getbbox(display_text)
    return max(0, right - left)


def truncate_localized_text(text: str, max_width: int, font_size: int) -> str:
    """Trim logical text while measuring it with the Arabic-capable font."""

    value = str(text or "")
    if max_width <= 0 or not value:
        return ""
    if localized_text_width(value, font_size) <= max_width:
        return value

    suffix = "..."
    trimmed = value
    while trimmed:
        candidate = f"{trimmed}{suffix}"
        if localized_text_width(candidate, font_size) <= max_width:
            return candidate
        trimmed = trimmed[:-1]
    return suffix if localized_text_width(suffix, font_size) <= max_width else ""


def draw_rtl_text(
    frame: np.ndarray,
    text: str,
    *,
    right_x: int,
    baseline_y: int,
    color_bgr: tuple[int, int, int],
    font_size: int,
) -> bool:
    """Draw Arabic/mixed text right-aligned on an OpenCV BGR image in place."""

    if not contains_arabic(text):
        return False

    display_text = shape_for_rtl_display(text)
    font = _arabic_font(font_size)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)
    drawing = ImageDraw.Draw(image)
    color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
    drawing.text(
        (int(right_x), int(baseline_y)),
        display_text,
        font=font,
        fill=color_rgb,
        anchor="rs",
    )
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return True
