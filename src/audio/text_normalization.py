"""Normalization used only for local bilingual safety matching."""

from __future__ import annotations

import re


_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_REPEATED = re.compile(r"(.)\1{2,}")
_WHITESPACE = re.compile(r"\s+")


def normalize_english(text: str) -> str:
    text = (text or "").casefold()
    text = _REPEATED.sub(r"\1\1", text)
    return _WHITESPACE.sub(" ", text).strip()


def normalize_arabic(text: str) -> str:
    text = _ARABIC_DIACRITICS.sub("", text or "")
    text = text.replace("ـ", "")
    text = text.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ؤ": "و", "ئ": "ي"}))
    text = _REPEATED.sub(r"\1\1", text)
    return _WHITESPACE.sub(" ", text).strip()


def normalize_bilingual(text: str) -> str:
    """Normalize Arabic and English without translating or persisting text."""
    return normalize_english(normalize_arabic(text))
