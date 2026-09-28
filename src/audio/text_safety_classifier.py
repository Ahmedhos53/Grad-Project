"""Local bilingual safety classification with explicit, configurable rules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import json
import time

from src.audio.text_normalization import normalize_bilingual


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULES_PATH = PROJECT_ROOT / "data" / "audio" / "safety_rules.json"
DEFAULT_TOXICITY_MODEL = "gorkem371/toxicity-classifier-xlmr-base-v3"


@dataclass(frozen=True)
class SafetyResult:
    flagged: bool = False
    categories: tuple[str, ...] = ()
    toxicity_confidence: float = 0.0
    rule_confidence: float = 0.0
    model_available: bool = False
    error: str = ""
    timestamp_ms: int = 0


def load_safety_rules(path: str | Path = DEFAULT_RULES_PATH) -> Mapping[str, Mapping[str, list[str]]]:
    with Path(path).open("r", encoding="utf-8") as source:
        return json.load(source)


class BilingualSafetyClassifier:
    """Flags possible unsafe cabin speech; it never determines speaker identity."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_TOXICITY_MODEL,
        rules_path: str | Path = DEFAULT_RULES_PATH,
        model_threshold: float = 0.72,
        use_model: bool = True,
    ):
        self.model_name = model_name
        self.rules = load_safety_rules(rules_path)
        self.model_threshold = model_threshold
        self.use_model = use_model
        self._tokenizer = None
        self._model = None
        self._model_error = ""

    def analyze(self, text: str, *, language: str = "unknown") -> SafetyResult:
        normalized = normalize_bilingual(text)
        if not normalized:
            return SafetyResult(timestamp_ms=int(time.monotonic() * 1000))

        matched_categories = self._match_rules(normalized)
        toxicity_confidence = self._predict_toxicity(normalized) if self.use_model else 0.0
        if toxicity_confidence >= self.model_threshold and not matched_categories:
            matched_categories.add("TOXIC_LANGUAGE")

        return SafetyResult(
            flagged=bool(matched_categories),
            categories=tuple(sorted(matched_categories)),
            toxicity_confidence=round(toxicity_confidence, 4),
            rule_confidence=1.0 if matched_categories else 0.0,
            model_available=self._model is not None,
            error=self._model_error,
            timestamp_ms=int(time.monotonic() * 1000),
        )

    def _match_rules(self, normalized_text: str) -> set[str]:
        matches: set[str] = set()
        for category, languages in self.rules.items():
            for terms in languages.values():
                if any(term in normalized_text for term in terms):
                    matches.add(category.upper())
                    break
        return matches

    def _predict_toxicity(self, normalized_text: str) -> float:
        if not self._ensure_model():
            return 0.0
        try:
            import torch

            encoded = self._tokenizer(
                normalized_text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            )
            with torch.no_grad():
                logits = self._model(**encoded).logits
                probabilities = torch.softmax(logits, dim=-1)[0]
            return float(probabilities[1].item())
        except Exception as exc:
            self._model_error = f"Safety inference failed: {exc}"
            return 0.0

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if self._model_error:
            return False
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.eval()
            return True
        except Exception as exc:
            self._model_error = f"Safety model unavailable: {exc}"
            return False
