"""High-level emotion detector: combines classifier + context analysis + caching."""
from __future__ import annotations

import hashlib
import json
import logging
import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from .classifier import EmotionClassifier, EmotionResult

logger = logging.getLogger(__name__)

# Punctuation → emotion intensity hints
PUNCT_HINTS = {
    "!!!": ("excited", 0.15),
    "!?":  ("surprised", 0.10),
    "!!":  ("excited", 0.10),
    "!":   ("excited", 0.05),
    "???": ("questioning", 0.10),
    "??":  ("questioning", 0.08),
    "...": ("calm", 0.05),
    "?":   ("questioning", 0.03),
}

# Keywords that shift emotion
KEYWORD_HINTS: Dict[str, Tuple[str, float]] = {
    # Happy/excited
    "congratulations": ("happy", 0.15),
    "amazing": ("excited", 0.10),
    "wonderful": ("happy", 0.10),
    "fantastic": ("excited", 0.10),
    "love": ("happy", 0.08),
    "celebrate": ("happy", 0.10),
    "won": ("excited", 0.10),
    "victory": ("excited", 0.12),
    "selected": ("excited", 0.12),
    "promoted": ("excited", 0.12),
    # Sad
    "miss": ("sad", 0.08),
    "lonely": ("sad", 0.10),
    "heartbreak": ("sad", 0.12),
    "grief": ("sad", 0.15),
    "crying": ("sad", 0.12),
    "tears": ("sad", 0.10),
    # Angry
    "furious": ("angry", 0.15),
    "unacceptable": ("angry", 0.12),
    "outrageous": ("angry", 0.12),
    "demand": ("angry", 0.08),
    # Fear
    "terrified": ("fear", 0.15),
    "nightmare": ("fear", 0.12),
    "scared": ("fear", 0.12),
    "panic": ("fear", 0.15),
    # Motivational
    "believe": ("motivational", 0.08),
    "champion": ("motivational", 0.10),
    "overcome": ("motivational", 0.10),
    "rise": ("motivational", 0.08),
    "strength": ("motivational", 0.08),
    # Calm
    "breathe": ("calm", 0.12),
    "peaceful": ("calm", 0.12),
    "gentle": ("calm", 0.08),
    "soothing": ("calm", 0.10),
    # Serious
    "critical": ("serious", 0.10),
    "urgent": ("serious", 0.10),
    "important": ("serious", 0.08),
    "immediate": ("serious", 0.10),
    "serious": ("serious", 0.12),
}


class EmotionDetector:
    """Full emotion detection pipeline with context analysis and result caching."""

    def __init__(
        self,
        classifier: Optional[EmotionClassifier] = None,
        cache_size: int = 1024,
    ):
        self.classifier = classifier or EmotionClassifier()
        self._cache: Dict[str, EmotionResult] = {}
        self._cache_size = cache_size

    def _text_hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def _analyze_context(self, text: str) -> Dict[str, float]:
        """Return per-emotion score adjustments from punctuation + keywords."""
        adjustments: Dict[str, float] = {}
        text_lower = text.lower()

        for punct, (emotion, boost) in PUNCT_HINTS.items():
            if punct in text:
                adjustments[emotion] = adjustments.get(emotion, 0.0) + boost

        for keyword, (emotion, boost) in KEYWORD_HINTS.items():
            pattern = r"\b" + re.escape(keyword) + r"\b"
            count = len(re.findall(pattern, text_lower))
            if count > 0:
                adjustments[emotion] = adjustments.get(emotion, 0.0) + boost * min(count, 3)

        # ALL-CAPS words signal intensity
        caps_words = len(re.findall(r"\b[A-Z]{3,}\b", text))
        if caps_words > 0:
            adjustments["excited"] = adjustments.get("excited", 0.0) + 0.05 * min(caps_words, 5)

        return adjustments

    def _merge_adjustments(
        self, result: EmotionResult, adjustments: Dict[str, float]
    ) -> EmotionResult:
        if not adjustments:
            return result

        scores = dict(result.scores)
        for emotion, boost in adjustments.items():
            scores[emotion] = min(scores.get(emotion, 0.0) + boost, 1.0)

        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        top_emotion = max(scores, key=scores.get)
        new_intensity = min(result.intensity + adjustments.get(top_emotion, 0.0) * 0.5, 1.0)

        return EmotionResult(
            emotion=top_emotion,
            intensity=new_intensity,
            confidence=scores[top_emotion],
            scores=scores,
            raw_label=result.raw_label,
        )

    def detect(self, text: str, use_cache: bool = True) -> EmotionResult:
        key = self._text_hash(text)
        if use_cache and key in self._cache:
            return self._cache[key]

        result = self.classifier.classify(text)
        adjustments = self._analyze_context(text)
        result = self._merge_adjustments(result, adjustments)

        if use_cache:
            if len(self._cache) >= self._cache_size:
                # Evict oldest entry
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[key] = result

        return result

    def detect_sentences(self, text: str) -> List[EmotionResult]:
        """Detect emotion per sentence for long texts."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [self.detect(s) for s in sentences if s.strip()]

    def detect_with_override(
        self, text: str, emotion_override: Optional[str] = None,
        intensity_override: Optional[float] = None,
    ) -> EmotionResult:
        result = self.detect(text)
        if emotion_override:
            scores = {emotion_override: 1.0}
            result = EmotionResult(
                emotion=emotion_override,
                intensity=intensity_override or result.intensity,
                confidence=1.0,
                scores=scores,
                raw_label=emotion_override,
            )
        elif intensity_override is not None:
            result = EmotionResult(
                emotion=result.emotion,
                intensity=intensity_override,
                confidence=result.confidence,
                scores=result.scores,
                raw_label=result.raw_label,
            )
        return result

    def detect_multilingual(
        self, text: str, language: str = "en"
    ) -> EmotionResult:
        return self.classifier.classify_multilingual(text, language)
