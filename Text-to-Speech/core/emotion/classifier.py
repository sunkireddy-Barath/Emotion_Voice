"""Emotion classifier using HuggingFace transformers.

Primary model : j-hartmann/emotion-english-distilroberta-base
                → 7 classes: anger, disgust, fear, joy, neutral, sadness, surprise

Fallback model: bhadresh-savani/distilbert-base-uncased-emotion
                → 6 classes: sadness, joy, love, anger, fear, surprise

Both are mapped to our 9 canonical emotion labels.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Canonical emotion labels used across the whole system
CANONICAL_EMOTIONS = [
    "neutral", "happy", "sad", "angry", "excited",
    "fear", "surprise", "calm", "serious",
]

# Map model-specific labels → canonical labels
HARTMANN_MAP = {
    "neutral": "neutral",
    "joy": "happy",
    "sadness": "sad",
    "anger": "angry",
    "fear": "fear",
    "surprise": "surprise",
    "disgust": "serious",
}

BHADRESH_MAP = {
    "joy": "happy",
    "sadness": "sad",
    "anger": "angry",
    "fear": "fear",
    "surprise": "surprise",
    "love": "calm",
}


@dataclass
class EmotionResult:
    emotion: str
    intensity: float
    confidence: float
    scores: Dict[str, float]
    raw_label: str

    def to_dict(self) -> Dict:
        return {
            "emotion": self.emotion,
            "intensity": round(self.intensity, 4),
            "confidence": round(self.confidence, 4),
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
        }


class EmotionClassifier:
    """Text-based emotion classifier with automatic model selection and fallback."""

    def __init__(
        self,
        model_name: str = "j-hartmann/emotion-english-distilroberta-base",
        fallback_model: str = "bhadresh-savani/distilbert-base-uncased-emotion",
        cache_dir: Optional[str | Path] = None,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.cache_dir = str(cache_dir) if cache_dir else None
        self.device = self._resolve_device(device)
        self._pipeline = None
        self._label_map: Dict[str, str] = {}
        self._loaded_model: Optional[str] = None

    def _resolve_device(self, device: str) -> int | str:
        if device == "auto":
            try:
                import torch
                return 0 if torch.cuda.is_available() else -1
            except ImportError:
                return -1
        if device == "cpu":
            return -1
        if device == "cuda":
            return 0
        return -1

    def _load_pipeline(self) -> None:
        from transformers import pipeline

        kwargs = {
            "task": "text-classification",
            "model": self.model_name,
            "top_k": None,
            "device": self.device,
        }
        if self.cache_dir:
            kwargs["model"] = self.model_name
            os.environ.setdefault("TRANSFORMERS_CACHE", self.cache_dir)

        try:
            logger.info(f"Loading emotion model: {self.model_name}")
            self._pipeline = pipeline(**kwargs)
            self._label_map = HARTMANN_MAP
            self._loaded_model = self.model_name
            logger.info("Emotion model loaded successfully")
        except Exception as e:
            logger.warning(f"Primary model failed ({e}), trying fallback: {self.fallback_model}")
            kwargs["model"] = self.fallback_model
            self._pipeline = pipeline(**kwargs)
            self._label_map = BHADRESH_MAP
            self._loaded_model = self.fallback_model
            logger.info(f"Fallback emotion model loaded: {self.fallback_model}")

    @property
    def pipeline(self):
        if self._pipeline is None:
            self._load_pipeline()
        return self._pipeline

    def _map_label(self, raw_label: str) -> str:
        raw_lower = raw_label.lower().strip()
        return self._label_map.get(raw_lower, "neutral")

    def _compute_intensity(self, confidence: float, emotion: str) -> float:
        """Scale raw confidence to intensity [0.1, 1.0]."""
        # High-energy emotions get a slight boost
        energy_boost = {
            "angry": 0.05, "excited": 0.05, "fear": 0.03,
            "happy": 0.03, "surprise": 0.02,
        }
        base = min(confidence * 1.1, 1.0)
        return min(base + energy_boost.get(emotion, 0.0), 1.0)

    def classify(self, text: str) -> EmotionResult:
        if not text or not text.strip():
            return EmotionResult(
                emotion="neutral", intensity=0.5, confidence=0.5,
                scores={"neutral": 0.5}, raw_label="neutral"
            )

        raw_results = self.pipeline(text[:512])[0]
        scores_raw = {r["label"].lower(): r["score"] for r in raw_results}

        # Map to canonical labels (merge scores for same canonical label)
        canonical_scores: Dict[str, float] = {}
        for raw_label, score in scores_raw.items():
            canon = self._map_label(raw_label)
            canonical_scores[canon] = canonical_scores.get(canon, 0.0) + score

        top_canon = max(canonical_scores, key=canonical_scores.get)
        confidence = canonical_scores[top_canon]

        # Find the raw label that maps to the winner
        top_raw = max(scores_raw, key=lambda k: scores_raw[k] if self._map_label(k) == top_canon else -1)
        intensity = self._compute_intensity(confidence, top_canon)

        return EmotionResult(
            emotion=top_canon,
            intensity=intensity,
            confidence=confidence,
            scores=canonical_scores,
            raw_label=top_raw,
        )

    def classify_batch(self, texts: List[str]) -> List[EmotionResult]:
        return [self.classify(t) for t in texts]

    def classify_multilingual(self, text: str, language: str = "en") -> EmotionResult:
        """For non-English text, translate first if translator available."""
        if language != "en":
            try:
                from transformers import pipeline as hf_pipeline
                translator = hf_pipeline(
                    "translation",
                    model=f"Helsinki-NLP/opus-mt-{language}-en",
                    device=self.device,
                )
                translated = translator(text[:512])[0]["translation_text"]
                return self.classify(translated)
            except Exception as e:
                logger.warning(f"Translation failed ({e}), classifying as-is")

        return self.classify(text)
