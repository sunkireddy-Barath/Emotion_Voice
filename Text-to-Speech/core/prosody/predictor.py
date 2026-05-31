"""Prosody predictor: maps text + emotion result → ProsodyTarget for TTS."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from ..emotion.classifier import EmotionResult
from .prosody_model import ProsodyModel, ProsodyTarget

logger = logging.getLogger(__name__)

# Rule-based prosody map used before neural model is trained
RULE_BASED_MAP: Dict[str, Dict[str, float]] = {
    "neutral":      {"pitch": 1.00, "energy": 1.00, "speed": 1.00, "pause_factor": 1.00},
    "happy":        {"pitch": 1.20, "energy": 1.30, "speed": 1.10, "pause_factor": 0.80},
    "sad":          {"pitch": 0.85, "energy": 0.70, "speed": 0.85, "pause_factor": 1.30},
    "angry":        {"pitch": 1.10, "energy": 1.60, "speed": 1.15, "pause_factor": 0.70},
    "excited":      {"pitch": 1.40, "energy": 1.50, "speed": 1.20, "pause_factor": 0.60},
    "fear":         {"pitch": 1.25, "energy": 0.90, "speed": 1.30, "pause_factor": 0.90},
    "surprise":     {"pitch": 1.35, "energy": 1.20, "speed": 1.00, "pause_factor": 0.85},
    "calm":         {"pitch": 0.95, "energy": 0.80, "speed": 0.90, "pause_factor": 1.20},
    "serious":      {"pitch": 0.90, "energy": 1.10, "speed": 0.92, "pause_factor": 1.10},
    "motivational": {"pitch": 1.30, "energy": 1.40, "speed": 1.05, "pause_factor": 0.75},
    "questioning":  {"pitch": 1.15, "energy": 0.95, "speed": 0.95, "pause_factor": 1.05},
    "storytelling": {"pitch": 1.10, "energy": 1.00, "speed": 0.95, "pause_factor": 1.15},
}


class ProsodyPredictor:
    """High-level prosody predictor.

    Uses rule-based map when neural model is not loaded.
    Falls back gracefully — the neural model is optional until trained.
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self._model: Optional[ProsodyModel] = None

        if model_path and Path(model_path).exists():
            self._load_model(model_path)

    def _load_model(self, path: str | Path) -> None:
        try:
            checkpoint = torch.load(str(path), map_location=self.device)
            model = ProsodyModel()
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            self._model = model.to(self.device)
            logger.info(f"Neural prosody model loaded from {path}")
        except Exception as e:
            logger.warning(f"Could not load neural prosody model ({e}), using rule-based")

    def _rule_based_predict(self, emotion: str, intensity: float) -> ProsodyTarget:
        base = RULE_BASED_MAP.get(emotion, RULE_BASED_MAP["neutral"])

        # Scale away from 1.0 based on intensity
        def scale(v: float) -> float:
            return 1.0 + (v - 1.0) * intensity

        return ProsodyTarget(
            pitch_curve=[],
            energy_curve=[],
            duration_curve=[],
            global_pitch=max(0.5, min(2.0, scale(base["pitch"]))),
            global_energy=max(0.4, min(2.0, scale(base["energy"]))),
            global_speed=max(0.5, min(2.0, scale(base["speed"]))),
            pause_factor=max(0.4, min(2.0, scale(base["pause_factor"]))),
        )

    def predict(
        self,
        text: str,
        emotion_result: EmotionResult,
        token_ids: Optional[List[int]] = None,
    ) -> ProsodyTarget:
        if self._model is None or token_ids is None:
            return self._rule_based_predict(emotion_result.emotion, emotion_result.intensity)

        self._model.eval()
        with torch.no_grad():
            t_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            e_id = torch.tensor(
                [self._model.get_emotion_id(emotion_result.emotion)],
                dtype=torch.long, device=self.device,
            )
            intensity = torch.tensor([emotion_result.intensity], dtype=torch.float, device=self.device)

            out = self._model(t_ids, e_id, intensity)

        return ProsodyTarget(
            pitch_curve=out["pitch_curve"][0].cpu().tolist(),
            energy_curve=out["energy_curve"][0].cpu().tolist(),
            duration_curve=out["duration_curve"][0].cpu().tolist(),
            global_pitch=float(out["pitch_scale"][0]),
            global_energy=float(out["energy_scale"][0]),
            global_speed=float(out["speed"][0]),
            pause_factor=float(out["pause_factor"][0]),
        )

    def predict_from_text(
        self, text: str, emotion: str = "neutral", intensity: float = 0.7
    ) -> ProsodyTarget:
        from ..emotion.classifier import EmotionResult
        result = EmotionResult(
            emotion=emotion, intensity=intensity, confidence=intensity,
            scores={emotion: intensity}, raw_label=emotion,
        )
        return self.predict(text, result)

    def adjust_for_language(
        self, prosody: ProsodyTarget, language: str
    ) -> ProsodyTarget:
        """Apply language-specific prosody adjustments."""
        adjustments = {
            "ta": {"pitch": 1.05, "speed": 0.95},  # Tamil: slightly higher pitch, slower
            "hi": {"pitch": 1.02, "speed": 0.97},
            "fr": {"pitch": 1.03, "speed": 1.02},
            "de": {"pitch": 0.98, "speed": 0.95},
        }
        adj = adjustments.get(language, {})
        return ProsodyTarget(
            pitch_curve=prosody.pitch_curve,
            energy_curve=prosody.energy_curve,
            duration_curve=prosody.duration_curve,
            global_pitch=prosody.global_pitch * adj.get("pitch", 1.0),
            global_energy=prosody.global_energy,
            global_speed=prosody.global_speed * adj.get("speed", 1.0),
            pause_factor=prosody.pause_factor,
        )
