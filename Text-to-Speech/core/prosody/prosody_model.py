"""Neural prosody prediction model: text + emotion → pitch/energy/speed/pause curves."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProsodyTarget:
    pitch_curve: List[float]      # normalized f0 per frame
    energy_curve: List[float]     # normalized energy per frame
    duration_curve: List[float]   # duration per phoneme in seconds
    global_pitch: float           # global pitch scale factor
    global_energy: float          # global energy scale factor
    global_speed: float           # speaking rate scale factor
    pause_factor: float           # pause duration multiplier

    def to_dict(self) -> dict:
        return {
            "pitch": round(self.global_pitch, 4),
            "energy": round(self.global_energy, 4),
            "speed": round(self.global_speed, 4),
            "pause_factor": round(self.pause_factor, 4),
            "pitch_curve_len": len(self.pitch_curve),
            "energy_curve_len": len(self.energy_curve),
        }


class VarianceAdaptor(nn.Module):
    """FastSpeech2-style variance adaptor for pitch, energy, and duration."""

    def __init__(self, d_model: int = 256, n_emotions: int = 9):
        super().__init__()
        self.d_model = d_model

        # Emotion embedding
        self.emotion_embed = nn.Embedding(n_emotions, d_model)

        # Pitch predictor
        self.pitch_predictor = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        # Energy predictor
        self.energy_predictor = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        # Duration predictor
        self.duration_predictor = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        # Global style predictor (pitch scale, energy scale, speed, pause)
        self.global_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 4),
            nn.Sigmoid(),
        )

    def _conv_block(self, x: torch.Tensor, conv: nn.Sequential) -> torch.Tensor:
        # x: [B, T, d_model]
        out = x.transpose(1, 2)  # [B, d_model, T]
        for layer in conv:
            if isinstance(layer, nn.Conv1d):
                out = layer(out)
            elif isinstance(layer, nn.ReLU):
                out = F.relu(out)
            elif isinstance(layer, nn.LayerNorm):
                out = layer(out.transpose(1, 2)).transpose(1, 2)
            elif isinstance(layer, nn.Linear):
                out = layer(out.transpose(1, 2)).transpose(1, 2)
        return out.transpose(1, 2).squeeze(-1)  # [B, T]

    def forward(
        self,
        x: torch.Tensor,         # [B, T, d_model] encoder output
        emotion_ids: torch.Tensor, # [B]
        intensity: torch.Tensor,  # [B]
    ) -> dict:
        # Add emotion conditioning
        e_emb = self.emotion_embed(emotion_ids).unsqueeze(1)  # [B, 1, d_model]
        e_emb = e_emb * intensity.unsqueeze(-1).unsqueeze(-1)
        x = x + e_emb

        pitch = self._conv_block(x, self.pitch_predictor)     # [B, T]
        energy = self._conv_block(x, self.energy_predictor)   # [B, T]
        duration = self._conv_block(x, self.duration_predictor)  # [B, T]

        # Global style from mean-pooled representation
        x_mean = x.mean(dim=1)  # [B, d_model]
        global_style = self.global_predictor(x_mean)  # [B, 4]

        # Scale global_style to meaningful ranges
        # pitch_scale: [0.7, 1.5], energy_scale: [0.6, 1.6], speed: [0.7, 1.4], pause: [0.6, 1.5]
        pitch_scale  = global_style[:, 0] * 0.8 + 0.7
        energy_scale = global_style[:, 1] * 1.0 + 0.6
        speed        = global_style[:, 2] * 0.7 + 0.7
        pause_factor = global_style[:, 3] * 0.9 + 0.6

        return {
            "pitch_curve": torch.sigmoid(pitch),
            "energy_curve": torch.sigmoid(energy),
            "duration_curve": F.softplus(duration),
            "pitch_scale": pitch_scale,
            "energy_scale": energy_scale,
            "speed": speed,
            "pause_factor": pause_factor,
        }


class ProsodyModel(nn.Module):
    """Full prosody prediction model with text encoder + variance adaptor."""

    EMOTION_LABELS = [
        "neutral", "happy", "sad", "angry", "excited",
        "fear", "surprise", "calm", "serious",
    ]

    def __init__(self, vocab_size: int = 256, d_model: int = 256, n_heads: int = 4,
                 n_layers: int = 4):
        super().__init__()
        self.d_model = d_model
        self.emotion_to_id = {e: i for i, e in enumerate(self.EMOTION_LABELS)}

        self.text_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.variance_adaptor = VarianceAdaptor(d_model, len(self.EMOTION_LABELS))

    def forward(
        self,
        token_ids: torch.Tensor,
        emotion_ids: torch.Tensor,
        intensity: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        x = self.text_embed(token_ids)
        x = self.pos_encoding(x)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.variance_adaptor(x, emotion_ids, intensity)

    def get_emotion_id(self, emotion: str) -> int:
        return self.emotion_to_id.get(emotion, 0)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        import math
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]
