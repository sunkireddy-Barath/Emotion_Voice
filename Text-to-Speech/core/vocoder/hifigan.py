"""HiFiGAN vocoder wrapper — converts mel spectrograms → waveform.

Uses Coqui TTS's built-in HiFiGAN when available, otherwise falls
back to direct mel-to-audio inversion via Griffin-Lim.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class HiFiGANVocoder:
    """HiFiGAN vocoder for high-fidelity waveform generation."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        config_path: Optional[str | Path] = None,
        device: str = "cpu",
        sample_rate: int = 22050,
    ):
        self.model_path = model_path
        self.config_path = config_path
        self.device = device
        self.sample_rate = sample_rate
        self._model = None
        self._backend: str = "none"

        self._initialize()

    def _initialize(self) -> None:
        if self._try_load_hifigan():
            return
        self._backend = "griffin_lim"
        logger.info("HiFiGAN not loaded — using Griffin-Lim fallback")

    def _try_load_hifigan(self) -> bool:
        try:
            from TTS.vocoder.models.hifigan_generator import HifiganGenerator
            from TTS.vocoder.configs.hifigan_config import HifiganConfig
            import json
            import torch

            if self.model_path and Path(self.model_path).exists():
                config = HifiganConfig()
                if self.config_path and Path(self.config_path).exists():
                    config.load_json(str(self.config_path))

                model = HifiganGenerator(config.model_params.in_channels)
                state_dict = torch.load(str(self.model_path), map_location=self.device)
                if "model" in state_dict:
                    model.load_state_dict(state_dict["model"])
                else:
                    model.load_state_dict(state_dict)

                model.eval()
                self._model = model
                self._backend = "hifigan"
                logger.info("HiFiGAN vocoder loaded")
                return True
        except Exception as e:
            logger.debug(f"HiFiGAN load failed: {e}")
        return False

    def mel_to_wave(self, mel: np.ndarray) -> np.ndarray:
        """Convert mel spectrogram [n_mels, T] → waveform [samples]."""
        if self._backend == "hifigan" and self._model is not None:
            return self._hifigan_infer(mel)
        return self._griffin_lim(mel)

    def _hifigan_infer(self, mel: np.ndarray) -> np.ndarray:
        import torch
        mel_tensor = torch.FloatTensor(mel).unsqueeze(0).to(self.device)
        with torch.no_grad():
            audio = self._model(mel_tensor).squeeze().cpu().numpy()
        return audio.astype(np.float32)

    def _griffin_lim(self, mel_db: np.ndarray, n_iter: int = 32) -> np.ndarray:
        """Griffin-Lim fallback using librosa."""
        import librosa
        mel_power = librosa.db_to_power(mel_db)
        audio = librosa.feature.inverse.mel_to_audio(
            mel_power,
            sr=self.sample_rate,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_iter=n_iter,
        )
        return audio.astype(np.float32)

    @property
    def backend(self) -> str:
        return self._backend
