"""Audio preprocessing pipeline: normalize → denoise → trim → extract features."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)


class AudioProcessor:
    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        normalize_db: float = -20.0,
        trim_top_db: int = 30,
        noise_reduction_strength: float = 0.15,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.fmin = fmin
        self.fmax = fmax
        self.normalize_db = normalize_db
        self.trim_top_db = trim_top_db
        self.noise_reduction_strength = noise_reduction_strength

    def load(self, path: str | Path) -> Tuple[np.ndarray, int]:
        audio, sr = librosa.load(str(path), sr=self.sample_rate, mono=True)
        return audio.astype(np.float32), sr

    def normalize(self, audio: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-9:
            return audio
        target_rms = 10 ** (self.normalize_db / 20.0)
        return audio * (target_rms / rms)

    def trim_silence(self, audio: np.ndarray) -> np.ndarray:
        trimmed, _ = librosa.effects.trim(audio, top_db=self.trim_top_db)
        return trimmed

    def remove_noise(self, audio: np.ndarray) -> np.ndarray:
        # Spectral subtraction using first 0.1s as noise profile
        noise_sample_len = min(int(0.1 * self.sample_rate), len(audio) // 4)
        if noise_sample_len < 128:
            return audio

        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length,
                             win_length=self.win_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        noise_stft = librosa.stft(audio[:noise_sample_len], n_fft=self.n_fft,
                                   hop_length=self.hop_length,
                                   win_length=self.win_length)
        noise_profile = np.mean(np.abs(noise_stft), axis=1, keepdims=True)

        cleaned_mag = np.maximum(
            magnitude - self.noise_reduction_strength * noise_profile, 0.0
        )
        cleaned_stft = cleaned_mag * np.exp(1j * phase)
        return librosa.istft(cleaned_stft, hop_length=self.hop_length,
                              win_length=self.win_length).astype(np.float32)

    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        return librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    def extract_pitch(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=self.sample_rate,
            hop_length=self.hop_length,
        )
        f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
        return f0, voiced_flag.astype(np.float32)

    def extract_energy(self, audio: np.ndarray) -> np.ndarray:
        frames = librosa.util.frame(audio, frame_length=self.win_length,
                                     hop_length=self.hop_length)
        energy = np.sqrt(np.mean(frames ** 2, axis=0)).astype(np.float32)
        return energy

    def extract_duration_features(self, audio: np.ndarray) -> Dict[str, float]:
        total_duration = len(audio) / self.sample_rate
        intervals = librosa.effects.split(audio, top_db=self.trim_top_db)
        speech_duration = sum((end - start) for start, end in intervals) / self.sample_rate
        silence_duration = total_duration - speech_duration
        speech_ratio = speech_duration / total_duration if total_duration > 0 else 0.0
        return {
            "total_duration": total_duration,
            "speech_duration": speech_duration,
            "silence_duration": silence_duration,
            "speech_ratio": speech_ratio,
        }

    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 40) -> np.ndarray:
        return librosa.feature.mfcc(
            y=audio, sr=self.sample_rate, n_mfcc=n_mfcc,
            n_fft=self.n_fft, hop_length=self.hop_length
        ).astype(np.float32)

    def compute_speaking_rate(self, audio: np.ndarray, text: str) -> float:
        words = len(text.split())
        duration_min = (len(audio) / self.sample_rate) / 60.0
        return words / duration_min if duration_min > 0 else 0.0

    def process(
        self, input_path: str | Path, output_dir: str | Path
    ) -> Dict[str, object]:
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = input_path.stem
        audio, sr = self.load(input_path)

        # Processing pipeline
        audio = self.remove_noise(audio)
        audio = self.normalize(audio)
        audio = self.trim_silence(audio)

        # Save cleaned WAV
        clean_path = output_dir / "cleaned_wav" / f"{stem}.wav"
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(clean_path), audio, self.sample_rate)

        # Extract features
        mel = self.extract_mel_spectrogram(audio)
        f0, voiced = self.extract_pitch(audio)
        energy = self.extract_energy(audio)
        mfcc = self.extract_mfcc(audio)
        duration_feats = self.extract_duration_features(audio)

        # Save features as numpy arrays
        feat_dir = output_dir / "prosody_features"
        feat_dir.mkdir(parents=True, exist_ok=True)
        np.save(feat_dir / f"{stem}_mel.npy", mel)
        np.save(feat_dir / f"{stem}_f0.npy", f0)
        np.save(feat_dir / f"{stem}_energy.npy", energy)
        np.save(feat_dir / f"{stem}_mfcc.npy", mfcc)

        return {
            "input": str(input_path),
            "output_wav": str(clean_path),
            "mel_shape": mel.shape,
            "f0_shape": f0.shape,
            "energy_shape": energy.shape,
            "duration": duration_feats,
            "sample_rate": self.sample_rate,
        }

    def process_batch(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        pattern: str = "*.wav",
    ) -> list:
        input_dir = Path(input_dir)
        files = sorted(input_dir.glob(pattern))
        logger.info(f"Processing {len(files)} files from {input_dir}")

        results = []
        for i, f in enumerate(files, 1):
            try:
                result = self.process(f, output_dir)
                results.append(result)
                logger.info(f"[{i}/{len(files)}] Processed: {f.name}")
            except Exception as e:
                logger.error(f"Failed to process {f.name}: {e}")

        return results
