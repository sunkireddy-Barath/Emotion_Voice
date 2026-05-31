"""Unit tests for audio processing pipeline."""
import numpy as np
import pytest
import tempfile
import soundfile as sf
from pathlib import Path
from data.processing.audio_processor import AudioProcessor


def make_sine(freq=440.0, sr=22050, duration=2.0, amplitude=0.3) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amplitude).astype(np.float32)


class TestAudioProcessor:
    def setup_method(self):
        self.proc = AudioProcessor(sample_rate=22050)

    def test_normalize(self):
        audio = make_sine(amplitude=0.1)
        normalized = self.proc.normalize(audio)
        rms = np.sqrt(np.mean(normalized ** 2))
        target_rms = 10 ** (-20 / 20.0)
        assert abs(rms - target_rms) < 0.01

    def test_normalize_silent(self):
        audio = np.zeros(22050, dtype=np.float32)
        result = self.proc.normalize(audio)
        assert np.allclose(result, 0.0)

    def test_trim_silence(self):
        # Add silence pads
        audio = make_sine(duration=1.0)
        silence = np.zeros(22050 // 4, dtype=np.float32)
        padded = np.concatenate([silence, audio, silence])
        trimmed = self.proc.trim_silence(padded)
        assert len(trimmed) < len(padded)
        assert len(trimmed) > 0

    def test_extract_mel_spectrogram(self):
        audio = make_sine(duration=1.0)
        mel = self.proc.extract_mel_spectrogram(audio)
        assert mel.ndim == 2
        assert mel.shape[0] == 80  # n_mels
        assert mel.shape[1] > 0

    def test_extract_energy(self):
        audio = make_sine(duration=1.0)
        energy = self.proc.extract_energy(audio)
        assert energy.ndim == 1
        assert len(energy) > 0
        assert np.all(energy >= 0)

    def test_extract_duration_features(self):
        audio = make_sine(duration=2.0)
        feats = self.proc.extract_duration_features(audio)
        assert "total_duration" in feats
        assert abs(feats["total_duration"] - 2.0) < 0.1
        assert 0.0 <= feats["speech_ratio"] <= 1.0

    def test_remove_noise(self):
        audio = make_sine(duration=1.0)
        noise = np.random.randn(len(audio)).astype(np.float32) * 0.01
        noisy = audio + noise
        cleaned = self.proc.remove_noise(noisy)
        # STFT-ISTFT reconstruction may differ by up to one FFT window
        assert abs(len(cleaned) - len(noisy)) <= self.proc.n_fft

    def test_process_file(self):
        audio = make_sine(duration=2.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = Path(tmpdir) / "test.wav"
            sf.write(str(in_path), audio, 22050)
            result = self.proc.process(in_path, Path(tmpdir) / "out")
            assert "output_wav" in result
            assert Path(result["output_wav"]).exists()

    def test_mfcc_shape(self):
        audio = make_sine(duration=1.0)
        mfcc = self.proc.extract_mfcc(audio, n_mfcc=40)
        assert mfcc.shape[0] == 40
