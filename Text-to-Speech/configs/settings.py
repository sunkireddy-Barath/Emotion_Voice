"""Central settings loader — reads configs/config.yaml + environment variables.

Environment variables override YAML values:
  API_KEY             → api.api_key
  REQUIRE_API_KEY     → api.require_api_key (true/false)
  DEVICE              → tts.device
  TTS_MODEL_TYPE      → tts.model_type
  DATABASE_URL        → storage.database_url
  MODELS_DIR          → tts.models_dir
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, field_validator

BASE_DIR = Path(__file__).parent.parent


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


class AudioSettings(BaseModel):
    sample_rates: List[int] = [22050, 24000]
    default_sample_rate: int = 22050
    channels: int = 1
    bit_depth: int = 16
    format: str = "wav"
    chunk_size: int = 1024


class PreprocessingSettings(BaseModel):
    normalize_db: float = -20.0
    silence_threshold_db: float = -40.0
    silence_min_duration_ms: int = 200
    noise_reduction_strength: float = 0.15
    trim_top_db: int = 30
    hop_length: int = 256
    win_length: int = 1024
    n_fft: int = 1024
    n_mels: int = 80
    fmin: int = 0
    fmax: int = 8000
    mel_fmin: float = 0.0
    mel_fmax: Optional[float] = None


class EmotionSettings(BaseModel):
    labels: List[str] = ["neutral", "happy", "sad", "angry", "excited",
                          "fear", "surprise", "calm", "serious"]
    model_name: str = "j-hartmann/emotion-english-distilroberta-base"
    fallback_model: str = "bhadresh-savani/distilbert-base-uncased-emotion"
    intensity_threshold: float = 0.5
    cache_dir: str = "models/emotion_classifier"
    device: str = "auto"

    @property
    def cache_path(self) -> Path:
        return BASE_DIR / self.cache_dir


class ProsodyMap(BaseModel):
    pitch: float = 1.0
    energy: float = 1.0
    speed: float = 1.0
    pause_factor: float = 1.0


class ProsodySettings(BaseModel):
    pitch_min: float = 50.0
    pitch_max: float = 600.0
    energy_min: float = 0.0
    energy_max: float = 1.0
    speed_min: float = 0.5
    speed_max: float = 2.0
    emotion_prosody_map: Dict[str, ProsodyMap] = {}

    def get_prosody_for_emotion(self, emotion: str) -> ProsodyMap:
        return self.emotion_prosody_map.get(emotion, ProsodyMap())


class TTSSettings(BaseModel):
    model_type: str = "vits"
    model_id: str = "tts_models/en/vctk/vits"
    multilingual_model_id: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    fallback_model: str = "tts_models/en/ljspeech/vits"
    models_dir: str = "models/tts_model"
    languages: List[str] = ["en", "ta", "hi", "fr", "de", "es"]
    default_language: str = "en"
    voice_samples_dir: str = "data/voice_samples"
    use_gpu: bool = False
    device: str = "cpu"

    @property
    def models_path(self) -> Path:
        return BASE_DIR / self.models_dir


class StreamingSettings(BaseModel):
    chunk_size_tokens: int = 20
    buffer_size_ms: int = 100
    target_latency_ms: int = 500
    max_latency_ms: int = 1000
    websocket_ping_interval: int = 20


class APISettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 2
    reload: bool = False
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    max_text_length: int = 5000
    rate_limit_per_minute: int = 60
    api_key: str = ""
    require_api_key: bool = False

    @property
    def effective_api_key(self) -> str:
        """Env var overrides YAML — API_KEY env var takes precedence."""
        return os.environ.get("API_KEY", self.api_key)

    @property
    def auth_enabled(self) -> bool:
        env_require = os.environ.get("REQUIRE_API_KEY", "").lower()
        if env_require in ("true", "1", "yes"):
            return True
        if env_require in ("false", "0", "no"):
            return False
        return bool(self.require_api_key and self.effective_api_key)


class VoiceCloningSettings(BaseModel):
    min_sample_duration_sec: float = 3.0
    max_sample_duration_sec: float = 30.0
    required_samples: int = 1
    recommended_samples: int = 5
    voice_embedding_dim: int = 512
    voice_profiles_dir: str = "data/voice_profiles"

    @property
    def profiles_path(self) -> Path:
        return BASE_DIR / self.voice_profiles_dir


class StorageSettings(BaseModel):
    database_url: str = "sqlite:///emotion_voice.db"
    redis_url: str = "redis://localhost:6379"
    object_storage_path: str = "data/storage"

    @property
    def effective_database_url(self) -> str:
        return os.environ.get("DATABASE_URL", self.database_url)


class Settings(BaseModel):
    audio: AudioSettings = AudioSettings()
    preprocessing: PreprocessingSettings = PreprocessingSettings()
    emotion: EmotionSettings = EmotionSettings()
    prosody: ProsodySettings = ProsodySettings()
    tts: TTSSettings = TTSSettings()
    streaming: StreamingSettings = StreamingSettings()
    api: APISettings = APISettings()
    voice_cloning: VoiceCloningSettings = VoiceCloningSettings()
    storage: StorageSettings = StorageSettings()

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_yaml(cls) -> "Settings":
        yaml_path = BASE_DIR / "configs" / "config.yaml"
        if not yaml_path.exists():
            return cls()

        raw = _load_yaml(yaml_path)

        # Apply environment variable overrides to raw config before parsing
        if "tts" not in raw:
            raw["tts"] = {}
        if os.environ.get("DEVICE"):
            raw["tts"]["device"] = os.environ["DEVICE"]
        if os.environ.get("TTS_MODEL_TYPE"):
            raw["tts"]["model_type"] = os.environ["TTS_MODEL_TYPE"]
        if os.environ.get("MODELS_DIR"):
            raw["tts"]["models_dir"] = os.environ["MODELS_DIR"]

        # Resolve prosody map from nested dicts
        if "prosody" in raw and "emotion_prosody_map" in raw["prosody"]:
            raw_map = raw["prosody"]["emotion_prosody_map"]
            if isinstance(raw_map, dict):
                raw["prosody"]["emotion_prosody_map"] = {
                    k: ProsodyMap(**v) if isinstance(v, dict) else v
                    for k, v in raw_map.items()
                }

        # Only pass known top-level keys
        known = cls.model_fields.keys()
        filtered = {k: v for k, v in raw.items() if k in known}
        return cls(**filtered)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_yaml()
