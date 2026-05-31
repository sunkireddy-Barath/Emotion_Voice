"""Singleton dependency providers — initialized once at startup, thread-safe."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Singletons — set by init_dependencies()
_tts_engine = None
_streaming_engine = None
_emotion_detector = None
_voice_manager = None
_db_url: Optional[str] = None


# ─── Getters (FastAPI Depends targets) ────────────────────────────────────────

def get_tts_engine():
    if _tts_engine is None:
        raise RuntimeError("TTS engine not initialized — app startup incomplete")
    return _tts_engine


def get_streaming_engine():
    if _streaming_engine is None:
        raise RuntimeError("Streaming engine not initialized")
    return _streaming_engine


def get_emotion_detector():
    if _emotion_detector is None:
        raise RuntimeError("Emotion detector not initialized")
    return _emotion_detector


def get_voice_mgr():
    if _voice_manager is None:
        raise RuntimeError("Voice manager not initialized")
    return _voice_manager


def get_db_session():
    """FastAPI dependency for database session."""
    from .db.database import get_db
    if _db_url is None:
        raise RuntimeError("Database not initialized")
    yield from get_db(_db_url)


# ─── Initialization ───────────────────────────────────────────────────────────

def init_dependencies(settings) -> None:
    """Initialize all singletons. Called once at app startup."""
    global _tts_engine, _streaming_engine, _emotion_detector, _voice_manager, _db_url

    _db_url = settings.storage.effective_database_url

    # ── Emotion detector ──────────────────────────────────────────────────────
    logger.info("Initializing emotion detector...")
    from core.emotion.classifier import EmotionClassifier
    from core.emotion.detector import EmotionDetector

    classifier = EmotionClassifier(
        model_name=settings.emotion.model_name,
        fallback_model=settings.emotion.fallback_model,
        cache_dir=str(settings.emotion.cache_path),
        device=settings.emotion.device,
    )
    _emotion_detector = EmotionDetector(classifier=classifier)
    logger.info("Emotion detector ready")

    # ── Prosody predictor ─────────────────────────────────────────────────────
    from core.prosody.predictor import ProsodyPredictor
    prosody_predictor = ProsodyPredictor(device=settings.tts.device)

    # ── TTS engine ────────────────────────────────────────────────────────────
    logger.info(f"Initializing TTS engine (model_type={settings.tts.model_type})...")
    from core.tts.engine import TTSEngine

    _tts_engine = TTSEngine(
        models_dir=str(settings.tts.models_path),
        device=settings.tts.device,
        model_type=settings.tts.model_type,
        emotion_detector=_emotion_detector,
        prosody_predictor=prosody_predictor,
    )
    logger.info(f"TTS engine ready: {_tts_engine.health()['model_type']}")

    # ── Streaming engine ──────────────────────────────────────────────────────
    from core.streaming.streamer import StreamingTTS

    _streaming_engine = StreamingTTS(
        tts_engine=_tts_engine,
        chunk_size_tokens=settings.streaming.chunk_size_tokens,
        target_latency_ms=settings.streaming.target_latency_ms,
    )

    # ── Voice manager ─────────────────────────────────────────────────────────
    logger.info("Initializing voice manager...")
    from core.tts.voice_manager import VoiceManager

    profiles_path = settings.voice_cloning.profiles_path
    profiles_path.mkdir(parents=True, exist_ok=True)
    _voice_manager = VoiceManager(profiles_dir=str(profiles_path))

    # Register existing voice profiles with TTS engine
    for profile in _voice_manager.list_voices():
        best = _voice_manager.get_best_sample(profile["voice_id"])
        if best and Path(best).exists():
            try:
                _tts_engine.register_voice(profile["voice_id"], best)
            except Exception as e:
                logger.warning(f"Could not register voice {profile['voice_id']}: {e}")

    logger.info(
        f"Voice manager ready: {len(_voice_manager.list_voices())} profiles loaded"
    )
    logger.info("All dependencies initialized successfully")
