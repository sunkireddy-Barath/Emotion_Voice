"""Integration tests for the FastAPI endpoints.

Uses httpx TestClient and mocks the TTS engine so no model downloads needed.
Run with: PYTHONPATH=. pytest tests/integration/ -v
"""
from __future__ import annotations

import io
import json
import wave
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_mock_wav(duration_sec: float = 0.5, sr: int = 22050) -> bytes:
    """Create a minimal valid WAV file."""
    num_samples = int(sr * duration_sec)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"\x00\x00" * num_samples)
    return buf.getvalue()


def build_mock_engine():
    from core.tts.engine import TTSResponse
    engine = MagicMock()
    engine.health.return_value = {
        "model_type": "vits_vctk",
        "model_name": "tts_models/en/vctk/vits",
        "device": "cpu",
        "voices_registered": 0,
        "ready": True,
        "supports_voice_cloning": False,
        "supported_languages": ["en", "ta"],
    }
    mock_resp = TTSResponse(
        audio_bytes=make_mock_wav(),
        sample_rate=22050,
        duration_sec=0.5,
        emotion="neutral",
        emotion_intensity=0.7,
        prosody={"pitch": 1.0, "energy": 1.0, "speed": 1.0, "pause_factor": 1.0},
        model_used="vits_vctk",
        latency_ms=120.0,
        request_id="test1234",
    )
    engine.synthesize.return_value = mock_resp
    engine.list_voices.return_value = []
    return engine


def build_mock_detector():
    from core.emotion.classifier import EmotionResult
    detector = MagicMock()
    result = EmotionResult(
        emotion="neutral", intensity=0.5, confidence=0.9,
        scores={"neutral": 0.9, "happy": 0.1}, raw_label="neutral"
    )
    detector.detect.return_value = result
    detector.detect_multilingual.return_value = result
    detector.detect_sentences.return_value = [result]
    return detector


@pytest.fixture
def client():
    """
    Fixture that creates a TestClient with fully mocked dependencies.
    Patches init_dependencies so the lifespan does not load real ML models.
    """
    from unittest.mock import patch
    import api.dependencies as deps

    # Pre-set mocks — lifespan is patched so it won't overwrite them
    deps._tts_engine = build_mock_engine()
    deps._emotion_detector = build_mock_detector()
    deps._voice_manager = MagicMock()
    deps._voice_manager.list_voices.return_value = []
    deps._voice_manager.get_best_sample.return_value = None
    deps._streaming_engine = MagicMock()
    deps._db_url = "sqlite:///:memory:"

    def _noop_init(settings):
        pass  # Don't load real models in tests

    with patch("api.dependencies.init_dependencies", side_effect=_noop_init), \
         patch("api.db.init_db", return_value=None):
        from api.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ─── Health ───────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_status(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data

    def test_root_returns_name(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "name" in data or "status" in data


# ─── TTS ──────────────────────────────────────────────────────────────────────

class TestTTSEndpoint:
    def test_synthesize_returns_wav(self, client):
        resp = client.post("/tts", json={"text": "Hello world"})
        assert resp.status_code == 200
        assert "audio/wav" in resp.headers["content-type"]
        assert len(resp.content) > 0

    def test_synthesize_has_emotion_header(self, client):
        resp = client.post("/tts", json={"text": "I am happy!"})
        assert resp.status_code == 200
        assert "x-emotion" in resp.headers

    def test_synthesize_has_request_id(self, client):
        resp = client.post("/tts", json={"text": "Hello"})
        assert "x-request-id" in resp.headers

    def test_synthesize_empty_text_is_422(self, client):
        resp = client.post("/tts", json={"text": ""})
        assert resp.status_code == 422

    def test_synthesize_json_endpoint(self, client):
        resp = client.post("/tts/json", json={"text": "Hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert "audio_base64" in data
        assert "emotion" in data
        assert "latency_ms" in data
        assert "request_id" in data
        assert "prosody" in data

    def test_invalid_emotion_422(self, client):
        resp = client.post("/tts", json={"text": "test", "emotion": "banana"})
        assert resp.status_code == 422

    def test_invalid_language_422(self, client):
        resp = client.post("/tts", json={"text": "test", "language": "xyz"})
        assert resp.status_code == 422

    def test_intensity_range_validated(self, client):
        resp = client.post("/tts", json={"text": "test", "intensity": 1.5})
        assert resp.status_code == 422

    def test_with_emotion_override(self, client):
        resp = client.post("/tts", json={"text": "test", "emotion": "happy", "intensity": 0.9})
        assert resp.status_code == 200


# ─── Emotion ──────────────────────────────────────────────────────────────────

class TestEmotionEndpoint:
    def test_analyze_returns_emotion(self, client):
        resp = client.post("/emotion-analysis", json={"text": "I am so happy today!"})
        assert resp.status_code == 200
        data = resp.json()
        assert "emotion" in data
        assert "intensity" in data
        assert "scores" in data
        assert 0.0 <= data["intensity"] <= 1.0

    def test_analyze_empty_text_422(self, client):
        resp = client.post("/emotion-analysis", json={"text": ""})
        assert resp.status_code == 422


# ─── Voices ───────────────────────────────────────────────────────────────────

class TestVoicesEndpoint:
    def test_list_voices_returns_200(self, client):
        resp = client.get("/voices")
        assert resp.status_code == 200
        data = resp.json()
        assert "voices" in data
        assert "total" in data


# ─── System ───────────────────────────────────────────────────────────────────

class TestSystemEndpoints:
    def test_info_endpoint(self, client):
        resp = client.get("/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "version" in data
        assert "supported_emotions" in data

    def test_metrics_endpoint(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
