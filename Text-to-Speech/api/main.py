"""Production FastAPI application.

Features:
  - API key authentication (set API_KEY env var)
  - Rate limiting (60 req/min per IP by default)
  - Request logging with latency
  - Prometheus metrics at /metrics
  - Database-backed synthesis history
  - Structured error responses
  - Graceful startup/shutdown

Endpoints:
  POST   /tts                   Synthesize speech (returns WAV bytes)
  POST   /tts/json              Synthesize speech (returns JSON + base64 audio)
  POST   /stream                SSE audio stream
  WS     /ws/stream             WebSocket audio stream
  POST   /emotion-analysis      Detect emotion from text
  GET    /voices                List voice profiles
  POST   /voices                Create voice profile
  POST   /voices/{id}/samples   Upload voice sample
  GET    /voices/{id}           Get voice profile details
  DELETE /voices/{id}           Delete voice profile
  GET    /voices/{id}/validate  Validate voice readiness
  POST   /voices/clone          Synthesize with cloned voice
  GET    /health                System health check
  GET    /metrics               Prometheus metrics
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.settings import get_settings
from .middleware.auth import APIKeyMiddleware
from .middleware.logging_middleware import LoggingMiddleware
from .middleware.rate_limit import RateLimitMiddleware
from .routers import tts, stream, emotion, voices

settings = get_settings()

# ─── Logging setup ────────────────────────────────────────────────────────────
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            "logs/emotion_voice.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
        ),
    ],
)
logger = logging.getLogger(__name__)


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("Starting Emotion-Aware Speech Foundation Model API")
    logger.info(f"  Model type : {settings.tts.model_type}")
    logger.info(f"  Device     : {settings.tts.device}")
    logger.info(f"  Auth       : {'ENABLED' if settings.api.auth_enabled else 'DISABLED'}")
    logger.info("=" * 60)

    from .dependencies import init_dependencies
    from .db import init_db

    # Initialize database
    db_url = settings.storage.effective_database_url
    init_db(db_url)

    # Initialize ML components
    init_dependencies(settings)

    logger.info("API ready — all systems operational")
    yield

    logger.info("Shutting down gracefully...")


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Emotion-Aware Speech Foundation Model",
    description=(
        "Production-grade, fully self-hosted TTS with:\n"
        "- Automatic emotion detection from text\n"
        "- 12 emotion styles with intensity control\n"
        "- Voice cloning from 3+ seconds of audio\n"
        "- 17-language support\n"
        "- Real-time WebSocket streaming\n"
        "- REST and SSE APIs\n\n"
        "**Authentication:** Set `X-API-Key` header with your API key."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── Middleware (order matters — first added = outermost) ─────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "X-API-Key"],
)
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=settings.api.rate_limit_per_minute,
)
app.add_middleware(LoggingMiddleware)
app.add_middleware(
    APIKeyMiddleware,
    api_key=settings.api.effective_api_key,
    enabled=settings.api.auth_enabled,
)

# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(tts.router)
app.include_router(stream.router)
app.include_router(emotion.router)
app.include_router(voices.router)


# ─── Core routes ──────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": "Emotion-Aware Speech Foundation Model",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get("/health", tags=["System"])
async def health():
    """System health check — always public, no auth required."""
    from .dependencies import get_tts_engine
    try:
        engine = get_tts_engine()
        h = engine.health()
        return {
            "status": "ok",
            **h,
            "version": "1.0.0",
            "timestamp": time.time(),
        }
    except RuntimeError:
        return {
            "status": "starting",
            "version": "1.0.0",
            "timestamp": time.time(),
        }


@app.get("/metrics", tags=["System"], include_in_schema=False)
async def metrics():
    """Prometheus metrics endpoint."""
    from .metrics import get_metrics_response
    content, content_type = get_metrics_response()
    return Response(content=content, media_type=content_type)


@app.get("/info", tags=["System"])
async def system_info():
    """System information and configuration summary."""
    from .dependencies import get_tts_engine
    try:
        engine = get_tts_engine()
        h = engine.health()
    except RuntimeError:
        h = {"model_type": "starting", "ready": False}

    return {
        "version": "1.0.0",
        "model": h.get("model_type", "unknown"),
        "model_ready": h.get("ready", False),
        "supports_voice_cloning": h.get("supports_voice_cloning", False),
        "supported_languages": h.get("supported_languages", ["en"]),
        "supported_emotions": [
            "neutral", "happy", "sad", "angry", "excited",
            "fear", "surprise", "calm", "serious",
            "motivational", "questioning", "storytelling",
        ],
        "auth_enabled": settings.api.auth_enabled,
        "rate_limit": f"{settings.api.rate_limit_per_minute} req/min",
    }
