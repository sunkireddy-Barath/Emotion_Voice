"""Prometheus metrics for the TTS API.

Metrics exposed:
  tts_requests_total          counter  — synthesis requests by status/model
  tts_latency_seconds         histogram— synthesis latency distribution
  tts_audio_duration_seconds  histogram— generated audio duration
  tts_emotion_requests_total  counter  — requests per emotion
  tts_streaming_chunks_total  counter  — streaming chunks sent
  active_websocket_connections gauge   — current WS connections
"""
from __future__ import annotations

try:
    from prometheus_client import (
        Counter, Histogram, Gauge,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        REGISTRY,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

import logging

logger = logging.getLogger(__name__)

if PROMETHEUS_AVAILABLE:
    _tts_requests = Counter(
        "tts_requests_total",
        "Total TTS synthesis requests",
        ["status", "model", "language"],
    )
    _tts_latency = Histogram(
        "tts_latency_seconds",
        "TTS synthesis latency in seconds",
        buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    )
    _audio_duration = Histogram(
        "tts_audio_duration_seconds",
        "Generated audio duration in seconds",
        buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    )
    _emotion_requests = Counter(
        "tts_emotion_requests_total",
        "Requests per detected emotion",
        ["emotion"],
    )
    _streaming_chunks = Counter(
        "tts_streaming_chunks_total",
        "Total streaming audio chunks sent",
    )
    _ws_connections = Gauge(
        "active_websocket_connections",
        "Currently active WebSocket connections",
    )
else:
    logger.warning("prometheus_client not installed — metrics disabled. "
                   "Install with: pip install prometheus-client")


def record_synthesis(
    status: str,
    model: str,
    language: str,
    latency_sec: float,
    audio_duration_sec: float,
    emotion: str,
) -> None:
    if not PROMETHEUS_AVAILABLE:
        return
    _tts_requests.labels(status=status, model=model, language=language).inc()
    _tts_latency.observe(latency_sec)
    _audio_duration.observe(audio_duration_sec)
    _emotion_requests.labels(emotion=emotion).inc()


def record_stream_chunk() -> None:
    if PROMETHEUS_AVAILABLE:
        _streaming_chunks.inc()


def ws_connect() -> None:
    if PROMETHEUS_AVAILABLE:
        _ws_connections.inc()


def ws_disconnect() -> None:
    if PROMETHEUS_AVAILABLE:
        _ws_connections.dec()


def get_metrics_response():
    """Return Prometheus metrics in text format."""
    if not PROMETHEUS_AVAILABLE:
        return "# prometheus_client not installed\n", "text/plain"
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
