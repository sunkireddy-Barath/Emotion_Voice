from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class EmotionResponse(BaseModel):
    emotion: str
    intensity: float
    confidence: float
    scores: Dict[str, float]


class ProsodyResponse(BaseModel):
    pitch: float
    energy: float
    speed: float
    pause_factor: float


class TTSResponse(BaseModel):
    audio_url: Optional[str] = None
    duration_sec: float
    sample_rate: int
    emotion: EmotionResponse
    prosody: ProsodyResponse
    model_used: str
    latency_ms: float


class EmotionAnalysisResponse(BaseModel):
    emotion: str
    intensity: float
    confidence: float
    scores: Dict[str, float]
    sentences: Optional[List[Dict]] = None


class VoiceResponse(BaseModel):
    voice_id: str
    name: str
    language: str
    description: str
    sample_count: int
    total_duration_sec: float
    created_at: str
    type: str = "cloned"


class VoiceListResponse(BaseModel):
    voices: List[VoiceResponse]
    total: int


class HealthResponse(BaseModel):
    status: str
    model_type: str
    device: str
    voices_registered: int
    version: str = "1.0.0"
