from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    voice_id: str = Field("default", description="Voice profile ID")
    language: str = Field("en", description="Language code (en, ta, hi, ...)")
    emotion: Optional[str] = Field(None, description="Override emotion (auto-detected if None)")
    intensity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Emotion intensity 0-1")
    pitch: Optional[float] = Field(None, ge=0.5, le=2.0, description="Pitch scale factor")
    energy: Optional[float] = Field(None, ge=0.3, le=2.0, description="Energy scale factor")
    speed: Optional[float] = Field(None, ge=0.3, le=2.0, description="Speed scale factor")
    sample_rate: int = Field(22050, description="Output sample rate")
    format: str = Field("wav", description="Output format: wav, mp3")

    @field_validator("emotion")
    @classmethod
    def validate_emotion(cls, v: Optional[str]) -> Optional[str]:
        valid = {"neutral", "happy", "sad", "angry", "excited", "fear",
                 "surprise", "calm", "serious", "motivational", "questioning", "storytelling"}
        if v is not None and v not in valid:
            raise ValueError(f"emotion must be one of: {sorted(valid)}")
        return v

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        valid = {"en", "ta", "hi", "fr", "de", "es", "it", "pt", "ru", "zh", "ja", "ko"}
        if v not in valid:
            raise ValueError(f"language must be one of: {sorted(valid)}")
        return v


class StreamRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice_id: str = "default"
    language: str = "en"
    emotion: Optional[str] = None
    intensity: Optional[float] = Field(None, ge=0.0, le=1.0)
    speed: Optional[float] = Field(None, ge=0.3, le=2.0)
    sample_rate: int = 22050


class EmotionAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    language: str = "en"
    per_sentence: bool = False


class VoiceCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    language: str = "en"
    description: str = ""


class VoiceCloneRequest(BaseModel):
    voice_id: str
    text: str = Field(..., min_length=1, max_length=5000)
    language: str = "en"
    emotion: Optional[str] = None
    intensity: Optional[float] = Field(None, ge=0.0, le=1.0)
