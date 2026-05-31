"""GET /voices, POST /voice-clone, POST /voices/{id}/samples — voice management."""
from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..schemas.requests import VoiceCreateRequest, VoiceCloneRequest
from ..schemas.responses import VoiceResponse, VoiceListResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/voices", tags=["Voices"])


def get_voice_manager():
    from ..dependencies import get_voice_mgr
    return get_voice_mgr()


def get_engine():
    from ..dependencies import get_tts_engine
    return get_tts_engine()


@router.get("", response_model=VoiceListResponse)
async def list_voices(manager=Depends(get_voice_manager)):
    """List all available voice profiles."""
    raw_voices = manager.list_voices()
    voices = []
    for v in raw_voices:
        voices.append(VoiceResponse(
            voice_id=v["voice_id"],
            name=v["name"],
            language=v.get("language", "en"),
            description=v.get("description", ""),
            sample_count=len(v.get("sample_files", [])),
            total_duration_sec=v.get("total_duration_sec", 0.0),
            created_at=v.get("created_at", ""),
            type="cloned",
        ))
    return VoiceListResponse(voices=voices, total=len(voices))


@router.post("", response_model=VoiceResponse, status_code=201)
async def create_voice(
    request: VoiceCreateRequest,
    manager=Depends(get_voice_manager),
):
    """Create a new voice profile (no samples yet)."""
    profile = manager.create_voice(
        name=request.name,
        language=request.language,
        description=request.description,
    )
    return VoiceResponse(
        voice_id=profile.voice_id,
        name=profile.name,
        language=profile.language,
        description=profile.description,
        sample_count=0,
        total_duration_sec=0.0,
        created_at=profile.created_at,
        type="cloned",
    )


@router.post("/{voice_id}/samples", status_code=201)
async def upload_sample(
    voice_id: str,
    file: UploadFile = File(...),
    manager=Depends(get_voice_manager),
    engine=Depends(get_engine),
):
    """Upload a voice sample WAV for cloning."""
    if not file.filename:
        raise HTTPException(status_code=422, detail="No filename provided")

    suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ".wav"
    allowed = {".wav", ".mp3", ".flac", ".ogg"}
    if suffix not in allowed:
        raise HTTPException(status_code=422, detail=f"Unsupported format: {suffix}")

    audio_bytes = await file.read()
    if len(audio_bytes) < 1024:
        raise HTTPException(status_code=422, detail="File too small")

    try:
        result = manager.add_sample(voice_id, audio_bytes, file.filename)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Voice not found: {voice_id}")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Register sample with TTS engine
    best = manager.get_best_sample(voice_id)
    if best:
        engine.register_voice(voice_id, best)

    return result


@router.get("/{voice_id}", response_model=VoiceResponse)
async def get_voice(voice_id: str, manager=Depends(get_voice_manager)):
    """Get voice profile details."""
    profile = manager.get_voice(voice_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Voice not found: {voice_id}")
    return VoiceResponse(
        voice_id=profile.voice_id,
        name=profile.name,
        language=profile.language,
        description=profile.description,
        sample_count=len(profile.sample_files),
        total_duration_sec=profile.total_duration_sec,
        created_at=profile.created_at,
        type="cloned",
    )


@router.delete("/{voice_id}", status_code=204)
async def delete_voice(voice_id: str, manager=Depends(get_voice_manager)):
    """Delete a voice profile and all its samples."""
    try:
        manager.delete_voice(voice_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Voice not found: {voice_id}")


@router.get("/{voice_id}/validate")
async def validate_voice(voice_id: str, manager=Depends(get_voice_manager)):
    """Check if voice has sufficient samples for cloning."""
    try:
        return manager.validate_voice(voice_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Voice not found: {voice_id}")


@router.post("/clone")
async def clone_voice(
    request: VoiceCloneRequest,
    manager=Depends(get_voice_manager),
    engine=Depends(get_engine),
):
    """Synthesize speech using a cloned voice profile."""
    from fastapi.responses import Response
    from core.tts.engine import TTSRequest as EngineRequest

    profile = manager.get_voice(request.voice_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Voice not found: {request.voice_id}")

    # Register voice if needed
    best = manager.get_best_sample(request.voice_id)
    if best:
        engine.register_voice(request.voice_id, best)

    try:
        eng_req = EngineRequest(
            text=request.text,
            voice_id=request.voice_id,
            language=request.language,
            emotion=request.emotion,
            intensity=request.intensity,
        )
        response = engine.synthesize(eng_req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return Response(
        content=response.audio_bytes,
        media_type="audio/wav",
        headers={
            "X-Voice-Id": request.voice_id,
            "X-Emotion": response.emotion,
            "X-Duration-Sec": str(response.duration_sec),
        },
    )
