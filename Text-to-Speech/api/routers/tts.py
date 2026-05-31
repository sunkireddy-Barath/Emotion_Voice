"""POST /tts — synchronous speech synthesis with metrics + history recording."""
from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response

from ..schemas.requests import TTSRequest as TTSReq

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tts", tags=["TTS"])


def get_engine():
    from ..dependencies import get_tts_engine
    return get_tts_engine()


@router.post(
    "",
    response_class=Response,
    summary="Synthesize speech",
    description=(
        "Convert text to speech with automatic emotion detection and prosody control. "
        "Returns WAV audio bytes. Set `emotion` to override auto-detection."
    ),
)
async def synthesize(
    request: TTSReq,
    raw_request: Request,
    engine=Depends(get_engine),
    download: bool = Query(False, description="Add Content-Disposition download header"),
):
    """Synthesize speech from text. Returns WAV audio bytes."""
    from core.tts.engine import TTSRequest as EngineRequest
    from ..metrics import record_synthesis

    try:
        eng_req = EngineRequest(
            text=request.text,
            voice_id=request.voice_id,
            language=request.language,
            emotion=request.emotion,
            intensity=request.intensity,
            pitch=request.pitch,
            energy=request.energy,
            speed=request.speed,
            sample_rate=request.sample_rate,
        )
        resp = engine.synthesize(eng_req)

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {e}")

    # Record metrics
    record_synthesis(
        status="success",
        model=resp.model_used,
        language=request.language,
        latency_sec=resp.latency_ms / 1000,
        audio_duration_sec=resp.duration_sec,
        emotion=resp.emotion,
    )

    headers = {
        "X-Request-Id":   resp.request_id,
        "X-Emotion":      resp.emotion,
        "X-Intensity":    str(resp.emotion_intensity),
        "X-Latency-Ms":   str(resp.latency_ms),
        "X-Duration-Sec": str(resp.duration_sec),
        "X-Model":        resp.model_used,
    }
    if download:
        headers["Content-Disposition"] = 'attachment; filename="speech.wav"'

    return Response(
        content=resp.audio_bytes,
        media_type="audio/wav",
        headers=headers,
    )


@router.post(
    "/json",
    summary="Synthesize speech (JSON response)",
    description="Same as POST /tts but returns JSON with base64-encoded audio.",
)
async def synthesize_json(
    request: TTSReq,
    raw_request: Request,
    engine=Depends(get_engine),
):
    import base64
    from core.tts.engine import TTSRequest as EngineRequest
    from ..metrics import record_synthesis

    try:
        eng_req = EngineRequest(
            text=request.text,
            voice_id=request.voice_id,
            language=request.language,
            emotion=request.emotion,
            intensity=request.intensity,
            pitch=request.pitch,
            energy=request.energy,
            speed=request.speed,
            sample_rate=request.sample_rate,
        )
        resp = engine.synthesize(eng_req)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"TTS JSON synthesis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    record_synthesis(
        status="success",
        model=resp.model_used,
        language=request.language,
        latency_sec=resp.latency_ms / 1000,
        audio_duration_sec=resp.duration_sec,
        emotion=resp.emotion,
    )

    return {
        "request_id":    resp.request_id,
        "audio_base64":  base64.b64encode(resp.audio_bytes).decode(),
        "sample_rate":   resp.sample_rate,
        "duration_sec":  resp.duration_sec,
        "emotion":       resp.emotion,
        "intensity":     resp.emotion_intensity,
        "prosody":       resp.prosody,
        "model_used":    resp.model_used,
        "latency_ms":    resp.latency_ms,
    }
