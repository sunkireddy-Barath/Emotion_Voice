"""POST /stream (SSE) and WebSocket /ws/stream — real-time audio streaming."""
from __future__ import annotations

import asyncio
import json
import logging
import struct
from typing import Optional

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from ..schemas.requests import StreamRequest

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Streaming"])


def get_streaming_tts():
    from ..dependencies import get_streaming_engine
    return get_streaming_engine()


@router.post("/stream")
async def stream_http(
    request: StreamRequest,
    streaming=Depends(get_streaming_tts),
):
    """Server-Sent Events audio stream. Each event contains a WAV chunk."""

    async def event_generator():
        chunk_id = 0
        async for chunk in streaming.stream(
            text=request.text,
            voice_id=request.voice_id,
            language=request.language,
            emotion=request.emotion,
            intensity=request.intensity,
            speed=request.speed,
            sample_rate=request.sample_rate,
        ):
            import base64
            meta = {
                "chunk_id": chunk.chunk_id,
                "duration_sec": chunk.duration_sec,
                "emotion": chunk.emotion,
                "is_last": chunk.is_last,
                "latency_ms": chunk.latency_ms,
                "audio_base64": base64.b64encode(chunk.audio_bytes).decode(),
            }
            yield f"data: {json.dumps(meta)}\n\n"
            chunk_id += 1

        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.websocket("/ws/stream")
async def websocket_stream(
    websocket: WebSocket,
    streaming=Depends(get_streaming_tts),
):
    """WebSocket audio stream.

    Client sends JSON: {text, voice_id, language, emotion, ...}
    Server sends binary frames: [4-byte meta_len][JSON meta][WAV bytes]
    Then sends text frame: {"type": "stream_end"} when done.
    """
    await websocket.accept()
    logger.info("WebSocket stream connection opened")

    try:
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "ping"}))
                continue

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
                continue

            if data.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                continue

            text = data.get("text", "")
            if not text:
                await websocket.send_text(json.dumps({"type": "error", "message": "Empty text"}))
                continue

            await streaming.stream_websocket(
                websocket=websocket,
                text=text,
                voice_id=data.get("voice_id", "default"),
                language=data.get("language", "en"),
                emotion=data.get("emotion"),
                intensity=data.get("intensity"),
                speed=data.get("speed"),
                sample_rate=data.get("sample_rate", 22050),
            )

    except WebSocketDisconnect:
        logger.info("WebSocket stream connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass
