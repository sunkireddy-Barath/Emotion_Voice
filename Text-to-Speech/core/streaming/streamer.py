"""Real-time streaming TTS engine.

Strategy:
  1. Split text into sentence chunks.
  2. Synthesize each chunk independently.
  3. Yield audio bytes as each chunk completes.
  4. Target first-chunk latency < 500ms.
"""
from __future__ import annotations

import asyncio
import io
import logging
import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    chunk_id: int
    audio_bytes: bytes
    sample_rate: int
    duration_sec: float
    emotion: str
    is_last: bool = False
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "sample_rate": self.sample_rate,
            "duration_sec": round(self.duration_sec, 3),
            "emotion": self.emotion,
            "is_last": self.is_last,
            "latency_ms": round(self.latency_ms, 1),
        }


class TextSplitter:
    """Split text into synthesis-ready chunks respecting sentence boundaries."""

    SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
    COMMA_PAUSE = re.compile(r",\s+")

    def split(
        self,
        text: str,
        max_chars: int = 200,
        min_chars: int = 20,
    ) -> List[str]:
        text = text.strip()
        if not text:
            return []

        # First split on sentence boundaries
        sentences = self.SENTENCE_END.split(text)
        chunks = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current) + len(sentence) <= max_chars:
                current = (current + " " + sentence).strip() if current else sentence
            else:
                if current:
                    chunks.append(current)
                # If single sentence is too long, split on commas
                if len(sentence) > max_chars:
                    sub_parts = self.COMMA_PAUSE.split(sentence)
                    sub_buf = ""
                    for part in sub_parts:
                        if len(sub_buf) + len(part) <= max_chars:
                            sub_buf = (sub_buf + ", " + part).strip(", ").strip() if sub_buf else part
                        else:
                            if sub_buf:
                                chunks.append(sub_buf)
                            sub_buf = part
                    if sub_buf:
                        chunks.append(sub_buf)
                    current = ""
                else:
                    current = sentence

        if current:
            chunks.append(current)

        # Merge very short chunks — but never exceed max_chars
        merged = []
        i = 0
        while i < len(chunks):
            if len(chunks[i]) < min_chars and merged:
                candidate = merged[-1] + " " + chunks[i]
                if len(candidate) <= max_chars:
                    merged[-1] = candidate
                else:
                    merged.append(chunks[i])
            else:
                merged.append(chunks[i])
            i += 1

        return [c.strip() for c in merged if c.strip()]


class StreamingTTS:
    """Async streaming wrapper around TTSEngine that yields audio chunks."""

    def __init__(
        self,
        tts_engine,
        chunk_size_tokens: int = 20,
        target_latency_ms: int = 500,
        max_queue_size: int = 8,
    ):
        self.tts_engine = tts_engine
        self.chunk_size_tokens = chunk_size_tokens
        self.target_latency_ms = target_latency_ms
        self.max_queue_size = max_queue_size
        self.splitter = TextSplitter()

    async def stream(
        self,
        text: str,
        voice_id: str = "default",
        language: str = "en",
        emotion: Optional[str] = None,
        intensity: Optional[float] = None,
        speed: Optional[float] = None,
        sample_rate: int = 22050,
    ) -> AsyncIterator[AudioChunk]:
        from ..tts.engine import TTSRequest

        chunks = self.splitter.split(text)
        if not chunks:
            return

        total = len(chunks)
        session_start = time.monotonic()

        for i, chunk_text in enumerate(chunks):
            t0 = time.monotonic()

            request = TTSRequest(
                text=chunk_text,
                voice_id=voice_id,
                language=language,
                emotion=emotion,
                intensity=intensity,
                speed=speed,
                sample_rate=sample_rate,
            )

            # Run synthesis in thread pool to not block the event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, self.tts_engine.synthesize, request
            )

            chunk_latency = (time.monotonic() - t0) * 1000

            yield AudioChunk(
                chunk_id=i,
                audio_bytes=response.audio_bytes,
                sample_rate=response.sample_rate,
                duration_sec=response.duration_sec,
                emotion=response.emotion,
                is_last=(i == total - 1),
                latency_ms=chunk_latency,
            )

            logger.debug(
                f"Chunk {i+1}/{total}: {len(chunk_text)} chars, "
                f"{response.duration_sec:.2f}s audio, {chunk_latency:.0f}ms latency"
            )

    async def stream_websocket(
        self,
        websocket,
        text: str,
        **kwargs,
    ) -> None:
        """Stream chunks directly over a WebSocket connection."""
        import json
        import struct

        try:
            async for chunk in self.stream(text, **kwargs):
                # Send metadata as JSON header
                meta = json.dumps(chunk.to_dict()).encode()
                header = struct.pack(">I", len(meta)) + meta
                await websocket.send_bytes(header + chunk.audio_bytes)

                if chunk.is_last:
                    await websocket.send_text(json.dumps({"type": "stream_end"}))

        except Exception as e:
            logger.error(f"WebSocket stream error: {e}")
            import json
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))

    def stream_sync(
        self,
        text: str,
        **kwargs,
    ):
        """Synchronous generator for non-async contexts."""
        from ..tts.engine import TTSRequest

        chunks = self.splitter.split(text)
        for i, chunk_text in enumerate(chunks):
            t0 = time.monotonic()
            request = TTSRequest(text=chunk_text, **kwargs)
            response = self.tts_engine.synthesize(request)
            yield AudioChunk(
                chunk_id=i,
                audio_bytes=response.audio_bytes,
                sample_rate=response.sample_rate,
                duration_sec=response.duration_sec,
                emotion=response.emotion,
                is_last=(i == len(chunks) - 1),
                latency_ms=(time.monotonic() - t0) * 1000,
            )
