"""Production TTS engine.

Model cascade (in order of quality / resource usage):
  1. XTTS v2       — multilingual, voice cloning, ~2.5 GB  (use when disk allows)
  2. Edge TTS      — Microsoft neural voices, emotion SSML styles (DEFAULT — requires internet)
  3. VITS VCTK     — multi-speaker English, ~200 MB        (offline fallback)
  4. VITS LJSpeech — single-speaker English, ~100 MB       (offline fallback)
  5. Silero TTS    — PyTorch hub, ~91 MB, human-like EN    (offline fallback)
  6. pyttsx3       — offline OS voices, 0 MB               (last resort)

Edge TTS uses Microsoft Azure Neural voices via the edge-tts package — fully free,
no API key, but requires internet. Each emotion maps to a specific voice + SSML style.
Emotion-aware prosody post-processing (pitch/speed/energy) is applied on top via librosa.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import os
import subprocess
import tempfile
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from ..emotion.detector import EmotionDetector
from ..emotion.classifier import EmotionResult
from ..prosody.predictor import ProsodyPredictor, ProsodyTarget

logger = logging.getLogger(__name__)

# Edge TTS — each emotion has a distinct voice + strong rate + strong pitch shift.
# Rate and pitch are deliberately large so the emotion is CLEARLY audible.
# Tested: pitch in Hz and rate in % both produce perceptibly different audio
# without inflating duration (unlike SSML express-as/prosody keyword values).
# Format: (voice_name, rate, pitch)
# Edge TTS — normal speed for all emotions. Emotion is expressed through
# voice CHARACTER (different voices) + pitch only. No speed changes.
# Format: (voice_name, rate, pitch)
EDGE_TTS_EMOTION_MAP: Dict[str, tuple] = {
    "neutral":      ("en-US-ChristopherNeural", "+0%", "+0Hz"),   # calm clear male
    "happy":        ("en-US-EmmaNeural",        "+0%", "+20Hz"),  # bright female, slightly higher
    "sad":          ("en-US-MichelleNeural",    "+0%", "-20Hz"),  # warm female, slightly lower
    "angry":        ("en-US-GuyNeural",         "+0%", "-15Hz"),  # deep male, slightly lower
    "excited":      ("en-US-EmmaNeural",        "+0%", "+25Hz"),  # bright female, naturally higher
    "fear":         ("en-US-AriaNeural",        "+0%", "+15Hz"),  # expressive female
    "surprise":     ("en-US-EmmaNeural",        "+0%", "+15Hz"),  # bright female
    "calm":         ("en-US-BrianNeural",       "+0%", "-10Hz"),  # warm male, slightly lower
    "serious":      ("en-US-SteffanNeural",     "+0%", "-15Hz"),  # professional male, lower
    "motivational": ("en-US-BrianNeural",       "+0%", "+10Hz"),  # warm male, slightly higher
    "questioning":  ("en-US-AriaNeural",        "+0%", "+10Hz"),  # expressive female
    "storytelling": ("en-US-MichelleNeural",    "+0%", "+0Hz"),   # warm narrative female
}

# Tamil — same approach: natural pitch shifts, voice character carries emotion
EDGE_TTS_TAMIL_MAP: Dict[str, tuple] = {
    "neutral":      ("ta-IN-ValluvarNeural",  "+0%", "+0Hz"),
    "happy":        ("ta-IN-PallaviNeural",   "+0%", "+20Hz"),
    "sad":          ("ta-IN-PallaviNeural",   "+0%", "-20Hz"),
    "angry":        ("ta-IN-ValluvarNeural",  "+0%", "-15Hz"),
    "excited":      ("ta-IN-PallaviNeural",   "+0%", "+25Hz"),
    "fear":         ("ta-IN-PallaviNeural",   "+0%", "+15Hz"),
    "surprise":     ("ta-IN-PallaviNeural",   "+0%", "+15Hz"),
    "calm":         ("ta-IN-ValluvarNeural",  "+0%", "-10Hz"),
    "serious":      ("ta-IN-ValluvarNeural",  "+0%", "-15Hz"),
    "motivational": ("ta-IN-ValluvarNeural",  "+0%", "+10Hz"),
    "questioning":  ("ta-IN-PallaviNeural",   "+0%", "+10Hz"),
    "storytelling": ("ta-IN-ValluvarNeural",  "+0%", "+0Hz"),
}

# Silero v3_en speakers mapped to emotion styles (en_0–en_7)
SILERO_EMOTION_SPEAKERS: Dict[str, str] = {
    "neutral":      "en_0",   # calm neutral male
    "happy":        "en_2",   # bright, upbeat female
    "sad":          "en_1",   # softer, slower male
    "angry":        "en_3",   # assertive male
    "excited":      "en_4",   # energetic female
    "fear":         "en_5",   # higher-pitched female
    "surprise":     "en_6",   # expressive male
    "calm":         "en_0",   # steady neutral
    "serious":      "en_3",   # deep assertive male
    "motivational": "en_7",   # projecting male
    "questioning":  "en_2",   # questioning female
    "storytelling": "en_6",   # narrative male
}

VCTK_EMOTION_SPEAKERS: Dict[str, str] = {
    # speaker IDs from VITS VCTK model — covering prosodic range
    "neutral":      "p225",  # calm female
    "happy":        "p228",  # bright female
    "sad":          "p245",  # slower male
    "angry":        "p248",  # assertive female
    "excited":      "p261",  # energetic female
    "fear":         "p270",  # high-pitched female
    "surprise":     "p280",  # expressive female
    "calm":         "p233",  # steady male
    "serious":      "p244",  # deep male
    "motivational": "p254",  # projecting male
    "questioning":  "p229",  # questioning female
    "storytelling": "p236",  # narrative female
}


@dataclass
class TTSRequest:
    text: str
    voice_id: str = "default"
    language: str = "en"
    emotion: Optional[str] = None
    intensity: Optional[float] = None
    pitch: Optional[float] = None
    energy: Optional[float] = None
    speed: Optional[float] = None
    sample_rate: int = 22050
    speaker_id: Optional[str] = None  # Direct speaker override for VCTK


@dataclass
class TTSResponse:
    audio_bytes: bytes
    sample_rate: int
    duration_sec: float
    emotion: str
    emotion_intensity: float
    prosody: dict
    model_used: str
    latency_ms: float
    request_id: str = ""


class TTSEngine:
    """Production emotion-aware TTS engine with model cascade and prosody post-processing."""

    SUPPORTED_LANGUAGES = {
        "en", "ta", "hi", "fr", "de", "es", "it", "pt", "ru",
        "zh-cn", "ja", "ko", "nl", "pl", "tr", "cs", "ar",
    }

    def __init__(
        self,
        models_dir: str | Path = "models/tts_model",
        device: str = "cpu",
        model_type: str = "vits",   # "vits", "xtts", or "auto"
        emotion_detector: Optional[EmotionDetector] = None,
        prosody_predictor: Optional[ProsodyPredictor] = None,
    ):
        self.models_dir = Path(models_dir)
        self.device = device
        self.requested_model_type = model_type
        self.emotion_detector = emotion_detector or EmotionDetector()
        self.prosody_predictor = prosody_predictor or ProsodyPredictor()

        self._tts = None
        self._model_type: str = "none"
        self._model_name: str = ""
        self._lock = threading.Lock()
        self._voice_embeddings: Dict[str, str] = {}  # voice_id → wav path
        self._tts_pyttsx3 = None
        self._silero_model = None
        self._silero_sample_rate: int = 24000
        self._edge_tts_available: bool = False

        self._initialize()

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def _initialize(self) -> None:
        if self.requested_model_type == "xtts":
            if self._try_load_xtts():
                return
            logger.warning("XTTS requested but unavailable — falling back")

        # Edge TTS: Microsoft Neural voices, emotion SSML styles, requires internet
        if self._try_load_edge_tts():
            return
        # Offline fallbacks
        if self._try_load_vits_vctk():
            return
        if self._try_load_vits_ljspeech():
            return
        # Silero TTS: PyTorch-native, ~91MB, human-like quality
        if self._try_load_silero():
            return
        if self._try_load_pyttsx3():
            return

        logger.error(
            "No TTS backend available.\n"
            "Install with: pip install TTS\n"
            "Then run: python scripts/download_models.py"
        )

    def _try_load_xtts(self) -> bool:
        try:
            from TTS.api import TTS
            model_id = "tts_models/multilingual/multi-dataset/xtts_v2"
            logger.info("Loading XTTS v2 (multilingual voice cloning)...")
            self._tts = TTS(model_name=model_id, progress_bar=True, gpu=(self.device != "cpu"))
            self._model_type = "xtts"
            self._model_name = model_id
            logger.info("XTTS v2 ready")
            return True
        except Exception as e:
            logger.debug(f"XTTS v2 load failed: {e}")
            return False

    def _try_load_vits_vctk(self) -> bool:
        try:
            from TTS.api import TTS
            model_id = "tts_models/en/vctk/vits"
            logger.info("Loading VITS VCTK (multi-speaker English)...")
            self._tts = TTS(model_name=model_id, progress_bar=True, gpu=(self.device != "cpu"))
            self._model_type = "vits_vctk"
            self._model_name = model_id
            logger.info("VITS VCTK ready")
            return True
        except Exception as e:
            logger.debug(f"VITS VCTK load failed: {e}")
            return False

    def _try_load_vits_ljspeech(self) -> bool:
        try:
            from TTS.api import TTS
            model_id = "tts_models/en/ljspeech/vits"
            logger.info("Loading VITS LJSpeech (single-speaker fallback)...")
            self._tts = TTS(model_name=model_id, progress_bar=True, gpu=(self.device != "cpu"))
            self._model_type = "vits"
            self._model_name = model_id
            logger.info("VITS LJSpeech ready")
            return True
        except Exception as e:
            logger.debug(f"VITS LJSpeech load failed: {e}")
            return False

    def _try_load_edge_tts(self) -> bool:
        try:
            import edge_tts  # noqa: F401 — verify installed
            import concurrent.futures

            # Run connectivity check in a worker thread (has its own event loop),
            # safe to call whether or not uvicorn already owns the main loop.
            async def _ping():
                comm = edge_tts.Communicate("test", "en-US-ChristopherNeural")
                async for chunk in comm.stream():
                    if chunk["type"] == "audio":
                        return True
                return False

            def _run_in_thread():
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(_ping())
                finally:
                    loop.close()

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                connected = ex.submit(_run_in_thread).result(timeout=30)

            if not connected:
                raise RuntimeError("No audio returned from connectivity check")

            self._edge_tts_available = True
            self._model_type = "edge_tts"
            self._model_name = "Microsoft Neural (edge-tts)"
            logger.info(
                "Edge TTS ready — Microsoft Neural voices with SSML emotion styles. "
                "12 emotions → unique voice + style (cheerful, sad, angry, excited, fearful...)."
            )
            return True
        except Exception as e:
            logger.warning(f"Edge TTS unavailable — falling back to offline TTS: {e}")
            return False

    def _try_load_silero(self) -> bool:
        try:
            import torch
            logger.info("Loading Silero TTS (PyTorch hub, downloading ~91 MB on first run)...")
            device = torch.device(self.device)
            model, example_text = torch.hub.load(
                repo_or_dir="snakers4/silero-models",
                model="silero_tts",
                language="en",
                speaker="v3_en",
                trust_repo=True,
            )
            model.to(device)
            self._silero_model = model
            self._silero_sample_rate = 24000
            self._silero_speaker = "en_0"
            self._model_type = "silero"
            self._model_name = "silero_tts_v3_en"
            logger.info("Silero TTS ready (24 kHz, English)")
            return True
        except Exception as e:
            logger.debug(f"Silero TTS load failed: {e}")
            return False

    def _try_load_pyttsx3(self) -> bool:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            # Configure best available voice
            voices = engine.getProperty("voices")
            if voices:
                engine.setProperty("voice", voices[0].id)
            engine.setProperty("rate", 180)
            self._tts_pyttsx3 = engine
            self._model_type = "pyttsx3"
            self._model_name = "pyttsx3"
            logger.warning(
                "Using pyttsx3 OS voices — quality is basic. "
                "Install TTS for significantly better quality: pip install TTS"
            )
            return True
        except Exception as e:
            logger.debug(f"pyttsx3 load failed: {e}")
            return False

    # -------------------------------------------------------------------------
    # Core synthesis
    # -------------------------------------------------------------------------

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        """Full pipeline: text → emotion → prosody → audio → post-process."""
        t0 = time.monotonic()
        request_id = hashlib.md5(f"{request.text}{time.time()}".encode()).hexdigest()[:8]

        # 1. Emotion detection
        emotion_result = self._detect_emotion(request)

        # 2. Prosody prediction
        prosody = self._predict_prosody(request, emotion_result)

        # 3. Synthesis
        with self._lock:
            audio, sr = self._synthesize_raw(request, emotion_result)

        # 4. Post-process: apply prosody
        # Edge TTS already encodes rate+pitch in the voice — skip all librosa processing.
        # Other backends get full pitch/speed/energy post-processing.
        if self._model_type != "edge_tts":
            audio = self._apply_prosody(audio, sr, prosody)
        else:
            # Trim leading/trailing silence for clean edge-tts output
            audio = self._trim_silence(audio, sr)

        # 5. Resample if needed
        if sr != request.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=request.sample_rate)
            sr = request.sample_rate

        # 6. Final normalize + clip
        audio = self._final_normalize(audio)

        audio_bytes = self._to_wav_bytes(audio, sr)
        latency_ms = round((time.monotonic() - t0) * 1000, 1)

        return TTSResponse(
            audio_bytes=audio_bytes,
            sample_rate=sr,
            duration_sec=round(len(audio) / sr, 3),
            emotion=emotion_result.emotion,
            emotion_intensity=round(emotion_result.intensity, 3),
            prosody=prosody.to_dict(),
            model_used=self._model_type,
            latency_ms=latency_ms,
            request_id=request_id,
        )

    def _detect_emotion(self, request: TTSRequest) -> EmotionResult:
        if request.emotion:
            return EmotionResult(
                emotion=request.emotion,
                intensity=request.intensity if request.intensity is not None else 0.85,
                confidence=1.0,
                scores={request.emotion: 1.0},
                raw_label=request.emotion,
            )
        return self.emotion_detector.detect(request.text)

    def _predict_prosody(self, request: TTSRequest, emotion_result: EmotionResult) -> ProsodyTarget:
        prosody = self.prosody_predictor.predict(request.text, emotion_result)
        prosody = self.prosody_predictor.adjust_for_language(prosody, request.language)

        # User explicit overrides take highest priority
        if request.pitch is not None:
            prosody.global_pitch = float(request.pitch)
        if request.energy is not None:
            prosody.global_energy = float(request.energy)
        if request.speed is not None:
            prosody.global_speed = float(request.speed)

        return prosody

    def _synthesize_raw(
        self, request: TTSRequest, emotion_result: EmotionResult
    ) -> Tuple[np.ndarray, int]:
        if self._model_type == "xtts":
            return self._synth_xtts(request)
        elif self._model_type == "edge_tts":
            return self._synth_edge_tts(request, emotion_result)
        elif self._model_type in ("vits_vctk", "vits"):
            return self._synth_vits(request, emotion_result)
        elif self._model_type == "silero":
            return self._synth_silero(request, emotion_result)
        elif self._model_type == "pyttsx3":
            return self._synth_pyttsx3(request)
        else:
            raise RuntimeError(
                "No TTS model loaded. Run: pip install TTS && "
                "python scripts/download_models.py"
            )

    def _synth_edge_tts(
        self, request: TTSRequest, emotion_result: Optional[EmotionResult] = None
    ) -> Tuple[np.ndarray, int]:
        import edge_tts

        emotion = (emotion_result.emotion if emotion_result else None) or "neutral"
        voice, rate, pitch = EDGE_TTS_EMOTION_MAP.get(
            emotion, ("en-US-ChristopherNeural", "+0%", "+0Hz")
        )

        # Language-specific voice selection
        lang = request.language or "en"
        if lang == "ta":
            voice, rate, pitch = EDGE_TTS_TAMIL_MAP.get(
                emotion, ("ta-IN-ValluvarNeural", "+0%", "+0Hz")
            )
        elif lang != "en":
            voice = {
                "hi": "hi-IN-MadhurNeural",
                "fr": "fr-FR-HenriNeural",
                "de": "de-DE-ConradNeural",
                "es": "es-ES-AlvaroNeural",
                "it": "it-IT-DiegoNeural",
                "pt": "pt-BR-AntonioNeural",
                "ru": "ru-RU-DmitryNeural",
                "zh-cn": "zh-CN-YunxiNeural",
                "ja": "ja-JP-KeitaNeural",
                "ko": "ko-KR-InJoonNeural",
                "nl": "nl-NL-MaartenNeural",
                "pl": "pl-PL-MarekNeural",
                "tr": "tr-TR-AhmetNeural",
                "cs": "cs-CZ-AntoninNeural",
                "ar": "ar-SA-HamedNeural",
            }.get(lang, voice)
            rate, pitch = "+0%", "+0Hz"

        text = request.text.strip()
        # volume="+50%" gives maximum clean loudness before our own normalization
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, volume="+50%")

        # Collect MP3 audio chunks in a dedicated thread (avoids event loop conflicts)
        import concurrent.futures

        async def _collect() -> list[bytes]:
            chunks: list[bytes] = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    chunks.append(chunk["data"])
            return chunks

        def _run():
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(_collect())
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            mp3_chunks = ex.submit(_run).result(timeout=30)

        if not mp3_chunks:
            raise RuntimeError("edge-tts returned no audio")

        mp3_data = b"".join(mp3_chunks)

        # ffmpeg: MP3 bytes → 24kHz mono WAV via stdin/stdout (no temp files)
        result = subprocess.run(
            ["ffmpeg", "-f", "mp3", "-i", "pipe:0",
             "-ar", "24000", "-ac", "1", "-f", "wav", "pipe:1",
             "-loglevel", "error"],
            input=mp3_data,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {result.stderr.decode()}")

        audio, sr = sf.read(io.BytesIO(result.stdout), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32), sr

    def _synth_xtts(self, request: TTSRequest) -> Tuple[np.ndarray, int]:
        speaker_wav = self._voice_embeddings.get(request.voice_id)
        language = request.language if request.language in self.SUPPORTED_LANGUAGES else "en"

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            kwargs: dict = {"text": request.text, "file_path": tmp_path, "language": language}
            if speaker_wav and Path(speaker_wav).exists():
                kwargs["speaker_wav"] = speaker_wav
            self._tts.tts_to_file(**kwargs)
            return self._load_wav(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _synth_vits(
        self, request: TTSRequest, emotion_result: EmotionResult
    ) -> Tuple[np.ndarray, int]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            kwargs: dict = {"text": request.text, "file_path": tmp_path}

            # For VCTK: pick speaker by emotion for most expressive output
            if self._model_type == "vits_vctk":
                speaker = (
                    request.speaker_id
                    or VCTK_EMOTION_SPEAKERS.get(emotion_result.emotion, "p225")
                )
                kwargs["speaker"] = speaker

            self._tts.tts_to_file(**kwargs)
            return self._load_wav(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _synth_silero(self, request: TTSRequest, emotion_result: Optional["EmotionResult"] = None) -> Tuple[np.ndarray, int]:
        import torch
        model = self._silero_model
        sr = self._silero_sample_rate
        text = request.text.strip()

        # Pick speaker by emotion
        emotion_name = emotion_result.emotion if emotion_result else "neutral"
        speaker = (
            request.speaker_id
            or SILERO_EMOTION_SPEAKERS.get(emotion_name, "en_0")
        )

        # Silero has a ~300-char limit per call; split on sentence boundaries if needed
        if len(text) > 250:
            chunks = []
            for sentence in text.replace("!", ".").replace("?", ".").split("."):
                sentence = sentence.strip()
                if sentence:
                    chunks.append(sentence + ".")
        else:
            chunks = [text]

        audio_parts = []
        with torch.no_grad():
            for chunk in chunks:
                if not chunk.strip():
                    continue
                audio = model.apply_tts(
                    text=chunk,
                    speaker=speaker,
                    sample_rate=sr,
                )
                audio_parts.append(audio.numpy())

        if not audio_parts:
            return np.zeros(sr, dtype=np.float32), sr

        audio = np.concatenate(audio_parts).astype(np.float32)
        return audio, sr

    def _synth_pyttsx3(self, request: TTSRequest) -> Tuple[np.ndarray, int]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            engine = self._tts_pyttsx3
            engine.save_to_file(request.text, tmp_path)
            engine.runAndWait()
            time.sleep(0.05)
            if Path(tmp_path).exists() and Path(tmp_path).stat().st_size > 100:
                return self._load_wav(tmp_path)
            # Fallback silence
            sr = 22050
            return np.zeros(max(sr, int(sr * len(request.text) / 15)), dtype=np.float32), sr
        except Exception as e:
            logger.error(f"pyttsx3 synthesis error: {e}")
            sr = 22050
            return np.zeros(sr, dtype=np.float32), sr
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # -------------------------------------------------------------------------
    # Post-processing
    # -------------------------------------------------------------------------

    def _trim_silence(self, audio: np.ndarray, sr: int, threshold_db: float = -40.0) -> np.ndarray:
        """Trim leading and trailing silence below threshold_db."""
        try:
            import librosa
            trimmed, _ = librosa.effects.trim(audio, top_db=abs(threshold_db), frame_length=512, hop_length=128)
            return trimmed.astype(np.float32)
        except Exception:
            return audio

    def _apply_prosody(
        self, audio: np.ndarray, sr: int, prosody: ProsodyTarget
    ) -> np.ndarray:
        """Apply pitch shift, time stretch, and energy scaling via librosa."""
        try:
            import librosa

            # Time stretch (speed)
            speed = float(prosody.global_speed)
            if abs(speed - 1.0) > 0.03:
                audio = librosa.effects.time_stretch(audio, rate=speed)

            # Pitch shift (in semitones, from scale factor)
            pitch_scale = float(prosody.global_pitch)
            if abs(pitch_scale - 1.0) > 0.03:
                n_steps = float(12 * np.log2(max(0.1, pitch_scale)))
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

            # Energy (volume) scaling
            energy = float(prosody.global_energy)
            if abs(energy - 1.0) > 0.03:
                audio = audio * energy

        except ImportError:
            logger.warning("librosa not available — prosody post-processing skipped")
        except Exception as e:
            logger.warning(f"Prosody post-processing failed: {e}")

        return audio.astype(np.float32)

    def _final_normalize(self, audio: np.ndarray) -> np.ndarray:
        """Peak normalize to 0 dBFS (maximum loudness without clipping)."""
        peak = np.max(np.abs(audio))
        if peak > 1e-6:
            audio = audio / peak  # normalize to exactly ±1.0 peak
        return np.clip(audio, -1.0, 1.0).astype(np.float32)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _load_wav(path: str) -> Tuple[np.ndarray, int]:
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32), sr

    @staticmethod
    def _to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    # -------------------------------------------------------------------------
    # Voice management
    # -------------------------------------------------------------------------

    def register_voice(self, voice_id: str, wav_path: str) -> None:
        if not Path(wav_path).exists():
            raise FileNotFoundError(f"Voice sample not found: {wav_path}")
        self._voice_embeddings[voice_id] = wav_path
        logger.info(f"Voice registered: {voice_id}")

    def unregister_voice(self, voice_id: str) -> None:
        self._voice_embeddings.pop(voice_id, None)

    def list_voices(self) -> List[Dict]:
        voices = [
            {
                "id": "default",
                "name": "Default",
                "type": "built-in",
                "model": self._model_type,
            }
        ]
        for vid, wav_path in self._voice_embeddings.items():
            voices.append({"id": vid, "name": vid, "type": "cloned", "wav": wav_path})
        return voices

    def health(self) -> Dict:
        return {
            "model_type": self._model_type,
            "model_name": self._model_name,
            "device": self.device,
            "voices_registered": len(self._voice_embeddings),
            "ready": self._model_type != "none",
            "supports_voice_cloning": self._model_type == "xtts",
            "supports_emotion_styles": self._model_type == "edge_tts",
            "supported_languages": sorted(self.SUPPORTED_LANGUAGES),
        }
