"""Voice profile manager: create, store, load, and validate cloned voice profiles."""
from __future__ import annotations

import json
import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class VoiceProfile:
    def __init__(
        self,
        voice_id: str,
        name: str,
        samples_dir: Path,
        language: str = "en",
        description: str = "",
    ):
        self.voice_id = voice_id
        self.name = name
        self.samples_dir = samples_dir
        self.language = language
        self.description = description
        self.created_at = datetime.utcnow().isoformat()
        self.sample_files: List[str] = []
        self.total_duration_sec: float = 0.0
        self.best_sample: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "voice_id": self.voice_id,
            "name": self.name,
            "language": self.language,
            "description": self.description,
            "created_at": self.created_at,
            "sample_files": self.sample_files,
            "total_duration_sec": round(self.total_duration_sec, 2),
            "best_sample": self.best_sample,
        }

    @classmethod
    def from_dict(cls, data: Dict, samples_dir: Path) -> "VoiceProfile":
        profile = cls(
            voice_id=data["voice_id"],
            name=data["name"],
            samples_dir=samples_dir,
            language=data.get("language", "en"),
            description=data.get("description", ""),
        )
        profile.created_at = data.get("created_at", "")
        profile.sample_files = data.get("sample_files", [])
        profile.total_duration_sec = data.get("total_duration_sec", 0.0)
        profile.best_sample = data.get("best_sample")
        return profile


class VoiceManager:
    """Manages voice profiles for cloning and personalization."""

    MIN_DURATION_SEC = 3.0
    MAX_DURATION_SEC = 30.0
    SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    def __init__(self, profiles_dir: str | Path):
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self._profiles: Dict[str, VoiceProfile] = {}
        self._load_all_profiles()

    def _load_all_profiles(self) -> None:
        for meta_file in self.profiles_dir.glob("*/profile.json"):
            try:
                with open(meta_file) as f:
                    data = json.load(f)
                voice_id = data["voice_id"]
                samples_dir = meta_file.parent / "samples"
                profile = VoiceProfile.from_dict(data, samples_dir)
                self._profiles[voice_id] = profile
            except Exception as e:
                logger.warning(f"Failed to load profile from {meta_file}: {e}")

        logger.info(f"Loaded {len(self._profiles)} voice profiles")

    def create_voice(
        self,
        name: str,
        language: str = "en",
        description: str = "",
        voice_id: Optional[str] = None,
    ) -> VoiceProfile:
        if not voice_id:
            voice_id = str(uuid.uuid4())[:8]

        voice_dir = self.profiles_dir / voice_id
        samples_dir = voice_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        profile = VoiceProfile(
            voice_id=voice_id,
            name=name,
            samples_dir=samples_dir,
            language=language,
            description=description,
        )
        self._profiles[voice_id] = profile
        self._save_profile(profile)
        logger.info(f"Voice profile created: {voice_id} ({name})")
        return profile

    def add_sample(
        self, voice_id: str, audio_bytes: bytes, filename: str
    ) -> Dict:
        profile = self._get_profile(voice_id)
        suffix = Path(filename).suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {suffix}. Use: {self.SUPPORTED_FORMATS}")

        dest = profile.samples_dir / filename
        dest.write_bytes(audio_bytes)

        # Validate and measure duration
        try:
            data, sr = sf.read(str(dest), dtype="float32")
            duration = len(data) / sr
            if duration < self.MIN_DURATION_SEC:
                dest.unlink()
                raise ValueError(f"Sample too short: {duration:.1f}s (min {self.MIN_DURATION_SEC}s)")
            if duration > self.MAX_DURATION_SEC:
                dest.unlink()
                raise ValueError(f"Sample too long: {duration:.1f}s (max {self.MAX_DURATION_SEC}s)")

            profile.sample_files.append(str(dest))
            profile.total_duration_sec += duration

            # Track best sample (longest clean sample, up to 15s)
            if (profile.best_sample is None or
                    (10.0 <= duration <= 15.0) or
                    (duration > 5.0 and profile.best_sample is None)):
                profile.best_sample = str(dest)

            self._save_profile(profile)
            return {"filename": filename, "duration": duration, "total_samples": len(profile.sample_files)}

        except sf.LibsndfileError as e:
            dest.unlink(missing_ok=True)
            raise ValueError(f"Invalid audio file: {e}")

    def get_best_sample(self, voice_id: str) -> Optional[str]:
        profile = self._get_profile(voice_id)
        return profile.best_sample

    def get_voice(self, voice_id: str) -> Optional[VoiceProfile]:
        return self._profiles.get(voice_id)

    def list_voices(self) -> List[Dict]:
        return [p.to_dict() for p in self._profiles.values()]

    def delete_voice(self, voice_id: str) -> None:
        if voice_id not in self._profiles:
            raise KeyError(f"Voice not found: {voice_id}")
        voice_dir = self.profiles_dir / voice_id
        shutil.rmtree(voice_dir, ignore_errors=True)
        del self._profiles[voice_id]
        logger.info(f"Voice deleted: {voice_id}")

    def validate_voice(self, voice_id: str) -> Dict:
        profile = self._get_profile(voice_id)
        issues = []
        if len(profile.sample_files) == 0:
            issues.append("No samples uploaded")
        if profile.total_duration_sec < self.MIN_DURATION_SEC * 3:
            issues.append(f"Insufficient audio: {profile.total_duration_sec:.1f}s (recommend 15s+)")
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "sample_count": len(profile.sample_files),
            "total_duration": profile.total_duration_sec,
        }

    def _get_profile(self, voice_id: str) -> VoiceProfile:
        if voice_id not in self._profiles:
            raise KeyError(f"Voice profile not found: {voice_id}")
        return self._profiles[voice_id]

    def _save_profile(self, profile: VoiceProfile) -> None:
        meta_path = self.profiles_dir / profile.voice_id / "profile.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)
