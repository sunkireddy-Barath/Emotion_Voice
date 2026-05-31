"""SQLAlchemy ORM models for voice profiles, synthesis history, and metrics."""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, String, Text, JSON, Index
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class VoiceProfileDB(Base):
    __tablename__ = "voice_profiles"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4())[:8])
    name = Column(String(100), nullable=False)
    language = Column(String(10), nullable=False, default="en")
    description = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    best_sample_path = Column(String(512), nullable=True)

    samples = relationship("VoiceSampleDB", back_populates="profile",
                           cascade="all, delete-orphan")
    syntheses = relationship("SynthesisHistoryDB", back_populates="voice_profile")

    __table_args__ = (
        Index("ix_voice_profiles_name", "name"),
    )

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def total_duration_sec(self) -> float:
        return sum(s.duration_sec for s in self.samples)


class VoiceSampleDB(Base):
    __tablename__ = "voice_samples"

    id = Column(Integer, primary_key=True, autoincrement=True)
    profile_id = Column(String(36), ForeignKey("voice_profiles.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    duration_sec = Column(Float, nullable=False)
    sample_rate = Column(Integer, default=22050)
    file_size_bytes = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    profile = relationship("VoiceProfileDB", back_populates="samples")

    __table_args__ = (
        Index("ix_voice_samples_profile", "profile_id"),
    )


class SynthesisHistoryDB(Base):
    __tablename__ = "synthesis_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(36), unique=True,
                        default=lambda: str(uuid.uuid4()))
    text = Column(Text, nullable=False)
    text_length = Column(Integer, default=0)
    voice_id = Column(String(36), ForeignKey("voice_profiles.id"), nullable=True)
    language = Column(String(10), default="en")
    emotion_requested = Column(String(50), nullable=True)
    emotion_detected = Column(String(50), nullable=True)
    emotion_intensity = Column(Float, nullable=True)
    pitch = Column(Float, nullable=True)
    energy = Column(Float, nullable=True)
    speed = Column(Float, nullable=True)
    model_used = Column(String(100), nullable=True)
    duration_sec = Column(Float, nullable=True)
    latency_ms = Column(Float, nullable=True)
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    client_ip = Column(String(50), nullable=True)

    voice_profile = relationship("VoiceProfileDB", back_populates="syntheses")

    __table_args__ = (
        Index("ix_synthesis_history_created", "created_at"),
        Index("ix_synthesis_history_voice", "voice_id"),
    )
