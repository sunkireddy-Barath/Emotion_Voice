"""Database connection, session management, and initialization."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session

from .models import Base

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None


def get_engine(database_url: str):
    global _engine
    if _engine is not None:
        return _engine

    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    _engine = create_engine(
        database_url,
        connect_args=connect_args,
        echo=False,
        pool_pre_ping=True,
    )

    # Enable WAL mode for SQLite (better concurrent performance)
    if database_url.startswith("sqlite"):
        @event.listens_for(_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return _engine


def init_db(database_url: str) -> None:
    """Create all tables. Safe to call multiple times."""
    engine = get_engine(database_url)
    Base.metadata.create_all(bind=engine)
    logger.info(f"Database initialized: {database_url}")


def get_session_factory(database_url: str) -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine(database_url)
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return _SessionLocal


def get_db(database_url: str) -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session."""
    factory = get_session_factory(database_url)
    db = factory()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
