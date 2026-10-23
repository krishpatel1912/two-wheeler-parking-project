from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from app.core.config import get_settings

settings = get_settings()

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

db_url = settings.database_url
engine_kwargs = {"future": True}

# If explicitly SQLite or PostgreSQL fails to connect, use local SQLite
if "sqlite" in db_url:
    engine_kwargs["connect_args"] = {"check_same_thread": False}
    engine = create_engine(db_url, **engine_kwargs)
else:
    try:
        test_engine = create_engine(db_url, pool_pre_ping=True, connect_args={"connect_timeout": 2})
        with test_engine.connect():
            pass
        engine = test_engine
    except Exception as exc:
        project_root = Path(__file__).resolve().parents[3]
        sqlite_path = project_root / "data" / "parking.db"
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        fallback_url = f"sqlite:///{sqlite_path}"
        logger.warning(
            f"[DB] PostgreSQL unavailable ({exc}). Seamlessly falling back to SQLite: {fallback_url}"
        )
        engine_kwargs["connect_args"] = {"check_same_thread": False}
        engine = create_engine(fallback_url, **engine_kwargs)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
