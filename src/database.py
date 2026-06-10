"""
Async SQLAlchemy database setup for prediction logging.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, JSON, Boolean
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    input_features = Column(JSON, nullable=False)
    probability = Column(Float, nullable=False)
    risk_band = Column(String(16), nullable=False)
    model_version = Column(String(32), nullable=False)
    prediction_time_ms = Column(Float, nullable=True)
    error = Column(String(512), nullable=True)


# Default: empty DATABASE_URL means DB logging is disabled
DATABASE_URL: str = ""

_engine = None
_async_session_maker = None


def init_db(database_url: str = ""):
    """Initialize async engine and session maker. Call at app startup."""
    global _engine, _async_session_maker, DATABASE_URL
    if not database_url:
        DATABASE_URL = ""
        _engine = None
        _async_session_maker = None
        return

    DATABASE_URL = database_url
    _engine = create_async_engine(database_url, echo=False, pool_size=5, max_overflow=10)
    _async_session_maker = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncSession:
    """Dependency that provides an async database session."""
    if _async_session_maker is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    async with _async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


async def create_tables():
    """Create all tables. Call within app lifespan."""
    if _engine is None:
        return
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """Dispose of the engine. Call at app shutdown."""
    global _engine, _async_session_maker
    if _engine:
        await _engine.dispose()
    _engine = None
    _async_session_maker = None


def is_db_initialized() -> bool:
    return _engine is not None


async def log_prediction(
    session: AsyncSession,
    input_features: dict,
    probability: float,
    risk_band: str,
    model_version: str,
    prediction_time_ms: float | None = None,
    error: str | None = None,
) -> PredictionLog:
    """Insert a prediction log entry."""
    log_entry = PredictionLog(
        input_features=input_features,
        probability=probability,
        risk_band=risk_band,
        model_version=model_version,
        prediction_time_ms=prediction_time_ms,
        error=error,
    )
    session.add(log_entry)
    await session.commit()
    await session.refresh(log_entry)
    return log_entry
