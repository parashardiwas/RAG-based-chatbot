"""
Async SQLAlchemy engine and session factory for PostgreSQL with pgvector.

Provides:
    - async_engine: AsyncEngine bound to DATABASE_URL
    - async_session_maker: sessionmaker producing AsyncSession instances
    - get_db(): FastAPI dependency that yields an AsyncSession
    - init_db(): creates all tables and enables the pgvector extension
"""

import os
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy import text

from app.db.models import Base

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/ragbot",
)

# ---------------------------------------------------------------------------
# Engine & session factory
# ---------------------------------------------------------------------------

async_engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
)

async_session_maker = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# ---------------------------------------------------------------------------
# Dependency injection helper
# ---------------------------------------------------------------------------


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that provides a transactional database session.

    Usage::

        @router.get("/items")
        async def list_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with async_session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


async def init_db() -> None:
    """Create all tables and enable the pgvector extension.

    Call this once at application startup (e.g. in a lifespan handler).
    """
    async with async_engine.begin() as conn:
        # Enable pgvector extension (requires superuser or CREATE privilege)
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # Create tables that don't exist yet
        await conn.run_sync(Base.metadata.create_all)
