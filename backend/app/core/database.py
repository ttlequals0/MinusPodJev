"""
Database configuration and session management.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from app.config import settings

_url = settings.database_url_async

# SQLite (aiosqlite) uses a NullPool, which rejects connection-pool sizing arguments.
_pool_kwargs: dict[str, int] = (
    {} if _url.startswith("sqlite") else {"pool_size": 5, "max_overflow": 10}
)

engine = create_async_engine(
    _url,
    echo=settings.DEBUG and not settings.TESTING,
    future=True,
    pool_pre_ping=True,
    **_pool_kwargs,
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Create base class for models
Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get database session.
    Yields an async database session and ensures it's closed after use.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
