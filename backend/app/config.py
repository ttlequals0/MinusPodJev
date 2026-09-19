"""
Application configuration module.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    # Application
    APP_NAME: str = Field(default="Jev Proxy", description="Application name")
    APP_VERSION: str = Field(default="0.1.0", description="Application version")
    APP_DESCRIPTION: str = Field(
        default="Thin HTTP proxy around the TypeSafe Jev System One endpoint",
        description="Application description",
    )

    # Environment
    ENVIRONMENT: str = Field(default="development", description="Environment name")
    PRODUCTION: bool = Field(default=False, description="Production mode flag")
    DEBUG: bool = Field(default=True, description="Debug mode flag")

    # Server
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    WORKERS: int = Field(default=1, description="Number of worker processes")

    # TypeSafe Jev upstream
    TYPESAFE_API_KEY: str | None = Field(
        default=None, description="Bearer token for the TypeSafe Jev API"
    )
    TYPESAFE_API_URL: str = Field(
        default="https://api.typesafe.ai/v1/systemone", description="Jev upstream URL"
    )
    JEV_MODEL: str = Field(default="jev-latest", description="Upstream Jev model name sent to TypeSafe")
    JEV_PUBLIC_MODEL: str = Field(
        default="typesafe/jev",
        description="Model id advertised on /models and echoed to OpenAI callers",
    )
    JEV_CACHE_PATH: str = Field(
        default="./jev_cache.json", description="Path of the on-disk JSON response cache"
    )
    JEV_TIMEOUT_SECONDS: float = Field(
        default=60.0, description="Upstream HTTP timeout in seconds"
    )
    JEV_ENTER: float = Field(
        default=0.95, description="Span-open probability threshold (noul >= enter)"
    )
    JEV_STAY: float = Field(
        default=0.40, description="Span-extend probability threshold (noul >= stay)"
    )
    JEV_CATEGORY_PASS: bool = Field(
        default=True, description="Run the per-span category second pass"
    )
    JEV_CATEGORY_CONTEXT: int = Field(
        default=2, description="Segments of before/after context in a category pass state"
    )
    JEV_DEFAULT_CATEGORY: str = Field(
        default="sponsor", description="Category used when the category pass is off"
    )

    # Sponsor lookup
    MINUSPOD_SPONSORS_URL: str | None = Field(
        default=None,
        description="Full URL of MinusPod's GET /sponsors; None uses the SEED_SPONSORS gazetteer",
    )
    MINUSPOD_API_TOKEN: str | None = Field(
        default=None, description="Optional bearer token for the MinusPod sponsors call"
    )
    SPONSOR_CACHE_TTL_SECONDS: float = Field(
        default=3600.0, description="TTL of the cached MinusPod sponsor matcher in seconds"
    )

    # Database
    DATABASE_TYPE: str = Field(default="sqlite", description="Database type (sqlite or postgresql)")
    DATABASE_URL: str | None = Field(default=None, description="Database connection URL")

    # PostgreSQL specific settings
    POSTGRES_HOST: str = Field(default="localhost", description="PostgreSQL host")
    POSTGRES_PORT: int = Field(default=5432, description="PostgreSQL port")
    POSTGRES_USER: str = Field(default="postgres", description="PostgreSQL user")
    POSTGRES_PASSWORD: str = Field(default="postgres", description="PostgreSQL password")
    POSTGRES_DB: str = Field(default="webapp", description="PostgreSQL database name")

    # SQLite specific settings
    SQLITE_DATABASE_PATH: str = Field(
        default="./webapp.db", description="SQLite database file path"
    )

    # Security
    SECRET_KEY: str = Field(
        default="change-me-in-production-use-a-long-random-string",
        description="Secret key for encryption",
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30, description="Access token expiration time in minutes"
    )
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default=7, description="Refresh token expiration time in days"
    )
    PASSWORD_MIN_LENGTH: int = Field(default=8, description="Minimum password length")

    # CORS
    CORS_ORIGINS: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        description="Allowed CORS origins",
    )

    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format",
    )

    # Rate limiting
    RATE_LIMIT_ENABLED: bool = Field(default=True, description="Enable rate limiting")
    RATE_LIMIT_REQUESTS: int = Field(default=100, description="Number of requests allowed")
    RATE_LIMIT_PERIOD: int = Field(default=60, description="Time period in seconds")

    # Testing
    TESTING: bool = Field(default=False, description="Testing mode flag")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="allow",
    )

    @property
    def database_url_async(self) -> str:
        """Get async database URL."""
        if self.DATABASE_URL:
            # Convert postgresql:// to postgresql+asyncpg://
            if self.DATABASE_URL.startswith("postgresql://"):
                return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
            return self.DATABASE_URL

        if self.DATABASE_TYPE == "postgresql":
            return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        else:
            return f"sqlite+aiosqlite:///{self.SQLITE_DATABASE_PATH}"

    @property
    def database_url_sync(self) -> str:
        """Get sync database URL (for Alembic migrations)."""
        if self.DATABASE_URL:
            # Keep postgresql:// for sync connections
            if self.DATABASE_URL.startswith("postgresql+asyncpg://"):
                return self.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
            return self.DATABASE_URL

        if self.DATABASE_TYPE == "postgresql":
            return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        else:
            return f"sqlite:///{self.SQLITE_DATABASE_PATH}"

    def require_jev_api_key(self) -> str:
        """Return the upstream bearer token or raise a clear config error."""
        if not self.TYPESAFE_API_KEY:
            raise ValueError(
                "TYPESAFE_API_KEY is not set. Export it or put it in .env before "
                "calling the Jev endpoint."
            )
        return self.TYPESAFE_API_KEY


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Create settings instance
settings = get_settings()
