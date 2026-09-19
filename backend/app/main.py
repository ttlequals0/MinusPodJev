"""
Main FastAPI application module.
"""

import logging
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.health import router as health_router
from app.api.openai import router as openai_router
from app.api.status import router as status_router
from app.api.v1.router import api_router as v1_router
from app.config import settings
from app.utils.redact import redact

_LOG_HANDLER_NAME = "jevproxy-stdout"


def _configure_logging() -> None:
    """Log to stdout at LOG_LEVEL. Idempotent: never double-adds a handler."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
    if any(h.get_name() == _LOG_HANDLER_NAME for h in root.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.set_name(_LOG_HANDLER_NAME)
    handler.setFormatter(logging.Formatter(settings.LOG_FORMAT))
    root.addHandler(handler)


_configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Log startup and shutdown."""
    logger.info("Starting up application...")
    yield
    logger.info("Shutting down application...")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    docs_url="/api/docs" if not settings.PRODUCTION else None,
    redoc_url="/api/redoc" if not settings.PRODUCTION else None,
    openapi_url="/api/openapi.json" if not settings.PRODUCTION else None,
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers. Health lives under /api so it is reachable through the nginx
# proxy (which forwards only /api/* to the backend) and matches the container
# HEALTHCHECK.
app.include_router(health_router, prefix="/api", tags=["health"])
app.include_router(status_router, prefix="/api", tags=["status"])
app.include_router(v1_router, prefix="/api/v1")

# OpenAI-compatible surface at the root, mounted twice so MinusPod reaches it
# whether or not its base_url includes /v1.
app.include_router(openai_router, tags=["openai"])
app.include_router(openai_router, prefix="/v1", tags=["openai"])


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    """Custom 404 handler."""
    return JSONResponse(
        status_code=404,
        content={"detail": "Not found"},
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Custom 500 handler."""
    logger.error("Internal server error: %s", redact(str(exc)))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
