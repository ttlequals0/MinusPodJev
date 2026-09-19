"""
API v1 main router.
"""

from fastapi import APIRouter

from app.api.v1.jev import router as jev_router

# Create main API router
api_router = APIRouter()

api_router.include_router(jev_router, tags=["jev"])


@api_router.get("/")
async def root() -> dict[str, str]:
    """API v1 root endpoint."""
    return {"message": "Welcome to the Jev proxy API v1"}
