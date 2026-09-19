"""
Start the application server.
"""

import uvicorn
from app.config import settings


def main():
    """Start the FastAPI application."""
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS if not settings.DEBUG else 1,
    )


if __name__ == "__main__":
    main()
