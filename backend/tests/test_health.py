"""
Tests for health check endpoints.
"""

from httpx import AsyncClient


async def test_health_check(client: AsyncClient):
    """Test the basic health check endpoint."""
    response = await client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "environment" in data
    assert "version" in data
