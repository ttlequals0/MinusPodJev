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


async def test_readiness_check(client: AsyncClient):
    """Test the readiness check endpoint."""
    response = await client.get("/api/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "database" in data
    assert "environment" in data
    assert "version" in data
