"""Tests for the DomainAdapter."""

from __future__ import annotations

import unittest.mock

import httpx
import pytest

from launch_engine.validation.adapters.domain import DomainAdapter


@pytest.fixture
def domain_adapter():
    """Return DomainAdapter instance."""
    return DomainAdapter()


@pytest.mark.asyncio
async def test_domain_available_via_notice(domain_adapter):
    """Test domain available via rdap.org notice."""
    mock_response = unittest.mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "notice": [{"title": "Object not found", "description": ["Domain not found"]}]
    }

    with unittest.mock.patch.object(
        domain_adapter._client, "get", return_value=mock_response
    ):
        result = await domain_adapter.validate("example.com")

    assert result.status == "available"
    assert result.domain == "example.com"
    assert result.details is not None
    assert "rdap_data" in result.details


@pytest.mark.asyncio
async def test_domain_taken(domain_adapter):
    """Test domain taken (no 'Object not found' notice)."""
    mock_response = unittest.mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "notice": [{"title": "Registration", "description": ["Domain registered"]}]
    }

    with unittest.mock.patch.object(
        domain_adapter._client, "get", return_value=mock_response
    ):
        result = await domain_adapter.validate("example.com")

    assert result.status == "taken"
    assert result.domain == "example.com"
    assert result.details is not None
    assert "rdap_data" in result.details


@pytest.mark.asyncio
async def test_domain_available_via_404(domain_adapter):
    """Test domain available via 404 status code."""
    mock_response = unittest.mock.Mock()
    mock_response.status_code = 404

    with unittest.mock.patch.object(
        domain_adapter._client, "get", return_value=mock_response
    ):
        result = await domain_adapter.validate("example.com")

    assert result.status == "available"
    assert result.domain == "example.com"


@pytest.mark.asyncio
async def test_domain_timeout(domain_adapter):
    """Test domain check timeout."""
    with unittest.mock.patch.object(
        domain_adapter._client,
        "get",
        side_effect=httpx.TimeoutException("Timeout"),
    ):
        result = await domain_adapter.validate("example.com")

    assert result.status == "unverifiable"
    assert result.domain == "example.com"
    assert "Timeout" in result.error


@pytest.mark.asyncio
async def test_domain_network_error(domain_adapter):
    """Test domain check network error."""
    with unittest.mock.patch.object(
        domain_adapter._client,
        "get",
        side_effect=httpx.NetworkError("Network error"),
    ):
        result = await domain_adapter.validate("example.com")

    assert result.status == "unverifiable"
    assert result.domain == "example.com"
    assert "Network error" in result.error


@pytest.mark.asyncio
async def test_domain_url_encoding(domain_adapter):
    """Test that domain inputs are correctly URL encoded."""
    mock_response = unittest.mock.Mock()
    mock_response.status_code = 404

    with unittest.mock.patch.object(
        domain_adapter._client, "get", return_value=mock_response
    ) as mock_get:
        await domain_adapter.validate("invalid/domain.com")
        mock_get.assert_called_once_with("https://rdap.org/domain/invalid%2Fdomain.com")


@pytest.mark.asyncio
async def test_domain_empty(domain_adapter):
    """Test empty domain."""
    result = await domain_adapter.validate("")
    assert result.status == "unverifiable"
    assert result.error == "Empty domain"


@pytest.mark.asyncio
async def test_domain_adapter_policy(domain_adapter):
    """Test that the adapter has the correct policy."""
    assert domain_adapter.version == "1.0.0"
    assert domain_adapter.policy.rate_limit_per_minute == 60
    assert domain_adapter.policy.cache_ttl_seconds == 86400
    assert domain_adapter.policy.timeout_seconds == 30.0


@pytest.mark.asyncio
async def test_domain_close(domain_adapter):
    """Test that we can close the adapter."""
    with unittest.mock.patch.object(
        domain_adapter._client, "aclose", return_value=unittest.mock.Mock()
    ) as mock_close:
        await domain_adapter.aclose()
        mock_close.assert_called_once()
