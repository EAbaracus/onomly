import time
import tempfile
import os
from datetime import datetime
from typing import Generator

import pytest
import aiosqlite

from launch_engine.cache import SQLiteCache
from launch_engine.core.validation import (
    ValidationResult,
    ValidationStatus,
    Confidence,
    ValidationChannel,
    Evidence,
)


@pytest.fixture
def sample_validation_result() -> ValidationResult:
    """Create a sample ValidationResult for testing."""
    now = datetime.now()
    return ValidationResult(
        target="test-target",
        channel=ValidationChannel.DOMAIN,
        status=ValidationStatus.AVAILABLE,
        confidence=Confidence.CONFIRMED,
        evidence=Evidence(
            source="test-source",
            url="https://example.com",
            checked_at=now,
            raw={"key": "value"},
        ),
        candidate_id="test-candidate-id",
        validation_id="test-validation-id",
        adapter_version="1.0.0",
        checked_at=now,
        manual_review_url="https://review.example.com",
    )


@pytest.fixture
def db_path() -> Generator[str, None, None]:
    """Create a temporary database path for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.mark.asyncio
async def test_initialization_creates_database(db_path: str):
    """Test that initialization creates the database and table."""
    cache = SQLiteCache(db_path)
    await cache.initialize()

    # Check that the database file exists
    assert os.path.exists(db_path)

    # Check that the cache table exists
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='cache'"
        )
        table = await cursor.fetchone()
        await cursor.close()
        assert table is not None
        assert table[0] == "cache"

    await cache.close()


@pytest.mark.asyncio
async def test_set_and_get_operations(
    db_path: str, sample_validation_result: ValidationResult
):
    """Test setting and getting a cache entry."""
    cache = SQLiteCache(db_path)
    await cache.initialize()

    try:
        key = "test-key"
        ttl = 60  # 60 seconds
        context_hash = "test-context-hash"

        # Set the value
        await cache.set(key, sample_validation_result, ttl, context_hash)

        # Get the value
        result = await cache.get(key)

        assert result is not None
        assert result.target == sample_validation_result.target
        assert result.channel == sample_validation_result.channel
        assert result.status == sample_validation_result.status
        assert result.confidence == sample_validation_result.confidence
        assert result.candidate_id == sample_validation_result.candidate_id
        assert result.validation_id == sample_validation_result.validation_id
        assert result.adapter_version == sample_validation_result.adapter_version
    finally:
        await cache.close()


@pytest.mark.asyncio
async def test_ttl_expiration(db_path: str, sample_validation_result: ValidationResult):
    """Test that cache entries expire after TTL."""
    cache = SQLiteCache(db_path)
    await cache.initialize()

    try:
        key = "test-key-ttl"
        ttl = 1  # 1 second
        context_hash = "test-context-hash-ttl"

        # Set the value
        await cache.set(key, sample_validation_result, ttl, context_hash)

        # Immediately get should work
        result = await cache.get(key)
        assert result is not None

        # Wait for expiration
        time.sleep(1.1)  # Wait slightly longer than TTL

        # After expiration, get should return None
        result = await cache.get(key)
        assert result is None
    finally:
        await cache.close()


@pytest.mark.asyncio
async def test_cache_miss_returns_none(db_path: str):
    """Test that getting a non-existent key returns None."""
    cache = SQLiteCache(db_path)
    await cache.initialize()

    try:
        result = await cache.get("non-existent-key")
        assert result is None
    finally:
        await cache.close()


@pytest.mark.asyncio
async def test_clear_removes_all_entries(
    db_path: str, sample_validation_result: ValidationResult
):
    """Test that clear removes all cache entries."""
    cache = SQLiteCache(db_path)
    await cache.initialize()

    try:
        # Add two entries
        await cache.set("key1", sample_validation_result, 60, "hash1")
        await cache.set("key2", sample_validation_result, 60, "hash2")

        # Verify both exist
        assert await cache.get("key1") is not None
        assert await cache.get("key2") is not None

        # Clear all entries
        await cache.clear()

        # Verify both are gone
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
    finally:
        await cache.close()


@pytest.mark.asyncio
async def test_version_tracking(
    db_path: str, sample_validation_result: ValidationResult
):
    """Test that version is stored and retrieved correctly."""
    cache = SQLiteCache(db_path)
    await cache.initialize()

    try:
        key = "test-key-version"
        ttl = 60
        context_hash = "test-context-hash-version"
        expected_version = "2.0.0"

        # Create a result with a specific version
        result_with_version = sample_validation_result.model_copy()
        result_with_version.adapter_version = expected_version

        # Set the value
        await cache.set(key, result_with_version, ttl, context_hash)

        # Directly query the database to check version
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute("SELECT version FROM cache WHERE key = ?", (key,))
            row = await cursor.fetchone()
            await cursor.close()
            assert row is not None
            assert row[0] == expected_version

        # Also verify via get
        result = await cache.get(key)
        assert result is not None
        assert result.adapter_version == expected_version
    finally:
        await cache.close()


@pytest.mark.asyncio
async def test_context_hash_storage(
    db_path: str, sample_validation_result: ValidationResult
):
    """Test that context_hash is stored and retrieved correctly."""
    cache = SQLiteCache(db_path)
    await cache.initialize()

    try:
        key = "test-key-context"
        ttl = 60
        expected_context_hash = "test-context-hash-12345"

        # Set the value
        await cache.set(key, sample_validation_result, ttl, expected_context_hash)

        # Directly query the database to check context_hash
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                "SELECT context_hash FROM cache WHERE key = ?", (key,)
            )
            row = await cursor.fetchone()
            await cursor.close()
            assert row is not None
            assert row[0] == expected_context_hash
    finally:
        await cache.close()


@pytest.mark.asyncio
async def test_close_method(db_path: str):
    """Test that close properly closes the database connection."""
    cache = SQLiteCache(db_path)
    await cache.initialize()
    assert cache.db is not None

    await cache.close()
    assert cache.db is None
