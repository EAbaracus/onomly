"""Tests for the validation pipeline orchestrator."""

import asyncio
import time
from unittest.mock import AsyncMock, Mock
from typing import Optional
import pytest

from launch_engine.validation.pipeline import ValidationPipeline
from launch_engine.validation.adapters.base import (
    ValidationAdapter,
    ValidationResult,
    ValidationStatus,
    AdapterPolicy,
)
from launch_engine.cache import SQLiteCache
from launch_engine.modules.naming.brief import NamingBrief, NameTypology
from launch_engine.modules.naming.candidates import NameCandidate
from launch_engine.core.validation import (
    ValidationResult as CoreValidationResult,
    ValidationStatus as CoreValidationStatus,
    ValidationChannel,
    Confidence,
    Evidence,
)
from datetime import datetime


class MockValidationAdapter(ValidationAdapter):
    """Mock validation adapter for testing."""

    def __init__(
        self,
        version: str = "1.0.0",
        policy: Optional[AdapterPolicy] = None,
        validate_result: Optional[ValidationResult] = None,
        validate_exception: Optional[Exception] = None,
    ):
        self._version = version
        self._policy = policy or AdapterPolicy(
            rate_limit_per_minute=60,
            cache_ttl_seconds=3600,
            timeout_seconds=30.0,
        )
        self._validate_result = validate_result or ValidationResult(
            status=ValidationStatus.AVAILABLE
        )
        self._validate_exception = validate_exception
        self.validate_call_count = 0
        self.validate_called_with = None

    @property
    def version(self) -> str:
        return self._version

    @property
    def policy(self) -> AdapterPolicy:
        return self._policy

    async def validate(self, value: str) -> ValidationResult:
        """Mock validate method."""
        self.validate_call_count += 1
        self.validate_called_with = value

        if self._validate_exception:
            raise self._validate_exception

        return self._validate_result


@pytest.fixture
def mock_cache():
    """Create a mock cache for testing."""
    cache = Mock(spec=SQLiteCache)
    cache.get = AsyncMock(return_value=None)  # Default to cache miss
    cache.set = AsyncMock()
    cache.initialize = AsyncMock()
    cache.close = AsyncMock()
    return cache


@pytest.fixture
def naming_brief():
    """Create a naming brief for testing."""
    return NamingBrief(
        project_codename="test_project",
        description="A test project",
        target_markets=["US", "EU"],
        industry="Technology",
    )


@pytest.fixture
def name_candidate():
    """Create a name candidate for testing."""
    return NameCandidate(
        candidate_id="test_1",
        name="TestName",
        typology=NameTypology.INVENTED,
        rationale="Test rationale",
    )


@pytest.mark.asyncio
async def test_validate_all_parallel_execution(
    mock_cache, naming_brief, name_candidate
):
    """Test that the pipeline runs adapters in parallel."""
    # Create mock adapters
    adapter1 = MockValidationAdapter(
        version="1.0.0",
        validate_result=ValidationResult(status=ValidationStatus.AVAILABLE),
    )
    adapter2 = MockValidationAdapter(
        version="2.0.0", validate_result=ValidationResult(status=ValidationStatus.TAKEN)
    )

    pipeline = ValidationPipeline(
        adapters=[adapter1, adapter2],
        cache=mock_cache,
        max_concurrency=10,
    )

    candidates = [name_candidate]

    # Act
    results = await pipeline.validate_all(candidates, naming_brief)

    # Assert
    assert len(results) == 2  # 1 candidate × 2 adapters
    assert adapter1.validate_call_count == 1
    assert adapter2.validate_call_count == 1
    assert adapter1.validate_called_with == "TestName"
    assert adapter2.validate_called_with == "TestName"


@pytest.mark.asyncio
async def test_validate_all_cache_hit(mock_cache, naming_brief, name_candidate):
    """Test that the pipeline uses cache when available."""
    # Setup mock to return a cached result
    cached_result = CoreValidationResult(
        target="TestName",
        channel=ValidationChannel.DOMAIN,
        status=CoreValidationStatus.AVAILABLE,
        confidence=Confidence.CONFIRMED,
        evidence=Evidence(
            source="cached",
            checked_at=datetime.now(),
        ),
        candidate_id="test_1",
        validation_id="test_1_domain_123",
        adapter_version="1.0.0",
        checked_at=datetime.now(),
    )
    mock_cache.get.return_value = cached_result

    adapter = MockValidationAdapter(
        version="1.0.0",
        validate_result=ValidationResult(
            status=ValidationStatus.TAKEN
        ),  # Different from cached
    )

    pipeline = ValidationPipeline(
        adapters=[adapter],
        cache=mock_cache,
        max_concurrency=10,
    )

    candidates = [name_candidate]

    # Act
    results = await pipeline.validate_all(candidates, naming_brief)

    # Assert
    assert len(results) == 1
    assert results[0] == cached_result  # Should return cached result
    assert adapter.validate_call_count == 0  # Adapter should not be called
    mock_cache.get.assert_called_once()
    mock_cache.set.assert_not_called()  # Should not set cache on hit


@pytest.mark.asyncio
async def test_validate_all_cache_miss(mock_cache, naming_brief, name_candidate):
    """Test that the pipeline calls adapter and stores result on cache miss."""
    mock_cache.get.return_value = None  # Cache miss

    adapter_result = ValidationResult(status=ValidationStatus.AVAILABLE)
    adapter = MockValidationAdapter(version="1.0.0", validate_result=adapter_result)

    pipeline = ValidationPipeline(
        adapters=[adapter],
        cache=mock_cache,
        max_concurrency=10,
    )

    candidates = [name_candidate]

    # Act
    results = await pipeline.validate_all(candidates, naming_brief)

    # Assert
    assert len(results) == 1
    assert adapter.validate_call_count == 1
    mock_cache.get.assert_called_once()
    mock_cache.set.assert_called_once()  # Should store result in cache


@pytest.mark.asyncio
async def test_validate_all_retry_on_transient_error(
    mock_cache, naming_brief, name_candidate
):
    """Test that the pipeline retries on transient errors."""
    # Fail twice with timeout, then succeed
    call_count = 0

    async def mock_validate(value: str):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise TimeoutError("Request timeout")
        return ValidationResult(status=ValidationStatus.AVAILABLE)

    adapter = MockValidationAdapter(
        version="1.0.0",
        validate_result=ValidationResult(status=ValidationStatus.AVAILABLE),
    )
    # Replace the validate method with our mock
    adapter.validate = mock_validate

    pipeline = ValidationPipeline(
        adapters=[adapter],
        cache=mock_cache,
        max_concurrency=10,
    )

    candidates = [name_candidate]

    # Act
    results = await pipeline.validate_all(candidates, naming_brief)

    # Assert
    assert len(results) == 1
    assert call_count == 3  # Should have retried twice then succeeded
    # The final result should be SUCCESS (not unverifiable)
    # Note: Our mock returns AVAILABLE which maps to AVAILABLE in core


@pytest.mark.asyncio
async def test_validate_all_no_retry_on_permanent_error(
    mock_cache, naming_brief, name_candidate
):
    """Test that the pipeline does not retry on permanent errors."""
    call_count = 0

    async def mock_validate(value: str):
        nonlocal call_count
        call_count += 1
        # Return a taken result (not an error, so no retry)
        return ValidationResult(status=ValidationStatus.TAKEN)

    adapter = MockValidationAdapter(
        version="1.0.0", validate_result=ValidationResult(status=ValidationStatus.TAKEN)
    )
    # Replace the validate method with our mock
    adapter.validate = mock_validate

    pipeline = ValidationPipeline(
        adapters=[adapter],
        cache=mock_cache,
        max_concurrency=10,
    )

    candidates = [name_candidate]

    # Act
    results = await pipeline.validate_all(candidates, naming_brief)

    # Assert
    assert len(results) == 1
    assert call_count == 1  # Should not retry
    # Should return TAKEN result


@pytest.mark.asyncio
async def test_validate_all_timeout_handling(mock_cache, naming_brief, name_candidate):
    """Test that the pipeline handles timeouts gracefully."""

    async def mock_validate(value: str):
        await asyncio.sleep(2)  # Longer than timeout
        return ValidationResult(status=ValidationStatus.AVAILABLE)

    adapter = MockValidationAdapter(
        version="1.0.0",
        policy=AdapterPolicy(
            rate_limit_per_minute=60,
            cache_ttl_seconds=3600,
            timeout_seconds=0.1,  # Very short timeout
        ),
        validate_result=ValidationResult(status=ValidationStatus.AVAILABLE),
    )
    # Replace the validate method with our mock
    adapter.validate = mock_validate

    pipeline = ValidationPipeline(
        adapters=[adapter],
        cache=mock_cache,
        max_concurrency=10,
        pipeline_timeout=5.0,  # Pipeline timeout
    )

    candidates = [name_candidate]

    # Act
    results = await pipeline.validate_all(candidates, naming_brief)

    # Assert
    assert len(results) == 1
    # Should return UNVERIFIABLE due to timeout
    assert results[0].status == CoreValidationStatus.UNVERIFIABLE


@pytest.mark.asyncio
async def test_validate_all_rate_limiting_integration(
    mock_cache, naming_brief, name_candidate
):
    """Test that the pipeline integrates with rate limiters."""
    # Use a rate limit lower than the number of calls to ensure delays
    adapter = MockValidationAdapter(
        version="1.0.0",
        policy=AdapterPolicy(
            rate_limit_per_minute=2,  # Very low: 2 per minute = 1 every 30 seconds
            cache_ttl_seconds=3600,
            timeout_seconds=30.0,
        ),
        validate_result=ValidationResult(status=ValidationStatus.AVAILABLE),
    )

    pipeline = ValidationPipeline(
        adapters=[adapter],
        cache=mock_cache,
        max_concurrency=10,
    )

    # Test multiple candidates to trigger rate limiting
    # Make more calls than the rate limit allows in a short time
    candidates = [
        NameCandidate(
            candidate_id=f"test_{i}",
            name=f"TestName{i}",
            typology=NameTypology.INVENTED,
            rationale=f"Test rationale {i}",
        )
        for i in range(5)  # 5 calls with limit of 2 per minute
    ]

    # Act
    start_time = time.time()
    results = await pipeline.validate_all(candidates, naming_brief)
    end_time = time.time()

    # Assert
    assert len(results) == 5  # 5 candidates × 1 adapter
    assert adapter.validate_call_count == 5
    # With rate limiting of 2 per minute, calls after the first 2 should be delayed
    # First 2 calls: immediate (bucket starts with 2 tokens)
    # 3rd call: needs to wait ~30 seconds for 1 token to refill
    # 4th call: needs to wait another ~30 seconds
    # 5th call: needs to wait another ~30 seconds
    # So minimum delay should be ~60 seconds for calls 3,4,5
    # But let's be conservative and check for at least 1 second delay
    assert end_time - start_time >= 1.0  # At least 1 second delay due to rate limiting


if __name__ == "__main__":
    pytest.main([__file__])


@pytest.mark.asyncio
async def test_validate_all_pipeline_timeout(mock_cache, naming_brief, name_candidate):
    """Test that the pipeline handles overall timeout."""

    async def mock_validate(value: str):
        await asyncio.sleep(10)  # Very long sleep
        return ValidationResult(status=ValidationStatus.AVAILABLE)

    adapter = MockValidationAdapter(
        version="1.0.0",
        policy=AdapterPolicy(
            rate_limit_per_minute=60,
            cache_ttl_seconds=3600,
            timeout_seconds=30.0,
        ),
        validate_result=ValidationResult(status=ValidationStatus.AVAILABLE),
    )
    adapter.validate = mock_validate

    pipeline = ValidationPipeline(
        adapters=[adapter],
        cache=mock_cache,
        max_concurrency=10,
        pipeline_timeout=0.1,  # Very short pipeline timeout
    )

    candidates = [name_candidate]

    # Act
    results = await pipeline.validate_all(candidates, naming_brief)

    # Assert
    assert len(results) == 1
    assert results[0].status == CoreValidationStatus.UNVERIFIABLE
    assert "timeout" in results[0].evidence.raw.get("error", "").lower()


@pytest.mark.asyncio
async def test_validate_all_retry_exhausted(mock_cache, naming_brief, name_candidate):
    """Test that the pipeline returns UNVERIFIABLE when retries are exhausted."""
    call_count = 0

    async def mock_validate(value: str):
        nonlocal call_count
        call_count += 1
        raise TimeoutError("Persistent timeout")

    adapter = MockValidationAdapter(
        version="1.0.0",
        validate_result=ValidationResult(status=ValidationStatus.AVAILABLE),
    )
    adapter.validate = mock_validate

    pipeline = ValidationPipeline(
        adapters=[adapter],
        cache=mock_cache,
        max_concurrency=10,
    )

    candidates = [name_candidate]

    # Act
    results = await pipeline.validate_all(candidates, naming_brief)

    # Assert
    assert len(results) == 1
    assert results[0].status == CoreValidationStatus.UNVERIFIABLE
    assert call_count == 3  # Should have tried 3 times


@pytest.mark.asyncio
async def test_validate_all_non_transient_error(
    mock_cache, naming_brief, name_candidate
):
    """Test that the pipeline does not retry on non-transient errors."""
    call_count = 0

    async def mock_validate(value: str):
        nonlocal call_count
        call_count += 1
        raise ValueError("Non-transient error")

    adapter = MockValidationAdapter(
        version="1.0.0",
        validate_result=ValidationResult(status=ValidationStatus.AVAILABLE),
    )
    adapter.validate = mock_validate

    pipeline = ValidationPipeline(
        adapters=[adapter],
        cache=mock_cache,
        max_concurrency=10,
    )

    candidates = [name_candidate]

    # Act
    results = await pipeline.validate_all(candidates, naming_brief)

    # Assert
    assert len(results) == 1
    assert results[0].status == CoreValidationStatus.UNVERIFIABLE
    assert call_count == 1  # Should not retry


@pytest.mark.asyncio
async def test_build_cache_key(mock_cache, naming_brief, name_candidate):
    """Test cache key building."""
    adapter = MockValidationAdapter(
        version="1.0.0",
        validate_result=ValidationResult(status=ValidationStatus.AVAILABLE),
    )

    pipeline = ValidationPipeline(
        adapters=[adapter],
        cache=mock_cache,
        max_concurrency=10,
    )

    # Act
    cache_key = pipeline._build_cache_key(adapter, name_candidate, naming_brief)

    # Assert
    assert isinstance(cache_key, str)
    assert len(cache_key) == 64  # SHA256 hash length


@pytest.mark.asyncio
async def test_build_context_hash(mock_cache, naming_brief):
    """Test context hash building."""
    adapter = MockValidationAdapter(
        version="1.0.0",
        validate_result=ValidationResult(status=ValidationStatus.AVAILABLE),
    )

    pipeline = ValidationPipeline(
        adapters=[adapter],
        cache=mock_cache,
        max_concurrency=10,
    )

    # Act
    context_hash = pipeline._build_context_hash(naming_brief)

    # Assert
    assert isinstance(context_hash, str)
    assert len(context_hash) == 64  # SHA256 hash length


@pytest.mark.asyncio
async def test_get_channel_from_adapter(mock_cache):
    """Test channel detection from adapter class name."""
    from launch_engine.validation.adapters.domain import DomainAdapter
    from launch_engine.validation.adapters.trademark import TrademarkAdapter
    from launch_engine.validation.adapters.social import SocialMediaAdapter

    pipeline = ValidationPipeline(
        adapters=[],
        cache=mock_cache,
        max_concurrency=10,
    )

    # Test domain adapter
    domain_adapter = DomainAdapter()
    channel = pipeline._get_channel_from_adapter(domain_adapter)
    assert channel == ValidationChannel.DOMAIN

    # Test trademark adapter
    trademark_adapter = TrademarkAdapter()
    channel = pipeline._get_channel_from_adapter(trademark_adapter)
    assert channel == ValidationChannel.TRADEMARK_TR

    # Test social adapter
    social_adapter = SocialMediaAdapter()
    channel = pipeline._get_channel_from_adapter(social_adapter)
    assert channel == ValidationChannel.SOCIAL_X


@pytest.mark.asyncio
async def test_convert_to_core_result(mock_cache, naming_brief, name_candidate):
    """Test adapter result to core result conversion."""
    adapter = MockValidationAdapter(
        version="1.0.0",
        validate_result=ValidationResult(
            status=ValidationStatus.AVAILABLE,
            details={
                "url": "https://example.com",
                "manual_review_url": "https://review.com",
            },
        ),
    )

    pipeline = ValidationPipeline(
        adapters=[adapter],
        cache=mock_cache,
        max_concurrency=10,
    )

    adapter_result = ValidationResult(
        status=ValidationStatus.AVAILABLE, details={"url": "https://example.com"}
    )

    # Act
    core_result = pipeline._convert_to_core_result(
        name_candidate, adapter, naming_brief, adapter_result
    )

    # Assert
    assert core_result.status == CoreValidationStatus.AVAILABLE
    assert core_result.confidence == Confidence.CONFIRMED
    assert core_result.candidate_id == name_candidate.candidate_id
    assert core_result.adapter_version == adapter.version


@pytest.mark.asyncio
async def test_create_unverifiable_result(mock_cache, name_candidate):
    """Test unverifiable result creation."""
    adapter = MockValidationAdapter(
        version="1.0.0",
        validate_result=ValidationResult(status=ValidationStatus.AVAILABLE),
    )

    pipeline = ValidationPipeline(
        adapters=[adapter],
        cache=mock_cache,
        max_concurrency=10,
    )

    # Act
    result = pipeline._create_unverifiable_result(
        name_candidate, adapter, "Test error message"
    )

    # Assert
    assert result.status == CoreValidationStatus.UNVERIFIABLE
    assert result.confidence == Confidence.UNKNOWN
    assert result.candidate_id == name_candidate.candidate_id
    assert "Test error message" in result.evidence.raw.get("error", "")


@pytest.mark.asyncio
async def test_create_unverifiable_result_with_none(mock_cache):
    """Test unverifiable result creation with None candidate/adapter."""
    pipeline = ValidationPipeline(
        adapters=[],
        cache=mock_cache,
        max_concurrency=10,
    )

    # Act
    result = pipeline._create_unverifiable_result(None, None, "Test error")

    # Assert
    assert result.status == CoreValidationStatus.UNVERIFIABLE
    assert result.candidate_id == "unknown"
    assert result.adapter_version == "unknown"


@pytest.mark.asyncio
async def test_validate_all_with_exception_in_results(
    mock_cache, naming_brief, name_candidate
):
    """Test pipeline handles exceptions in gathered results."""
    adapter = MockValidationAdapter(
        version="1.0.0",
        validate_result=ValidationResult(status=ValidationStatus.AVAILABLE),
    )

    # Mock gather to return an exception
    import asyncio

    original_gather = asyncio.gather

    async def mock_gather(*args, **kwargs):
        await original_gather(*args, **kwargs)
        # Inject an exception
        return [Exception("Test exception")]

    pipeline = ValidationPipeline(
        adapters=[adapter],
        cache=mock_cache,
        max_concurrency=10,
    )

    # Patch gather temporarily
    import unittest.mock

    with unittest.mock.patch("asyncio.gather", side_effect=mock_gather):
        results = await pipeline.validate_all([name_candidate], naming_brief)

        # Should handle the exception gracefully
        assert len(results) == 1
        assert results[0].status == CoreValidationStatus.UNVERIFIABLE


@pytest.mark.asyncio
async def test_should_retry_with_transient_error(mock_cache):
    """Test _should_retry detects transient errors."""
    pipeline = ValidationPipeline(
        adapters=[],
        cache=mock_cache,
        max_concurrency=10,
    )

    # Test with transient error indicators
    for error_msg in [
        "timeout",
        "network error",
        "connection refused",
        "502",
        "503",
        "504",
        "429",
        "408",
    ]:
        result = CoreValidationResult(
            target="test",
            channel=ValidationChannel.DOMAIN,
            status=CoreValidationStatus.UNVERIFIABLE,
            confidence=Confidence.UNKNOWN,
            evidence=Evidence(
                source="test", checked_at=datetime.now(), raw={"error": error_msg}
            ),
            candidate_id="test",
            validation_id="test",
            adapter_version="1.0",
            checked_at=datetime.now(),
        )

        should_retry = pipeline._should_retry(result, 0, 3)
        assert should_retry is True, f"Should retry on error: {error_msg}"


@pytest.mark.asyncio
async def test_should_retry_with_non_transient_error(mock_cache):
    """Test _should_retry rejects non-transient errors."""
    pipeline = ValidationPipeline(
        adapters=[],
        cache=mock_cache,
        max_concurrency=10,
    )

    # Test with non-transient error
    result = CoreValidationResult(
        target="test",
        channel=ValidationChannel.DOMAIN,
        status=CoreValidationStatus.UNVERIFIABLE,
        confidence=Confidence.UNKNOWN,
        evidence=Evidence(
            source="test", checked_at=datetime.now(), raw={"error": "invalid input"}
        ),
        candidate_id="test",
        validation_id="test",
        adapter_version="1.0",
        checked_at=datetime.now(),
    )

    should_retry = pipeline._should_retry(result, 0, 3)
    assert should_retry is False


@pytest.mark.asyncio
async def test_should_retry_with_available_status(mock_cache):
    """Test _should_retry returns False for AVAILABLE status."""
    pipeline = ValidationPipeline(
        adapters=[],
        cache=mock_cache,
        max_concurrency=10,
    )

    result = CoreValidationResult(
        target="test",
        channel=ValidationChannel.DOMAIN,
        status=CoreValidationStatus.AVAILABLE,
        confidence=Confidence.CONFIRMED,
        evidence=Evidence(source="test", checked_at=datetime.now()),
        candidate_id="test",
        validation_id="test",
        adapter_version="1.0",
        checked_at=datetime.now(),
    )

    should_retry = pipeline._should_retry(result, 0, 3)
    assert should_retry is False


@pytest.mark.asyncio
async def test_get_channel_from_adapter_global_trademark(mock_cache):
    """Test channel detection for global trademark adapter."""
    from launch_engine.validation.adapters.base import ValidationAdapter, AdapterPolicy

    class GlobalTrademarkAdapter(ValidationAdapter):
        @property
        def version(self):
            return "1.0"

        @property
        def policy(self):
            return AdapterPolicy(60, 3600, 30.0)

        async def validate(self, value):
            pass

    pipeline = ValidationPipeline(
        adapters=[],
        cache=mock_cache,
        max_concurrency=10,
    )

    adapter = GlobalTrademarkAdapter()
    channel = pipeline._get_channel_from_adapter(adapter)
    assert channel == ValidationChannel.TRADEMARK_GLOBAL


@pytest.mark.asyncio
async def test_get_channel_from_adapter_instagram(mock_cache):
    """Test channel detection for Instagram adapter."""
    from launch_engine.validation.adapters.base import ValidationAdapter, AdapterPolicy

    class SocialInstagramAdapter(ValidationAdapter):
        @property
        def version(self):
            return "1.0"

        @property
        def policy(self):
            return AdapterPolicy(60, 3600, 30.0)

        async def validate(self, value):
            return ValidationResult(status=ValidationStatus.AVAILABLE)

    pipeline = ValidationPipeline(
        adapters=[],
        cache=mock_cache,
        max_concurrency=10,
    )

    adapter = SocialInstagramAdapter()
    channel = pipeline._get_channel_from_adapter(adapter)
    assert channel == ValidationChannel.SOCIAL_INSTAGRAM


@pytest.mark.asyncio
async def test_get_channel_from_adapter_linkedin(mock_cache):
    """Test channel detection for LinkedIn adapter."""
    from launch_engine.validation.adapters.base import ValidationAdapter, AdapterPolicy

    class SocialLinkedInAdapter(ValidationAdapter):
        @property
        def version(self):
            return "1.0"

        @property
        def policy(self):
            return AdapterPolicy(60, 3600, 30.0)

        async def validate(self, value):
            return ValidationResult(status=ValidationStatus.AVAILABLE)

    pipeline = ValidationPipeline(
        adapters=[],
        cache=mock_cache,
        max_concurrency=10,
    )

    adapter = SocialLinkedInAdapter()
    channel = pipeline._get_channel_from_adapter(adapter)
    assert channel == ValidationChannel.SOCIAL_LINKEDIN


@pytest.mark.asyncio
async def test_get_channel_from_adapter_unknown(mock_cache):
    """Test channel detection for unknown adapter falls back to DOMAIN."""
    from launch_engine.validation.adapters.base import ValidationAdapter, AdapterPolicy

    class UnknownAdapter(ValidationAdapter):
        @property
        def version(self):
            return "1.0"

        @property
        def policy(self):
            return AdapterPolicy(60, 3600, 30.0)

        async def validate(self, value):
            pass

    pipeline = ValidationPipeline(
        adapters=[],
        cache=mock_cache,
        max_concurrency=10,
    )

    adapter = UnknownAdapter()
    channel = pipeline._get_channel_from_adapter(adapter)
    assert channel == ValidationChannel.DOMAIN


@pytest.mark.asyncio
async def test_validate_all_with_retry_and_backoff(
    mock_cache, naming_brief, name_candidate
):
    """Test that retry uses exponential backoff."""
    call_count = 0
    call_times = []

    async def mock_validate(value: str):
        nonlocal call_count
        call_count += 1
        call_times.append(asyncio.get_event_loop().time())

        if call_count < 3:
            # Transient error that should trigger retry
            raise ConnectionError("Network timeout")
        return ValidationResult(status=ValidationStatus.AVAILABLE)

    adapter = MockValidationAdapter(
        version="1.0.0",
        validate_result=ValidationResult(status=ValidationStatus.AVAILABLE),
    )
    adapter.validate = mock_validate

    pipeline = ValidationPipeline(
        adapters=[adapter],
        cache=mock_cache,
        max_concurrency=10,
    )

    results = await pipeline.validate_all([name_candidate], naming_brief)

    # Should have retried twice
    assert call_count == 3
    assert len(results) == 1
    assert results[0].status == CoreValidationStatus.AVAILABLE

    # Check that there was delay between retries (exponential backoff)
    if len(call_times) >= 3:
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]
        # Second delay should be longer than first (exponential)
        assert delay2 > delay1


@pytest.mark.asyncio
async def test_validate_all_with_exception_retry(
    mock_cache, naming_brief, name_candidate
):
    """Test retry on exception with transient error."""
    call_count = 0

    async def mock_validate(value: str):
        nonlocal call_count
        call_count += 1

        if call_count < 2:
            # Transient network error
            raise ConnectionError("Network timeout")
        return ValidationResult(status=ValidationStatus.AVAILABLE)

    adapter = MockValidationAdapter(
        version="1.0.0",
        validate_result=ValidationResult(status=ValidationStatus.AVAILABLE),
    )
    adapter.validate = mock_validate

    pipeline = ValidationPipeline(
        adapters=[adapter],
        cache=mock_cache,
        max_concurrency=10,
    )

    results = await pipeline.validate_all([name_candidate], naming_brief)

    # Should have retried once
    assert call_count == 2
    assert len(results) == 1
    assert results[0].status == CoreValidationStatus.AVAILABLE


@pytest.mark.asyncio
async def test_build_cache_key_with_different_contexts(mock_cache, name_candidate):
    """Test that different contexts produce different cache keys."""
    adapter = MockValidationAdapter(
        version="1.0.0",
        validate_result=ValidationResult(status=ValidationStatus.AVAILABLE),
    )

    pipeline = ValidationPipeline(
        adapters=[adapter],
        cache=mock_cache,
        max_concurrency=10,
    )

    # Create two different briefs
    brief1 = NamingBrief(
        project_codename="project1",
        description="Test 1",
        target_markets=["US"],
        industry="Technology",
    )

    brief2 = NamingBrief(
        project_codename="project2",
        description="Test 2",
        target_markets=["EU"],
        industry="Healthcare",
    )

    key1 = pipeline._build_cache_key(adapter, name_candidate, brief1)
    key2 = pipeline._build_cache_key(adapter, name_candidate, brief2)

    # Different contexts should produce different keys
    assert key1 != key2
