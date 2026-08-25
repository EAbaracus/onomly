import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from launch_engine.core.validation import (
    ValidationStatus,
    Confidence,
    ValidationChannel,
    Evidence,
    ValidationResult,
    CacheEntry,
)


def test_validation_status_enum():
    """Test ValidationStatus enum values."""
    assert ValidationStatus.AVAILABLE == "available"
    assert ValidationStatus.TAKEN == "taken"
    assert ValidationStatus.UNVERIFIABLE == "unverifiable"
    assert ValidationStatus.ERROR == "error"


def test_confidence_enum():
    """Test Confidence enum values."""
    assert Confidence.CONFIRMED == "confirmed"
    assert Confidence.LIKELY == "likely"
    assert Confidence.UNKNOWN == "unknown"


def test_validation_channel_enum():
    """Test ValidationChannel enum values."""
    assert ValidationChannel.DOMAIN == "domain"
    assert ValidationChannel.TRADEMARK_TR == "trademark_tr"
    assert ValidationChannel.TRADEMARK_GLOBAL == "trademark_global"
    assert ValidationChannel.SOCIAL_X == "social_x"
    assert ValidationChannel.SOCIAL_INSTAGRAM == "social_instagram"
    assert ValidationChannel.SOCIAL_LINKEDIN == "social_linkedin"


def test_evidence_model():
    """Test Evidence model."""
    now = datetime.now(timezone.utc)
    evidence = Evidence(
        source="test", url="http://example.com", checked_at=now, raw={"key": "value"}
    )
    assert evidence.source == "test"
    assert evidence.url == "http://example.com"
    assert evidence.checked_at == now
    assert evidence.raw == {"key": "value"}

    # Test optional fields
    evidence2 = Evidence(source="test", checked_at=now)
    assert evidence2.url is None
    assert evidence2.raw is None


def test_validation_result_model():
    """Test ValidationResult model."""
    now = datetime.now(timezone.utc)
    evidence = Evidence(source="test", checked_at=now)
    result = ValidationResult(
        target="example.com",
        channel=ValidationChannel.DOMAIN,
        status=ValidationStatus.AVAILABLE,
        confidence=Confidence.CONFIRMED,
        evidence=evidence,
        candidate_id="123",
        validation_id="val-1",
        adapter_version="1.0",
        checked_at=now,
        manual_review_url="http://review.example.com",
    )
    assert result.target == "example.com"
    assert result.channel == ValidationChannel.DOMAIN
    assert result.status == ValidationStatus.AVAILABLE
    assert result.confidence == Confidence.CONFIRMED
    assert result.evidence == evidence
    assert result.candidate_id == "123"
    assert result.validation_id == "val-1"
    assert result.adapter_version == "1.0"
    assert result.checked_at == now
    assert result.manual_review_url == "http://review.example.com"

    # Test optional fields (manual_review_url)
    result2 = ValidationResult(
        target="example.com",
        channel=ValidationChannel.DOMAIN,
        status=ValidationStatus.AVAILABLE,
        confidence=Confidence.CONFIRMED,
        evidence=evidence,
        candidate_id="456",
        validation_id="val-2",
        adapter_version="1.0",
        checked_at=now,
    )
    assert result2.candidate_id == "456"
    assert result2.manual_review_url is None


def test_cache_entry_model():
    """Test CacheEntry model."""
    now = datetime.now(timezone.utc)
    evidence = Evidence(source="test", checked_at=now)
    result = ValidationResult(
        target="example.com",
        channel=ValidationChannel.DOMAIN,
        status=ValidationStatus.AVAILABLE,
        confidence=Confidence.CONFIRMED,
        evidence=evidence,
        candidate_id="789",
        validation_id="val-1",
        adapter_version="1.0",
        checked_at=now,
    )
    cache_entry = CacheEntry(
        key="key-1",
        result=result,
        expires_at=now,
        adapter_version="1.0",
        policy_version="1.0",
        created_at=now,
    )
    assert cache_entry.key == "key-1"
    assert cache_entry.result == result
    assert cache_entry.expires_at == now
    assert cache_entry.adapter_version == "1.0"
    assert cache_entry.policy_version == "1.0"
    assert cache_entry.created_at == now


def test_model_validation_errors():
    """Test that models raise validation errors on missing required fields."""
    # Evidence requires source and checked_at
    with pytest.raises(ValidationError):
        Evidence()  # missing source and checked_at

    # ValidationResult requires target, channel, status, confidence, evidence, validation_id, adapter_version, checked_at
    now = datetime.now(timezone.utc)
    Evidence(source="test", checked_at=now)
    with pytest.raises(ValidationError):
        ValidationResult()  # missing all required

    # CacheEntry requires key, result, expires_at, adapter_version, policy_version, created_at
    with pytest.raises(ValidationError):
        CacheEntry()  # missing all required
