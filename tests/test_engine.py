"""Tests for the LaunchEngine class."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from launch_engine.engine import LaunchEngine
from launch_engine.modules.naming.brief import NamingBrief, NameTypology
from launch_engine.modules.naming.candidates import NameCandidate, NameCandidateList
from launch_engine.core.validation import (
    ValidationResult,
    ValidationStatus,
    ValidationChannel,
    Confidence,
    Evidence,
)
from launch_engine.validation.adapters.base import ValidationAdapter


@pytest.fixture
def naming_brief() -> NamingBrief:
    """Create a sample naming brief for testing."""
    return NamingBrief(
        project_codename="test_project",
        description="A test project",
        target_markets=["USA", "Europe"],
        industry="Technology",
        brand_personality="Innovative",
        avoid_terms=["test", "avoid"],
        phonetic_constraints=None,  # type: ignore
        preferred_typologies=[],
        candidate_count=10,
        language="English",
    )


def test_initialization_default_adapters(naming_brief: NamingBrief) -> None:
    """Test LaunchEngine initialization with default adapters."""
    with (
        patch("launch_engine.engine.LLMAdapter") as mock_llm,
        patch("launch_engine.engine.SQLiteCache") as mock_cache,
        patch("launch_engine.engine.BrandNamingModule") as mock_bn,
        patch("launch_engine.engine.ValidationPipeline") as mock_vp,
        patch("launch_engine.validation.adapters.domain.DomainAdapter") as mock_domain,
        patch(
            "launch_engine.validation.adapters.trademark.TrademarkAdapter"
        ) as mock_trademark,
        patch(
            "launch_engine.validation.adapters.social.SocialMediaAdapter"
        ) as mock_social,
    ):

        # Configure mocks
        mock_llm_instance = MagicMock()
        mock_llm_instance.provider = "ollama"
        mock_llm_instance.model_id = "ollama/test_model"
        mock_llm.return_value = mock_llm_instance

        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance

        mock_bn_instance = MagicMock()
        mock_bn.return_value = mock_bn_instance

        mock_vp_instance = MagicMock()
        mock_vp.return_value = mock_vp_instance

        # Set up domain adapter mock
        mock_domain_instance = MagicMock()
        mock_domain_instance.policy = MagicMock()
        mock_domain_instance.policy.rate_limit_per_minute = 60
        mock_domain.return_value = mock_domain_instance

        # Set up trademark adapter mock
        mock_trademark_instance = MagicMock()
        mock_trademark_instance.policy = MagicMock()
        mock_trademark_instance.policy.rate_limit_per_minute = 10
        mock_trademark.return_value = mock_trademark_instance

        # Set up social adapter mock
        mock_social_instance = MagicMock()
        mock_social_instance.policy = MagicMock()
        mock_social_instance.policy.rate_limit_per_minute = 30
        mock_social.return_value = mock_social_instance

        engine = LaunchEngine(
            llm_provider="ollama",
            llm_model="test_model",
            cache_db_path=":memory:",
        )

        # Check that the adapters were created
        assert len(engine.adapters) == 3
        mock_domain.assert_called_once()
        mock_trademark.assert_called_once()
        mock_social.assert_called_once()

        # Check that components were initialized
        mock_llm.assert_called_once_with(provider="ollama", model="test_model")
        mock_cache.assert_called_once_with(db_path=":memory:")
        mock_bn.assert_called_once_with(llm_adapter=mock_llm_instance)
        mock_vp.assert_called_once()
        args, kwargs = mock_vp.call_args
        adapters_arg = (
            kwargs.get("adapters") or args[0] if args else kwargs.get("adapters")
        )
        assert len(adapters_arg) == 3
        # Check that the adapters are the ones we created (order: domain, trademark, social)
        assert adapters_arg[0] == mock_domain_instance
        assert adapters_arg[1] == mock_trademark_instance
        assert adapters_arg[2] == mock_social_instance


def test_initialization_custom_adapters(naming_brief: NamingBrief) -> None:
    """Test LaunchEngine initialization with custom adapters."""
    # Create custom adapters with policy attribute
    custom_adapters = []
    for _ in range(2):
        adapter = MagicMock(spec=ValidationAdapter)
        adapter.policy = MagicMock()
        adapter.policy.rate_limit_per_minute = 60
        custom_adapters.append(adapter)

    with (
        patch("launch_engine.engine.LLMAdapter"),
        patch("launch_engine.engine.SQLiteCache"),
        patch("launch_engine.engine.BrandNamingModule"),
        patch("launch_engine.engine.ValidationPipeline"),
    ):

        engine = LaunchEngine(
            llm_provider="ollama",
            llm_model="test_model",
            cache_db_path=":memory:",
            adapters=custom_adapters,
        )

        assert engine.adapters == custom_adapters


@pytest.mark.asyncio
async def test_generate_names_success(naming_brief: NamingBrief) -> None:
    """Test successful name generation."""
    mock_candidates = [
        NameCandidate(
            candidate_id="test_cand",
            name="Test",
            typology=NameTypology.INVENTED,
            rationale="Test",
        )
    ]
    mock_candidate_list = NameCandidateList(
        brief_ref="test_project",
        candidates=mock_candidates,
        llm_model_used="ollama/test_model",
        llm_provider="ollama",
        generated_at=datetime.now(),
    )

    with (
        patch("launch_engine.engine.LLMAdapter") as mock_llm,
        patch("launch_engine.engine.SQLiteCache"),
        patch("launch_engine.engine.BrandNamingModule") as mock_bn,
        patch("launch_engine.engine.ValidationPipeline"),
        patch("launch_engine.validation.adapters.domain.DomainAdapter") as mock_domain,
        patch(
            "launch_engine.validation.adapters.trademark.TrademarkAdapter"
        ) as mock_trademark,
        patch(
            "launch_engine.validation.adapters.social.SocialMediaAdapter"
        ) as mock_social,
    ):

        # Set up adapter mocks with policy
        mock_domain_instance = MagicMock()
        mock_domain_instance.policy = MagicMock()
        mock_domain_instance.policy.rate_limit_per_minute = 60
        mock_domain.return_value = mock_domain_instance

        mock_trademark_instance = MagicMock()
        mock_trademark_instance.policy = MagicMock()
        mock_trademark_instance.policy.rate_limit_per_minute = 10
        mock_trademark.return_value = mock_trademark_instance

        mock_social_instance = MagicMock()
        mock_social_instance.policy = MagicMock()
        mock_social_instance.policy.rate_limit_per_minute = 30
        mock_social.return_value = mock_social_instance

        # Set up LLM mock
        mock_llm_instance = MagicMock()
        mock_llm_instance.provider = "ollama"
        mock_llm_instance.model_id = "ollama/test_model"
        mock_llm.return_value = mock_llm_instance

        # Set up BrandNamingModule mock
        mock_bn_instance = MagicMock()
        mock_bn_instance.run = AsyncMock(return_value=mock_candidate_list)
        mock_bn.return_value = mock_bn_instance

        engine = LaunchEngine(
            llm_provider="ollama",
            llm_model="test_model",
            cache_db_path=":memory:",
        )

        result = await engine.generate_names(naming_brief)

        assert result == mock_candidate_list
        mock_bn_instance.run.assert_called_once_with(naming_brief)


@pytest.mark.asyncio
async def test_generate_names_error(naming_brief: NamingBrief) -> None:
    """Test name generation error handling."""
    with (
        patch("launch_engine.engine.LLMAdapter") as mock_llm,
        patch("launch_engine.engine.SQLiteCache"),
        patch("launch_engine.engine.BrandNamingModule") as mock_bn,
        patch("launch_engine.engine.ValidationPipeline"),
        patch("launch_engine.validation.adapters.domain.DomainAdapter") as mock_domain,
        patch(
            "launch_engine.validation.adapters.trademark.TrademarkAdapter"
        ) as mock_trademark,
        patch(
            "launch_engine.validation.adapters.social.SocialMediaAdapter"
        ) as mock_social,
    ):

        # Set up adapter mocks with policy
        mock_domain_instance = MagicMock()
        mock_domain_instance.policy = MagicMock()
        mock_domain_instance.policy.rate_limit_per_minute = 60
        mock_domain.return_value = mock_domain_instance

        mock_trademark_instance = MagicMock()
        mock_trademark_instance.policy = MagicMock()
        mock_trademark_instance.policy.rate_limit_per_minute = 10
        mock_trademark.return_value = mock_trademark_instance

        mock_social_instance = MagicMock()
        mock_social_instance.policy = MagicMock()
        mock_social_instance.policy.rate_limit_per_minute = 30
        mock_social.return_value = mock_social_instance

        # Set up LLM mock
        mock_llm_instance = MagicMock()
        mock_llm_instance.provider = "ollama"
        mock_llm_instance.model_id = "ollama/test_model"
        mock_llm.return_value = mock_llm_instance

        # Set up BrandNamingModule mock to raise exception
        mock_bn_instance = MagicMock()
        mock_bn_instance.run = AsyncMock(side_effect=Exception("Test error"))
        mock_bn.return_value = mock_bn_instance

        engine = LaunchEngine(
            llm_provider="ollama",
            llm_model="test_model",
            cache_db_path=":memory:",
        )

        result = await engine.generate_names(naming_brief)

        # Should return an empty candidate list
        assert isinstance(result, NameCandidateList)
        assert result.candidates == []
        assert result.brief_ref == naming_brief.project_codename
        assert result.llm_model_used == "ollama/test_model"
        assert result.llm_provider == "ollama"


@pytest.mark.asyncio
async def test_validate_names_success(naming_brief: NamingBrief) -> None:
    """Test successful validation."""
    mock_candidates = [
        NameCandidate(
            candidate_id="test_cand",
            name="Test",
            typology=NameTypology.INVENTED,
            rationale="Test",
        )
    ]
    mock_validation_results = [
        ValidationResult(
            target="Test",
            channel=ValidationChannel.DOMAIN,
            status=ValidationStatus.AVAILABLE,
            confidence=Confidence.CONFIRMED,
            evidence=Evidence(source="test", url=None, checked_at=datetime.now()),
            candidate_id="test_cand",
            validation_id="test_validation_domain",
            adapter_version="1.0",
            checked_at=datetime.now(),
        ),
        ValidationResult(
            target="Test",
            channel=ValidationChannel.TRADEMARK_TR,
            status=ValidationStatus.AVAILABLE,
            confidence=Confidence.CONFIRMED,
            evidence=Evidence(source="test", url=None, checked_at=datetime.now()),
            candidate_id="test_cand",
            validation_id="test_validation_trademark",
            adapter_version="1.0",
            checked_at=datetime.now(),
        ),
        ValidationResult(
            target="Test",
            channel=ValidationChannel.SOCIAL_X,
            status=ValidationStatus.AVAILABLE,
            confidence=Confidence.CONFIRMED,
            evidence=Evidence(source="test", url=None, checked_at=datetime.now()),
            candidate_id="test_cand",
            validation_id="test_validation_social",
            adapter_version="1.0",
            checked_at=datetime.now(),
        ),
    ]

    with (
        patch("launch_engine.engine.LLMAdapter"),
        patch("launch_engine.engine.SQLiteCache"),
        patch("launch_engine.engine.BrandNamingModule"),
        patch("launch_engine.engine.ValidationPipeline") as mock_vp,
        patch("launch_engine.validation.adapters.domain.DomainAdapter") as mock_domain,
        patch(
            "launch_engine.validation.adapters.trademark.TrademarkAdapter"
        ) as mock_trademark,
        patch(
            "launch_engine.validation.adapters.social.SocialMediaAdapter"
        ) as mock_social,
    ):

        # Set up adapter mocks with policy
        mock_domain_instance = MagicMock()
        mock_domain_instance.policy = MagicMock()
        mock_domain_instance.policy.rate_limit_per_minute = 60
        mock_domain.return_value = mock_domain_instance

        mock_trademark_instance = MagicMock()
        mock_trademark_instance.policy = MagicMock()
        mock_trademark_instance.policy.rate_limit_per_minute = 10
        mock_trademark.return_value = mock_trademark_instance

        mock_social_instance = MagicMock()
        mock_social_instance.policy = MagicMock()
        mock_social_instance.policy.rate_limit_per_minute = 30
        mock_social.return_value = mock_social_instance

        # Set up ValidationPipeline mock
        mock_vp_instance = MagicMock()
        mock_vp_instance.validate_all = AsyncMock(return_value=mock_validation_results)
        mock_vp.return_value = mock_vp_instance

        engine = LaunchEngine(
            llm_provider="ollama",
            llm_model="test_model",
            cache_db_path=":memory:",
        )

        result = await engine.validate_names(mock_candidates, naming_brief)

        assert result == mock_validation_results
        mock_vp_instance.validate_all.assert_called_once_with(
            mock_candidates, naming_brief
        )


@pytest.mark.asyncio
async def test_validate_names_error(naming_brief: NamingBrief) -> None:
    """Test validation error handling."""
    mock_candidates = [
        NameCandidate(
            candidate_id="test_cand",
            name="Test",
            typology=NameTypology.INVENTED,
            rationale="Test",
        )
    ]

    with (
        patch("launch_engine.engine.LLMAdapter"),
        patch("launch_engine.engine.SQLiteCache"),
        patch("launch_engine.engine.BrandNamingModule"),
        patch("launch_engine.engine.ValidationPipeline") as mock_vp,
        patch("launch_engine.validation.adapters.domain.DomainAdapter") as mock_domain,
        patch(
            "launch_engine.validation.adapters.trademark.TrademarkAdapter"
        ) as mock_trademark,
        patch(
            "launch_engine.validation.adapters.social.SocialMediaAdapter"
        ) as mock_social,
    ):

        # Set up adapter mocks with policy
        mock_domain_instance = MagicMock()
        mock_domain_instance.policy = MagicMock()
        mock_domain_instance.policy.rate_limit_per_minute = 60
        mock_domain.return_value = mock_domain_instance

        mock_trademark_instance = MagicMock()
        mock_trademark_instance.policy = MagicMock()
        mock_trademark_instance.policy.rate_limit_per_minute = 10
        mock_trademark.return_value = mock_trademark_instance

        mock_social_instance = MagicMock()
        mock_social_instance.policy = MagicMock()
        mock_social_instance.policy.rate_limit_per_minute = 30
        mock_social.return_value = mock_social_instance

        # Set up ValidationPipeline mock to raise exception
        mock_vp_instance = MagicMock()
        mock_vp_instance.validate_all = AsyncMock(side_effect=Exception("Test error"))
        mock_vp.return_value = mock_vp_instance

        engine = LaunchEngine(
            llm_provider="ollama",
            llm_model="test_model",
            cache_db_path=":memory:",
        )

        result = await engine.validate_names(mock_candidates, naming_brief)

        # Should return unverifiable results (one per candidate)
        assert len(result) == len(mock_candidates)
        for validation_result in result:
            assert validation_result.status == ValidationStatus.UNVERIFIABLE
            assert validation_result.confidence == Confidence.UNKNOWN
            assert "Test error" in validation_result.evidence.raw.get("error", "")


@pytest.mark.asyncio
async def test_run_full_pipeline(naming_brief: NamingBrief) -> None:
    """Test the full pipeline execution."""
    mock_candidates = [
        NameCandidate(
            candidate_id="test_cand",
            name="Test",
            typology=NameTypology.INVENTED,
            rationale="Test",
        )
    ]
    mock_candidate_list = NameCandidateList(
        brief_ref="test_project",
        candidates=mock_candidates,
        llm_model_used="ollama/test_model",
        llm_provider="ollama",
        generated_at=datetime.now(),
    )
    mock_validation_results = [
        ValidationResult(
            target="Test",
            channel=ValidationChannel.DOMAIN,
            status=ValidationStatus.AVAILABLE,
            confidence=Confidence.CONFIRMED,
            evidence=Evidence(source="test", url=None, checked_at=datetime.now()),
            candidate_id="test_cand",
            validation_id="test_validation_domain",
            adapter_version="1.0",
            checked_at=datetime.now(),
        ),
        ValidationResult(
            target="Test",
            channel=ValidationChannel.TRADEMARK_TR,
            status=ValidationStatus.AVAILABLE,
            confidence=Confidence.CONFIRMED,
            evidence=Evidence(source="test", url=None, checked_at=datetime.now()),
            candidate_id="test_cand",
            validation_id="test_validation_trademark",
            adapter_version="1.0",
            checked_at=datetime.now(),
        ),
        ValidationResult(
            target="Test",
            channel=ValidationChannel.SOCIAL_X,
            status=ValidationStatus.AVAILABLE,
            confidence=Confidence.CONFIRMED,
            evidence=Evidence(source="test", url=None, checked_at=datetime.now()),
            candidate_id="test_cand",
            validation_id="test_validation_social",
            adapter_version="1.0",
            checked_at=datetime.now(),
        ),
    ]

    with (
        patch("launch_engine.engine.LLMAdapter") as mock_llm,
        patch("launch_engine.engine.SQLiteCache"),
        patch("launch_engine.engine.BrandNamingModule") as mock_bn,
        patch("launch_engine.engine.ValidationPipeline") as mock_vp,
        patch("launch_engine.validation.adapters.domain.DomainAdapter") as mock_domain,
        patch(
            "launch_engine.validation.adapters.trademark.TrademarkAdapter"
        ) as mock_trademark,
        patch(
            "launch_engine.validation.adapters.social.SocialMediaAdapter"
        ) as mock_social,
    ):

        # Set up adapter mocks with policy
        mock_domain_instance = MagicMock()
        mock_domain_instance.policy = MagicMock()
        mock_domain_instance.policy.rate_limit_per_minute = 60
        mock_domain.return_value = mock_domain_instance

        mock_trademark_instance = MagicMock()
        mock_trademark_instance.policy = MagicMock()
        mock_trademark_instance.policy.rate_limit_per_minute = 10
        mock_trademark.return_value = mock_trademark_instance

        mock_social_instance = MagicMock()
        mock_social_instance.policy = MagicMock()
        mock_social_instance.policy.rate_limit_per_minute = 30
        mock_social.return_value = mock_social_instance

        # Set up LLM mock
        mock_llm_instance = MagicMock()
        mock_llm_instance.provider = "ollama"
        mock_llm_instance.model_id = "ollama/test_model"
        mock_llm.return_value = mock_llm_instance

        # Set up BrandNamingModule mock
        mock_bn_instance = MagicMock()
        mock_bn_instance.run = AsyncMock(return_value=mock_candidate_list)
        mock_bn.return_value = mock_bn_instance

        # Set up ValidationPipeline mock
        mock_vp_instance = MagicMock()
        mock_vp_instance.validate_all = AsyncMock(return_value=mock_validation_results)
        mock_vp.return_value = mock_vp_instance

        engine = LaunchEngine(
            llm_provider="ollama",
            llm_model="test_model",
            cache_db_path=":memory:",
        )

        candidates, validation_results = await engine.run_full_pipeline(naming_brief)

        assert candidates == mock_candidate_list
        assert validation_results == mock_validation_results
        mock_bn_instance.run.assert_called_once_with(naming_brief)
        mock_vp_instance.validate_all.assert_called_once_with(
            mock_candidates, naming_brief
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
