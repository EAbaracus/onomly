"""Integration tests for the full Launch Engine pipeline."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from launch_engine.engine import LaunchEngine
from launch_engine.modules.naming.brief import NamingBrief, NameTypology
from launch_engine.modules.naming.candidates import (
    NameCandidate,
    NameCandidateList,
    InternalAssessment,
)
from launch_engine.core.validation import (
    ValidationResult,
    ValidationStatus,
    ValidationChannel,
    Confidence,
    Evidence,
)


@pytest.fixture
def naming_brief():
    """Create a sample naming brief for testing."""
    return NamingBrief(
        project_codename="test_integration",
        description="A test integration project",
        target_markets=["USA", "Europe"],
        industry="Technology",
        brand_personality="Innovative",
        avoid_terms=["test", "avoid"],
        candidate_count=3,
    )


@pytest.fixture
def mock_candidates():
    """Create mock candidates for testing."""
    return [
        NameCandidate(
            candidate_id="cand_001",
            name="Innovatech",
            typology=NameTypology.PORTMANTEAU,
            rationale="Combines innovation and technology",
            phonetic_notes="Easy to pronounce in English",
            tagline_options=["Innovating Tomorrow", "Tech Forward"],
            brand_story_seed="Born from the need to bridge innovation and technology...",
            internal_assessment=InternalAssessment(
                score=0.92,
                rationale="Strong brand potential with clear market positioning",
                source="llm_self_assessment",
            ),
        ),
        NameCandidate(
            candidate_id="cand_002",
            name="NexaFlow",
            typology=NameTypology.COMPOUND,
            rationale="Suggests next-generation workflow",
            phonetic_notes="Smooth pronunciation across languages",
            tagline_options=["Flow Into the Future"],
            brand_story_seed="NexaFlow represents the seamless integration...",
            internal_assessment=InternalAssessment(
                score=0.88,
                rationale="Good memorability and market fit",
                source="llm_self_assessment",
            ),
        ),
        NameCandidate(
            candidate_id="cand_003",
            name="Synthex",
            typology=NameTypology.INVENTED,
            rationale="Unique invented name with tech connotations",
            phonetic_notes="Distinctive sound profile",
            tagline_options=["Synthesize Success"],
            brand_story_seed="Synthex emerges from the synthesis of ideas...",
            internal_assessment=InternalAssessment(
                score=0.85,
                rationale="Unique and memorable",
                source="llm_self_assessment",
            ),
        ),
    ]


@pytest.fixture
def mock_validation_results():
    """Create mock validation results for testing."""
    return [
        # Domain validations
        ValidationResult(
            target="innovatech.com",
            channel=ValidationChannel.DOMAIN,
            status=ValidationStatus.AVAILABLE,
            confidence=Confidence.CONFIRMED,
            evidence=Evidence(
                source="domain",
                url="https://rdap.org/innovatech.com",
                checked_at=datetime.now(),
                raw={"status": "available"},
            ),
            candidate_id="cand_001",
            validation_id="val_001",
            adapter_version="1.0.0",
            checked_at=datetime.now(),
        ),
        ValidationResult(
            target="nexaflow.com",
            channel=ValidationChannel.DOMAIN,
            status=ValidationStatus.TAKEN,
            confidence=Confidence.CONFIRMED,
            evidence=Evidence(
                source="domain",
                url="https://rdap.org/nexaflow.com",
                checked_at=datetime.now(),
                raw={"status": "registered"},
            ),
            candidate_id="cand_002",
            validation_id="val_002",
            adapter_version="1.0.0",
            checked_at=datetime.now(),
        ),
        # Trademark validations
        ValidationResult(
            target="Innovatech",
            channel=ValidationChannel.TRADEMARK_TR,
            status=ValidationStatus.AVAILABLE,
            confidence=Confidence.LIKELY,
            evidence=Evidence(
                source="trademark_tr",
                url="https://turkpatent.gov.tr",
                checked_at=datetime.now(),
                raw={"matches": 0},
            ),
            candidate_id="cand_001",
            validation_id="val_003",
            adapter_version="1.0.0",
            checked_at=datetime.now(),
        ),
        # Social media validations
        ValidationResult(
            target="@innovatech",
            channel=ValidationChannel.SOCIAL_X,
            status=ValidationStatus.AVAILABLE,
            confidence=Confidence.CONFIRMED,
            evidence=Evidence(
                source="social_x",
                url="https://x.com/innovatech",
                checked_at=datetime.now(),
                raw={"status": "available"},
            ),
            candidate_id="cand_001",
            validation_id="val_004",
            adapter_version="1.0.0",
            checked_at=datetime.now(),
        ),
    ]


class TestFullPipelineIntegration:
    """Integration tests for the complete Launch Engine pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_end_to_end(
        self, naming_brief, mock_candidates, mock_validation_results
    ):
        """Test the complete pipeline from brief to validated candidates."""
        with (
            patch("launch_engine.engine.LLMAdapter") as mock_llm_class,
            patch("launch_engine.engine.SQLiteCache") as mock_cache_class,
            patch("launch_engine.engine.BrandNamingModule") as mock_bn_class,
            patch("launch_engine.engine.ValidationPipeline") as mock_vp_class,
            patch(
                "launch_engine.validation.adapters.domain.DomainAdapter"
            ) as mock_domain_class,
            patch(
                "launch_engine.validation.adapters.trademark.TrademarkAdapter"
            ) as mock_trademark_class,
            patch(
                "launch_engine.validation.adapters.social.SocialMediaAdapter"
            ) as mock_social_class,
        ):

            # Setup LLM mock
            mock_llm = MagicMock()
            mock_llm.provider = "ollama"
            mock_llm.model_id = "ollama/qwen3:14b"
            mock_llm_class.return_value = mock_llm

            # Setup cache mock
            mock_cache = MagicMock()
            mock_cache.initialize = AsyncMock()
            mock_cache_class.return_value = mock_cache

            # Setup BrandNamingModule mock
            mock_bn = MagicMock()
            mock_candidate_list = NameCandidateList(
                brief_ref=naming_brief.project_codename,
                candidates=mock_candidates,
                llm_model_used="ollama/qwen3:14b",
                llm_provider="ollama",
                generated_at=datetime.now(),
            )
            mock_bn.run = AsyncMock(return_value=mock_candidate_list)
            mock_bn_class.return_value = mock_bn

            # Setup ValidationPipeline mock
            mock_vp = MagicMock()
            mock_vp.validate_all = AsyncMock(return_value=mock_validation_results)
            mock_vp_class.return_value = mock_vp

            # Setup adapter mocks
            for mock_adapter_class in [
                mock_domain_class,
                mock_trademark_class,
                mock_social_class,
            ]:
                mock_adapter = MagicMock()
                mock_adapter.policy = MagicMock()
                mock_adapter.policy.rate_limit_per_minute = 60
                mock_adapter_class.return_value = mock_adapter

            # Create engine and run full pipeline
            engine = LaunchEngine(
                llm_provider="ollama",
                llm_model="qwen3:14b",
                cache_db_path=":memory:",
            )

            candidates, validation_results = await engine.run_full_pipeline(
                naming_brief
            )

            # Verify results
            assert candidates.brief_ref == naming_brief.project_codename
            assert len(candidates.candidates) == 3
            assert candidates.llm_provider == "ollama"

            assert len(validation_results) == 4
            assert all(isinstance(r, ValidationResult) for r in validation_results)

            # Verify mocks were called correctly
            mock_bn.run.assert_called_once_with(naming_brief)
            mock_vp.validate_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_pipeline_with_partial_failures(self, naming_brief, mock_candidates):
        """Test pipeline behavior when some validations fail."""
        with (
            patch("launch_engine.engine.LLMAdapter") as mock_llm_class,
            patch("launch_engine.engine.SQLiteCache") as mock_cache_class,
            patch("launch_engine.engine.BrandNamingModule") as mock_bn_class,
            patch("launch_engine.engine.ValidationPipeline") as mock_vp_class,
            patch(
                "launch_engine.validation.adapters.domain.DomainAdapter"
            ) as mock_domain_class,
            patch(
                "launch_engine.validation.adapters.trademark.TrademarkAdapter"
            ) as mock_trademark_class,
            patch(
                "launch_engine.validation.adapters.social.SocialMediaAdapter"
            ) as mock_social_class,
        ):

            # Setup mocks
            mock_llm = MagicMock()
            mock_llm.provider = "ollama"
            mock_llm.model_id = "ollama/qwen3:14b"
            mock_llm_class.return_value = mock_llm

            mock_cache = MagicMock()
            mock_cache.initialize = AsyncMock()
            mock_cache_class.return_value = mock_cache

            mock_candidate_list = NameCandidateList(
                brief_ref=naming_brief.project_codename,
                candidates=mock_candidates,
                llm_model_used="ollama/qwen3:14b",
                llm_provider="ollama",
                generated_at=datetime.now(),
            )
            mock_bn = MagicMock()
            mock_bn.run = AsyncMock(return_value=mock_candidate_list)
            mock_bn_class.return_value = mock_bn

            # Simulate partial validation failures
            partial_results = [
                ValidationResult(
                    target="innovatech.com",
                    channel=ValidationChannel.DOMAIN,
                    status=ValidationStatus.AVAILABLE,
                    confidence=Confidence.CONFIRMED,
                    evidence=Evidence(source="domain", checked_at=datetime.now()),
                    candidate_id="cand_001",
                    validation_id="val_001",
                    adapter_version="1.0.0",
                    checked_at=datetime.now(),
                ),
                ValidationResult(
                    target="innovatech",
                    channel=ValidationChannel.TRADEMARK_TR,
                    status=ValidationStatus.UNVERIFIABLE,
                    confidence=Confidence.UNKNOWN,
                    evidence=Evidence(
                        source="trademark_tr",
                        checked_at=datetime.now(),
                        raw={"error": "Service unavailable"},
                    ),
                    candidate_id="cand_001",
                    validation_id="val_002",
                    adapter_version="1.0.0",
                    checked_at=datetime.now(),
                    manual_review_url="https://turkpatent.gov.tr",
                ),
            ]

            mock_vp = MagicMock()
            mock_vp.validate_all = AsyncMock(return_value=partial_results)
            mock_vp_class.return_value = mock_vp

            for mock_adapter_class in [
                mock_domain_class,
                mock_trademark_class,
                mock_social_class,
            ]:
                mock_adapter = MagicMock()
                mock_adapter.policy = MagicMock()
                mock_adapter.policy.rate_limit_per_minute = 60
                mock_adapter_class.return_value = mock_adapter

            engine = LaunchEngine(
                llm_provider="ollama",
                llm_model="qwen3:14b",
                cache_db_path=":memory:",
            )

            candidates, validation_results = await engine.run_full_pipeline(
                naming_brief
            )

            # Verify pipeline completed despite partial failures
            assert len(candidates.candidates) == 3
            assert len(validation_results) == 2

            # Verify mixed statuses
            statuses = [r.status for r in validation_results]
            assert ValidationStatus.AVAILABLE in statuses
            assert ValidationStatus.UNVERIFIABLE in statuses

    @pytest.mark.asyncio
    async def test_cache_integration(self, naming_brief, mock_candidates):
        """Test that cache is properly integrated into the pipeline."""
        with (
            patch("launch_engine.engine.LLMAdapter") as mock_llm_class,
            patch("launch_engine.engine.SQLiteCache") as mock_cache_class,
            patch("launch_engine.engine.BrandNamingModule") as mock_bn_class,
            patch("launch_engine.engine.ValidationPipeline") as mock_vp_class,
            patch(
                "launch_engine.validation.adapters.domain.DomainAdapter"
            ) as mock_domain_class,
            patch(
                "launch_engine.validation.adapters.trademark.TrademarkAdapter"
            ) as mock_trademark_class,
            patch(
                "launch_engine.validation.adapters.social.SocialMediaAdapter"
            ) as mock_social_class,
        ):

            # Setup mocks
            mock_llm = MagicMock()
            mock_llm.provider = "ollama"
            mock_llm.model_id = "ollama/qwen3:14b"
            mock_llm_class.return_value = mock_llm

            mock_cache = MagicMock()
            mock_cache.initialize = AsyncMock()
            mock_cache.get = AsyncMock(return_value=None)  # Cache miss
            mock_cache.set = AsyncMock()
            mock_cache_class.return_value = mock_cache

            mock_candidate_list = NameCandidateList(
                brief_ref=naming_brief.project_codename,
                candidates=mock_candidates,
                llm_model_used="ollama/qwen3:14b",
                llm_provider="ollama",
                generated_at=datetime.now(),
            )
            mock_bn = MagicMock()
            mock_bn.run = AsyncMock(return_value=mock_candidate_list)
            mock_bn_class.return_value = mock_bn

            validation_results = [
                ValidationResult(
                    target="innovatech.com",
                    channel=ValidationChannel.DOMAIN,
                    status=ValidationStatus.AVAILABLE,
                    confidence=Confidence.CONFIRMED,
                    evidence=Evidence(source="domain", checked_at=datetime.now()),
                    candidate_id="cand_001",
                    validation_id="val_001",
                    adapter_version="1.0.0",
                    checked_at=datetime.now(),
                ),
            ]
            mock_vp = MagicMock()
            mock_vp.validate_all = AsyncMock(return_value=validation_results)
            mock_vp_class.return_value = mock_vp

            for mock_adapter_class in [
                mock_domain_class,
                mock_trademark_class,
                mock_social_class,
            ]:
                mock_adapter = MagicMock()
                mock_adapter.policy = MagicMock()
                mock_adapter.policy.rate_limit_per_minute = 60
                mock_adapter_class.return_value = mock_adapter

            engine = LaunchEngine(
                llm_provider="ollama",
                llm_model="qwen3:14b",
                cache_db_path=":memory:",
            )

            await engine.run_full_pipeline(naming_brief)

            # Verify cache was created and passed to pipeline
            assert engine.cache is not None
            assert engine.validation_pipeline is not None

    @pytest.mark.asyncio
    async def test_error_recovery_in_pipeline(self, naming_brief):
        """Test that pipeline recovers gracefully from errors."""
        with (
            patch("launch_engine.engine.LLMAdapter") as mock_llm_class,
            patch("launch_engine.engine.SQLiteCache") as mock_cache_class,
            patch("launch_engine.engine.BrandNamingModule") as mock_bn_class,
            patch("launch_engine.engine.ValidationPipeline"),
            patch(
                "launch_engine.validation.adapters.domain.DomainAdapter"
            ) as mock_domain_class,
            patch(
                "launch_engine.validation.adapters.trademark.TrademarkAdapter"
            ) as mock_trademark_class,
            patch(
                "launch_engine.validation.adapters.social.SocialMediaAdapter"
            ) as mock_social_class,
        ):

            # Setup mocks
            mock_llm = MagicMock()
            mock_llm.provider = "ollama"
            mock_llm.model_id = "ollama/qwen3:14b"
            mock_llm_class.return_value = mock_llm

            mock_cache = MagicMock()
            mock_cache.initialize = AsyncMock()
            mock_cache_class.return_value = mock_cache

            # Simulate generation failure
            mock_bn = MagicMock()
            mock_bn.run = AsyncMock(side_effect=Exception("LLM service unavailable"))
            mock_bn_class.return_value = mock_bn

            for mock_adapter_class in [
                mock_domain_class,
                mock_trademark_class,
                mock_social_class,
            ]:
                mock_adapter = MagicMock()
                mock_adapter.policy = MagicMock()
                mock_adapter.policy.rate_limit_per_minute = 60
                mock_adapter_class.return_value = mock_adapter

            engine = LaunchEngine(
                llm_provider="ollama",
                llm_model="qwen3:14b",
                cache_db_path=":memory:",
            )

            # Pipeline should handle error gracefully
            candidates, validation_results = await engine.run_full_pipeline(
                naming_brief
            )

            # Should return empty results but not crash
            assert len(candidates.candidates) == 0
            assert len(validation_results) == 0


class TestCrossComponentIntegration:
    """Tests for integration between specific components."""

    @pytest.mark.asyncio
    async def test_naming_module_to_validation_pipeline(
        self, naming_brief, mock_candidates
    ):
        """Test data flow from naming module to validation pipeline."""
        with (
            patch("launch_engine.engine.LLMAdapter") as mock_llm_class,
            patch("launch_engine.engine.SQLiteCache") as mock_cache_class,
            patch("launch_engine.engine.BrandNamingModule") as mock_bn_class,
            patch("launch_engine.engine.ValidationPipeline") as mock_vp_class,
            patch(
                "launch_engine.validation.adapters.domain.DomainAdapter"
            ) as mock_domain_class,
            patch(
                "launch_engine.validation.adapters.trademark.TrademarkAdapter"
            ) as mock_trademark_class,
            patch(
                "launch_engine.validation.adapters.social.SocialMediaAdapter"
            ) as mock_social_class,
        ):

            # Setup mocks
            mock_llm = MagicMock()
            mock_llm.provider = "ollama"
            mock_llm.model_id = "ollama/qwen3:14b"
            mock_llm_class.return_value = mock_llm

            mock_cache = MagicMock()
            mock_cache.initialize = AsyncMock()
            mock_cache_class.return_value = mock_cache

            mock_candidate_list = NameCandidateList(
                brief_ref=naming_brief.project_codename,
                candidates=mock_candidates,
                llm_model_used="ollama/qwen3:14b",
                llm_provider="ollama",
                generated_at=datetime.now(),
            )
            mock_bn = MagicMock()
            mock_bn.run = AsyncMock(return_value=mock_candidate_list)
            mock_bn_class.return_value = mock_bn

            mock_vp = MagicMock()
            mock_vp.validate_all = AsyncMock(return_value=[])
            mock_vp_class.return_value = mock_vp

            for mock_adapter_class in [
                mock_domain_class,
                mock_trademark_class,
                mock_social_class,
            ]:
                mock_adapter = MagicMock()
                mock_adapter.policy = MagicMock()
                mock_adapter.policy.rate_limit_per_minute = 60
                mock_adapter_class.return_value = mock_adapter

            engine = LaunchEngine(
                llm_provider="ollama",
                llm_model="qwen3:14b",
                cache_db_path=":memory:",
            )

            await engine.run_full_pipeline(naming_brief)

            # Verify candidates were passed to validation pipeline
            call_args = mock_vp.validate_all.call_args
            passed_candidates = call_args[0][0]
            passed_brief = call_args[0][1]

            assert len(passed_candidates) == 3
            assert passed_brief == naming_brief
            assert all(isinstance(c, NameCandidate) for c in passed_candidates)
