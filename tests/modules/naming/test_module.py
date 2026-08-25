"""
Tests for the BrandNamingModule.
"""

import json
from unittest.mock import Mock, AsyncMock
import pytest

from launch_engine.modules.naming.module import BrandNamingModule
from launch_engine.modules.naming.brief import (
    NamingBrief,
    NameTypology,
    PhoneticConstraints,
)
from launch_engine.modules.naming.candidates import (
    NameCandidate,
    NameCandidateList,
)
from launch_engine.llm import LLMAdapter


@pytest.fixture
def mock_llm_adapter():
    """Create a mock LLM adapter."""
    adapter = Mock(spec=LLMAdapter)
    adapter.provider = "openai"
    adapter.model = "gpt-4"
    adapter.model_id = "openai/gpt-4"
    adapter.generate = AsyncMock()
    return adapter


@pytest.fixture
def sample_brief():
    """Create a sample naming brief for testing."""
    return NamingBrief(
        project_codename="TestProject",
        description="A test project for brand naming",
        target_markets=["US", "EU"],
        industry="Technology",
        brand_personality="Innovative",
        phonetic_constraints=PhoneticConstraints(
            max_syllables=3, max_length=10, avoid_sounds=["z", "x"]
        ),
        avoid_terms=["test", "bad"],
        preferred_typologies=[NameTypology.INVENTED, NameTypology.METAPHORICAL],
        candidate_count=5,
        language="en",
    )


@pytest.fixture
def naming_module(mock_llm_adapter):
    """Create a BrandNamingModule instance."""
    return BrandNamingModule(llm_adapter=mock_llm_adapter)


def test_module_initialization(naming_module, mock_llm_adapter):
    """Test module initialization with LLMAdapter."""
    assert naming_module.name == "brand_naming"
    assert naming_module.llm_adapter == mock_llm_adapter


@pytest.mark.asyncio
async def test_run_returns_namelist(naming_module, mock_llm_adapter, sample_brief):
    """Test that run() returns a NameCandidateList."""
    # Mock LLM responses for divergent and convergent stages
    mock_llm_adapter.generate.side_effect = [
        # Divergent generation response
        json.dumps(
            [
                {
                    "name": "Novatek",
                    "typology": "INVENTED",
                    "rationale": "Modern and technical sounding",
                    "phonetic_notes": "Two syllables, easy to pronounce",
                    "tagline_options": [
                        "Innovate with Novatek",
                        "Novatek: Tech Forward",
                    ],
                    "brand_story_seed": "A story about technological innovation",
                },
                {
                    "name": "Brighto",  # 7 chars, fits within max_length=10
                    "typology": "INVENTED",
                    "rationale": "Suggests brightness and optimism",
                    "phonetic_notes": "Two syllables",
                    "tagline_options": ["Stay Bright with Brighto"],
                    "brand_story_seed": "A story about bringing light to technology",
                },
            ]
        ),
        # Convergent scoring response
        json.dumps(
            [
                {
                    "candidate_id": "cand_001",
                    "score": 0.85,
                    "rationale": "Strong fit with innovative brand personality",
                },
                {
                    "candidate_id": "cand_002",
                    "score": 0.75,
                    "rationale": "Good fit but slightly less distinctive",
                },
            ]
        ),
    ]

    # Run the module
    result = await naming_module.run(sample_brief)

    # Assertions
    assert isinstance(result, NameCandidateList)
    assert result.brief_ref == sample_brief.project_codename
    assert (
        len(result.candidates) == 2
    )  # Limited by candidate_count=5 but we only have 2
    assert result.llm_model_used == mock_llm_adapter.model_id
    assert result.llm_provider == mock_llm_adapter.provider

    # Check first candidate
    assert result.candidates[0].name == "Novatek"
    assert result.candidates[0].typology == NameTypology.INVENTED
    assert result.candidates[0].internal_assessment is not None
    assert result.candidates[0].internal_assessment.score == 0.85

    # Check second candidate
    assert result.candidates[1].name == "Brighto"
    assert result.candidates[1].typology == NameTypology.INVENTED
    assert result.candidates[1].internal_assessment is not None
    assert result.candidates[1].internal_assessment.score == 0.75


@pytest.mark.asyncio
async def test_divergent_generation_with_mocked_llm(
    naming_module, mock_llm_adapter, sample_brief
):
    """Test divergent generation with mocked LLM."""
    # Mock LLM response
    mock_response = json.dumps(
        [
            {
                "name": "TestName",
                "typology": "INVENTED",
                "rationale": "A test name",
                "phonetic_notes": "One syllable",
                "tagline_options": ["Test it"],
                "brand_story_seed": "A test story",
            }
        ]
    )
    mock_llm_adapter.generate.return_value = mock_response

    # Call divergent generation
    candidates = await naming_module._divergent_generate(sample_brief)

    # Assertions
    assert len(candidates) == 1
    assert candidates[0].name == "TestName"
    assert candidates[0].typology == NameTypology.INVENTED
    assert candidates[0].candidate_id == "cand_001"
    assert mock_llm_adapter.generate.called


def test_midfilter_removes_invalid_candidates(naming_module, sample_brief):
    """Test that midfilter removes invalid candidates."""
    # Create test candidates
    candidates = [
        NameCandidate(
            candidate_id="cand_001",
            name="testname",  # Contains avoided term "test"
            typology=NameTypology.INVENTED,
            rationale="A test name",
        ),
        NameCandidate(
            candidate_id="cand_002",
            name="xenon",  # Contains avoided sound "x"
            typology=NameTypology.INVENTED,
            rationale="A chemical name",
        ),
        NameCandidate(
            candidate_id="cand_003",
            name="verylongname",  # Exceeds max_length of 10
            typology=NameTypology.INVENTED,
            rationale="A long name",
        ),
        NameCandidate(
            candidate_id="cand_004",
            name="super",  # Valid name
            typology=NameTypology.INVENTED,
            rationale="A good name",
        ),
    ]

    # Apply midfilter
    filtered = naming_module._midfilter(candidates, sample_brief)

    # Assertions
    assert len(filtered) == 1
    assert filtered[0].name == "super"
    assert filtered[0].candidate_id == "cand_004"


@pytest.mark.asyncio
async def test_convergent_scoring_with_mocked_llm(
    naming_module, mock_llm_adapter, sample_brief
):
    """Test convergent scoring with mocked LLM."""
    # Create test candidates
    candidates = [
        NameCandidate(
            candidate_id="cand_001",
            name="GoodName",
            typology=NameTypology.INVENTED,
            rationale="A good name",
        ),
        NameCandidate(
            candidate_id="cand_002",
            name="BetterName",
            typology=NameTypology.INVENTED,
            rationale="A better name",
        ),
    ]

    # Mock LLM response for scoring
    mock_response = json.dumps(
        [
            {"candidate_id": "cand_001", "score": 0.7, "rationale": "Decent fit"},
            {"candidate_id": "cand_002", "score": 0.9, "rationale": "Excellent fit"},
        ]
    )
    mock_llm_adapter.generate.return_value = mock_response

    # Call convergent scoring
    scored = await naming_module._convergent_score(candidates, sample_brief)

    # Assertions
    assert len(scored) == 2
    # Should be sorted by score descending
    assert scored[0].name == "BetterName"  # Higher score first
    assert scored[0].internal_assessment.score == 0.9
    assert scored[1].name == "GoodName"
    assert scored[1].internal_assessment.score == 0.7
    assert mock_llm_adapter.generate.called


@pytest.mark.asyncio
async def test_error_handling_when_llm_fails(
    naming_module, mock_llm_adapter, sample_brief
):
    """Test error handling when LLM fails."""
    # Make LLM generate raise an exception
    mock_llm_adapter.generate.side_effect = RuntimeError("API Error")

    # Run the module - should handle gracefully
    result = await naming_module.run(sample_brief)

    # Assertions
    assert isinstance(result, NameCandidateList)
    assert result.brief_ref == sample_brief.project_codename
    assert len(result.candidates) == 0  # Empty list due to LLM failure
    assert result.llm_model_used == mock_llm_adapter.model_id
    assert result.llm_provider == mock_llm_adapter.provider


def test_prefilter_validates_and_cleans(naming_module):
    """Test that _prefilter validates brief and cleans avoid list."""
    # Create brief with duplicate and empty avoid terms
    brief = NamingBrief(
        project_codename="Test",
        description="Test",
        target_markets=["US"],
        industry="Tech",
        avoid_terms=["test", "TEST", "", "bad", "test"],  # Duplicates and empty
        preferred_typologies=[
            NameTypology.INVENTED,
            NameTypology.INVENTED,
        ],  # Duplicate
    )

    # Apply prefilter
    naming_module._prefilter(brief)

    # Assertions - check content, not order
    assert set(brief.avoid_terms) == {
        "test",
        "bad",
    }  # Lowercased, deduplicated, empty removed
    assert len(brief.avoid_terms) == 2
    assert brief.preferred_typologies == [NameTypology.INVENTED]  # Deduplicated


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
