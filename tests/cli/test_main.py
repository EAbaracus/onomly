"""Tests for CLI interface."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from launch_engine.cli.main import app
from launch_engine.modules.naming.brief import NameTypology
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
from datetime import datetime, timezone

runner = CliRunner()


@pytest.fixture
def sample_candidate_list():
    """Create a sample NameCandidateList for testing."""
    candidates = [
        NameCandidate(
            candidate_id="cand_001",
            name="Veyra",
            typology=NameTypology.INVENTED,
            rationale="Modern and memorable",
            internal_assessment=InternalAssessment(
                score=0.85,
                rationale="Strong brand potential",
                source="llm_self_assessment",
            ),
        ),
        NameCandidate(
            candidate_id="cand_002",
            name="Nexora",
            typology=NameTypology.SUGGESTIVE,
            rationale="Suggests innovation",
            internal_assessment=InternalAssessment(
                score=0.78,
                rationale="Good market fit",
                source="llm_self_assessment",
            ),
        ),
    ]
    return NameCandidateList(
        brief_ref="test_project",
        candidates=candidates,
        llm_model_used="ollama/qwen3:14b",
        llm_provider="ollama",
        generated_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_validation_results():
    """Create sample validation results for testing."""
    return [
        ValidationResult(
            target="veyra.com",
            channel=ValidationChannel.DOMAIN,
            status=ValidationStatus.AVAILABLE,
            confidence=Confidence.CONFIRMED,
            evidence=Evidence(
                source="domain",
                url="https://rdap.org/veyra.com",
                checked_at=datetime.now(timezone.utc),
            ),
            candidate_id="cand_001",
            validation_id="val_001",
            adapter_version="1.0.0",
            checked_at=datetime.now(timezone.utc),
        ),
        ValidationResult(
            target="veyra",
            channel=ValidationChannel.SOCIAL_X,
            status=ValidationStatus.TAKEN,
            confidence=Confidence.CONFIRMED,
            evidence=Evidence(
                source="social_x",
                url="https://x.com/veyra",
                checked_at=datetime.now(timezone.utc),
            ),
            candidate_id="cand_001",
            validation_id="val_002",
            adapter_version="1.0.0",
            checked_at=datetime.now(timezone.utc),
        ),
    ]


def test_generate_names_table_output(sample_candidate_list):
    """Test generate-names command with table output."""
    with patch("launch_engine.cli.main.LaunchEngine") as mock_engine_class:
        mock_engine = MagicMock()
        mock_engine.generate_names = AsyncMock(return_value=sample_candidate_list)
        mock_engine_class.return_value = mock_engine

        result = runner.invoke(
            app,
            [
                "generate-names",
                "--project-codename",
                "test_project",
                "--description",
                "Test project description",
                "--target-markets",
                "USA,Europe",
                "--industry",
                "Technology",
                "--output-format",
                "table",
            ],
        )

        assert result.exit_code == 0
        assert "Veyra" in result.stdout
        assert "Nexora" in result.stdout
        assert "INVENTED" in result.stdout
        mock_engine.generate_names.assert_called_once()


def test_generate_names_json_output(sample_candidate_list):
    """Test generate-names command with JSON output."""
    with patch("launch_engine.cli.main.LaunchEngine") as mock_engine_class:
        mock_engine = MagicMock()
        mock_engine.generate_names = AsyncMock(return_value=sample_candidate_list)
        mock_engine_class.return_value = mock_engine

        result = runner.invoke(
            app,
            [
                "generate-names",
                "--project-codename",
                "test_project",
                "--description",
                "Test project description",
                "--target-markets",
                "USA,Europe",
                "--industry",
                "Technology",
                "--output-format",
                "json",
            ],
        )

        assert result.exit_code == 0
        # Verify JSON is valid
        output_data = json.loads(result.stdout)
        assert output_data["brief_ref"] == "test_project"
        assert len(output_data["candidates"]) == 2
        assert output_data["candidates"][0]["name"] == "Veyra"


def test_generate_names_csv_output(sample_candidate_list):
    """Test generate-names command with CSV output."""
    with patch("launch_engine.cli.main.LaunchEngine") as mock_engine_class:
        mock_engine = MagicMock()
        mock_engine.generate_names = AsyncMock(return_value=sample_candidate_list)
        mock_engine_class.return_value = mock_engine

        result = runner.invoke(
            app,
            [
                "generate-names",
                "--project-codename",
                "test_project",
                "--description",
                "Test project description",
                "--target-markets",
                "USA,Europe",
                "--industry",
                "Technology",
                "--output-format",
                "csv",
            ],
        )

        assert result.exit_code == 0
        assert "name,typology,rationale,score" in result.stdout
        assert "Veyra" in result.stdout
        assert "INVENTED" in result.stdout


def test_validate_table_output(
    sample_candidate_list, sample_validation_results, tmp_path
):
    """Test validate command with table output."""
    # Create temporary candidates file
    candidates_file = tmp_path / "candidates.json"
    candidates_file.write_text(sample_candidate_list.model_dump_json())

    with patch("launch_engine.cli.main.LaunchEngine") as mock_engine_class:
        mock_engine = MagicMock()
        mock_engine.validate_names = AsyncMock(return_value=sample_validation_results)
        mock_engine_class.return_value = mock_engine

        result = runner.invoke(
            app,
            [
                "validate",
                "--candidates-file",
                str(candidates_file),
                "--target-markets",
                "USA,Europe",
                "--industry",
                "Technology",
                "--output-format",
                "table",
            ],
        )

        assert result.exit_code == 0
        assert "veyra.com" in result.stdout
        assert "AVAILABLE" in result.stdout
        assert "TAKEN" in result.stdout
        mock_engine.validate_names.assert_called_once()


def test_validate_json_output(
    sample_candidate_list, sample_validation_results, tmp_path
):
    """Test validate command with JSON output."""
    # Create temporary candidates file
    candidates_file = tmp_path / "candidates.json"
    candidates_file.write_text(sample_candidate_list.model_dump_json())

    with patch("launch_engine.cli.main.LaunchEngine") as mock_engine_class:
        mock_engine = MagicMock()
        mock_engine.validate_names = AsyncMock(return_value=sample_validation_results)
        mock_engine_class.return_value = mock_engine

        result = runner.invoke(
            app,
            [
                "validate",
                "--candidates-file",
                str(candidates_file),
                "--target-markets",
                "USA,Europe",
                "--industry",
                "Technology",
                "--output-format",
                "json",
            ],
        )

        assert result.exit_code == 0
        # Verify JSON is valid
        output_data = json.loads(result.stdout)
        assert len(output_data) == 2
        assert output_data[0]["target"] == "veyra.com"
        assert output_data[0]["status"] == "available"


def test_validate_csv_output(
    sample_candidate_list, sample_validation_results, tmp_path
):
    """Test validate command with CSV output."""
    # Create temporary candidates file
    candidates_file = tmp_path / "candidates.json"
    candidates_file.write_text(sample_candidate_list.model_dump_json())

    with patch("launch_engine.cli.main.LaunchEngine") as mock_engine_class:
        mock_engine = MagicMock()
        mock_engine.validate_names = AsyncMock(return_value=sample_validation_results)
        mock_engine_class.return_value = mock_engine

        result = runner.invoke(
            app,
            [
                "validate",
                "--candidates-file",
                str(candidates_file),
                "--target-markets",
                "USA,Europe",
                "--industry",
                "Technology",
                "--output-format",
                "csv",
            ],
        )

        assert result.exit_code == 0
        assert "target,channel,status,confidence,manual_review_url" in result.stdout
        assert "veyra.com" in result.stdout
        assert "available" in result.stdout


def test_validate_file_not_found():
    """Test validate command with non-existent file."""
    result = runner.invoke(
        app,
        [
            "validate",
            "--candidates-file",
            "/nonexistent/file.json",
            "--target-markets",
            "USA,Europe",
            "--industry",
            "Technology",
        ],
    )

    assert result.exit_code == 1
    assert "Error" in result.stdout


def test_cache_clear():
    """Test cache clear command."""
    with patch("launch_engine.cache.SQLiteCache") as mock_cache_class:
        mock_cache = MagicMock()
        mock_cache.initialize = AsyncMock()
        mock_cache.clear = AsyncMock()
        mock_cache.close = AsyncMock()
        mock_cache_class.return_value = mock_cache

        result = runner.invoke(app, ["cache", "clear"])

        assert result.exit_code == 0
        assert "cleared" in result.stdout.lower()
        mock_cache.initialize.assert_called_once()
        mock_cache.clear.assert_called_once()
        mock_cache.close.assert_called_once()


def test_cache_stats():
    """Test cache stats command."""
    with patch("launch_engine.cache.SQLiteCache") as mock_cache_class:
        mock_cache = MagicMock()
        mock_cache.initialize = AsyncMock()
        mock_cache.close = AsyncMock()
        mock_cache_class.return_value = mock_cache

        result = runner.invoke(app, ["cache", "stats"])

        assert result.exit_code == 0
        assert "Cache database" in result.stdout
        mock_cache.initialize.assert_called_once()
        mock_cache.close.assert_called_once()


def test_cache_invalid_action():
    """Test cache command with invalid action."""
    result = runner.invoke(app, ["cache", "invalid"])

    assert result.exit_code == 1
    assert "Unknown action" in result.stdout or "Error" in result.stdout


def test_adapters_command():
    """Test adapters command."""
    result = runner.invoke(app, ["adapters"])

    assert result.exit_code == 0
    assert "DomainAdapter" in result.stdout
    assert "TrademarkAdapter" in result.stdout
    assert "SocialMediaAdapter" in result.stdout
    assert "DOMAIN" in result.stdout
    assert "TRADEMARK" in result.stdout


def test_generate_names_with_optional_params(sample_candidate_list):
    """Test generate-names with optional parameters."""
    with patch("launch_engine.cli.main.LaunchEngine") as mock_engine_class:
        mock_engine = MagicMock()
        mock_engine.generate_names = AsyncMock(return_value=sample_candidate_list)
        mock_engine_class.return_value = mock_engine

        result = runner.invoke(
            app,
            [
                "generate-names",
                "--project-codename",
                "test_project",
                "--description",
                "Test project description",
                "--target-markets",
                "USA,Europe",
                "--industry",
                "Technology",
                "--brand-personality",
                "innovative,modern",
                "--avoid-terms",
                "test,avoid",
                "--candidate-count",
                "15",
                "--llm-provider",
                "openai",
                "--llm-model",
                "gpt-4",
                "--output-format",
                "table",
            ],
        )

        assert result.exit_code == 0
        # Verify the brief was created with correct parameters
        call_args = mock_engine.generate_names.call_args
        brief = call_args[0][0]
        assert brief.project_codename == "test_project"
        assert brief.brand_personality == "innovative,modern"
        assert brief.avoid_terms == ["test", "avoid"]
        assert brief.candidate_count == 15


def test_generate_names_error_handling():
    """Test generate-names error handling."""
    with patch("launch_engine.cli.main.LaunchEngine") as mock_engine_class:
        mock_engine = MagicMock()
        mock_engine.generate_names = AsyncMock(side_effect=Exception("Test error"))
        mock_engine_class.return_value = mock_engine

        result = runner.invoke(
            app,
            [
                "generate-names",
                "--project-codename",
                "test_project",
                "--description",
                "Test project description",
                "--target-markets",
                "USA,Europe",
                "--industry",
                "Technology",
            ],
        )

        assert result.exit_code == 1
        assert "Error" in result.stdout


def test_models_command_lists_catalog():
    """Test the `models` command prints the catalog without prompting."""
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "Onomly LLM models" in result.stdout
    assert "9router" in result.stdout
    assert "Nemotron" in result.stdout


def test_configure_selects_by_number(tmp_path, monkeypatch):
    """Test `configure` picks a model from the numbered list and persists it."""
    from launch_engine import config as config_mod

    monkeypatch.setattr(config_mod, "config_path", lambda: tmp_path / "config.json")

    result = runner.invoke(app, ["configure"], input="2\n")
    assert result.exit_code == 0
    assert "Saved:" in result.stdout
    saved = config_mod.load_config(tmp_path / "config.json")
    assert saved.configured is True
    from launch_engine import models as model_catalog

    assert saved.model_id == model_catalog.MODELS[1].id


def test_configure_accepts_raw_id(tmp_path, monkeypatch):
    """Test `configure` accepts a raw provider/model id input."""
    from launch_engine import config as config_mod

    monkeypatch.setattr(config_mod, "config_path", lambda: tmp_path / "config.json")

    result = runner.invoke(app, ["configure"], input="ollama/llama3:8b\n")
    assert result.exit_code == 0
    saved = config_mod.load_config(tmp_path / "config.json")
    assert saved.llm_provider == "ollama"
    assert saved.llm_model == "llama3:8b"


def test_configure_invalid_selection(tmp_path, monkeypatch):
    """Test `configure` rejects an out-of-range / malformed selection."""
    from launch_engine import config as config_mod

    monkeypatch.setattr(config_mod, "config_path", lambda: tmp_path / "config.json")

    result = runner.invoke(app, ["configure"], input="999\n")
    assert result.exit_code == 1
    assert "Invalid selection" in result.stdout
