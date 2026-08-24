import json
from pathlib import Path

import pytest

from launch_engine.config import OnomlyConfig, load_config, save_config


def test_load_config_missing_file(tmp_path: Path):
    """Test loading configuration when the file doesn't exist."""
    config_file = tmp_path / "config.json"
    assert not config_file.exists()

    config = load_config(path=config_file)

    assert isinstance(config, OnomlyConfig)
    # Default values from the dataclass definition
    assert config.configured is False
    # we don't want to assert llm_provider and llm_model explicitly as they default to DEFAULT_MODEL's values which might change
    # but we can verify it returns a valid config


def test_load_config_valid_json(tmp_path: Path):
    """Test loading configuration from a valid JSON file."""
    config_file = tmp_path / "config.json"
    config_data = {
        "llm_provider": "anthropic",
        "llm_model": "claude-3-5-sonnet-latest",
        "configured": True
    }
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f)

    config = load_config(path=config_file)

    assert config.llm_provider == "anthropic"
    assert config.llm_model == "claude-3-5-sonnet-latest"
    assert config.configured is True


def test_load_config_invalid_json(tmp_path: Path):
    """Test loading configuration when the JSON is malformed."""
    config_file = tmp_path / "config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        f.write("{ invalid json")

    config = load_config(path=config_file)

    assert isinstance(config, OnomlyConfig)
    assert config.configured is False


def test_load_config_missing_fields(tmp_path: Path):
    """Test loading configuration when some JSON fields are missing."""
    config_file = tmp_path / "config.json"
    config_data = {
        "configured": True
    }
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f)

    config = load_config(path=config_file)

    assert config.configured is True
    # The other fields should fall back to defaults
    assert config.llm_provider
    assert config.llm_model


def test_save_config(tmp_path: Path):
    """Test saving configuration to disk."""
    config_file = tmp_path / "subdir" / "config.json"

    config = OnomlyConfig(
        llm_provider="openai",
        llm_model="gpt-4o",
        configured=True
    )

    saved_path = save_config(config, path=config_file)

    assert saved_path == config_file
    assert config_file.exists()

    with open(config_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["llm_provider"] == "openai"
    assert data["llm_model"] == "gpt-4o"
    assert data["configured"] is True
