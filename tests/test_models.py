"""Tests for the models module."""

from launch_engine.models import find_by_id, DEFAULT_MODEL


def test_find_by_id_success():
    """Test finding an existing model by ID."""
    model_id = "9router/google/gemini-2.5-flash:free"
    result = find_by_id(model_id)

    assert result is not None
    assert result.id == model_id
    assert result.provider == "9router"
    assert result.model == "google/gemini-2.5-flash:free"


def test_find_by_id_default_model():
    """Test finding the default model by ID."""
    result = find_by_id(DEFAULT_MODEL.id)

    assert result is not None
    assert result == DEFAULT_MODEL


def test_find_by_id_not_found():
    """Test finding a non-existent model by ID returns None."""
    result = find_by_id("nonexistent/model")

    assert result is None
