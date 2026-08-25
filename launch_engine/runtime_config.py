"""Runtime config loader for Launch Engine MCP server.

Reads 9router credentials from Hermes config without hardcoding secrets.
Falls back to environment variables if already set.
"""

from __future__ import annotations

import os
from pathlib import Path

_HERMES_CONFIG = Path.home() / "AppData" / "Local" / "hermes" / "config.yaml"


def _load_9router_from_hermes_config() -> tuple[str | None, str | None]:
    """Extract NINEROUTER_URL and NINEROUTER_KEY from Hermes config.yaml.

    Returns (None, None) if config is missing or keys absent.
    """
    if not _HERMES_CONFIG.exists():
        return None, None
    try:
        import yaml  # lightweight, already a dep of hermes
    except ImportError:
        return None, None
    try:
        with open(_HERMES_CONFIG, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        providers = (cfg or {}).get("providers", {})
        nr = providers.get("9router", {})
        return nr.get("base_url"), nr.get("api_key")
    except Exception:
        return None, None


def ensure_9router_env() -> None:
    """Ensure 9router credentials are mapped into the env LiteLLM expects.

    LiteLLM's openai-compatible path reads OPENAI_API_KEY / OPENAI_API_BASE.
    9router exposes the same surface, so we mirror NINEROUTER_* → OPENAI_*.
    Priority: existing OPENAI_* env > Hermes config.yaml > localhost default.
    Secrets are never hardcoded here.
    """
    url, key = _load_9router_from_hermes_config()

    # Base URL
    if url and "OPENAI_API_BASE" not in os.environ:
        os.environ["OPENAI_API_BASE"] = url
    elif "OPENAI_API_BASE" not in os.environ:
        os.environ["OPENAI_API_BASE"] = "http://localhost:20128/v1"

    # API key
    if key and "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = key
    # If no key from config, leave OPENAI_API_KEY unset only if already in env.
    # 9router may run with auth disabled; LiteLLM still needs a non-empty value.
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ.get("NINEROUTER_KEY", "sk-noauth")
