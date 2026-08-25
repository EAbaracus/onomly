"""Launch Engine MCP server (stdio transport).

Exposes LaunchEngine functionality as MCP tools so Hermes can call
brand-naming and validation pipeline directly. Runs in the same venv
as Hermes (C:/Users/eltun/AppData/Local/hermes/hermes-agent/venv).

Uses the high-level mcp.server.mcpserver.MCPServer API.

Tools:
  - generate_names(brief_json) -> NameCandidateList JSON
  - validate_names(candidates_json, brief_json) -> ValidationResult[] JSON
  - run_full_pipeline(brief_json) -> {candidates, results} JSON
  - list_adapters() -> adapter metadata JSON
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import asdict

# MCP servers communicate over stdio (stdout). LiteLLM and other libs may
# write logs to stdout and corrupt the JSON-RPC stream. Silence them and
# drop any root handlers bound to stdout before importing chatty deps.
for _name in ("litellm", "httpx", "openai", "uvicorn"):
    logging.getLogger(_name).setLevel(logging.CRITICAL + 1)
_root = logging.getLogger()
for _h in list(_root.handlers):
    if getattr(_h, "stream", None) is sys.stdout:
        _root.removeHandler(_h)

from mcp.server.mcpserver import MCPServer
from launch_engine.engine import LaunchEngine
from launch_engine.modules.naming.brief import NamingBrief
from launch_engine.modules.naming.candidates import NameCandidate
from launch_engine.validation.adapters.domain import DomainAdapter
from launch_engine.validation.adapters.trademark import TrademarkAdapter
from launch_engine.validation.adapters.social import SocialMediaAdapter
from launch_engine.runtime_config import ensure_9router_env

ensure_9router_env()

APP = MCPServer("launch-engine")

LLM_PROVIDER = "9router"
LLM_MODEL = "kc/nvidia/nemotron-3-super-120b-a12b:free"
CACHE_DB = "launch_engine_cache.db"


def _make_engine() -> LaunchEngine:
    return LaunchEngine(
        llm_provider=LLM_PROVIDER,
        llm_model=LLM_MODEL,
        cache_db_path=CACHE_DB,
        adapters=[
            DomainAdapter(),
            TrademarkAdapter(),
            SocialMediaAdapter(),
        ],
    )


def _brief_from_json(data: dict) -> NamingBrief:
    return NamingBrief(**data)


def _candidates_from_json(data: list[dict]) -> list[NameCandidate]:
    return [NameCandidate(**c) for c in data]


@APP.tool()
async def generate_names(brief_json: str) -> str:
    """Generate brand name candidates from a naming brief.

    Args:
        brief_json: JSON string of NamingBrief fields (project_codename,
            description, target_markets, industry, optional brand_story_seed,
            preferred_typologies, avoid_terms, phonetic_constraints, name_count).
    """
    brief = _brief_from_json(json.loads(brief_json))
    engine = _make_engine()
    result = await engine.generate_names(brief)
    return result.model_dump_json(indent=2)


@APP.tool()
async def validate_names(candidates_json: str, brief_json: str) -> str:
    """Validate brand name candidates across domain, trademark, social media.

    Args:
        candidates_json: JSON array of NameCandidate dicts (candidate_id, name, typology).
        brief_json: JSON string of NamingBrief fields.
    """
    brief = _brief_from_json(json.loads(brief_json))
    candidates = _candidates_from_json(json.loads(candidates_json))
    engine = _make_engine()
    results = await engine.validate_names(candidates, brief)
    return json.dumps([r.model_dump() for r in results], default=str, indent=2)


@APP.tool()
async def run_full_pipeline(brief_json: str) -> str:
    """Generate then validate names in one call.

    Args:
        brief_json: JSON string of NamingBrief fields.
    """
    brief = _brief_from_json(json.loads(brief_json))
    engine = _make_engine()
    candidates, results = await engine.run_full_pipeline(brief)
    payload = {
        "candidates": json.loads(candidates.model_dump_json()),
        "results": [r.model_dump() for r in results],
    }
    return json.dumps(payload, default=str, indent=2)


@APP.tool()
async def list_adapters() -> str:
    """List available validation adapters and their channels."""
    adapters = [
        {
            "name": "DomainAdapter",
            "channel": "DOMAIN",
            "policy": asdict(DomainAdapter().policy),
        },
        {
            "name": "TrademarkAdapter",
            "channel": "TRADEMARK_TR",
            "policy": asdict(TrademarkAdapter().policy),
        },
        {
            "name": "SocialMediaAdapter",
            "channel": "SOCIAL_X",
            "policy": asdict(SocialMediaAdapter().policy),
        },
    ]
    return json.dumps(adapters, default=str, indent=2)


async def main() -> None:
    await APP.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())
