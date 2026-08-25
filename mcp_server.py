"""
MCP Server for the Launch Engine Brand Naming Pipeline.

This script runs a Model Context Protocol (MCP) server that exposes the
launch engine's capabilities to tools like Claude.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import asdict

# MCP servers communicate over stdio (stdout). LiteLLM and other libs may
# write logs to stdout and corrupt the JSON-RPC stream. Silence them and
# prevent stdout handlers from propagating.
for _name in ("litellm", "httpx", "openai", "uvicorn"):
    logging.getLogger(_name).setLevel(logging.CRITICAL + 1)
_root = logging.getLogger()
for _h in list(_root.handlers):
    if getattr(_h, "stream", None) is sys.stdout:
        _root.removeHandler(_h)

from mcp.server.mcpserver import MCPServer  # noqa: E402
from launch_engine.engine import LaunchEngine  # noqa: E402
from launch_engine.modules.naming.brief import NamingBrief  # noqa: E402
from launch_engine.modules.naming.candidates import NameCandidate  # noqa: E402
from launch_engine.validation.adapters.domain import DomainAdapter  # noqa: E402
from launch_engine.validation.adapters.trademark import TrademarkAdapter  # noqa: E402
from launch_engine.validation.adapters.social import SocialMediaAdapter  # noqa: E402
from launch_engine.runtime_config import ensure_9router_env  # noqa: E402

ensure_9router_env()

APP = MCPServer("launch-engine")

LLM_PROVIDER = "9router"
LLM_MODEL = "openai/gpt-4o-mini"
CACHE_DB = "onomly_mcp_cache.db"


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
async def generate_brand_names(brief_json: str) -> str:
    """
    Generate brand name candidates based on a naming brief.
    Use this to get initial ideas before validating them.

    Args:
        brief_json: A JSON string containing NamingBrief fields
            (project_codename, description, target_markets, industry,
            preferred_typologies, avoid_terms, phonetic_constraints, name_count).
    """
    brief = _brief_from_json(json.loads(brief_json))
    engine = _make_engine()
    result = await engine.generate_names(brief)
    return result.model_dump_json(indent=2)


@APP.tool()
async def validate_brand_names(candidates_json: str, brief_json: str) -> str:
    """
    Validate a list of brand name candidates across domains, social, and TM.

    Args:
        candidates_json: JSON array of NameCandidate dicts.
        brief_json: JSON string of NamingBrief fields.
    """
    brief = _brief_from_json(json.loads(brief_json))
    candidates = _candidates_from_json(json.loads(candidates_json))
    engine = _make_engine()
    results = await engine.validate_names(candidates, brief)
    return json.dumps([r.model_dump() for r in results], default=str, indent=2)


@APP.tool()
async def run_full_naming_pipeline(brief_json: str) -> str:
    """
    Run the complete pipeline: generation -> validation in one step.

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
async def check_validation_adapter_policies() -> str:
    """
    Check the rate limits and timeout policies configured for validation adapters.
    Use this to understand why validation might be slow.
    """
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


async def main():
    # Import here to avoid polluting global namespace for older mcp versions
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await APP.run(
            read_stream,
            write_stream,
            APP.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
