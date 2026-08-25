"""CLI interface for Launch Engine."""

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from launch_engine.engine import LaunchEngine
from launch_engine.modules.naming.brief import NamingBrief
from launch_engine.modules.naming.candidates import NameCandidateList
from launch_engine.core.validation import ValidationResult
from launch_engine.runtime_config import ensure_9router_env
from launch_engine import models as model_catalog
from launch_engine.config import OnomlyConfig, load_config, save_config
from launch_engine.models import ModelEntry

ensure_9router_env()

app = typer.Typer(help="Onomly - Brand naming and validation CLI")
console = Console()


def _resolve_llm(provider: str | None, model: str | None) -> tuple[str, str]:
    """Resolve provider/model from CLI overrides or saved config.

    Precedence: explicit CLI flag > saved config > catalog default.
    """
    cfg = load_config()
    prov = provider or cfg.llm_provider
    mod = model or cfg.llm_model
    return prov, mod


def _print_first_run_hint(output_format: str) -> None:
    """Warn about the default model on first run, but only for human-readable
    output so it never corrupts json/csv streams."""
    if output_format != "table":
        return
    cfg = load_config()
    if not cfg.configured:
        console.print(
            "[yellow]No model configured yet.[/yellow] "
            "Using default: [cyan]"
            f"{cfg.model_id}"
            "[/cyan]. Run [bold]onomly configure[/bold] to pick a model."
        )


@app.command()
def generate_names(
    project_codename: str = typer.Option(..., help="Project codename"),
    description: str = typer.Option(..., help="Project description"),
    target_markets: str = typer.Option(
        ..., help="Comma-separated target markets (e.g., 'USA,Europe')"
    ),
    industry: str = typer.Option(..., help="Industry (e.g., 'Technology')"),
    brand_personality: Optional[str] = typer.Option(
        None, help="Brand personality traits"
    ),
    avoid_terms: Optional[str] = typer.Option(
        None, help="Comma-separated terms to avoid"
    ),
    candidate_count: int = typer.Option(10, help="Number of candidates to generate"),
    llm_provider: str = typer.Option(
        None,
        help="LLM provider (9router, openai, anthropic, ollama). Default: saved config or 9router.",
    ),
    llm_model: str = typer.Option(
        None, help="LLM model name. Default: saved config or 9router free model."
    ),
    cache_db: str = typer.Option("launch_engine_cache.db", help="Cache database path"),
    output_format: str = typer.Option("table", help="Output format: table, json, csv"),
):
    """Generate brand name candidates based on a naming brief."""
    llm_provider, llm_model = _resolve_llm(llm_provider, llm_model)
    _print_first_run_hint(output_format)
    try:
        # Parse comma-separated values
        markets = [m.strip() for m in target_markets.split(",")]
        avoid = [t.strip() for t in avoid_terms.split(",")] if avoid_terms else []

        # Create naming brief
        brief = NamingBrief(
            project_codename=project_codename,
            description=description,
            target_markets=markets,
            industry=industry,
            brand_personality=brand_personality,
            avoid_terms=avoid,
            candidate_count=candidate_count,
        )

        # Initialize engine
        engine = LaunchEngine(
            llm_provider=llm_provider,
            llm_model=llm_model,
            cache_db_path=cache_db,
        )

        # Run generation
        result = asyncio.run(engine.generate_names(brief))

        # Output results
        if output_format == "json":
            console.print(result.model_dump_json(indent=2))
        elif output_format == "csv":
            _output_csv(result)
        else:  # table
            _output_table(result)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def validate(
    candidates_file: str = typer.Option(..., help="Path to JSON file with candidates"),
    target_markets: str = typer.Option(..., help="Comma-separated target markets"),
    industry: str = typer.Option(..., help="Industry"),
    llm_provider: str = typer.Option(
        None,
        help="LLM provider (9router, openai, anthropic, ollama). Default: saved config or 9router.",
    ),
    llm_model: str = typer.Option(
        None, help="LLM model name. Default: saved config or 9router free model."
    ),
    cache_db: str = typer.Option("launch_engine_cache.db", help="Cache database path"),
    output_format: str = typer.Option("table", help="Output format: table, json, csv"),
):
    """Validate brand name candidates."""
    llm_provider, llm_model = _resolve_llm(llm_provider, llm_model)
    _print_first_run_hint(output_format)
    try:
        # Load candidates from file
        candidates_path = Path(candidates_file)
        if not candidates_path.exists():
            console.print(f"[red]Error:[/red] File not found: {candidates_file}")
            raise typer.Exit(1)

        with open(candidates_path) as f:
            data = json.load(f)

        candidates_list = NameCandidateList.model_validate(data)
        markets = [m.strip() for m in target_markets.split(",")]

        # Create minimal brief for validation
        brief = NamingBrief(
            project_codename=candidates_list.brief_ref,
            description="Validation only",
            target_markets=markets,
            industry=industry,
        )

        # Initialize engine
        engine = LaunchEngine(
            llm_provider=llm_provider,
            llm_model=llm_model,
            cache_db_path=cache_db,
        )

        # Run validation
        results = asyncio.run(engine.validate_names(candidates_list.candidates, brief))

        # Output results
        if output_format == "json":
            output = [r.model_dump() for r in results]
            console.print(json.dumps(output, indent=2, default=str))
        elif output_format == "csv":
            _output_validation_csv(results)
        else:  # table
            _output_validation_table(results)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def cache(
    action: str = typer.Argument(..., help="Action: stats, clear"),
    cache_db: str = typer.Option("launch_engine_cache.db", help="Cache database path"),
):
    """Manage the validation cache."""
    from launch_engine.cache import SQLiteCache

    try:
        cache = SQLiteCache(db_path=cache_db)
        asyncio.run(cache.initialize())

        if action == "stats":
            # Get basic cache stats
            console.print(f"Cache database: {cache_db}")
            console.print("[yellow]Note: Detailed stats not yet implemented[/yellow]")
        elif action == "clear":
            asyncio.run(cache.clear())
            console.print("[green]Cache cleared successfully[/green]")
        else:
            console.print(f"[red]Error:[/red] Unknown action: {action}")
            console.print("Available actions: stats, clear")
            raise typer.Exit(1)

        asyncio.run(cache.close())

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def adapters():
    """List available validation adapters."""
    from launch_engine.validation.adapters.domain import DomainAdapter
    from launch_engine.validation.adapters.trademark import TrademarkAdapter
    from launch_engine.validation.adapters.social import SocialMediaAdapter

    table = Table(title="Available Validation Adapters")
    table.add_column("Adapter", style="cyan")
    table.add_column("Channel", style="green")
    table.add_column("Rate Limit", style="yellow")
    table.add_column("Cache TTL", style="blue")

    domain = DomainAdapter()
    table.add_row(
        "DomainAdapter",
        "DOMAIN",
        f"{domain.policy.rate_limit_per_minute}/min",
        f"{domain.policy.cache_ttl_seconds}s",
    )

    trademark = TrademarkAdapter()
    table.add_row(
        "TrademarkAdapter",
        "TRADEMARK_TR",
        f"{trademark.policy.rate_limit_per_minute}/min",
        f"{trademark.policy.cache_ttl_seconds}s",
    )

    social = SocialMediaAdapter()
    table.add_row(
        "SocialMediaAdapter",
        "SOCIAL_X, SOCIAL_INSTAGRAM, SOCIAL_LINKEDIN",
        f"{social.policy.rate_limit_per_minute}/min",
        f"{social.policy.cache_ttl_seconds}s",
    )

    console.print(table)


@app.command()
def configure():
    """Interactive first-run setup: choose the LLM model from a list."""
    console.print("[bold]Onomly — choose your LLM model[/bold]\n")
    table = Table(title="Available models")
    table.add_column("#", style="dim", width=4)
    table.add_column("Model", style="cyan")
    table.add_column("Provider", style="green")
    table.add_column("Key needed", style="yellow")
    table.add_column("Note", style="white")

    for i, entry in enumerate(model_catalog.MODELS, 1):
        table.add_row(
            str(i),
            entry.label,
            entry.provider,
            "yes" if entry.needs_key else "no",
            entry.note,
        )
    console.print(table)

    choice = typer.prompt(
        "Select a model by number (or type a provider/model id)",
        default="1",
    )
    chosen: ModelEntry | None = None
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(model_catalog.MODELS):
            chosen = model_catalog.MODELS[idx]
    if chosen is None:
        # Treat input as a raw provider/model id (e.g. "ollama/llama3:8b").
        if "/" in choice:
            prov, mod = choice.split("/", 1)
            chosen = ModelEntry(
                provider=prov.strip(),
                model=mod.strip(),
                label=choice,
                needs_key=(prov.strip() not in ("9router", "ollama")),
            )
    if chosen is None:
        console.print(f"[red]Invalid selection: {choice}[/red]")
        raise typer.Exit(1)

    if chosen.needs_key:
        console.print(
            f"[yellow]Note:[/yellow] {chosen.label} requires an API key in the "
            "environment (e.g. OPENAI_API_KEY / ANTHROPIC_API_KEY). Onomly does not "
            "store secrets."
        )

    cfg = OnomlyConfig(
        llm_provider=chosen.provider,
        llm_model=chosen.model,
        configured=True,
    )
    saved = save_config(cfg)
    console.print(
        f"[green]Saved:[/green] {chosen.id}\nConfig written to [dim]{saved}[/dim]"
    )


@app.command()
def models():
    """List the available LLM models (non-interactive)."""
    table = Table(title="Onomly LLM models")
    table.add_column("#", style="dim", width=4)
    table.add_column("Model", style="cyan")
    table.add_column("Provider", style="green")
    table.add_column("Key needed", style="yellow")
    table.add_column("Note", style="white")
    for i, entry in enumerate(model_catalog.MODELS, 1):
        table.add_row(
            str(i),
            entry.label,
            entry.provider,
            "yes" if entry.needs_key else "no",
            entry.note,
        )
    console.print(table)
    console.print(f"\n[dim]Default: {model_catalog.DEFAULT_MODEL.id}[/dim]")


def _output_table(result: NameCandidateList):
    """Output candidates as a rich table."""
    table = Table(title=f"Brand Name Candidates - {result.brief_ref}")
    table.add_column("#", style="dim", width=3)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Typology", style="green")
    table.add_column("Rationale", style="white")
    table.add_column("Score", style="yellow", justify="right")

    for i, candidate in enumerate(result.candidates, 1):
        score = (
            f"{candidate.internal_assessment.score:.2f}"
            if candidate.internal_assessment
            else "N/A"
        )

        table.add_row(
            str(i),
            candidate.name,
            candidate.typology.value,
            (
                candidate.rationale[:50] + "..."
                if len(candidate.rationale) > 50
                else candidate.rationale
            ),
            str(score),
        )

    console.print(table)
    console.print(
        f"\n[dim]Generated {len(result.candidates)} candidates using {result.llm_provider}/{result.llm_model_used}[/dim]"
    )


def _output_csv(result: NameCandidateList):
    """Output candidates as CSV."""
    console.print("name,typology,rationale,score")
    for candidate in result.candidates:
        score = (
            candidate.internal_assessment.score if candidate.internal_assessment else ""
        )
        if isinstance(score, float):
            score = f"{score:.2f}"

        # Escape CSV fields
        rationale = candidate.rationale.replace('"', '""')
        console.print(
            f'"{candidate.name}","{candidate.typology.value}","{rationale}","{score}"'
        )


def _output_validation_table(results: list[ValidationResult]):
    """Output validation results as a rich table."""
    table = Table(title="Validation Results")
    table.add_column("Target", style="cyan")
    table.add_column("Channel", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Confidence", style="blue")
    table.add_column("Manual Review", style="magenta")

    for result in results:
        status_style = {
            "available": "[green]AVAILABLE[/green]",
            "taken": "[red]TAKEN[/red]",
            "unverifiable": "[yellow]UNVERIFIABLE[/yellow]",
        }.get(result.status.value, result.status.value)

        manual_review = "Yes" if result.manual_review_url else "No"

        table.add_row(
            result.target,
            result.channel.value,
            status_style,
            result.confidence.value,
            manual_review,
        )

    console.print(table)


def _output_validation_csv(results: list[ValidationResult]):
    """Output validation results as CSV."""
    console.print("target,channel,status,confidence,manual_review_url")
    for result in results:
        manual_url = result.manual_review_url or ""
        console.print(
            f'"{result.target}","{result.channel.value}","{result.status.value}","{result.confidence.value}","{manual_url}"'
        )


if __name__ == "__main__":
    app()
