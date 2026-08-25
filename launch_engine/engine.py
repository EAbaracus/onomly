"""Onomly - Application layer orchestrating the brand naming pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Tuple

from launch_engine.llm import LLMAdapter
from launch_engine.cache import SQLiteCache
from launch_engine.modules.naming.module import BrandNamingModule
from launch_engine.validation.pipeline import ValidationPipeline
from launch_engine.validation.adapters.base import ValidationAdapter
from launch_engine.modules.naming.brief import NamingBrief
from launch_engine.modules.naming.candidates import NameCandidate, NameCandidateList
from launch_engine.core.validation import (
    ValidationResult,
    ValidationStatus,
    ValidationChannel,
    Confidence,
    Evidence,
)


class LaunchEngine:
    """Main application layer that orchestrates all components of the brand naming pipeline."""

    def __init__(
        self,
        llm_provider: str,
        llm_model: str,
        cache_db_path: str,
        adapters: Optional[List[ValidationAdapter]] = None,
    ) -> None:
        """Initialize the LaunchEngine with all required components.

        Args:
            llm_provider: The LLM provider (ollama, openai, anthropic, 9router).
            llm_model: The model name to use.
            cache_db_path: Path to the SQLite database file for caching.
            adapters: List of validation adapters to use. If None, defaults to
                      domain, trademark, and social adapters.
        """
        # Initialize LLM adapter
        self.llm_adapter = LLMAdapter(
            provider=llm_provider,
            model=llm_model,
        )

        # Initialize cache
        self.cache = SQLiteCache(db_path=cache_db_path)

        # Initialize validation adapters (default to domain, trademark, social)
        if adapters is None:
            from launch_engine.validation.adapters.domain import DomainAdapter
            from launch_engine.validation.adapters.trademark import TrademarkAdapter
            from launch_engine.validation.adapters.social import SocialMediaAdapter

            self.adapters = [
                DomainAdapter(),
                TrademarkAdapter(),
                SocialMediaAdapter(),
            ]
        else:
            self.adapters = adapters

        # Initialize brand naming module
        self.brand_naming_module = BrandNamingModule(llm_adapter=self.llm_adapter)

        # Initialize validation pipeline
        self.validation_pipeline = ValidationPipeline(
            adapters=self.adapters,
            cache=self.cache,
        )

    async def generate_names(self, brief: NamingBrief) -> NameCandidateList:
        """Generate name candidates based on the naming brief.

        Args:
            brief: Naming brief containing project details and constraints.

        Returns:
            NameCandidateList containing generated and scored candidates.
        """
        try:
            return await self.brand_naming_module.run(brief)
        except Exception as e:
            # Handle errors gracefully - return empty candidate list
            print(f"Error generating names: {e}")
            return NameCandidateList(
                brief_ref=brief.project_codename,
                candidates=[],
                llm_model_used=self.llm_adapter.model_id,
                llm_provider=self.llm_adapter.provider,
                generated_at=datetime.now(),
            )

    async def validate_names(
        self, candidates: List[NameCandidate], brief: NamingBrief
    ) -> List[ValidationResult]:
        """Validate name candidates using the validation pipeline.

        Args:
            candidates: List of name candidates to validate.
            brief: Naming brief containing context for validation.

        Returns:
            List of ValidationResult objects.
        """
        try:
            return await self.validation_pipeline.validate_all(candidates, brief)
        except Exception as e:
            # Handle errors gracefully - return unverifiable results for all candidates
            print(f"Error validating names: {e}")
            unverifiable_results = []
            for candidate in candidates:
                evidence = Evidence(
                    source="ValidationPipeline error",
                    url=None,
                    checked_at=datetime.now(),
                    raw={"error": str(e)},
                )
                unverifiable_results.append(
                    ValidationResult(
                        target=candidate.name,
                        channel=ValidationChannel.DOMAIN,  # Default channel
                        status=ValidationStatus.UNVERIFIABLE,
                        confidence=Confidence.UNKNOWN,
                        evidence=evidence,
                        candidate_id=candidate.candidate_id,
                        validation_id=f"error_{candidate.candidate_id}_{int(datetime.now().timestamp())}",
                        adapter_version="unknown",
                        checked_at=datetime.now(),
                        manual_review_url=None,
                    )
                )
            return unverifiable_results

    async def run_full_pipeline(
        self, brief: NamingBrief
    ) -> Tuple[NameCandidateList, List[ValidationResult]]:
        """Run the full brand naming pipeline: generate then validate names.

        Args:
            brief: Naming brief containing project details and constraints.

        Returns:
            Tuple of (NameCandidateList, List[ValidationResult]).
        """
        # Generate names
        candidates = await self.generate_names(brief)
        # Validate names
        validation_results = await self.validate_names(candidates.candidates, brief)
        return candidates, validation_results
