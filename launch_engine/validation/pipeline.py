"""Validation pipeline orchestrator."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import List, Optional

from launch_engine.validation.adapters.base import (
    ValidationAdapter,
    ValidationResult,
    ValidationStatus,
)
from launch_engine.validation.rate_limiter import RateLimiter
from launch_engine.cache import SQLiteCache
from launch_engine.modules.naming.brief import NamingBrief
from launch_engine.modules.naming.candidates import NameCandidate
from launch_engine.core.validation import (
    ValidationResult as CoreValidationResult,
    ValidationStatus as CoreValidationStatus,
    ValidationChannel,
    Confidence,
    Evidence,
)
from datetime import datetime


class ValidationPipeline:
    """Orchestrates validation adapters with retry, rate limiting, and caching."""

    def __init__(
        self,
        adapters: List[ValidationAdapter],
        cache: SQLiteCache,
        max_concurrency: int = 10,
        pipeline_timeout: float = 120.0,
    ):
        """Initialize the validation pipeline.

        Args:
            adapters: List of validation adapters to use.
            cache: SQLite cache instance for storing validation results.
            max_concurrency: Maximum number of concurrent validations.
            pipeline_timeout: Timeout in seconds for the entire pipeline.
        """
        self.adapters = adapters
        self.cache = cache
        self.max_concurrency = max_concurrency
        self.pipeline_timeout = pipeline_timeout
        self._semaphore = asyncio.Semaphore(max_concurrency)
        # Create rate limiters for each adapter based on their policies
        self._rate_limiters = {
            adapter: RateLimiter(adapter.policy.rate_limit_per_minute)
            for adapter in adapters
        }

    async def validate_all(
        self, candidates: List[NameCandidate], brief: NamingBrief
    ) -> List[CoreValidationResult]:
        """Validate all candidates using all adapters in parallel.

        Args:
            candidates: List of name candidates to validate.
            brief: Naming brief containing context for validation.

        Returns:
            List of CoreValidationResult objects.
        """
        # Create tasks for all candidate-adapter combinations
        tasks = []
        for candidate in candidates:
            for adapter in self.adapters:
                task = self._validate_candidate_with_adapter(candidate, adapter, brief)
                tasks.append(task)

        # Wait for all tasks to complete with pipeline timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.pipeline_timeout,
            )
        except TimeoutError:
            # If pipeline times out, return unverifiable results for all
            results = []
            for candidate in candidates:
                for adapter in self.adapters:
                    results.append(
                        self._create_unverifiable_result(
                            candidate, adapter, "Pipeline timeout exceeded"
                        )
                    )

        # Process results, handling exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                # This shouldn't happen with return_exceptions=True, but just in case
                processed_results.append(
                    self._create_unverifiable_result(
                        None, None, f"Unexpected error: {str(result)}"
                    )
                )
            else:
                processed_results.append(result)

        return processed_results

    async def _validate_candidate_with_adapter(
        self,
        candidate: NameCandidate,
        adapter: ValidationAdapter,
        brief: NamingBrief,
    ) -> CoreValidationResult:
        """Validate a single candidate with a single adapter.

        Args:
            candidate: The name candidate to validate.
            adapter: The validation adapter to use.
            brief: Naming brief containing context.

        Returns:
            CoreValidationResult object.
        """
        async with self._semaphore:
            # Check cache first
            cache_key = self._build_cache_key(adapter, candidate, brief)
            cached_result = await self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Apply rate limiting
            await self._rate_limiters[adapter].acquire()

            # Validate with retry logic
            result = await self._validate_with_retry(candidate, adapter, brief)

            # Store in cache
            context_hash = self._build_context_hash(brief)
            await self.cache.set(
                cache_key,
                result,
                ttl=adapter.policy.cache_ttl_seconds,
                context_hash=context_hash,
            )

            return result

    async def _validate_with_retry(
        self,
        candidate: NameCandidate,
        adapter: ValidationAdapter,
        brief: NamingBrief,
    ) -> CoreValidationResult:
        """Validate with exponential backoff retry logic.

        Args:
            candidate: The name candidate to validate.
            adapter: The validation adapter to use.
            brief: Naming brief containing context.

        Returns:
            CoreValidationResult object.
        """
        max_attempts = 3
        base_delay = 1.0  # seconds

        for attempt in range(max_attempts):
            try:
                # Apply adapter-specific timeout
                result = await asyncio.wait_for(
                    adapter.validate(candidate.name),
                    timeout=adapter.policy.timeout_seconds,
                )

                # Convert adapter result to core validation result
                core_result = self._convert_to_core_result(
                    candidate, adapter, brief, result
                )

                # Check if we should retry based on error type
                if self._should_retry(core_result, attempt, max_attempts):
                    if attempt < max_attempts - 1:
                        delay = base_delay * (2**attempt)  # Exponential backoff
                        await asyncio.sleep(delay)
                        continue

                return core_result

            except TimeoutError:
                if attempt < max_attempts - 1:
                    delay = base_delay * (2**attempt)
                    await asyncio.sleep(delay)
                    continue
                else:
                    # Final attempt timed out
                    error_msg = f"Validation timeout after {max_attempts} attempts"
                    return self._create_unverifiable_result(
                        candidate, adapter, error_msg
                    )
            except Exception as e:
                # Handle network errors, etc.
                if self._is_transient_error(e) and attempt < max_attempts - 1:
                    delay = base_delay * (2**attempt)
                    await asyncio.sleep(delay)
                    continue
                else:
                    # Non-transient error or max retries exceeded
                    return self._create_unverifiable_result(
                        candidate, adapter, f"Validation failed: {str(e)}"
                    )

        # Should not reach here, but just in case
        return self._create_unverifiable_result(
            candidate, adapter, f"Validation failed after {max_attempts} attempts"
        )

    def _should_retry(
        self, result: CoreValidationResult, attempt: int, max_attempts: int
    ) -> bool:
        """Determine if we should retry based on the result.

        Args:
            result: The validation result.
            attempt: Current attempt number (0-indexed).
            max_attempts: Maximum number of attempts.

        Returns:
            True if we should retry, False otherwise.
        """
        # Retry on UNVERIFIABLE status (indicates transient error)
        if result.status == CoreValidationStatus.UNVERIFIABLE:
            # Check if it's a transient error based on error message
            if result.evidence.raw and isinstance(result.evidence.raw, dict):
                error_msg = str(result.evidence.raw.get("error", "")).lower()
            else:
                error_msg = ""

            transient_indicators = [
                "timeout",
                "network",
                "connection",
                "502",
                "503",
                "504",
                "429",
                "408",
            ]
            return any(indicator in error_msg for indicator in transient_indicators)
        return False

    def _is_transient_error(self, exception: Exception) -> bool:
        """Check if an exception is transient and worth retrying.

        Args:
            exception: The exception to check.

        Returns:
            True if transient, False otherwise.
        """
        error_str = str(exception).lower()
        transient_indicators = [
            "timeout",
            "network",
            "connection",
            "502",
            "503",
            "504",
            "429",
            "408",
        ]
        return any(indicator in error_str for indicator in transient_indicators)

    def _build_cache_key(
        self, adapter: ValidationAdapter, candidate: NameCandidate, brief: NamingBrief
    ) -> str:
        """Build a cache key for the validation.

        Args:
            adapter: The validation adapter.
            candidate: The name candidate.
            brief: Naming brief containing context.

        Returns:
            Cache key string.
        """
        context_hash = self._build_context_hash(brief)
        normalized_target = candidate.name.lower().strip()

        # Convert policy to dict - handle both Pydantic models and dataclasses
        if hasattr(adapter.policy, "model_dump"):
            policy_dict = adapter.policy.model_dump()
        elif hasattr(adapter.policy, "__dict__"):
            policy_dict = adapter.policy.__dict__
        else:
            policy_dict = {}

        key_data = {
            "channel": self._get_channel_from_adapter(adapter).value,
            "adapter_version": adapter.version,
            "policy_version": json.dumps(policy_dict, sort_keys=True),
            "normalized_target": normalized_target,
            "context_hash": context_hash,
        }
        key_json = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_json.encode()).hexdigest()

    def _build_context_hash(self, brief: NamingBrief) -> str:
        """Build a hash of the validation context.

        Args:
            brief: Naming brief containing context.

        Returns:
            Context hash string.
        """
        context_data = {
            "target_markets": sorted(brief.target_markets),
            "industry": brief.industry,
            "validation_policy": getattr(brief, "validation_policy", {}),
        }
        context_json = json.dumps(context_data, sort_keys=True)
        return hashlib.sha256(context_json.encode()).hexdigest()

    def _get_channel_from_adapter(
        self, adapter: ValidationAdapter
    ) -> ValidationChannel:
        """Get the validation channel from an adapter.

        Args:
            adapter: The validation adapter.

        Returns:
            Validation channel.
        """
        # This would typically come from the adapter itself
        # For now, we'll infer from the adapter type or use a default
        adapter_name = adapter.__class__.__name__.lower()
        if "domain" in adapter_name:
            return ValidationChannel.DOMAIN
        elif "trademark" in adapter_name:
            if "global" in adapter_name:
                return ValidationChannel.TRADEMARK_GLOBAL
            else:
                return ValidationChannel.TRADEMARK_TR
        elif "social" in adapter_name:
            if "x" in adapter_name or "twitter" in adapter_name:
                return ValidationChannel.SOCIAL_X
            elif "instagram" in adapter_name:
                return ValidationChannel.SOCIAL_INSTAGRAM
            elif "linkedin" in adapter_name:
                return ValidationChannel.SOCIAL_LINKEDIN
            else:
                return ValidationChannel.SOCIAL_X  # default
        else:
            return ValidationChannel.DOMAIN  # fallback

    def _convert_to_core_result(
        self,
        candidate: NameCandidate,
        adapter: ValidationAdapter,
        brief: NamingBrief,
        adapter_result: ValidationResult,
    ) -> CoreValidationResult:
        """Convert adapter validation result to core validation result.

        Args:
            candidate: The name candidate.
            adapter: The validation adapter.
            brief: Naming brief containing context.
            adapter_result: Result from the adapter validation.

        Returns:
            CoreValidationResult object.
        """
        # Map adapter status to core status
        status_mapping = {
            ValidationStatus.AVAILABLE: CoreValidationStatus.AVAILABLE,
            ValidationStatus.TAKEN: CoreValidationStatus.TAKEN,
            ValidationStatus.UNVERIFIABLE: CoreValidationStatus.UNVERIFIABLE,
        }
        core_status = status_mapping.get(
            adapter_result.status, CoreValidationStatus.UNVERIFIABLE
        )

        # Determine confidence based on status and error presence
        has_error = adapter_result.error is not None and adapter_result.error != ""
        if (
            core_status == CoreValidationStatus.AVAILABLE
            or core_status == CoreValidationStatus.TAKEN
        ):
            confidence_str = "confirmed" if not has_error else "likely"
        else:
            confidence_str = "unknown"

        # Map string confidence to enum
        confidence_mapping = {
            "confirmed": Confidence.CONFIRMED,
            "likely": Confidence.LIKELY,
            "unknown": Confidence.UNKNOWN,
        }
        confidence = confidence_mapping[confidence_str]

        # Create evidence
        evidence = Evidence(
            source=f"{adapter.__class__.__name__} v{adapter.version}",
            url=adapter_result.details.get("url") if adapter_result.details else None,
            checked_at=datetime.now(),
            raw=adapter_result.details if adapter_result.details else None,
        )

        # Determine channel
        channel = self._get_channel_from_adapter(adapter)

        # Get manual review URL if available
        manual_review_url = None
        if adapter_result.details:
            manual_review_url = adapter_result.details.get("manual_review_url")

        return CoreValidationResult(
            target=candidate.name,
            channel=channel,
            status=core_status,
            confidence=confidence,
            evidence=evidence,
            candidate_id=candidate.candidate_id,
            validation_id=f"{candidate.candidate_id}_{channel.value}_{int(time.time())}",
            adapter_version=adapter.version,
            checked_at=datetime.now(),
            manual_review_url=manual_review_url,
        )

    def _create_unverifiable_result(
        self,
        candidate: Optional[NameCandidate],
        adapter: Optional[ValidationAdapter],
        error_message: str,
    ) -> CoreValidationResult:
        """Create an unverifiable validation result.

        Args:
            candidate: The name candidate (if available).
            adapter: The validation adapter (if available).
            error_message: Error message to include.

        Returns:
            CoreValidationResult with UNVERIFIABLE status.
        """
        target = candidate.name if candidate else "unknown"
        candidate_id = candidate.candidate_id if candidate else "unknown"
        adapter_version = adapter.version if adapter else "unknown"
        channel = (
            self._get_channel_from_adapter(adapter)
            if adapter
            else ValidationChannel.DOMAIN
        )

        evidence = Evidence(
            source="ValidationPipeline error",
            url=None,
            checked_at=datetime.now(),
            raw={"error": error_message},
        )

        return CoreValidationResult(
            target=target,
            channel=channel,
            status=CoreValidationStatus.UNVERIFIABLE,
            confidence=Confidence.UNKNOWN,
            evidence=evidence,
            candidate_id=candidate_id,
            validation_id=f"{candidate_id}_{channel.value}_{int(time.time())}_error",
            adapter_version=adapter_version,
            checked_at=datetime.now(),
            manual_review_url=None,
        )
