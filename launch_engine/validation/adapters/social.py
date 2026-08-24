"""Social media username checker."""

from __future__ import annotations

import logging
import urllib.parse

import httpx

from .base import AdapterPolicy, ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)


class SocialMediaAdapter:
    """Check username availability on social media platforms."""

    version = "1.0.0"
    policy = AdapterPolicy(
        rate_limit_per_minute=30,
        cache_ttl_seconds=43200,  # 12 hours
        timeout_seconds=30.0,
    )

    # Platforms to check: (name, url_format)
    PLATFORMS = [
        ("twitter", "https://twitter.com/{username}"),
        ("instagram", "https://instagram.com/{username}"),
        ("linkedin", "https://linkedin.com/in/{username}"),
    ]

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=self.policy.timeout_seconds,
            follow_redirects=True,
        )

    async def validate(self, username: str) -> ValidationResult:
        """Check if a username is available on social media platforms.

        Args:
            username: The username to check.

        Returns:
            ValidationResult with status AVAILABLE, TAKEN, or UNVERIFIABLE.
        """
        # Normalize username: lower case and strip whitespace
        username = username.strip().lower()
        if not username:
            return ValidationResult(
                status=ValidationStatus.UNVERIFIABLE,
                error="Empty username",
            )

        taken_platforms = []
        available_platforms = []
        errors = []

        for platform_name, url_format in self.PLATFORMS:
            encoded_username = urllib.parse.quote(username)
            url = url_format.format(username=encoded_username)
            try:
                response = await self._client.head(url)
                if 200 <= response.status_code < 300:  # 2xx
                    taken_platforms.append(platform_name)
                elif response.status_code == 404:
                    available_platforms.append(platform_name)
                else:
                    # Other status codes (e.g., 429, 500) we treat as an error
                    # for this platform.
                    errors.append(
                        f"{platform_name}: unexpected HTTP {response.status_code}"
                    )
                    logger.warning(
                        "Unexpected status %s for %s (%s)",
                        response.status_code,
                        url,
                        platform_name,
                    )
            except httpx.TimeoutException:
                errors.append(f"{platform_name}: timeout")
                logger.warning("Timeout checking %s (%s)", url, platform_name)
            except httpx.NetworkError as e:
                errors.append(f"{platform_name}: network error")
                logger.warning(
                    "Network error checking %s (%s): %s",
                    url,
                    platform_name,
                    str(e),
                )
            except Exception:  # pylint: disable=broad-except
                errors.append(f"{platform_name}: unexpected error")
                logger.exception(
                    "Unexpected error checking %s (%s)",
                    url,
                    platform_name,
                )

        # Determine overall result
        if taken_platforms:
            # Username is taken on at least one platform
            return ValidationResult(
                status=ValidationStatus.TAKEN,
                social_media=username,
                details={
                    "taken_platforms": taken_platforms,
                    "available_platforms": available_platforms,
                    "errors": errors,
                },
            )
        if not errors and len(available_platforms) == len(self.PLATFORMS):
            # All platforms available and no errors
            return ValidationResult(
                status=ValidationStatus.AVAILABLE,
                social_media=username,
                details={
                    "taken_platforms": taken_platforms,
                    "available_platforms": available_platforms,
                    "errors": errors,
                },
            )
        # Otherwise, we have errors and not all platforms are available
        # (or we have mixed results)
        return ValidationResult(
            status=ValidationStatus.UNVERIFIABLE,
            social_media=username,
            details={
                "taken_platforms": taken_platforms,
                "available_platforms": available_platforms,
                "errors": errors,
            },
            error="Unable to determine availability due to errors",
        )

    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
