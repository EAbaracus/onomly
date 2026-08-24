"""Domain availability checker using RDAP."""

from __future__ import annotations

import logging
import urllib.parse

import httpx

from .base import AdapterPolicy, ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)


class DomainAdapter:
    """Check domain availability via RDAP."""

    version = "1.0.0"
    policy = AdapterPolicy(
        rate_limit_per_minute=60,
        cache_ttl_seconds=86400,  # 24 hours
        timeout_seconds=30.0,
    )

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=self.policy.timeout_seconds)

    async def validate(self, domain: str) -> ValidationResult:
        """Check if a domain is available via RDAP.

        Args:
            domain: The domain name to check (e.g., "example.com").

        Returns:
            ValidationResult with status AVAILABLE, TAKEN, or UNVERIFIABLE.
        """
        # Normalize domain: lower case and strip whitespace
        domain = domain.strip().lower()
        if not domain:
            return ValidationResult(
                status=ValidationStatus.UNVERIFIABLE,
                error="Empty domain",
            )

        # Default result (unverifiable)
        result = ValidationResult(
            status=ValidationStatus.UNVERIFIABLE,
            domain=domain,
        )

        # Try to use rdap.org as a proxy RDAP service
        encoded_domain = urllib.parse.quote(domain, safe="")
        url = f"https://rdap.org/domain/{encoded_domain}"
        try:
            response = await self._client.get(url)
            if response.status_code == 200:
                data = response.json()
                notices = data.get("notice", [])
                # Check if any notice indicates the object was not found
                if any(notice.get("title") == "Object not found" for notice in notices):
                    result = ValidationResult(
                        status=ValidationStatus.AVAILABLE,
                        domain=domain,
                        details={"rdap_data": data},
                    )
                else:
                    result = ValidationResult(
                        status=ValidationStatus.TAKEN,
                        domain=domain,
                        details={"rdap_data": data},
                    )
            elif response.status_code == 404:
                # Some RDAP servers return 404 for unavailable domains.
                result = ValidationResult(
                    status=ValidationStatus.AVAILABLE,
                    domain=domain,
                )
            else:
                # Other status codes (e.g., 429, 500) we treat as unverifiable.
                result = ValidationResult(
                    status=ValidationStatus.UNVERIFIABLE,
                    domain=domain,
                    error=f"Unexpected HTTP status {response.status_code}",
                )
                logger.warning(
                    "Unexpected RDAP status %s for domain %s: %s",
                    response.status_code,
                    domain,
                    response.text,
                )
        except httpx.TimeoutException:
            result = ValidationResult(
                status=ValidationStatus.UNVERIFIABLE,
                domain=domain,
                error="Timeout",
            )
            logger.warning("Timeout checking domain %s", domain)
        except httpx.NetworkError as e:
            result = ValidationResult(
                status=ValidationStatus.UNVERIFIABLE,
                domain=domain,
                error=f"Network error: {str(e)}",
            )
            logger.warning("Network error checking domain %s: %s", domain, str(e))
        except Exception as e:  # pylint: disable=broad-except
            result = ValidationResult(
                status=ValidationStatus.UNVERIFIABLE,
                domain=domain,
                error=f"Unexpected error: {str(e)}",
            )
            logger.exception("Unexpected error checking domain %s", domain)

        return result

    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
