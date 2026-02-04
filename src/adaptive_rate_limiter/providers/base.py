# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""Provider interface for API-specific rate limit handling."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DiscoveredBucket:
    """Rate limit bucket discovered from a provider.

    This dataclass represents buckets discovered via provider APIs (e.g.,
    from discover_limits()). It differs from types.rate_limit.RateLimitBucket
    which is used for internal queue management.

    Attributes:
        bucket_id: Unique identifier for this bucket (e.g., 'gpt-5', 'claude-haiku-4.5, 'venice-uncensored')
        rpm_limit: Requests per minute limit (None if unknown)
        tpm_limit: Tokens per minute limit (None if unknown)
        rpm_remaining: Remaining requests in current window (None if unknown)
        tpm_remaining: Remaining tokens in current window (None if unknown)
    """

    bucket_id: str
    rpm_limit: int | None = None
    tpm_limit: int | None = None
    rpm_remaining: int | None = None
    tpm_remaining: int | None = None


@dataclass
class RateLimitInfo:
    """Parsed rate limit information from API response."""

    rpm_remaining: int | None = None
    rpm_limit: int | None = None
    rpm_reset: float | None = None  # Unix timestamp
    tpm_remaining: int | None = None
    tpm_limit: int | None = None
    tpm_reset: float | None = None  # Unix timestamp
    retry_after: int | None = None  # Seconds
    is_rate_limited: bool = False  # True if 429 response
    timestamp: float = field(default_factory=time.time)


class ProviderInterface(ABC):
    """
    Abstract interface for API provider integration.

    Providers are responsible for:
    1. Discovering rate limits (if supported by the API)
    2. Parsing rate limit information from responses (headers or body)
    3. Mapping model IDs to rate limit buckets
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique provider name (e.g., 'venice', 'openai', 'anthropic')."""
        pass

    @abstractmethod
    async def discover_limits(
        self,
        force_refresh: bool = False,
        timeout: float = 30.0,
    ) -> dict[str, DiscoveredBucket]:
        """Discover rate limits for all buckets from the provider.

        This method should query the provider's API to fetch current rate
        limit information for all available rate limit buckets. Providers
        that don't support limit discovery should return an empty dictionary.

        Args:
            force_refresh: If True, bypass any cached limits and fetch fresh
                data from the provider. Default is False (use cache if available).
            timeout: Maximum time in seconds to wait for the provider response.
                Default is 30.0 seconds.

        Returns:
            A dictionary mapping bucket IDs (str) to their DiscoveredBucket
            objects. Returns an empty dict if the provider doesn't support
            discovery or if no buckets are available.

        Raises:
            BackendConnectionError: If the provider cannot be reached or
                the connection times out.
            ProviderError: If the provider returns an error response.

        Note:
            Implementations should cache results when force_refresh is False
            to avoid excessive API calls. The cache should be invalidated
            after a reasonable TTL (e.g., 5-15 minutes).
        """
        pass

    @abstractmethod
    def parse_rate_limit_response(
        self,
        headers: dict[str, str],
        body: dict[str, Any] | None = None,
        status_code: int | None = None,
    ) -> RateLimitInfo:
        """Parse rate limit information from an HTTP response.

        This is the ONLY place where HTTP headers are parsed for rate limit
        information. The resulting RateLimitInfo is passed to
        Backend.update_state() to update the rate limiter's internal state.

        Args:
            headers: HTTP response headers as a dictionary. Keys are header
                names (case-insensitive matching recommended). Common headers
                include 'x-ratelimit-remaining-requests', 'x-ratelimit-limit-tokens',
                'retry-after', etc.
            body: Optional parsed JSON response body. Some providers include
                rate limit info in the response body rather than headers.
                Default is None.
            status_code: Optional HTTP status code. Used to detect 429 (rate
                limited) responses and set is_rate_limited=True. Default is None.

        Returns:
            A RateLimitInfo object containing the parsed rate limit data.
            Fields that cannot be determined from the response should be
            left as None. The is_rate_limited field should be True if
            status_code is 429.

        Raises:
            ValueError: If headers contain malformed rate limit values that
                cannot be parsed (e.g., non-numeric strings where integers
                are expected).

        Note:
            Implementations should handle missing headers gracefully by
            returning RateLimitInfo with None values for unknown fields.
            Header name matching should be case-insensitive for robustness.
        """
        pass

    @abstractmethod
    async def get_bucket_for_model(
        self, model_id: str, resource_type: str | None = None
    ) -> str:
        """Get the rate limit bucket ID for a specific model.

        Maps a model identifier to its corresponding rate limit bucket.
        Different models may share the same bucket (e.g., all gpt-5 variants
        might share a 'gpt-5' bucket) or have individual buckets.

        Args:
            model_id: The model identifier as used in API requests (e.g.,
                'gpt-5', 'claude-haiku-4.5', 'venice-uncensored'').
            resource_type: Optional resource type to consider when determining
                the bucket (e.g., 'requests', 'tokens', 'images'). Some providers
                have different rate limits for different resource types.
                Default is None (use the default resource type).

        Returns:
            The bucket ID string that should be used for rate limiting
            requests to this model. If no specific bucket exists, return
            a default bucket ID (e.g., 'default' or the model_id itself).

        Raises:
            ValueError: If model_id is empty or invalid.
            ProviderError: If bucket lookup requires an API call that fails.

        Note:
            Implementations may cache model-to-bucket mappings for performance.
            If a model is unknown, returning the model_id as the bucket_id
            is a reasonable fallback that ensures isolation.
        """
        pass
