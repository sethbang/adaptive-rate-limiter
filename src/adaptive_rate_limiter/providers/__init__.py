# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""Provider abstractions for API-specific rate limit handling.

This subpackage defines the interface for integrating different API providers
with the adaptive rate limiter. Providers are responsible for:

- Discovering rate limits from provider APIs (when supported)
- Parsing rate limit information from HTTP response headers/body
- Mapping model identifiers to rate limit buckets

Exported classes:
    ProviderInterface: Abstract base class for provider implementations.
    DiscoveredBucket: Rate limit bucket discovered from a provider.
    RateLimitInfo: Parsed rate limit data from API responses.
"""

from .base import DiscoveredBucket, ProviderInterface, RateLimitInfo

__all__ = [
    "DiscoveredBucket",
    "ProviderInterface",
    "RateLimitInfo",
]
