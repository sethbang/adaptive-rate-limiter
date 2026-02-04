# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
Backend implementations for rate limiting storage.

This module provides the abstract base class and concrete implementations
for rate limiting state storage backends.

Available backends:
- BaseBackend: Abstract base class defining the backend interface
- MemoryBackend: In-memory backend for single-instance deployments
- RedisBackend: Redis-based backend for distributed deployments (requires redis extra)

Supporting types:
- HealthCheckResult: Structured result from backend health checks
- validate_safety_margin: Utility function for validating safety margin values
- FallbackRateLimiter: Per-request rate limiter for Redis fallback mode
- InFlightRequest: Tracks in-flight requests for orphan recovery
- ModelLimits: Per-model rate limits from API

Note: RedisBackend and related types are lazily imported to avoid requiring
the redis package when only using MemoryBackend.
"""

from typing import TYPE_CHECKING, cast

from adaptive_rate_limiter.backends.base import (
    BaseBackend,
    HealthCheckResult,
    validate_safety_margin,
)
from adaptive_rate_limiter.backends.memory import MemoryBackend

# Lazy imports for optional redis backend
if TYPE_CHECKING:
    from adaptive_rate_limiter.backends.redis import (
        FallbackRateLimiter,
        InFlightRequest,
        ModelLimits,
        RedisBackend,
    )

__all__ = [
    # Base classes
    "BaseBackend",
    "FallbackRateLimiter",
    "HealthCheckResult",
    "InFlightRequest",
    # Memory backend
    "MemoryBackend",
    "ModelLimits",
    # Redis backend (lazy loaded)
    "RedisBackend",
    "validate_safety_margin",
]


def __getattr__(name: str) -> type:
    """Lazy import for optional redis backend components."""
    if name in (
        "RedisBackend",
        "FallbackRateLimiter",
        "InFlightRequest",
        "ModelLimits",
    ):
        try:
            from adaptive_rate_limiter.backends import redis as redis_module

            return cast(type, getattr(redis_module, name))
        except ImportError as e:  # pragma: no cover
            raise ImportError(  # pragma: no cover
                f"'{name}' requires the 'redis' extra. "
                "Install with: pip install adaptive-rate-limiter[redis]"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
