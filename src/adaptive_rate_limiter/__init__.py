# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""Adaptive Rate Limiter - Intelligent rate limiting for API clients.

This library provides production-ready rate limiting with adaptive capacity
management, multi-provider support, and distributed backends.

Key Features:
    - Automatic rate limit discovery and tracking
    - Intelligent request scheduling with queue management
    - Support for streaming responses with token accounting
    - Multiple backend options (memory, Redis)
    - Provider-agnostic design with protocol-based interfaces
    - Reservation tracking for capacity management

Quick Start:
    >>> from adaptive_rate_limiter import ClientProtocol, create_scheduler
    >>> from adaptive_rate_limiter.scheduler import RateLimiterConfig
    >>>
    >>> class MyClient(ClientProtocol):
    ...     @property
    ...     def base_url(self) -> str:
    ...         return "https://api.example.com"
    ...     @property
    ...     def timeout(self) -> float:
    ...         return 30.0
    ...     def get_headers(self) -> dict:
    ...         return {"Authorization": "Bearer ..."}
    >>>
    >>> scheduler = create_scheduler(client=MyClient(), mode="intelligent")
    >>> async with scheduler:
    ...     result = await scheduler.submit_request(metadata, request_func)

Main Exports:
    - Scheduler, create_scheduler: Core scheduling components
    - MemoryBackend, RedisBackend: Storage backends
    - RateLimiterConfig: Configuration options
    - ClientProtocol: Protocol for API clients
    - StreamingReservationContext: Streaming response handling
    - ReservationTracker: Capacity reservation management

Note: RedisBackend requires the 'redis' extra. Install with:
    pip install adaptive-rate-limiter[redis]

Version: 1.0.0
"""

__version__ = "1.0.0"

from typing import TYPE_CHECKING

from .backends import (
    BaseBackend,
    MemoryBackend,
)
from .exceptions import (
    BackendConnectionError,
    BackendOperationError,
    BucketNotFoundError,
    CapacityExceededError,
    ConfigurationError,
    QueueOverflowError,
    RateLimiterError,
    ReservationCapacityError,
    TooManyFailedRequestsError,
)
from .protocols import (
    ClassifierProtocol,
    ClientProtocol,
    RequestMetadata,
    StreamingResponseProtocol,
)
from .providers import (
    DiscoveredBucket,
    ProviderInterface,
    RateLimitInfo,
)
from .reservation import (
    ReservationContext,
    ReservationTracker,
)
from .scheduler import (
    RateLimiterConfig,
    Scheduler,
    create_scheduler,
)
from .streaming import (
    RateLimitedAsyncIterator,
    StreamingReservationContext,
)
from .types.resource import (
    AUDIO,
    EMBEDDING,
    GENERIC,
    IMAGE,
    RESOURCE_TYPES,
    TEXT,
    ResourceType,
)

# Lazy import for optional redis backend
if TYPE_CHECKING:
    from .backends import RedisBackend

__all__ = [
    "AUDIO",
    "EMBEDDING",
    "GENERIC",
    "IMAGE",
    "RESOURCE_TYPES",
    "TEXT",
    "BackendConnectionError",
    "BackendOperationError",
    # Backends
    "BaseBackend",
    "BucketNotFoundError",
    "CapacityExceededError",
    "ClassifierProtocol",
    # Protocols
    "ClientProtocol",
    "ConfigurationError",
    "DiscoveredBucket",
    "MemoryBackend",
    # Providers
    "ProviderInterface",
    "QueueOverflowError",
    "RateLimitInfo",
    "RateLimitedAsyncIterator",
    "RateLimiterConfig",
    # Exceptions
    "RateLimiterError",
    "RedisBackend",  # Lazy loaded - requires redis extra
    "RequestMetadata",
    "ReservationCapacityError",
    # Reservation
    "ReservationContext",
    "ReservationTracker",
    # Resource types
    "ResourceType",
    # Scheduler
    "Scheduler",
    # Streaming
    "StreamingReservationContext",
    "StreamingResponseProtocol",
    "TooManyFailedRequestsError",
    "create_scheduler",
]


def __getattr__(name: str) -> type:
    """Lazy import for optional redis backend."""
    if name == "RedisBackend":
        from .backends import RedisBackend

        return RedisBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
