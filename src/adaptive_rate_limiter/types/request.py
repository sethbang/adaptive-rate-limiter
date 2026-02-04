# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
Request metadata types for rate limiting.

This module defines request metadata used for classification, routing, and queue
management in the adaptive rate limiting system.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class RequestMetadata:
    """
    Comprehensive metadata for request classification, routing, and queue management.

    This dataclass contains all the information needed to classify, prioritize, route,
    and track requests through the intelligent queue system. It serves as the primary
    data structure for request lifecycle management.

    The metadata enables sophisticated request handling including:
    * Resource type classification for appropriate queue selection
    * Token estimation for rate limit calculations
    * Priority-based scheduling and queue ordering
    * Timeout handling and request lifecycle management
    * Client-specific tracking and debugging

    Attributes:
        request_id: Unique identifier for this request instance
        model_id: Model identifier for the target model
        resource_type: Classification of the resource being accessed (string)
        estimated_tokens: Estimated token consumption for text requests (None for non-text)
        priority: Request priority level (higher numbers = higher priority)
        submitted_at: UTC timestamp when the request was submitted to the queue
        timeout: Maximum time to wait for request completion (seconds)
        client_id: Optional client identifier for multi-tenant scenarios
        endpoint: API endpoint path for debugging and metrics
        requires_model: Whether this request requires a specific model (vs. generic endpoint)

    Priority Levels:
        * 0: Normal priority (default)
        * 1-9: Higher priority (for important operations)
        * Negative: Lower priority (for background operations)
    """

    request_id: str
    model_id: str
    resource_type: str  # Uses string instead of Enum for flexibility
    estimated_tokens: int | None = None  # For text requests
    priority: int = 0  # Higher = more important
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    timeout: float | None = 60.0
    client_id: str | None = None
    endpoint: str | None = None
    requires_model: bool = True


__all__ = ["RequestMetadata"]
