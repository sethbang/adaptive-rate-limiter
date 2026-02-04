# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
Streaming support for rate limiting.

This module provides components for tracking rate limit reservations during
streaming response consumption. It implements the refund-based accounting
model where capacity is released based on actual token consumption.

Classes:
    StreamingReservationContext: Context object for tracking a streaming
        request's rate limit reservation through its lifecycle.
    RateLimitedAsyncIterator: Async iterator wrapper that adds rate limit
        tracking to streaming responses.
    StreamingInFlightTracker: Manages in-flight streaming request tracking
        and background cleanup of hung streams.
    StreamingInFlightEntry: Dataclass for tracking an in-flight streaming request.
"""

from .context import StreamingReservationContext
from .iterator import RateLimitedAsyncIterator
from .tracker import StreamingInFlightEntry, StreamingInFlightTracker

__all__ = [
    "RateLimitedAsyncIterator",
    "StreamingInFlightEntry",
    "StreamingInFlightTracker",
    "StreamingReservationContext",
]
