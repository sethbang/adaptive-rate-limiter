# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
Streaming in-flight entry for the Adaptive Rate Limiter.

This module provides the dataclass used to track in-flight streaming requests
for background cleanup (hung-stream detection).

Classes:
    StreamingInFlightEntry: Dataclass tracking an in-flight streaming request.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Any


@dataclass
class StreamingInFlightEntry:
    """
    Tracks an in-flight streaming request for background cleanup.

    This dataclass is used by the background cleanup task to identify
    stale streaming entries that haven't had activity for > 5 minutes.

    Attributes:
        reservation_id: Unique identifier for this reservation
        bucket_id: The rate limit bucket this reservation belongs to
        reserved_tokens: Number of tokens reserved at request start
        started_at: Monotonic timestamp when streaming started
        last_activity_at: Monotonic timestamp of last chunk received
            (updated on each chunk)
        wrapper_ref: Weak reference to the iterator wrapper for cleanup detection

    All timestamps use ``time.monotonic()`` so staleness/abandonment
    detection is immune to wall-clock (NTP) adjustments.
    """

    reservation_id: str
    bucket_id: str
    reserved_tokens: int
    started_at: float
    last_activity_at: float  # Updated on each chunk
    wrapper_ref: weakref.ref[Any]  # Weak reference to wrapper for cleanup


__all__ = [
    "StreamingInFlightEntry",
]
