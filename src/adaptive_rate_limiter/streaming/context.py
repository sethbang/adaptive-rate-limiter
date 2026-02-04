# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
Streaming reservation context for tracking rate limit reservations during streaming.

This module provides the StreamingReservationContext dataclass that tracks
a streaming request's rate limit reservation through its lifecycle.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..backends.base import BaseBackend

    # Type for metrics callback: (reserved, actual, extraction_succeeded, bucket_id, duration)
    MetricsCallback = Callable[[int, int, bool, str | None, float | None], None]
    ErrorMetricsCallback = Callable[[str | None], None]


@dataclass
class StreamingReservationContext:
    """
    Context for tracking a streaming request's rate limit reservation.

    Created by the Scheduler when detecting a streaming response,
    passed to RateLimitedAsyncIterator for release on completion.

    Attributes:
        reservation_id: Unique identifier for this reservation
        bucket_id: The rate limit bucket this reservation belongs to
        request_id: The original request ID
        reserved_tokens: Number of tokens reserved at request start
        backend: Reference to the rate limit backend for release
        created_at: Timestamp when the reservation was created
        metrics_callback: Optional callback to record completion metrics
        error_metrics_callback: Optional callback to record error metrics

    Runtime tracking attributes:
        final_tokens: Actual token count extracted from stream completion
        chunk_count: Number of chunks received
        last_chunk_at: Timestamp of the last chunk received
    """

    reservation_id: str
    bucket_id: str
    request_id: str
    reserved_tokens: int
    backend: BaseBackend
    created_at: float = field(default_factory=time.time)

    # Runtime tracking - these are updated as the stream progresses
    final_tokens: int | None = field(default=None, repr=False)
    chunk_count: int = field(default=0, repr=False)
    last_chunk_at: float | None = field(default=None, repr=False)

    # Metrics callbacks - optional for backward compatibility
    # Signature: (reserved: int, actual: int, extraction_succeeded: bool, bucket_id: Optional[str], duration: Optional[float]) -> None
    metrics_callback: MetricsCallback | None = field(default=None, repr=False)
    # Signature: (bucket_id: Optional[str]) -> None
    error_metrics_callback: ErrorMetricsCallback | None = field(
        default=None, repr=False
    )

    def record_chunk(self) -> None:
        """
        Record chunk receipt for activity tracking.

        This method should be called for each chunk received to update
        the activity timestamp. The background cleanup task uses this
        to detect hung streams.
        """
        self.last_chunk_at = time.time()
        self.chunk_count += 1

    def set_final_tokens(self, tokens: int) -> None:
        """
        Set final token count from usage field.

        Args:
            tokens: The total token count from the stream's usage field.
                   Only non-negative values are accepted.
        """
        if tokens >= 0:
            self.final_tokens = tokens

    @property
    def actual_tokens_for_release(self) -> int:
        """
        Tokens to use for release: actual if known, else reserved (fallback).

        When final_tokens is known (extracted from stream completion),
        use that value. Otherwise, fall back to reserved_tokens which
        results in a zero refund (conservative approach).

        Returns:
            The token count to use for capacity release calculation.
        """
        return (
            self.final_tokens if self.final_tokens is not None else self.reserved_tokens
        )

    @property
    def duration_seconds(self) -> float | None:
        """
        Stream duration from creation to now, or None if no chunks received.

        Returns:
            Duration in seconds if streaming has started (chunk_count > 0),
            None otherwise.
        """
        if self.chunk_count == 0:
            return None
        return time.time() - self.created_at


__all__ = ["StreamingReservationContext"]
