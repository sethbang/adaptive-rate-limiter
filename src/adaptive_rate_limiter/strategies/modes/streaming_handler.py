# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
Streaming response handler for the intelligent mode strategy.

This module extracts streaming detection, wrapping, and capacity tracking
from IntelligentModeStrategy to improve code organization and testability.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Any,
)

if TYPE_CHECKING:
    from ...backends.base import BaseBackend
    from ...reservation.context import ReservationContext
    from ...reservation.tracker import ReservationTracker
    from ...types.request import RequestMetadata

from ...streaming.context import StreamingReservationContext
from ...streaming.iterator import RateLimitedAsyncIterator

logger = logging.getLogger(__name__)


class StreamingHandler:
    """
    Handles streaming response detection, wrapping, and capacity tracking.

    Responsibilities:
    - Detect if a response is a streaming response (heuristic-based)
    - Wrap streaming responses with rate limit tracking
    - Coordinate with reservation tracker for capacity management
    - Provide metrics callbacks for streaming completion/error tracking

    This class is designed as a stateless helper that receives dependencies
    via constructor injection.
    """

    def __init__(
        self,
        reservation_tracker: ReservationTracker,
        backend: BaseBackend,
        streaming_metrics: Any,  # StreamingMetrics - avoid circular import
        register_callback: Callable[
            [str, str, int, Any], Any
        ],  # async callback for in-flight registration
    ):
        """
        Initialize the streaming handler.

        Args:
            reservation_tracker: Tracker for managing reservations
            backend: Backend for capacity release operations
            streaming_metrics: Metrics instance for recording streaming stats
            register_callback: Async callback to register streaming in-flight entries
        """
        self._reservation_tracker = reservation_tracker
        self._backend = backend
        self._streaming_metrics = streaming_metrics
        self._register_callback = register_callback

    def detect_streaming_response(self, result: Any) -> bool:
        """
        Detect if the response is a streaming response.

        This is a HEURISTIC for advisory metrics only. It checks for:
        - Async generators and generators
        - Objects with a truthy 'stream' attribute
        - Objects with async/sync iterator protocols (excluding common iterables)

        Args:
            result: The API response to check

        Returns:
            True if the response appears to be a streaming response
        """
        if result is None:
            return False

        # Check for common streaming response patterns
        if inspect.isasyncgen(result) or inspect.isgenerator(result):
            return True

        # Check for stream attribute (common in SDK response objects)
        if hasattr(result, "stream") and result.stream:
            return True

        # Check for iterator protocol
        if hasattr(result, "__aiter__") or hasattr(result, "__iter__"):
            # Exclude common non-streaming iterables
            return not isinstance(
                result, (str, bytes, dict, list, tuple, range, set, frozenset)
            )

        return False

    async def get_reservation(
        self, request_id: str, bucket_id: str
    ) -> ReservationContext | None:
        """
        Get reservation WITHOUT clearing it.

        Used in streaming support where the iterator takes ownership
        and handles the release when iteration completes.

        Args:
            request_id: The request identifier
            bucket_id: The bucket identifier

        Returns:
            ReservationContext if found, None otherwise
        """
        return await self._reservation_tracker.get(request_id, bucket_id)

    async def wrap_streaming_response(
        self,
        result: Any,
        reservation: ReservationContext,
        metadata: RequestMetadata,
    ) -> Any:
        """
        Wrap a streaming response with rate limit tracking.

        The wrapper intercepts iteration to extract tokens and release
        capacity on completion. Handles different response types:
        - Stream classes with _iterator attribute
        - Direct async iterables
        - Unknown types (fallback to immediate release)

        Args:
            result: The streaming response to wrap
            reservation: The reservation context for capacity tracking
            metadata: Request metadata for logging/tracking

        Returns:
            The wrapped response (or original if wrapping not possible)
        """

        # Create metrics callbacks
        def on_completion(
            reserved: int,
            actual: int,
            extraction_succeeded: bool,
            bucket_id: str | None,
            duration: float | None,
        ) -> None:
            """Callback invoked when streaming completes successfully."""
            self._streaming_metrics.record_completion(
                reserved, actual, extraction_succeeded
            )

        def on_error(bucket_id: str | None) -> None:
            """Callback invoked when streaming errors/fallback."""
            self._streaming_metrics.record_error()

        # Create streaming context from reservation with metrics callbacks
        streaming_ctx = StreamingReservationContext(
            reservation_id=reservation.reservation_id,
            bucket_id=reservation.bucket_id,
            request_id=metadata.request_id,
            reserved_tokens=reservation.estimated_tokens,
            backend=self._backend,
            metrics_callback=on_completion,
            error_metrics_callback=on_error,
        )

        # Determine what to wrap based on result type
        if hasattr(result, "_iterator"):
            # Result is a Stream class - wrap its internal iterator
            original_iterator = result._iterator
            wrapped_iterator = RateLimitedAsyncIterator(
                original_iterator, streaming_ctx
            )
            result._iterator = wrapped_iterator

            # Register for background cleanup
            await self._register_callback(
                reservation.reservation_id,
                reservation.bucket_id,
                reservation.estimated_tokens,
                wrapped_iterator,
            )

            logger.debug(
                f"Wrapped streaming response for {metadata.request_id} "
                f"(Stream class with _iterator)"
            )
            return result
        elif hasattr(result, "__aiter__"):
            # Result is an async iterable - wrap it directly
            wrapped = RateLimitedAsyncIterator(result, streaming_ctx)

            # Register for background cleanup
            await self._register_callback(
                reservation.reservation_id,
                reservation.bucket_id,
                reservation.estimated_tokens,
                wrapped,
            )

            logger.debug(
                f"Wrapped streaming response for {metadata.request_id} (async iterable)"
            )
            return wrapped
        else:
            # Unknown type - log warning and release reservation immediately
            logger.warning(
                f"Unknown streaming response type {type(result).__name__} for "
                f"{metadata.request_id}, rate limit tracking will not be applied. "
                f"Releasing reservation immediately."
            )
            # Release reservation since we can't track it through the iterator
            try:
                await self._backend.release_reservation(
                    reservation.bucket_id, reservation.reservation_id
                )
            except Exception as e:
                logger.warning(
                    f"Failed to release reservation for unknown streaming type: {e}"
                )

            # Also clear from local tracker since we released
            await self._reservation_tracker.get_and_clear(
                metadata.request_id, reservation.bucket_id
            )
            return result


__all__ = ["StreamingHandler"]
