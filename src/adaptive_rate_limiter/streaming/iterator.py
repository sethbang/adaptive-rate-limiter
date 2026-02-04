# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
Rate-limited async iterator wrapper for streaming responses.

This module provides the RateLimitedAsyncIterator class for tracking rate limit
reservations during streaming response consumption. It implements the refund-based
accounting model where capacity is released based on actual token consumption.

The wrapper intercepts chunk iteration to:
1. Track token usage from final chunk
2. Release capacity on completion (normal or error)
3. Support aclose() for explicit cleanup

Key Design Decisions:
- Refund-based accounting: At stream completion, release capacity using
  (reserved_tokens - actual_tokens) as the refund
- Conservative fallback: If token extraction fails, use reserved_tokens as
  actual (refund=0) to prevent over-allocation
- asyncio.shield: Protect release operations from cancellation
- Idempotent release: Use _released flag to prevent double release
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from typing import (
    Any,
    Generic,
    TypeVar,
    cast,
)

from .context import StreamingReservationContext

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Long-running stream warning threshold (20 minutes = 1200 seconds)
_LONG_RUNNING_STREAM_THRESHOLD_SECONDS = 1200


class RateLimitedAsyncIterator(AsyncIterator[T], Generic[T]):
    """
    Async iterator wrapper that adds rate limit tracking to streaming responses.

    This class wraps the raw SSE iterator from HTTP responses and:
    - Extracts tokens from the final chunk's usage field
    - Releases reservation capacity on completion
    - Handles cleanup on all exit paths (normal, exception, explicit close)

    The wrapper is transparent to downstream consumers - it yields the same
    chunks as the underlying iterator, just with rate limit tracking added.

    Usage:
        ctx = StreamingReservationContext(
            reservation_id="...",
            bucket_id="...",
            request_id="...",
            reserved_tokens=4000,
            backend=backend_instance,
        )
        wrapped = RateLimitedAsyncIterator(raw_iterator, ctx)

        async for chunk in wrapped:
            # Process chunk - tokens are extracted and tracked automatically
            pass
        # On completion, capacity is released with refund

    Note:
        For early break from iteration, use a context manager pattern:
        async with stream:
            async for chunk in stream:
                if should_stop:
                    break  # __aexit__ calls aclose() for cleanup
    """

    __slots__ = (
        "__weakref__",
        "_closed",
        "_ctx",
        "_inner",
        "_released",
        "_warned_long_running",
    )

    def __init__(
        self,
        inner: AsyncIterator[T],
        context: StreamingReservationContext,
    ) -> None:
        """
        Initialize the rate-limited async iterator wrapper.

        Args:
            inner: The underlying async iterator to wrap (raw SSE stream)
            context: The streaming reservation context for capacity tracking
        """
        self._inner = inner
        self._ctx = context
        self._released = False
        self._closed = False
        self._warned_long_running = False

    def _update_activity(self) -> None:
        """
        Update last activity time for background cleanup.

        Records chunk receipt on the context and updates the in-flight
        tracker's activity timestamp.
        """
        self._ctx.record_chunk()

    async def __anext__(self) -> T:
        """
        Get next chunk with token extraction and cleanup on completion.

        This method:
        1. Gets the next chunk from the underlying iterator
        2. Updates activity tracking
        3. Checks for long-running stream warnings
        4. Extracts tokens from chunks with usage data
        5. On StopAsyncIteration: releases capacity with refund
        6. On Exception: releases capacity with fallback (refund=0)

        Returns:
            The next chunk from the underlying iterator

        Raises:
            StopAsyncIteration: When iteration completes (after capacity release)
            Exception: Any exception from the underlying iterator (after fallback release)
        """
        try:
            chunk = await self._inner.__anext__()

            self._update_activity()

            # Check for long-running stream warning
            self._check_long_running_warning()

            # Extract tokens from chunk (may contain usage on final chunk)
            tokens = self._extract_tokens(chunk)
            if tokens is not None:
                self._ctx.set_final_tokens(tokens)

            return chunk

        except StopAsyncIteration:
            # Normal completion - release with actual tokens (refund-based)
            await self._release_capacity()
            raise

        except Exception as e:
            # Error path - release with fallback (refund=0)
            logger.warning(
                f"Stream error for {self._ctx.request_id}, "
                f"releasing with fallback: {type(e).__name__}: {e}"
            )
            await self._release_capacity_fallback()
            raise

    def _check_long_running_warning(self) -> None:
        """
        Check if stream has exceeded the long-running threshold and warn once.

        Streams approaching the 30-minute request mapping TTL should be
        flagged for operational awareness. This warns at 20 minutes.
        """
        if self._warned_long_running:
            return

        duration = self._ctx.duration_seconds
        if duration and duration > _LONG_RUNNING_STREAM_THRESHOLD_SECONDS:
            self._warned_long_running = True
            logger.warning(
                f"Stream {self._ctx.request_id} running for {duration:.0f}s, "
                f"approaching request mapping TTL. Chunk count: {self._ctx.chunk_count}"
            )

    def _extract_tokens(self, chunk: Any) -> int | None:
        """
        Extract total_tokens from chunk.usage if present.

        The API may include a usage field in the final streaming chunk
        when stream_options.include_usage is true. This method handles
        both object attribute access and dictionary access patterns.

        Args:
            chunk: The streaming chunk to extract tokens from

        Returns:
            The total_tokens value if found and valid, None otherwise
        """
        # Try object attribute access (Pydantic models, dataclasses)
        if hasattr(chunk, "usage") and chunk.usage is not None:
            usage = chunk.usage
            if hasattr(usage, "total_tokens") and usage.total_tokens is not None:
                try:
                    return int(usage.total_tokens)
                except (ValueError, TypeError):
                    pass

        # Try dictionary access (raw dict responses)
        elif isinstance(chunk, dict):
            usage = chunk.get("usage")
            if usage and isinstance(usage, dict):
                total = usage.get("total_tokens")
                if total is not None:
                    try:
                        return int(total)
                    except (ValueError, TypeError):
                        pass

        return None

    async def _release_capacity(self) -> None:
        """
        Release capacity with refund-based accounting.

        This is the normal completion path. The refund is calculated as:
            refund = reserved_tokens - actual_tokens

        For streams that consumed less than reserved, this returns
        capacity to the pool. For streams that consumed exactly what
        was reserved (or more), the refund is zero or negative
        (clamped to zero by the backend).

        The release operation is protected with asyncio.shield to
        prevent cancellation from interrupting the cleanup.
        """
        if self._released:
            return
        self._released = True

        actual = self._ctx.actual_tokens_for_release
        reserved = self._ctx.reserved_tokens
        duration = self._ctx.duration_seconds
        extraction_succeeded = self._ctx.final_tokens is not None

        try:
            refund = reserved - actual

            # Shield the release from task cancellation
            await asyncio.shield(
                self._ctx.backend.release_streaming_reservation(
                    self._ctx.bucket_id,
                    self._ctx.reservation_id,
                    reserved_tokens=reserved,
                    actual_tokens=actual,
                )
            )

            duration_str = f"{duration:.1f}s" if duration is not None else "N/A"

            logger.debug(
                f"Stream {self._ctx.request_id} completed: "
                f"consumed {actual}/{reserved} tokens "
                f"(refund: {refund}) in {duration_str}, "
                f"chunks: {self._ctx.chunk_count}"
            )

            # Record metrics
            if self._ctx.metrics_callback:
                try:
                    self._ctx.metrics_callback(
                        reserved,
                        actual,
                        extraction_succeeded,
                        self._ctx.bucket_id,
                        duration,
                    )
                except Exception as metrics_err:
                    logger.debug(f"Metrics callback failed: {metrics_err}")

        except Exception as e:
            logger.warning(
                f"Failed to release streaming reservation for {self._ctx.request_id}: "
                f"{type(e).__name__}: {e}"
            )

    async def _release_capacity_fallback(self) -> None:
        """
        Release capacity with reserved=actual (zero refund).

        This is the fallback path used when:
        - Token extraction failed
        - Stream errored before completion
        - Early close without token data

        By setting actual_tokens = reserved_tokens, the refund is zero,
        which is conservative - we assume maximum consumption to prevent
        over-allocation.

        The release operation is protected with asyncio.shield to
        prevent cancellation from interrupting the cleanup.
        """
        if self._released:
            return
        self._released = True

        reserved = self._ctx.reserved_tokens

        try:
            # Shield the release from task cancellation
            await asyncio.shield(
                self._ctx.backend.release_streaming_reservation(
                    self._ctx.bucket_id,
                    self._ctx.reservation_id,
                    reserved_tokens=reserved,
                    actual_tokens=reserved,  # Zero refund
                )
            )

            logger.debug(
                f"Stream {self._ctx.request_id} released with fallback (refund=0), "
                f"chunks processed: {self._ctx.chunk_count}"
            )

            # Record error/fallback metrics
            if self._ctx.error_metrics_callback:
                try:
                    self._ctx.error_metrics_callback(self._ctx.bucket_id)
                except Exception as metrics_err:
                    logger.debug(f"Error metrics callback failed: {metrics_err}")

        except Exception as e:
            logger.warning(
                f"Failed fallback release for {self._ctx.request_id}: "
                f"{type(e).__name__}: {e}"
            )

    async def aclose(self) -> None:
        """
        Explicit close with cleanup.

        Called when:
        - User breaks from async for early (via context manager __aexit__)
        - Explicit close() call on the stream
        - Garbage collection with active stream

        This method is idempotent - multiple calls are safe.

        The close operation:
        1. Releases capacity with fallback if not already released
        2. Closes the inner iterator if it supports aclose()
        """
        if self._closed:
            return
        self._closed = True

        # Release with fallback if not already released
        if not self._released:
            await self._release_capacity_fallback()

        # Close inner iterator if supported
        if hasattr(self._inner, "aclose"):
            try:
                await cast(AsyncGenerator[T, None], self._inner).aclose()
            except Exception as e:
                # Log but don't propagate - we're in cleanup
                logger.debug(
                    f"Error closing inner iterator for {self._ctx.request_id}: "
                    f"{type(e).__name__}: {e}"
                )

    def __aiter__(self) -> RateLimitedAsyncIterator[T]:
        """Return self as the async iterator."""
        return self

    @property
    def context(self) -> StreamingReservationContext:
        """
        Access the streaming reservation context.

        Useful for debugging and testing.
        """
        return self._ctx


__all__ = ["RateLimitedAsyncIterator"]
