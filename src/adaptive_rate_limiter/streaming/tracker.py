# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
Streaming in-flight tracker for the Adaptive Rate Limiter.

This module provides components for tracking in-flight streaming requests to
detect hung streams and clean them up. It implements the background cleanup
pattern.

Classes:
    StreamingInFlightEntry: Dataclass tracking an in-flight streaming request.
    StreamingInFlightTracker: Manages in-flight streaming request tracking and cleanup.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..backends.base import BaseBackend

logger = logging.getLogger(__name__)


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
        started_at: Timestamp when streaming started
        last_activity_at: Timestamp of last chunk received (updated on each chunk)
        wrapper_ref: Weak reference to the iterator wrapper for cleanup detection
    """

    reservation_id: str
    bucket_id: str
    reserved_tokens: int
    started_at: float
    last_activity_at: float  # Updated on each chunk
    wrapper_ref: weakref.ref[Any]  # Weak reference to wrapper for cleanup


class StreamingInFlightTracker:
    """
    Manages in-flight streaming request tracking and cleanup.

    This class provides background cleanup of hung streams by tracking all
    active streaming requests and periodically checking for stale entries.

    A stream is considered stale if it hasn't had any activity (chunk received)
    for a configurable timeout period (default 5 minutes).

    Attributes:
        backend: Backend for releasing streaming reservations
        cleanup_interval: Seconds between cleanup scans (default 60)
        activity_timeout: Seconds of inactivity before a stream is stale (default 300)

    Example:
        >>> tracker = StreamingInFlightTracker(backend)
        >>> await tracker.start_cleanup()
        >>>
        >>> # Register a streaming request
        >>> await tracker.register(
        ...     reservation_id="res-123",
        ...     bucket_id="tier:xs",
        ...     reserved_tokens=4000,
        ...     wrapper=iterator_wrapper,
        ... )
        >>>
        >>> # Update activity on each chunk
        >>> await tracker.update_activity("res-123")
        >>>
        >>> # Deregister when complete
        >>> await tracker.deregister("res-123")
        >>>
        >>> await tracker.stop_cleanup()
    """

    def __init__(
        self,
        backend: BaseBackend,
        cleanup_interval: int = 60,
        activity_timeout: int = 300,
        metrics_callback: Any | None = None,
    ):
        """
        Initialize the streaming in-flight tracker.

        Args:
            backend: Backend for releasing streaming reservations
            cleanup_interval: Seconds between cleanup scans (default 60)
            activity_timeout: Seconds of inactivity before stale (default 300)
            metrics_callback: Optional callback for recording stale cleanups.
                Signature: (bucket_id: Optional[str]) -> None
        """
        self._backend = backend
        self._cleanup_interval = cleanup_interval
        self._activity_timeout = activity_timeout
        self._metrics_callback = metrics_callback

        # In-flight tracking
        self._streaming_in_flight: dict[str, StreamingInFlightEntry] = {}
        self._lock = asyncio.Lock()

        # Cleanup task management
        self._cleanup_task: asyncio.Task[None] | None = None
        self._running = False

    @property
    def active_count(self) -> int:
        """Return the number of currently tracked streaming requests."""
        return len(self._streaming_in_flight)

    async def start_cleanup(self) -> None:
        """
        Start the background cleanup task.

        This starts a periodic task that scans for and releases stale
        streaming entries.
        """
        if self._cleanup_task is None or self._cleanup_task.done():
            self._running = True
            self._cleanup_task = asyncio.create_task(
                self._cleanup_loop(),
                name="streaming_cleanup",
            )
            logger.debug("Started streaming in-flight cleanup task")

    async def stop_cleanup(self) -> None:
        """
        Stop the background cleanup task.

        This cancels the periodic cleanup task and waits for it to complete.
        """
        self._running = False
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task
        self._cleanup_task = None
        logger.debug("Stopped streaming in-flight cleanup task")

    async def register(
        self,
        reservation_id: str,
        bucket_id: str,
        reserved_tokens: int,
        wrapper: Any,
    ) -> None:
        """
        Register a streaming wrapper in the in-flight tracker.

        Called when creating a new RateLimitedAsyncIterator. The wrapper_ref
        is stored as a weak reference so the tracker doesn't prevent garbage
        collection.

        Args:
            reservation_id: The reservation identifier
            bucket_id: The rate limit bucket identifier
            reserved_tokens: Tokens that were reserved
            wrapper: The RateLimitedAsyncIterator instance (stored as weakref)
        """
        now = time.time()
        entry = StreamingInFlightEntry(
            reservation_id=reservation_id,
            bucket_id=bucket_id,
            reserved_tokens=reserved_tokens,
            started_at=now,
            last_activity_at=now,
            wrapper_ref=weakref.ref(wrapper),
        )

        async with self._lock:
            self._streaming_in_flight[reservation_id] = entry

        logger.debug(
            f"Registered streaming in-flight entry: {reservation_id} "
            f"(bucket={bucket_id}, tokens={reserved_tokens})"
        )

    async def deregister(self, reservation_id: str) -> None:
        """
        Remove a streaming entry from the in-flight tracker.

        NOTE: This method is intended for explicit external cleanup scenarios.
        The RateLimitedAsyncIterator does NOT call this method directly.
        Instead, the iterator sets its `_released` flag to True after
        releasing capacity to the backend, and the background cleanup task
        (_cleanup_stale) detects this flag and removes the entry from the
        tracker automatically. This design avoids circular dependencies
        and keeps deregistration synchronized with the cleanup cycle.

        Args:
            reservation_id: The reservation identifier to remove
        """
        async with self._lock:
            entry = self._streaming_in_flight.pop(reservation_id, None)

        if entry:
            logger.debug(f"Deregistered streaming in-flight entry: {reservation_id}")

    async def update_activity(self, reservation_id: str) -> None:
        """
        Update last_activity_at for a streaming entry.

        Called by RateLimitedAsyncIterator on each chunk to indicate
        the stream is still active. This prevents premature cleanup.

        Args:
            reservation_id: The reservation identifier to update
        """
        async with self._lock:
            entry = self._streaming_in_flight.get(reservation_id)
            if entry:
                entry.last_activity_at = time.time()

    async def _cleanup_loop(self) -> None:
        """
        Background task to clean up stale streaming entries.

        Runs every cleanup_interval seconds and calls _cleanup_stale
        to release hung streams.
        """
        while self._running:
            try:
                await asyncio.sleep(self._cleanup_interval)
                cleaned = await self._cleanup_stale()
                if cleaned > 0:
                    logger.info(f"Streaming cleanup released {cleaned} stale entries")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Streaming cleanup error: {e}")

    async def _cleanup_stale(self) -> int:
        """
        Clean up streaming entries inactive for > activity_timeout seconds.

        Scans _streaming_in_flight for entries that are stale. Also handles
        cleanup of GC'd wrappers and completed streams.

        For each stale entry:
        1. Remove from _streaming_in_flight dict
        2. Call backend.release_streaming_reservation() with actual=reserved (conservative)
        3. Use asyncio.shield() for safety during cancellation

        Note:
            Due to a potential race condition, the iterator may release capacity
            between when entries are collected (inside the lock) and when backend
            release calls are made (outside the lock). This could result in duplicate
            release calls for the same reservation_id. Therefore, implementations of
            ``backend.release_streaming_reservation()`` MUST be idempotent - duplicate
            release calls for the same reservation_id should be safely ignored or
            handled gracefully (e.g., by checking if the reservation still exists
            before releasing).

        Returns:
            Number of stale entries cleaned up.
        """
        now = time.time()
        cutoff = now - self._activity_timeout
        stale_entries: list[StreamingInFlightEntry] = []

        # Collect stale entries INSIDE lock
        async with self._lock:
            for res_id, entry in list(self._streaming_in_flight.items()):
                # 1. Check if wrapper is still alive
                wrapper = entry.wrapper_ref()
                if wrapper is None:
                    # Wrapper GC'd - treat as stale/abandoned
                    stale_entries.append(self._streaming_in_flight.pop(res_id))
                    continue

                # 2. Check if already released by iterator
                if getattr(wrapper, "_released", False):
                    # Already released - just remove from tracker
                    self._streaming_in_flight.pop(res_id)
                    continue

                # 3. Sync activity from wrapper context
                # The iterator updates its context.last_chunk_at on each chunk
                if hasattr(wrapper, "_ctx") and wrapper._ctx.last_chunk_at:
                    entry.last_activity_at = max(
                        entry.last_activity_at, wrapper._ctx.last_chunk_at
                    )

                # 4. Check staleness
                if entry.last_activity_at < cutoff:
                    stale_entries.append(self._streaming_in_flight.pop(res_id))

        # Release OUTSIDE lock to avoid holding lock during backend calls
        for entry in stale_entries:
            age = now - entry.started_at
            logger.warning(
                f"Cleaning up stale streaming entry {entry.reservation_id} "
                f"(age: {age:.1f}s, bucket: {entry.bucket_id}, "
                f"inactivity: {now - entry.last_activity_at:.1f}s)"
            )

            # Record metrics for observability
            if self._metrics_callback:
                try:
                    self._metrics_callback(entry.bucket_id)
                except Exception as e:
                    logger.debug(f"Metrics callback error: {e}")

            # Use conservative fallback: actual=reserved (zero refund)
            # This prevents over-allocation by assuming maximum consumption
            try:
                await asyncio.shield(
                    self._backend.release_streaming_reservation(
                        entry.bucket_id,
                        entry.reservation_id,
                        reserved_tokens=entry.reserved_tokens,
                        actual_tokens=entry.reserved_tokens,  # Conservative: refund=0
                    )
                )
            except asyncio.CancelledError:
                # Shield caught it, release completed, now re-raise
                raise
            except Exception as e:
                # Log and continue - orphan recovery will handle residual cases
                logger.warning(
                    f"Failed to release stale streaming entry {entry.reservation_id}: {e}"
                )

        return len(stale_entries)

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about the tracker.

        Returns:
            Dictionary with tracker statistics.
        """
        return {
            "active_streams": len(self._streaming_in_flight),
            "cleanup_interval": self._cleanup_interval,
            "activity_timeout": self._activity_timeout,
            "cleanup_running": self._running,
        }


__all__ = [
    "StreamingInFlightEntry",
    "StreamingInFlightTracker",
]
