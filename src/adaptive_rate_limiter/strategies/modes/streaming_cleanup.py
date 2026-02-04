# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
Streaming cleanup manager for the intelligent mode strategy.

This module extracts streaming in-flight tracking and cleanup logic
from IntelligentModeStrategy to improve code organization and testability.
"""

from __future__ import annotations

import asyncio
import logging
import time
import weakref
from typing import (
    TYPE_CHECKING,
    Any,
)

if TYPE_CHECKING:
    from ...backends.base import BaseBackend

import contextlib

from ...streaming.tracker import StreamingInFlightEntry

logger = logging.getLogger(__name__)


class StreamingCleanupManager:
    """
    Manages cleanup of stale streaming entries.

    Responsibilities:
    - Track streaming responses in-flight with their reservation info
    - Background cleanup of stale/hung streaming entries
    - Activity tracking to detect stale streams
    - Safe release of capacity for abandoned streams

    This class encapsulates all streaming cleanup state and logic,
    providing a clean interface for the main strategy.
    """

    def __init__(
        self,
        backend: BaseBackend,
        streaming_metrics: Any,  # StreamingMetrics - avoid circular import
        cleanup_interval: float = 60.0,
        activity_timeout: float = 300.0,
    ):
        """
        Initialize the streaming cleanup manager.

        Args:
            backend: Backend for capacity release operations
            streaming_metrics: Metrics instance for recording cleanup stats
            cleanup_interval: Interval between cleanup runs (seconds)
            activity_timeout: Inactivity threshold for stale detection (seconds)
        """
        self._backend = backend
        self._streaming_metrics = streaming_metrics
        self._cleanup_interval = cleanup_interval
        self._activity_timeout = activity_timeout

        # Streaming in-flight tracking
        self._streaming_in_flight: dict[str, StreamingInFlightEntry] = {}
        self._streaming_in_flight_lock = asyncio.Lock()

        # Background task
        self._cleanup_task: asyncio.Task[None] | None = None
        self._running = False

    def start(self) -> None:
        """Start the background cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._running = True
            self._cleanup_task = asyncio.create_task(
                self._cleanup_loop(), name="streaming_cleanup"
            )

    async def stop(self) -> None:
        """Stop the background cleanup task."""
        self._running = False
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task

    async def register(
        self,
        reservation_id: str,
        bucket_id: str,
        reserved_tokens: int,
        wrapper: Any,
    ) -> None:
        """
        Register a streaming wrapper in the in-flight tracker.

        Args:
            reservation_id: Unique identifier for the reservation
            bucket_id: The bucket this streaming request belongs to
            reserved_tokens: Number of tokens reserved for this request
            wrapper: The iterator wrapper (stored as weakref)
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

        async with self._streaming_in_flight_lock:
            self._streaming_in_flight[reservation_id] = entry

    async def deregister(self, reservation_id: str) -> None:
        """
        Remove a streaming entry from the in-flight tracker.

        Args:
            reservation_id: The reservation ID to remove
        """
        async with self._streaming_in_flight_lock:
            self._streaming_in_flight.pop(reservation_id, None)

    async def update_activity(self, reservation_id: str) -> None:
        """
        Update last_activity_at for a streaming entry.

        Args:
            reservation_id: The reservation ID to update
        """
        async with self._streaming_in_flight_lock:
            entry = self._streaming_in_flight.get(reservation_id)
            if entry:
                entry.last_activity_at = time.time()

    async def _cleanup_loop(self) -> None:
        """Background task to clean up stale streaming entries."""
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

        Returns:
            Number of stale entries cleaned up.
        """
        now = time.time()
        cutoff = now - self._activity_timeout
        stale_entries: list[StreamingInFlightEntry] = []

        # Collect stale entries INSIDE lock
        async with self._streaming_in_flight_lock:
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

            # Record metrics
            self._streaming_metrics.record_stale_cleanup(bucket_id=entry.bucket_id)

            # Use conservative fallback: actual=reserved (zero refund)
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
                raise
            except Exception as e:
                logger.warning(
                    f"Failed to release stale streaming entry {entry.reservation_id}: {e}"
                )

        return len(stale_entries)


__all__ = ["StreamingCleanupManager"]
