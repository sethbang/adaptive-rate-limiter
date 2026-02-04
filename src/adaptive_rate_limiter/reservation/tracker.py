# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""ReservationTracker for managing reservation contexts with compound key indexing."""

import asyncio
import contextlib
import heapq
import logging
import time

from ..exceptions import ReservationCapacityError
from .context import ReservationContext

logger = logging.getLogger(__name__)


class ReservationTracker:
    """
    Tracks reservation contexts with compound key indexing.

    Primary storage: Dict[(request_id, bucket_id), ReservationContext]
    Secondary index: Dict[request_id, Set[(request_id, bucket_id)]]

    ARCHITECTURE NOTE:
    This class is the LOCAL LAYER of a two-layer reservation tracking system.
    The DISTRIBUTED LAYER is managed by the Redis backend's _in_flight tracking
    and orphan recovery.

    The local layer provides:
    - Fast O(1) lookup by (request_id, bucket_id) compound key
    - O(1) lookup by request_id alone via secondary index
    - Memory management via stale reservation cleanup
    - Thread-safe operations via asyncio.Lock

    This tracker does NOT make backend calls for releasing reservations.
    That is handled by the caller (e.g., IntelligentModeStrategy).
    """

    def __init__(
        self,
        max_reservations: int = 10000,
        max_reservation_age: float = 240.0,
        stale_cleanup_interval: float = 60.0,
    ):
        """
        Initialize the ReservationTracker.

        Args:
            max_reservations: Maximum number of reservations to track (default: 10000)
            max_reservation_age: Maximum age in seconds before a reservation is considered stale (default: 240 = 4 minutes)
            stale_cleanup_interval: Interval in seconds between stale reservation cleanup runs (default: 60)
        """
        self._max_reservations = max_reservations
        self._max_reservation_age = max_reservation_age
        self._stale_cleanup_interval = stale_cleanup_interval

        # Primary storage - keyed by (request_id, bucket_id) tuple
        self._reservation_contexts: dict[tuple[str, str], ReservationContext] = {}

        # Secondary index for O(1) lookup by request_id alone
        self._request_id_index: dict[str, set[tuple[str, str]]] = {}

        # Time-ordered min-heap for O(log n) stale cleanup
        # Entries are (created_at, (request_id, bucket_id)) tuples
        # Note: Heap may contain stale entries for already-removed keys
        # (we validate entries against _reservation_contexts during cleanup)
        self._time_heap: list[tuple[float, tuple[str, str]]] = []

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

        # Cleanup task state
        self._cleanup_task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        """Start the background cleanup task."""
        if self._running:
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.debug("ReservationTracker cleanup task started")

    async def stop(self) -> None:
        """Stop the background cleanup task."""
        if not self._running:
            return

        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task
            self._cleanup_task = None
        logger.debug("ReservationTracker cleanup task stopped")

    async def store(
        self,
        request_id: str,
        bucket_id: str,
        reservation_id: str,
        estimated_tokens: int,
    ) -> None:
        """
        Store a reservation context.

        Args:
            request_id: The request ID
            bucket_id: The bucket ID
            reservation_id: The reservation ID
            estimated_tokens: The number of tokens reserved

        Raises:
            ReservationCapacityError: If the tracker is at maximum capacity
        """
        async with self._lock:
            # Check capacity before adding
            if len(self._reservation_contexts) >= self._max_reservations:
                # Try to clean up stale reservations first
                cleaned = await self._cleanup_stale_unlocked()
                if (
                    cleaned == 0
                    and len(self._reservation_contexts) >= self._max_reservations
                ):
                    raise ReservationCapacityError(
                        f"Reservation tracker at capacity ({self._max_reservations})"
                    )

            key = (request_id, bucket_id)
            context = ReservationContext(
                reservation_id=reservation_id,
                bucket_id=bucket_id,
                estimated_tokens=estimated_tokens,
            )

            # Store in primary storage
            self._reservation_contexts[key] = context

            # Update secondary index
            if request_id not in self._request_id_index:
                self._request_id_index[request_id] = set()
            self._request_id_index[request_id].add(key)

            # Add to time-ordered heap for O(log n) stale cleanup
            heapq.heappush(self._time_heap, (context.created_at, key))

            logger.debug(
                "Stored reservation context: request_id=%s, bucket_id=%s, reservation_id=%s, tokens=%d",
                request_id,
                bucket_id,
                reservation_id,
                estimated_tokens,
            )

    async def get(
        self,
        request_id: str,
        bucket_id: str,
    ) -> ReservationContext | None:
        """
        Get a reservation context without clearing it.

        Args:
            request_id: The request ID
            bucket_id: The bucket ID

        Returns:
            The ReservationContext if found, None otherwise
        """
        async with self._lock:
            key = (request_id, bucket_id)
            return self._reservation_contexts.get(key)

    async def get_and_clear(
        self,
        request_id: str,
        bucket_id: str | None = None,
    ) -> ReservationContext | None:
        """
        Get and clear a reservation context (idempotent).

        If bucket_id is provided, retrieves the specific reservation.
        If bucket_id is None, retrieves the first reservation found for the request_id.

        Args:
            request_id: The request ID
            bucket_id: Optional bucket ID for specific lookup

        Returns:
            The ReservationContext if found and cleared, None otherwise
        """
        async with self._lock:
            if bucket_id is not None:
                # Direct lookup by compound key
                key = (request_id, bucket_id)
                context = self._reservation_contexts.pop(key, None)

                if context is not None:
                    # Update secondary index
                    if request_id in self._request_id_index:
                        self._request_id_index[request_id].discard(key)
                        if not self._request_id_index[request_id]:
                            del self._request_id_index[request_id]

                    logger.debug(
                        "Cleared reservation context: request_id=%s, bucket_id=%s",
                        request_id,
                        bucket_id,
                    )

                return context
            else:
                # Lookup via secondary index
                if request_id not in self._request_id_index:
                    return None

                keys = self._request_id_index[request_id]
                if not keys:
                    del self._request_id_index[request_id]
                    return None

                # Get first key and retrieve context
                key = next(iter(keys))
                context = self._reservation_contexts.pop(key, None)

                if context is not None:
                    keys.discard(key)
                    if not keys:
                        del self._request_id_index[request_id]

                    logger.debug(
                        "Cleared reservation context (lookup): request_id=%s, bucket_id=%s",
                        request_id,
                        key[1],
                    )

                return context

    async def clear_all_for_request(
        self,
        request_id: str,
    ) -> list[ReservationContext]:
        """
        Clear all reservations for a given request.

        Used when a request fails and all its reservations need to be released.

        Args:
            request_id: The request ID

        Returns:
            List of all cleared ReservationContext objects
        """
        async with self._lock:
            if request_id not in self._request_id_index:
                return []

            keys = list(self._request_id_index[request_id])
            contexts = []

            for key in keys:
                context = self._reservation_contexts.pop(key, None)
                if context is not None:
                    contexts.append(context)

            # Clean up secondary index
            del self._request_id_index[request_id]

            if contexts:
                logger.debug(
                    "Cleared all reservations for request: request_id=%s, count=%d",
                    request_id,
                    len(contexts),
                )

            return contexts

    async def _cleanup_loop(self) -> None:
        """Background task that periodically cleans up stale reservations."""
        while self._running:
            try:
                await asyncio.sleep(self._stale_cleanup_interval)
                if self._running:
                    await self._cleanup_stale()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Error in stale reservation cleanup: %s", e)

    async def _cleanup_stale(self) -> int:
        """
        Clean up stale reservations.

        Returns:
            Number of stale reservations cleaned up
        """
        async with self._lock:
            return await self._cleanup_stale_unlocked()

    async def _cleanup_stale_unlocked(self) -> int:
        """
        Clean up stale reservations using the time-ordered heap (O(log n) per stale entry).

        This method assumes the lock is already held.

        Uses a min-heap ordered by created_at to efficiently find stale entries
        without scanning the entire dictionary. The heap may contain stale
        references to already-removed keys, which are validated and skipped.

        Returns:
            Number of stale reservations cleaned up
        """
        now = time.time()
        cutoff = now - self._max_reservation_age
        cleaned_count = 0

        # Pop from heap while entries are stale
        while self._time_heap:
            created_at, key = self._time_heap[0]

            # If the oldest entry isn't stale, we're done
            if created_at >= cutoff:
                break

            # Pop the stale entry from the heap
            heapq.heappop(self._time_heap)

            # Validate against primary storage (heap may have stale references)
            context = self._reservation_contexts.get(key)
            if context is None:
                # Key was already removed (e.g., via get_and_clear)
                continue

            # Verify this is the same context (created_at matches)
            # This handles the case where a key was removed and re-added
            if context.created_at != created_at:
                # Different context at this key, skip (new entry has its own heap entry)
                continue

            # Remove the stale context
            del self._reservation_contexts[key]
            cleaned_count += 1

            # Update secondary index
            request_id = key[0]
            if request_id in self._request_id_index:
                self._request_id_index[request_id].discard(key)
                if not self._request_id_index[request_id]:
                    del self._request_id_index[request_id]

        if cleaned_count > 0:
            logger.info(
                "Cleaned up %d stale reservations (older than %d seconds)",
                cleaned_count,
                self._max_reservation_age,
            )

        # Check stale ratio and log warning if high
        try:
            ratio = self.stale_entry_ratio
            if ratio > 0.5:
                logger.warning(
                    "High stale reservation ratio in heap: %.2f%% (%d/%d). "
                    "Consider calling compact_heap() or adjusting cleanup interval.",
                    ratio * 100,
                    len(self._time_heap) - len(self._reservation_contexts),
                    len(self._time_heap),
                )
        except Exception as e:
            logger.warning("Failed to check stale entry ratio: %s", e)

        return cleaned_count

    async def get_and_clear_stale(self, cutoff_time: float) -> list[ReservationContext]:
        """
        Atomically retrieve and remove all stale reservations using time-ordered heap.

        This method provides an encapsulated way for external callers
        (e.g., IntelligentModeStrategy) to get stale reservations for
        backend release operations, without directly accessing internal
        storage dictionaries.

        Uses a min-heap ordered by created_at for O(log n) per stale entry
        instead of O(n) full scan.

        Args:
            cutoff_time: Unix timestamp. Reservations created before this
                time are considered stale and will be removed.

        Returns:
            List of stale ReservationContext objects that were removed.
            The caller is responsible for releasing these to the backend.
        """
        stale_contexts: list[ReservationContext] = []

        async with self._lock:
            # Pop from heap while entries are stale
            while self._time_heap:
                created_at, key = self._time_heap[0]

                # If the oldest entry isn't stale, we're done
                if created_at >= cutoff_time:
                    break

                # Pop the stale entry from the heap
                heapq.heappop(self._time_heap)

                # Validate against primary storage (heap may have stale references)
                context = self._reservation_contexts.get(key)
                if context is None:
                    # Key was already removed (e.g., via get_and_clear)
                    continue

                # Verify this is the same context (created_at matches)
                if context.created_at != created_at:
                    # Different context at this key, skip
                    continue

                # Remove the stale context
                del self._reservation_contexts[key]
                stale_contexts.append(context)

                # Update secondary index
                request_id = key[0]
                if request_id in self._request_id_index:
                    self._request_id_index[request_id].discard(key)
                    if not self._request_id_index[request_id]:
                        del self._request_id_index[request_id]

        return stale_contexts

    @property
    def stale_entry_ratio(self) -> float:
        """Return ratio of stale entries to total heap entries."""
        if not self._time_heap:
            return 0.0
        stale_count = sum(
            1 for _, key in self._time_heap if key not in self._reservation_contexts
        )
        return stale_count / len(self._time_heap)

    def compact_heap(self) -> int:
        """Remove stale entries from heap. Returns number of entries removed."""
        valid_entries = [
            (ts, key)
            for ts, key in self._time_heap
            if key in self._reservation_contexts
        ]
        removed = len(self._time_heap) - len(valid_entries)
        self._time_heap = valid_entries
        heapq.heapify(self._time_heap)
        return removed

    @property
    def reservation_count(self) -> int:
        """Return the current number of tracked reservations."""
        return len(self._reservation_contexts)

    @property
    def request_count(self) -> int:
        """Return the current number of unique requests with reservations."""
        return len(self._request_id_index)

    def _rebuild_time_heap(self) -> None:
        """
        Rebuild the time heap from current context data.

        This is an internal method primarily for testing scenarios where
        context timestamps are modified directly. In production, timestamps
        are immutable after creation, so the heap stays synchronized.

        WARNING: This method is O(n log n) and should not be called in
        production hot paths.
        """
        self._time_heap = [
            (context.created_at, key)
            for key, context in self._reservation_contexts.items()
        ]
        heapq.heapify(self._time_heap)
