# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
High-performance cache for state management.

Provides configurable locking strategies and O(log n) expiration cleanup.
"""

import asyncio
import contextlib
import heapq
import logging
import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any

from ...observability.constants import (
    CACHE_EVICTIONS_TOTAL,
    CACHE_HITS_TOTAL,
    CACHE_MISSES_TOTAL,
)
from ...observability.protocols import MetricsCollectorProtocol
from ..config import StateConfig
from .models import StateEntry, StateMetrics

logger = logging.getLogger(__name__)  # adaptive_rate_limiter.scheduler.state.cache


class Cache:
    """
    High-performance cache for state management with configurable locking strategy.

    Supports two locking modes based on configuration:

    **Lock-Free Read Mode** (config.lock_free_reads=True):
    - Read operations (get) do not acquire locks for maximum performance
    - Write operations use asyncio.Lock for mutual exclusion without blocking event loop
    - Best for read-heavy workloads with occasional writes

    **Full Locking Mode** (config.lock_free_reads=False):
    - All operations use asyncio.Lock for complete synchronization
    - Best for write-heavy workloads or when strict ordering is required

    Thread Safety Note:
        All write operations use asyncio.Lock regardless of mode, ensuring the event
        loop is never blocked by lock contention. The lock_free_reads mode only affects
        whether reads acquire the lock, not the type of lock used for writes.
    """

    def __init__(
        self,
        config: StateConfig,
        metrics_collector: MetricsCollectorProtocol | None = None,
    ) -> None:
        self.config = config
        self._cache: dict[str, StateEntry] = {}
        self._versions: dict[str, list[StateEntry]] = {}
        # Use OrderedDict for O(1) eviction of oldest entry (issue state_mgmt_007)
        self._creation_times: OrderedDict[str, float] = OrderedDict()

        # Min-heap for O(log n) expired entry detection (issue state_mgmt_008)
        # Entries are (expires_at_timestamp, key) tuples
        # Note: Heap may contain stale entries for removed/updated keys
        self._expiration_heap: list[tuple[float, str]] = []

        # Use asyncio.Lock consistently for both modes to avoid blocking event loop
        # The lock_free_reads mode controls whether reads acquire the lock, not the lock type
        self._lock = asyncio.Lock()

        # Dedicated lock for thread-safe metrics updates
        self._metrics_lock = threading.Lock()

        # Background cleanup
        self._cleanup_task: asyncio.Task[None] | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self._task_cancelled = False
        self._task_stop_requested = False

        # Metrics
        self.metrics = StateMetrics()

        # Optional unified metrics collector for Prometheus export
        self._metrics_collector = metrics_collector

        logger.info(f"Cache initialized with {config.cache_policy.value} policy")

    async def start(self) -> None:
        """Start background cleanup task."""
        if self._running:
            return

        try:
            current_loop = asyncio.get_running_loop()

            if (
                self._event_loop
                and self._event_loop != current_loop
                and self._cleanup_task
                and not self._cleanup_task.done()
            ):
                await self._cancel_cross_loop_task(
                    self._cleanup_task,
                    self._event_loop,
                    timeout=self.config.cleanup_task_wait_timeout,
                )

            self._event_loop = current_loop
            self._running = True
            self._task_cancelled = False
            self._task_stop_requested = False

            self._cleanup_task = asyncio.create_task(
                self._cleanup_loop(),
                name=f"cache_cleanup_{id(self)}_loop_{id(current_loop)}",
            )

            logger.info(f"Cache background cleanup started in loop {id(current_loop)}")

        except RuntimeError as e:
            logger.warning(f"Could not start cache cleanup task: {e}")

    async def _cancel_cross_loop_task(
        self,
        task: asyncio.Task[None],
        task_loop: asyncio.AbstractEventLoop,
        timeout: float = 2.0,
    ) -> bool:
        """Cancel a task that belongs to a different event loop."""
        if task.done():
            return True

        self._task_cancelled = True

        try:
            task_loop.call_soon_threadsafe(task.cancel)
        except RuntimeError as e:
            logger.warning(f"Could not cancel task in loop {id(task_loop)}: {e}")
            return False

        start_time = time.time()
        while not task.done() and (time.time() - start_time) < timeout:  # noqa: ASYNC110
            await asyncio.sleep(0.1)

        return task.done()

    async def stop(self) -> None:
        """Stop background cleanup."""
        if not self._running:
            return

        if self._task_stop_requested:
            return

        self._task_stop_requested = True
        self._running = False

        if not self._cleanup_task:
            await self.clear()
            self._task_stop_requested = False
            return

        try:
            current_loop = asyncio.get_running_loop()

            if self._event_loop and self._event_loop != current_loop:
                await self._cancel_cross_loop_task(
                    self._cleanup_task,
                    self._event_loop,
                    timeout=self.config.cleanup_task_cancel_timeout,
                )
                self._cleanup_task = None
                self._event_loop = None
                await self.clear()
                self._task_stop_requested = False
                return

            if not self._cleanup_task.done():
                self._cleanup_task.cancel()
                with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                    await asyncio.wait_for(
                        asyncio.shield(self._cleanup_task),
                        timeout=self.config.cleanup_task_wait_timeout,
                    )

        except RuntimeError:
            pass
        finally:
            self._cleanup_task = None
            self._event_loop = None
            self._task_cancelled = False
            self._task_stop_requested = False

        await self.clear()
        logger.info("Cache stopped and cleared")

    async def close(self) -> None:
        """Close the cache with graceful shutdown."""
        logger.info("Closing cache (graceful shutdown)...")
        await self.stop()
        logger.info("Cache closed")

    async def __aenter__(self) -> "Cache":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def get(self, key: str) -> StateEntry | None:
        """Get entry from cache."""
        if self.config.lock_free_reads:
            try:
                entry = self._cache.get(key)
                if entry and not entry.is_expired:
                    with self._metrics_lock:
                        self.metrics.cache_hits += 1
                    if self._metrics_collector:
                        self._metrics_collector.inc_counter(CACHE_HITS_TOTAL)
                    return entry
                elif entry and entry.is_expired:
                    await self._remove_expired(key)
                with self._metrics_lock:
                    self.metrics.cache_misses += 1
                if self._metrics_collector:
                    self._metrics_collector.inc_counter(CACHE_MISSES_TOTAL)
                return None
            except (KeyError, AttributeError, ValueError):
                with self._metrics_lock:
                    self.metrics.cache_misses += 1
                if self._metrics_collector:
                    self._metrics_collector.inc_counter(CACHE_MISSES_TOTAL)
                return None
        else:
            async with self._lock:
                entry = self._cache.get(key)
                if entry and not entry.is_expired:
                    with self._metrics_lock:
                        self.metrics.cache_hits += 1
                    if self._metrics_collector:
                        self._metrics_collector.inc_counter(CACHE_HITS_TOTAL)
                    return entry
                elif entry and entry.is_expired:
                    await self._remove_expired_unsafe(key)
                with self._metrics_lock:
                    self.metrics.cache_misses += 1
                if self._metrics_collector:
                    self._metrics_collector.inc_counter(CACHE_MISSES_TOTAL)
                return None

    async def set(self, entry: StateEntry, ttl_override: float | None = None) -> bool:
        """Set entry in cache."""
        if ttl_override:
            entry.expires_at = datetime.now(timezone.utc) + timedelta(
                seconds=ttl_override
            )
        elif self.config.cache_ttl:
            entry.expires_at = datetime.now(timezone.utc) + timedelta(
                seconds=self.config.cache_ttl
            )

        # Always use asyncio.Lock for writes to avoid blocking event loop
        async with self._lock:
            return await self._set_unsafe(entry)

    async def _set_unsafe(self, entry: StateEntry) -> bool:
        """Set entry without locking (must be called under lock)."""
        key = entry.key

        if (
            self.config.max_cache_size
            and len(self._cache) >= self.config.max_cache_size
            and key not in self._cache
        ):
            await self._evict_oldest()

        if self.config.enable_versioning:
            if key not in self._versions:
                self._versions[key] = []
            self._versions[key].append(entry)

            if len(self._versions[key]) > self.config.max_versions:
                self._versions[key] = self._versions[key][-self.config.max_versions :]

        self._cache[key] = entry
        # For OrderedDict, if key exists we need to move to end to maintain insertion order
        if key in self._creation_times:
            self._creation_times.move_to_end(key)
        self._creation_times[key] = time.time()

        # Add to expiration heap if entry has expiration (for O(log n) cleanup)
        if entry.expires_at is not None:
            expires_timestamp = entry.expires_at.timestamp()
            heapq.heappush(self._expiration_heap, (expires_timestamp, key))

        return True

    async def atomic_update(
        self, key: str, data: dict[str, Any], merge: bool = True
    ) -> StateEntry | None:
        """
        Atomically update existing entry or create new one.

        This method provides atomic read-modify-write semantics using
        copy-on-write pattern to ensure lock-free readers never observe
        partially-updated state.
        """
        # Always use asyncio.Lock for atomic updates to avoid blocking event loop
        async with self._lock:
            old_entry = self._cache.get(key)
            if old_entry:
                # Copy-on-write: create a new entry instead of mutating in place
                # This ensures lock-free readers see either the old or new state,
                # never a partially-updated state
                new_data = {**old_entry.data, **data} if merge else data

                entry = StateEntry(
                    key=key,
                    data=new_data,
                    state_type=old_entry.state_type,
                    version=old_entry.version + 1,
                    created_at=old_entry.created_at,
                    updated_at=datetime.now(timezone.utc),
                    expires_at=old_entry.expires_at,
                    metadata=old_entry.metadata,
                    namespace=old_entry.namespace,
                )
            else:
                entry = StateEntry(key=key, data=data, namespace=self.config.namespace)
            await self._set_unsafe(entry)
            return entry

    async def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        # Always use asyncio.Lock for deletes to avoid blocking event loop
        async with self._lock:
            return await self._delete_unsafe(key)

    async def _delete_unsafe(self, key: str) -> bool:
        """Delete entry without locking."""
        if key in self._cache:
            del self._cache[key]
            self._creation_times.pop(key, None)
            if key in self._versions:
                del self._versions[key]
            return True
        return False

    async def clear(self) -> None:
        """Clear entire cache."""
        # Always use asyncio.Lock for clear to avoid blocking event loop
        async with self._lock:
            self._cache.clear()
            self._versions.clear()
            self._creation_times.clear()
            self._expiration_heap.clear()

    async def bulk_get(self, keys: list[str]) -> dict[str, StateEntry | None]:
        """
        Get multiple entries efficiently (non-atomic convenience method).

        **Thread Safety Note:**
            This method iterates through keys with individual get() calls.
            It is NOT atomic - state may change between individual gets if
            another writer modifies entries concurrently. This is by design
            for better performance in read-heavy scenarios.

            If you need a consistent snapshot of multiple entries, use
            :meth:`atomic_bulk_get` instead, which holds the lock for the
            entire operation.

        Args:
            keys: List of cache keys to retrieve

        Returns:
            Dict mapping keys to their StateEntry values (or None if not found)

        See Also:
            :meth:`atomic_bulk_get`: For atomic read operations requiring consistency
        """
        results = {}
        for key in keys:
            results[key] = await self.get(key)
        return results

    async def bulk_set(self, entries: list[StateEntry]) -> dict[str, bool]:
        """
        Set multiple entries efficiently (non-atomic convenience method).

        **Thread Safety Note:**
            This method iterates through entries with individual set() calls.
            It is NOT atomic - other operations may interleave between individual
            sets if concurrent writers are active. This is by design for better
            performance when atomicity is not required.

            If you need all entries to be written as a single atomic operation,
            use :meth:`atomic_bulk_set` instead, which holds the lock for the
            entire operation.

        Args:
            entries: List of StateEntry objects to store

        Returns:
            Dict mapping entry keys to success status (True if set succeeded)

        See Also:
            :meth:`atomic_bulk_set`: For atomic write operations requiring consistency
        """
        results = {}
        for entry in entries:
            results[entry.key] = await self.set(entry)
        return results

    async def atomic_bulk_get(self, keys: list[str]) -> dict[str, StateEntry | None]:
        """
        Get multiple entries atomically with a consistent snapshot.

        This method holds the lock for the entire operation, ensuring that
        the returned values represent a consistent point-in-time snapshot.
        No writes can interleave during the read operation.

        **Performance Trade-off:**
            This method blocks all writers while reading. For read-heavy
            workloads where occasional staleness is acceptable, prefer
            :meth:`bulk_get` for better throughput.

        Args:
            keys: List of cache keys to retrieve

        Returns:
            Dict mapping keys to their StateEntry values (or None if not found)

        See Also:
            :meth:`bulk_get`: Non-atomic version for better performance
        """
        results: dict[str, StateEntry | None] = {}
        async with self._lock:
            for key in keys:
                entry = self._cache.get(key)
                if entry and not entry.is_expired:
                    with self._metrics_lock:
                        self.metrics.cache_hits += 1
                    results[key] = entry
                elif entry and entry.is_expired:
                    # Remove expired entry (already under lock)
                    await self._remove_expired_unsafe(key)
                    with self._metrics_lock:
                        self.metrics.cache_misses += 1
                    results[key] = None
                else:
                    with self._metrics_lock:
                        self.metrics.cache_misses += 1
                    results[key] = None
        return results

    async def atomic_bulk_set(
        self, entries: list[StateEntry], ttl_override: float | None = None
    ) -> dict[str, bool]:
        """
        Set multiple entries atomically as a single operation.

        This method holds the lock for the entire operation, ensuring that
        all entries are written together without any interleaving operations
        from other writers.

        **Performance Trade-off:**
            This method blocks all readers (if lock_free_reads=False) and
            writers while writing. For write-heavy workloads where atomicity
            is not required, prefer :meth:`bulk_set` for better throughput.

        Args:
            entries: List of StateEntry objects to store
            ttl_override: Optional TTL in seconds to apply to all entries

        Returns:
            Dict mapping entry keys to success status (True if set succeeded)

        See Also:
            :meth:`bulk_set`: Non-atomic version for better performance
        """
        results: dict[str, bool] = {}
        async with self._lock:
            for entry in entries:
                # Apply TTL if specified
                if ttl_override:
                    entry.expires_at = datetime.now(timezone.utc) + timedelta(
                        seconds=ttl_override
                    )
                elif self.config.cache_ttl:
                    entry.expires_at = datetime.now(timezone.utc) + timedelta(
                        seconds=self.config.cache_ttl
                    )
                # Use unsafe set since we already hold the lock
                results[entry.key] = await self._set_unsafe(entry)
        return results

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_size = len(self._cache)
        expired_count = sum(1 for entry in self._cache.values() if entry.is_expired)

        return {
            "size": total_size,
            "max_size": self.config.max_cache_size,
            "expired_entries": expired_count,
            "metrics": self.metrics.__dict__,
            "config": {
                "cache_ttl": self.config.cache_ttl,
                "cache_policy": self.config.cache_policy.value,
                "lock_free_reads": self.config.lock_free_reads,
            },
        }

    # === Public Metrics Recording API ===
    # These methods encapsulate _metrics_lock access for StateManager

    def record_flush_retry(self) -> None:
        """Record a flush retry (thread-safe).

        Called by StateManager when a backend write fails and will be retried.
        """
        with self._metrics_lock:
            self.metrics.flush_retries += 1

    def record_flush_drop(self) -> None:
        """Record a dropped flush (thread-safe).

        Called by StateManager when an update is dropped after max retries.
        """
        with self._metrics_lock:
            self.metrics.flush_drops += 1

    def record_flush_success(self, writes: int) -> None:
        """Record successful flush with backend writes AND bulk operation (thread-safe).

        This method atomically updates both backend_writes and bulk_operations
        in a single lock acquisition to maintain the same atomicity guarantees
        as the original inline code.

        Args:
            writes: Number of successful backend writes in this batch.
        """
        with self._metrics_lock:
            self.metrics.backend_writes += writes
            self.metrics.bulk_operations += 1

    async def _evict_oldest(self) -> None:
        """Evict oldest entry to make room using O(1) OrderedDict access."""
        if not self._creation_times:
            return

        # OrderedDict maintains insertion order, so the first key is the oldest
        # next(iter(...)) gets the first key in O(1)
        oldest_key = next(iter(self._creation_times))
        await self._delete_unsafe(oldest_key)
        with self._metrics_lock:
            self.metrics.cache_evictions += 1
        if self._metrics_collector:
            self._metrics_collector.inc_counter(CACHE_EVICTIONS_TOTAL)

    async def _remove_expired(self, key: str) -> None:
        """Remove expired entry (async-safe)."""
        # Always use asyncio.Lock for expired removal to avoid blocking event loop
        async with self._lock:
            await self._remove_expired_unsafe(key)

    async def _remove_expired_unsafe(self, key: str) -> None:
        """Remove expired entry without locking.

        Note:
            In lock-free read mode, there's a race condition where an expired entry
            detected during get() may be replaced with a fresh entry by another
            coroutine before this method is called. To prevent accidentally deleting
            valid cached data, we re-check that the entry exists AND is still expired
            before removal.
        """
        # Re-check that entry exists AND is still expired to prevent race condition
        # where another coroutine may have replaced the expired entry with a fresh one
        entry = self._cache.get(key)
        if entry is None or not entry.is_expired:
            return

        del self._cache[key]
        self._creation_times.pop(key, None)
        if key in self._versions:
            del self._versions[key]

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        try:
            while self._running and not self._task_cancelled:
                if self._task_stop_requested:
                    break

                try:
                    await asyncio.sleep(self.config.cleanup_interval)
                except asyncio.CancelledError:
                    break

                if not self._running or self._task_cancelled:
                    break

                try:
                    await self._cleanup_expired()
                except Exception as e:
                    logger.error(f"Error during cache cleanup: {e}", exc_info=True)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Fatal error in cleanup loop: {e}", exc_info=True)

    async def _cleanup_expired(self) -> None:
        """Clean up expired entries using O(log n) heap-based detection."""
        cleaned_count = 0
        current_time = time.time()

        # Pop from heap while entries are expired
        while self._expiration_heap:
            expires_at, key = self._expiration_heap[0]

            # If the earliest expiration is in the future, we're done
            if expires_at > current_time:
                break

            # Pop the expired entry from the heap
            heapq.heappop(self._expiration_heap)

            # Validate against cache (heap may have stale references)
            entry = self._cache.get(key)
            if entry is None:
                # Key was already removed (e.g., via delete)
                continue

            # Verify this entry's expiration matches the heap entry
            # (handles case where an entry was updated with new expiration)
            if entry.expires_at is None:
                continue
            if entry.expires_at.timestamp() != expires_at:
                # Different expiration, skip (new entry has its own heap entry)
                continue

            # Remove the expired entry
            await self._remove_expired(key)
            cleaned_count += 1

        if cleaned_count > 0:
            logger.debug(f"Cleaned up {cleaned_count} expired cache entries")


__all__ = ["Cache"]
