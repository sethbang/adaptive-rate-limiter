# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
Rate limit reset watcher for the intelligent mode strategy.

This module extracts the rate limit reset monitoring logic from
IntelligentModeStrategy to improve code organization and testability.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Any,
)

if TYPE_CHECKING:
    from ...scheduler.state import StateManager

logger = logging.getLogger(__name__)


class RateLimitResetWatcher:
    """
    Manages rate limit reset watchers for proactive queue release.

    Responsibilities:
    - Schedule wake-up tasks when rate limits reset
    - Track which buckets have active watchers (deduplication)
    - Calculate earliest reset time across all buckets
    - Coordinate with scheduler loop via wakeup callback

    This enables proactive queue release instead of reactive polling,
    improving throughput when capacity becomes available.
    """

    def __init__(
        self,
        state_manager: StateManager,
        wakeup_callback: Callable[[], Any],  # Async callback to wake scheduler
    ):
        """
        Initialize the reset watcher.

        Args:
            state_manager: State manager for checking bucket states
            wakeup_callback: Async callback to invoke when reset occurs
        """
        self._state_manager = state_manager
        self._wakeup_callback = wakeup_callback

        # Task tracking
        self._reset_tasks: set[asyncio.Task[Any]] = set()
        self._buckets_waiting: set[str] = set()

    @property
    def buckets_waiting_for_reset(self) -> set[str]:
        """Get the set of buckets currently waiting for reset."""
        return self._buckets_waiting

    async def schedule_watcher(self, bucket_id: str, reset_timestamp: float) -> None:
        """
        Schedule a wake-up task for when a bucket's rate limit resets.

        Enables proactive queue release instead of reactive polling.

        Args:
            bucket_id: The bucket identifier to watch
            reset_timestamp: Unix timestamp when the rate limit resets
        """
        # Calculate wait time
        now = time.time()
        # Add a small buffer (0.1s) to ensure server clock has rolled over
        wait_time = max(0, reset_timestamp - now + 0.1)

        # Handle immediate reset
        if wait_time <= 0:
            # Immediate wakeup (fire-and-forget with exception handling)
            task = asyncio.create_task(self._wakeup_callback())
            task.add_done_callback(
                lambda t: t.exception() if not t.cancelled() else None
            )
            return

        # Deduplication: If we already have a watcher for this bucket, skip
        if bucket_id in self._buckets_waiting:
            return

        # Define the watcher coroutine
        async def reset_watcher() -> None:
            try:
                await asyncio.sleep(wait_time)
                logger.debug(f"Rate limit reset timer expired for bucket {bucket_id}")

                # Cleanup tracking
                self._buckets_waiting.discard(bucket_id)

                # Wake up scheduler
                await self._wakeup_callback()

            except asyncio.CancelledError:
                logger.debug(f"Reset watcher for {bucket_id} cancelled")
            finally:
                # Ensure cleanup happens even on error
                self._buckets_waiting.discard(bucket_id)

        # Create and track task
        task = asyncio.create_task(reset_watcher())
        self._reset_tasks.add(task)
        self._buckets_waiting.add(bucket_id)

        # Self-cleanup callback
        task.add_done_callback(self._reset_tasks.discard)

        logger.debug(
            f"Scheduled rate limit reset watcher for bucket {bucket_id} "
            f"(reset in {wait_time:.2f}s)"
        )

    async def get_earliest_reset_time(self) -> float | None:
        """
        Get the earliest reset time across all buckets.

        Returns:
            Unix timestamp of earliest reset, or None if no reset times available
        """
        if not self._state_manager:
            return None

        earliest_reset = None

        # Check all buckets being tracked
        for bucket_id in self._buckets_waiting:
            state = await self._state_manager.get_state(bucket_id)
            if state and state.reset_at:
                reset_ts = state.reset_at.timestamp()
                if earliest_reset is None or reset_ts < earliest_reset:
                    earliest_reset = reset_ts

        return earliest_reset

    async def stop(self) -> None:
        """Cancel all watcher tasks and clean up."""
        # Cancel all rate limit reset watcher tasks
        for task in list(self._reset_tasks):
            if not task.done():
                task.cancel()

        # Wait for all watchers to complete cancellation
        if self._reset_tasks:
            await asyncio.gather(*self._reset_tasks, return_exceptions=True)

        # Clear tracking sets
        self._reset_tasks.clear()
        self._buckets_waiting.clear()


__all__ = ["RateLimitResetWatcher"]
