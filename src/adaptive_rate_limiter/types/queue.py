# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
Queue types for rate limiting and scheduling.

This module defines the core queue data structures used by the scheduler
and rate limiting system.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from .rate_limit import RateLimitBucket
from .request import RequestMetadata

if TYPE_CHECKING:
    from asyncio import Future, PriorityQueue


@dataclass
class QueuedRequest:
    """
    A request waiting in the queue system.

    Wraps request metadata along with the callable that will execute the
    request and a future that will be resolved when the request completes.

    Attributes:
        metadata: Request metadata for classification and tracking
        request_func: Async callable that executes the actual request
        future: Future that will be resolved with the request result
        queue_entry_time: UTC timestamp when request was added to queue
    """

    metadata: RequestMetadata
    request_func: Callable[[], Awaitable[Any]]
    future: "Future[Any]"
    queue_entry_time: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class QueueInfo:
    """
    Information about a queue for scheduling decisions with safe priority tracking.

    This dataclass contains metadata for queue management and scheduling, including
    thread-safe priority tracking that eliminates the need for unsafe queue peeking.

    Priority Tracking:
        Instead of accessing the internal _queue attribute (which causes race conditions),
        this class tracks priority information through metadata updates during enqueue
        and dequeue operations. This provides safe, accurate priority information for
        scheduling decisions without risking IndexError or corruption.

    Thread Safety:
        All priority tracking metadata updates are protected by an asyncio.Lock to ensure
        atomicity of compound operations (+=, etc.). This prevents race conditions under
        concurrent access from multiple coroutines.

    Attributes:
        queue_key: Unique key identifying this queue
        model_id: Model identifier this queue serves
        resource_type: Resource type classification (string)
        queue: The underlying asyncio.PriorityQueue
        rate_config: Rate limit configuration for this queue
        queue_depth: Current number of items in queue
        last_request_time: Timestamp of last processed request
        current_priority: Most recent priority value enqueued
        priority_sum: Running sum for average calculation
        total_enqueued: Total items ever enqueued
        total_dequeued: Total items ever dequeued
        last_enqueue_time: Timestamp of last enqueue
        last_dequeue_time: Timestamp of last dequeue
        max_priority_seen: Maximum priority value ever seen
        min_priority_seen: Minimum priority value ever seen
    """

    queue_key: str
    model_id: str
    resource_type: str  # Uses string instead of Enum
    queue: "PriorityQueue[Any]"
    rate_config: RateLimitBucket  # Renamed from RateLimitConfig
    queue_depth: int = 0
    last_request_time: datetime | None = None

    # Priority tracking metadata (protected by _metadata_lock)
    current_priority: float = 0.0  # Most recent priority value enqueued
    priority_sum: float = 0.0  # Running sum for average calculation
    total_enqueued: int = 0  # Total items ever enqueued
    total_dequeued: int = 0  # Total items ever dequeued
    last_enqueue_time: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_dequeue_time: datetime | None = None

    # Priority statistics for monitoring and debugging
    max_priority_seen: float = float("-inf")
    min_priority_seen: float = float("inf")

    # Lock for protecting metadata updates
    _metadata_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock, init=False, repr=False, compare=False
    )

    @property
    def avg_priority(self) -> float:
        """
        Calculate average priority of items enqueued.

        This provides a representative priority value for the queue that can be used
        in scheduling decisions without accessing the queue's internal structure.

        Returns:
            float: Average priority of all items ever enqueued, or 0.0 if no items
        """
        return (
            self.priority_sum / self.total_enqueued if self.total_enqueued > 0 else 0.0
        )

    @property
    def current_size(self) -> int:
        """
        Safe access to queue size using built-in qsize().

        Uses the queue's thread-safe qsize() method instead of accessing internal state.

        Returns:
            int: Current number of items in the queue
        """
        return self.queue.qsize()

    @property
    def is_empty(self) -> bool:
        """
        Check if queue is empty safely.

        Uses the queue's thread-safe empty() method instead of checking length.

        Returns:
            bool: True if queue is empty, False otherwise
        """
        return self.queue.empty()

    @property
    def items_pending(self) -> int:
        """
        Number of items still in queue (enqueued - dequeued).

        This provides an alternative measure of queue depth based on the difference
        between total enqueued and dequeued items.

        Returns:
            int: Number of items currently pending in the queue
        """
        return self.total_enqueued - self.total_dequeued

    async def update_on_enqueue(self, priority: float) -> None:
        """
        Update metadata when item is enqueued.

        This method must be called whenever an item is added to the queue to maintain
        accurate priority tracking. It updates all relevant statistics atomically.

        Thread Safety:
            All compound operations (+=, max, min) are protected by an asyncio.Lock
            to prevent race conditions under concurrent access. This ensures atomic
            read-modify-write operations and prevents lost updates.

        Args:
            priority: The priority value of the item being enqueued (higher = more urgent)
        """
        async with self._metadata_lock:
            self.current_priority = priority
            self.priority_sum += priority
            self.total_enqueued += 1
            self.last_enqueue_time = datetime.now(timezone.utc)
            self.max_priority_seen = max(self.max_priority_seen, priority)
            self.min_priority_seen = min(self.min_priority_seen, priority)

    async def update_on_dequeue(self) -> None:
        """
        Update metadata when item is dequeued.

        This method must be called whenever an item is removed from the queue to maintain
        accurate tracking of queue activity.

        Thread Safety:
            All compound operations (+=) are protected by an asyncio.Lock to prevent
            race conditions under concurrent access. This ensures atomic read-modify-write
            operations and prevents lost updates.
        """
        async with self._metadata_lock:
            self.total_dequeued += 1
            self.last_dequeue_time = datetime.now(timezone.utc)

    def get_priority_for_scheduling(self) -> float:
        """
        Get priority value for scheduling decisions.

        This is the primary method that scheduling strategies should use to obtain
        priority information. It returns the average priority, which represents the
        typical urgency of items in this queue without requiring unsafe queue access.

        Returns:
            float: Average priority value for scheduling (higher = more urgent)
        """
        return self.avg_priority


@dataclass
class ScheduleResult:
    """
    Result from the intelligent scheduler.

    Contains the result of a scheduling decision, including which request
    should be processed next and any wait time if needed.

    Attributes:
        request: The queued request to process, or None if no request available
        wait_time: Suggested wait time in seconds before retrying
        should_retry: Whether the caller should retry scheduling later
    """

    request: QueuedRequest | None = None
    wait_time: float = 0.0
    should_retry: bool = True


__all__ = [
    "QueueInfo",
    "QueuedRequest",
    "ScheduleResult",
]
