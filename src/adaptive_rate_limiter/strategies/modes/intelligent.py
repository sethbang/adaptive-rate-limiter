# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
Intelligent mode scheduling strategy for the adaptive rate limiter.

This module implements the INTELLIGENT mode strategy which provides advanced
queuing with sophisticated rate limiting. It uses bucket-based queuing,
intelligent scheduling algorithms, and comprehensive rate limit management.

Key features:
- Multi-queue management (one queue per bucket/model)
- Rate limit capacity tracking and reservation
- Streaming response wrapping for accurate token accounting
- Background cleanup tasks for stale reservations and hung streams
- Event-driven scheduling with rate limit reset watchers
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
)

if TYPE_CHECKING:
    from ...protocols.client import ClientProtocol
    from ...scheduler.config import RateLimiterConfig
    from ...scheduler.state import StateManager

import contextlib

from ...exceptions import (
    RateLimiterError,
)
from ...observability.metrics import StreamingMetrics
from ...protocols.classifier import ClassifierProtocol
from ...providers.base import ProviderInterface
from ...reservation.context import ReservationContext
from ...reservation.tracker import ReservationTracker
from ...strategies.scheduling import BaseSchedulingStrategy, create_strategy
from ...types.queue import QueuedRequest, QueueInfo, ScheduleResult
from ...types.rate_limit import RateLimitBucket
from ...types.request import RequestMetadata
from .base import BaseSchedulingModeStrategy
from .reset_watcher import RateLimitResetWatcher
from .streaming_cleanup import StreamingCleanupManager
from .streaming_handler import StreamingHandler

logger = logging.getLogger(__name__)


@dataclass
class ReservationMetrics:
    """
    Metrics for reservation tracking and fallback behavior.

    Provides comprehensive observability for:
    - Reservation lifecycle (stale cleanups, emergency cleanups)
    - Fallback behavior (backpressure rejections, update fallbacks)
    - Streaming detection (for prioritization)
    """

    # Counters by bucket
    stale_cleanups: dict[str, int] = field(default_factory=dict)
    emergency_cleanups: int = 0
    backpressure_rejections: dict[str, int] = field(default_factory=dict)
    update_fallbacks: dict[str, dict[str, int]] = field(
        default_factory=dict
    )  # bucket -> reason -> count

    # Streaming detection metrics
    streaming_requests: int = 0
    non_streaming_requests: int = 0

    def record_stale_cleanup(self, bucket_id: str) -> None:
        self.stale_cleanups[bucket_id] = self.stale_cleanups.get(bucket_id, 0) + 1

    def record_emergency_cleanup(self) -> None:
        self.emergency_cleanups += 1

    def record_backpressure_rejection(self, bucket_id: str) -> None:
        self.backpressure_rejections[bucket_id] = (
            self.backpressure_rejections.get(bucket_id, 0) + 1
        )

    def record_update_fallback(self, bucket_id: str | None, reason: str) -> None:
        bucket_key = bucket_id or "unknown"
        if bucket_key not in self.update_fallbacks:
            self.update_fallbacks[bucket_key] = {}
        self.update_fallbacks[bucket_key][reason] = (
            self.update_fallbacks[bucket_key].get(reason, 0) + 1
        )

    def record_request_type(self, is_streaming: bool) -> None:
        """Track streaming vs non-streaming for prioritization."""
        if is_streaming:
            self.streaming_requests += 1
        else:
            self.non_streaming_requests += 1

    def get_streaming_ratio(self) -> float:
        """Returns streaming request ratio (0.0-1.0)."""
        total = self.streaming_requests + self.non_streaming_requests
        if total == 0:
            return 0.0
        return self.streaming_requests / total

    def get_stats(self) -> dict[str, Any]:
        return {
            "stale_cleanups": dict(self.stale_cleanups),
            "emergency_cleanups": self.emergency_cleanups,
            "backpressure_rejections": dict(self.backpressure_rejections),
            "update_fallbacks": {k: dict(v) for k, v in self.update_fallbacks.items()},
            "streaming_requests": self.streaming_requests,
            "non_streaming_requests": self.non_streaming_requests,
            "streaming_ratio": self.get_streaming_ratio(),
        }


# NOTE: StreamingMetrics is imported from observability.metrics
# The canonical implementation lives there; no duplicate needed here.


class IntelligentModeStrategy(BaseSchedulingModeStrategy):
    """
    INTELLIGENT mode strategy: Advanced queuing with sophisticated rate limiting.

    Uses bucket-based queuing, intelligent scheduling algorithms, and comprehensive
    rate limit management with state tracking.

    This strategy is ideal for:
    - High-volume production environments
    - Multi-tenant systems with rate limiting
    - Applications requiring fair scheduling across models

    Key Features:
    - Multi-queue management (one queue per bucket/model)
    - Atomic capacity check and reservation
    - Streaming response wrapping with refund-based accounting
    - Background cleanup for stale reservations and hung streams
    - Event-driven scheduling with rate limit reset watchers

    Attributes:
        MAX_RESERVATION_AGE: Maximum age in seconds before a reservation is stale
        MAX_RESERVATIONS: Maximum number of reservations to track
        STALE_CLEANUP_INTERVAL: Interval in seconds between cleanup runs
    """

    # Reservation tracking configuration (ClassVar allows test modification)
    # CRITICAL: Must be < max_request_timeout (300s) to prevent double-release race
    MAX_RESERVATION_AGE: ClassVar[int] = (
        240  # 4 minutes - SHORTER than Redis orphan recovery (5 min)
    )
    MAX_RESERVATIONS: ClassVar[int] = 10000  # Prevent unbounded memory growth
    STALE_CLEANUP_INTERVAL: ClassVar[int] = 60  # Run cleanup every minute

    def __init__(
        self,
        scheduler: Any,  # BaseScheduler - kept as Any to avoid circular imports
        config: RateLimiterConfig,
        client: ClientProtocol,
        provider: ProviderInterface,
        classifier: ClassifierProtocol,
        state_manager: StateManager,
        scheduling_strategy: BaseSchedulingStrategy | None = None,
    ):
        """
        Initialize the IntelligentModeStrategy.

        Args:
            scheduler: Reference to the main scheduler for shared functionality.
            config: Scheduler configuration containing rate limiting parameters.
            client: Client instance following ClientProtocol for API integration.
            provider: Provider interface for bucket discovery and header parsing.
            classifier: Request classifier for metadata extraction.
            state_manager: State manager for rate limit state tracking.
            scheduling_strategy: Optional scheduling strategy (default: weighted_round_robin)
        """
        super().__init__(scheduler, config, client)

        # Required dependencies for INTELLIGENT mode
        self.provider = provider
        self.classifier = classifier
        self.state_manager = state_manager
        self.scheduling_strategy = scheduling_strategy or create_strategy(
            "weighted_round_robin"
        )

        # Advanced queuing with O(1) operations using deques
        self.fast_queues: dict[str, deque[QueuedRequest]] = defaultdict(deque)
        self.queue_info: dict[str, QueueInfo] = {}

        # Optimization buffers for batch processing
        self._batch_buffer: list[QueuedRequest | None] = [None] * 50
        self._eligible_buffer: list[str] = [""] * 100

        # Fast tracking flags for O(1) queue state checks
        self._queue_has_items: dict[str, bool] = {}
        self._models_with_capacity: set[str] = set()

        # Request reservation system for accurate rate limit tracking
        self._reserved_capacity: dict[str, dict[str, int]] = defaultdict(
            lambda: {"requests": 0, "tokens": 0}
        )

        # Per-queue locks for thread safety
        self._queue_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

        # Batch processing parameters
        self._batch_size = getattr(config, "batch_size", 50)
        self.max_from_queue = 1
        self._loop_sleep_time = getattr(config, "scheduler_interval", 0.001)

        # Safety margin for rate limits
        self._safety_margin = getattr(config, "rate_limit_buffer_ratio", 0.9)

        # Event-driven scheduling
        self._wakeup_event = asyncio.Event()
        self._wakeup_lock = asyncio.Lock()

        # Capacity check failure tracking
        self._capacity_check_failures: dict[str, int] = {}
        self.max_capacity_check_failures = 5

        # Concurrency protection
        self._active_requests = 0
        self.max_concurrent_requests = getattr(config, "max_concurrent_executions", 100)
        self._request_timeout = getattr(config, "request_timeout", 300.0)

        # Activity tracking for diagnostics
        self._last_activity_time = time.time()
        self._idle_cycles = 0
        self._last_log_time = 0

        # Task tracking system for concurrency management
        self._active_tasks: dict[str, asyncio.Task[Any]] = {}
        self._task_lock = asyncio.Lock()
        self._active_request_count = 0

        # Concurrency control with semaphore
        self._request_semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        # Task cleanup management
        self._cleanup_interval = 1.0
        self._cleanup_task: asyncio.Task[None] | None = None

        # Cold start probing
        self._bucket_probes: set[str] = set()

        # Per-bucket initialization locks to prevent cold-start race conditions
        self._bucket_init_locks: dict[str, asyncio.Lock] = {}
        self._initialized_buckets: set[str] = set()

        # ===== RESERVATION TRACKING =====
        # Reservation tracking via composition (delegates storage operations)
        self._reservation_tracker = ReservationTracker(
            max_reservations=self.MAX_RESERVATIONS,
            max_reservation_age=self.MAX_RESERVATION_AGE,
            stale_cleanup_interval=self.STALE_CLEANUP_INTERVAL,
        )

        # Metrics for observability
        self._reservation_metrics = ReservationMetrics()

        # Stale reservation cleanup task (handles backend release calls)
        self._stale_cleanup_task: asyncio.Task[None] | None = None

        # ===== STREAMING METRICS =====
        self._streaming_metrics = StreamingMetrics()

        # ===== HELPER CLASSES (Composition) =====
        # Rate limit reset watcher for proactive queue release
        self._reset_watcher = RateLimitResetWatcher(
            state_manager=state_manager,
            wakeup_callback=self._safe_set_wakeup_event,
        )

        # Streaming cleanup manager for hung stream detection
        self._streaming_cleanup_manager = StreamingCleanupManager(
            backend=state_manager.backend,
            streaming_metrics=self._streaming_metrics,
            cleanup_interval=60,  # seconds
            activity_timeout=300,  # 5 minutes
        )

        # Streaming handler for wrapping/detecting streams
        self._streaming_handler = StreamingHandler(
            reservation_tracker=self._reservation_tracker,
            backend=state_manager.backend,
            streaming_metrics=self._streaming_metrics,
            register_callback=self._streaming_cleanup_manager.register,
        )

    async def _ensure_bucket_initialized(self, bucket_id: str) -> None:
        """
        Ensure a bucket is initialized with a lock to prevent cold-start races.

        This prevents multiple concurrent requests from all seeing "empty state"
        and firing simultaneously before the first response establishes the rate limit.
        """
        if bucket_id in self._initialized_buckets:
            return  # Fast path - no lock needed

        # Use setdefault for atomic lock creation to prevent race conditions
        # where multiple coroutines might try to create a lock for the same bucket
        lock = self._bucket_init_locks.setdefault(bucket_id, asyncio.Lock())

        async with lock:
            if bucket_id in self._initialized_buckets:
                return  # Double-check pattern

            # Mark as initialized so future requests skip the lock
            self._initialized_buckets.add(bucket_id)

    async def submit_request(
        self, metadata: RequestMetadata, request_func: Callable[[], Awaitable[Any]]
    ) -> ScheduleResult:
        """
        Submit request in INTELLIGENT mode - advanced queuing.

        Creates a queued request and adds it to the appropriate queue based on
        the request's bucket assignment. The scheduler loop will process the
        queue and execute the request when capacity is available.

        Args:
            metadata: Request metadata containing routing information
            request_func: Async function to execute the request

        Returns:
            ScheduleResult containing the queued request and scheduling info

        Raises:
            QueueOverflowError: If the queue is full and overflow_policy is "reject"
        """
        # Get queue key based on bucket discovery
        queue_key = await self._get_queue_key(metadata)

        # Create queue if needed
        if queue_key not in self.fast_queues:
            await self._create_fast_queue(queue_key, metadata)

        # Create queued request
        future: asyncio.Future[Any] = asyncio.Future()
        queued_request = QueuedRequest(
            metadata=metadata,
            request_func=request_func,
            future=future,
            queue_entry_time=datetime.now(timezone.utc),
        )

        # Check and handle queue overflow
        queue = self.fast_queues[queue_key]
        queue_info = self.queue_info.get(queue_key)

        if await self._check_queue_overflow(len(queue), queue_key) and (
            getattr(self.config, "overflow_policy", "reject") == "drop_oldest" and queue
        ):
            dropped_request = queue.popleft()
            # Update metadata for dropped request
            if queue_info:
                await queue_info.update_on_dequeue()
            if not dropped_request.future.cancelled():
                dropped_request.future.set_exception(
                    RateLimiterError("Request dropped due to queue overflow")
                )

        # Add to queue
        queue.append(queued_request)
        self._queue_has_items[queue_key] = True

        # Update metadata for safe priority tracking
        if queue_info:
            priority = float(metadata.priority)
            await queue_info.update_on_enqueue(priority)

        # Signal scheduler loop (fire-and-forget with exception handling)
        task = asyncio.create_task(self._safe_set_wakeup_event())
        task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

        # Update metrics
        await self._update_metrics_for_request(metadata, "requests_scheduled")

        return ScheduleResult(request=queued_request, wait_time=0.0, should_retry=True)

    async def run_scheduling_loop(self) -> None:
        """
        Main scheduler loop for INTELLIGENT mode.

        Continuously processes eligible queues, selecting requests based on the
        scheduling strategy and executing them when capacity is available.
        """
        while self._running:
            try:
                await self._loop_intelligent_mode()

                # Small yield to prevent CPU hogging
                await asyncio.sleep(self._loop_sleep_time)

            except asyncio.CancelledError:
                logger.info("Intelligent scheduler loop cancelled")
                break
            except (AttributeError, ValueError, OSError, TypeError) as e:
                logger.exception(f"Intelligent scheduler error: {e}")
                await asyncio.sleep(0.01)

    async def start(self) -> None:
        """
        Start intelligent mode strategy.

        Initializes background tasks for cleanup, stale reservation monitoring,
        and streaming cleanup.
        """
        self._running = True
        # Start the cleanup task
        await self._start_cleanup_task()
        # Start the stale reservation cleanup task
        self._start_stale_cleanup_task()
        # Start the streaming cleanup task (via helper)
        self._streaming_cleanup_manager.start()

    async def stop(self) -> None:
        """
        Stop intelligent mode strategy.

        Cancels all background tasks and cleans up resources.
        """
        self._running = False

        # Stop reset watcher (cancels all watcher tasks)
        await self._reset_watcher.stop()

        # Clean up the cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task

        # Clean up the stale reservation cleanup task
        if self._stale_cleanup_task and not self._stale_cleanup_task.done():
            self._stale_cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stale_cleanup_task

        # Stop streaming cleanup manager (via helper)
        await self._streaming_cleanup_manager.stop()

        # Stop state manager to clean up its background tasks
        if self.state_manager:
            await self.state_manager.stop()

    def get_metrics(self) -> dict[str, Any]:
        """
        Get intelligent mode metrics.

        Returns:
            Dictionary containing comprehensive mode metrics including
            queue counts, active requests, and reservation/streaming metrics.
        """
        return {
            "mode": "intelligent",
            "total_queues": len(self.fast_queues),
            "queues_with_items": sum(
                1 for has_items in self._queue_has_items.values() if has_items
            ),
            "idle_cycles": self._idle_cycles,
            "active_requests": self._active_requests,
            "capacity_check_failures": len(self._capacity_check_failures),
            "reservation_metrics": self._reservation_metrics.get_stats(),
            "streaming_metrics": self._streaming_metrics.get_stats(),
        }

    def get_streaming_metrics(self) -> StreamingMetrics:
        """
        Get the streaming metrics instance.

        Returns:
            The StreamingMetrics instance tracking streaming request metrics.
        """
        return self._streaming_metrics

    # ===== MAIN SCHEDULING LOOP =====

    async def _loop_intelligent_mode(self) -> None:
        """Scheduler loop for INTELLIGENT mode."""
        # Update metrics
        if self.scheduler.metrics_enabled:
            self.scheduler.metrics["scheduler_loops"] += 1

        # Find eligible queues
        eligible_queues = await self._find_eligible_queues_intelligent()

        if not eligible_queues:
            # No work to do - wait for activity
            self._idle_cycles += 1

            # Log occasionally to show we are alive but idle
            if self._idle_cycles % 100 == 0:
                logger.debug(
                    f"Scheduler idle (cycles: {self._idle_cycles}). "
                    f"Queues with items: {[k for k, v in self._queue_has_items.items() if v]}"
                )

            # Calculate adaptive wait time based on rate limits
            wait_time = 1.0  # Default wait

            earliest_reset = await self._get_earliest_reset_time()
            if earliest_reset:
                now = time.time()
                if earliest_reset > now:
                    # Wait until reset, but cap at 60s to ensure responsiveness
                    wait_time = min(earliest_reset - now, 60.0)
                    # Ensure we don't spin too fast if reset is very close
                    wait_time = max(wait_time, self._loop_sleep_time)

            await self._handle_idle_state_intelligent(timeout=wait_time)
            return

        # Reset idle tracking
        self._idle_cycles = 0
        self._last_activity_time = time.time()

        # Select and process queues
        selected_queue_keys = await self._select_queues_for_processing(eligible_queues)
        await self._process_selected_queues_intelligent(selected_queue_keys)

    # ===== QUEUE MANAGEMENT =====

    async def _get_queue_key(self, metadata: RequestMetadata) -> str:
        """Get the bucket-based queue key for a given request."""
        if not self.provider or not metadata.requires_model:
            return f"default_{metadata.resource_type}"

        if not metadata.model_id or metadata.model_id == "unknown":
            return f"default_{metadata.resource_type}"

        bucket_id = await self.provider.get_bucket_for_model(
            metadata.model_id, metadata.resource_type
        )
        return f"{bucket_id}:{metadata.resource_type}"

    async def _create_fast_queue(
        self, queue_key: str, metadata: RequestMetadata
    ) -> None:
        """Create a fast queue for INTELLIGENT mode."""
        self.fast_queues[queue_key] = deque()

        # Create a placeholder priority queue for the QueueInfo
        priority_queue: asyncio.PriorityQueue[Any] = asyncio.PriorityQueue()

        self.queue_info[queue_key] = QueueInfo(
            queue_key=queue_key,
            model_id=metadata.model_id,
            resource_type=metadata.resource_type,
            queue=priority_queue,
            rate_config=RateLimitBucket(
                model_id=metadata.model_id,
                resource_type=metadata.resource_type,
                rpm_limit=1000,  # Default limit
            ),
            queue_depth=0,
        )
        self._queue_has_items[queue_key] = False

    async def _check_queue_overflow(self, queue_size: int, queue_key: str) -> bool:
        """Check if queue would overflow and handle based on policy."""
        max_queue_size = getattr(self.config, "max_queue_size", 1000)

        if queue_size >= max_queue_size:
            # Update metrics
            if self.scheduler.metrics_enabled:
                self.scheduler.metrics["queue_overflows"] += 1

            overflow_policy = getattr(self.config, "overflow_policy", "reject")
            if overflow_policy == "reject":
                raise RateLimiterError(f"Queue {queue_key} is full")
            return True  # Indicates overflow occurred
        return False

    async def _safe_set_wakeup_event(self) -> None:
        """Thread-safe method to set the wakeup event."""
        try:
            async with self._wakeup_lock:
                self._wakeup_event.set()
        except (AttributeError, RuntimeError, OSError) as e:
            logger.warning(f"Failed to set wakeup event: {e}")

    async def _update_metrics_for_request(
        self, metadata: RequestMetadata, metric_name: str
    ) -> None:
        """Update metrics for a request event."""
        if self.scheduler.metrics_enabled:
            self.scheduler.metrics[metric_name] += 1

    def _extract_bucket_id_from_queue_key(self, queue_key: str) -> str | None:
        """
        Extract bucket_id from queue_key to avoid redundant provider calls.

        Queue keys are constructed as "{bucket_id}:{resource_type}" in _get_queue_key().
        Default queues use format "default_{resource_type}" and have no bucket_id.

        Args:
            queue_key: The queue key string

        Returns:
            The bucket_id portion, or None for default queues
        """
        if queue_key.startswith("default_"):
            return None

        # Split on first ":" only - bucket_id may contain ":" characters
        parts = queue_key.split(":", 1)
        if len(parts) >= 1 and parts[0]:
            return parts[0]
        return None

    async def _find_eligible_queues_intelligent(self) -> dict[str, QueueInfo]:
        """Find eligible queues for INTELLIGENT mode."""
        eligible = {}

        # Create snapshot to prevent dict mutation during iteration
        queue_snapshot = dict(self._queue_has_items)

        for queue_key, has_items in queue_snapshot.items():
            if not has_items:
                continue

            queue_info = self.queue_info.get(queue_key)
            if not queue_info:
                logger.warning(f"Queue {queue_key} has items but no queue_info")
                continue

            # Check circuit breaker
            if (
                not self.scheduler._circuit_breaker_always_closed
                and self.scheduler.circuit_breaker
            ) and not await self.scheduler.circuit_breaker.can_execute(
                queue_info.model_id
            ):
                continue

            # Check rate limits via state manager
            if self.provider and self.state_manager:
                # Extract bucket_id from queue_key to avoid redundant provider calls
                bucket_id = self._extract_bucket_id_from_queue_key(queue_key)

                if bucket_id:
                    state = await self.state_manager.get_state(bucket_id)
                    # Skip only if we KNOW we're out of capacity
                    # If state is None (no prior requests), allow processing to bootstrap state
                    if (
                        state
                        and state.remaining_requests is not None
                        and state.remaining_requests <= 0
                    ):
                        # Check for imminent reset
                        if state.reset_at:
                            time_until_reset = (
                                state.reset_at - datetime.now(timezone.utc)
                            ).total_seconds()

                            # Threshold: 2x scheduler interval (approx 20ms)
                            imminent_threshold = self._loop_sleep_time * 2

                            if time_until_reset <= imminent_threshold:
                                # DO NOT CONTINUE - Allow to fall through and be added to eligible
                                pass
                            else:
                                # Not imminent, ensure watcher exists and skip
                                reset_ts = state.reset_at.timestamp()
                                if (
                                    bucket_id
                                    not in self._reset_watcher.buckets_waiting_for_reset
                                ):
                                    await self._reset_watcher.schedule_watcher(
                                        bucket_id, reset_ts
                                    )
                                continue
                        else:
                            # No reset time available, skip the queue
                            continue

            # Update queue depth and add to eligible
            queue = self.fast_queues.get(queue_key)
            if queue:
                queue_info.queue_depth = len(queue)
                eligible[queue_key] = queue_info

        return eligible

    async def _handle_idle_state_intelligent(self, timeout: float = 1.0) -> None:
        """
        Handle idle state for INTELLIGENT mode.

        Args:
            timeout: Maximum time to wait for wakeup event (seconds)
        """
        # Wait for wakeup event with timeout
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._wakeup_event.wait(), timeout=timeout)

        # Clear the event for next iteration
        async with self._wakeup_lock:
            self._wakeup_event.clear()

    async def _select_queues_for_processing(
        self, eligible_queues: dict[str, QueueInfo]
    ) -> list[str]:
        """Select queues for processing using scheduling strategy."""
        selected_queue_keys = []
        batch_count = 0

        eligible_list = list(eligible_queues.values())
        while eligible_list and batch_count < self._batch_size:
            selected_queue_info = await self.scheduling_strategy.select(eligible_list)

            if not selected_queue_info:
                break

            selected_queue_keys.append(selected_queue_info.queue_key)

            # Remove from eligible list to avoid re-selection
            eligible_list = [
                q for q in eligible_list if q.queue_key != selected_queue_info.queue_key
            ]
            batch_count += 1

        return selected_queue_keys

    async def _process_selected_queues_intelligent(
        self, selected_queue_keys: list[str]
    ) -> None:
        """Process requests from selected queues in INTELLIGENT mode."""
        for queue_key in selected_queue_keys:
            await self._process_single_queue_intelligent(queue_key)

    async def _process_single_queue_intelligent(self, queue_key: str) -> bool:
        """Process requests from a single queue in INTELLIGENT mode."""
        async with self._queue_locks[queue_key]:
            queue = self.fast_queues.get(queue_key)

            if not queue or len(queue) == 0:
                self._queue_has_items[queue_key] = False
                return False

            # Process up to max_from_queue requests
            requests_processed = 0
            max_from_queue = min(3, self._batch_size)

            while queue and requests_processed < max_from_queue:
                if not await self._try_process_next_request_intelligent(
                    queue, queue_key
                ):
                    break
                requests_processed += 1

            # Update queue status
            if not queue or len(queue) == 0:
                self._queue_has_items[queue_key] = False
            else:
                self._queue_has_items[queue_key] = True

        return requests_processed > 0

    async def _try_process_next_request_intelligent(
        self, queue: deque[QueuedRequest], queue_key: str
    ) -> bool:
        """Try to process the next request from a queue in INTELLIGENT mode."""
        # Check if we have capacity BEFORE dequeuing
        if not await self._can_accept_request():
            return False

        if not queue:
            return False

        request = queue[0]  # Peek at first request

        # Cold Start Probing Logic
        # Extract bucket_id from queue_key to avoid redundant provider calls
        bucket_id = self._extract_bucket_id_from_queue_key(queue_key)

        if bucket_id and self.state_manager:
            state = await self.state_manager.get_state(bucket_id)
            # If state is unverified (never updated from headers), enforce probing
            if state and not getattr(state, "is_verified", True):
                if bucket_id in self._bucket_probes:
                    # Probe already active, wait
                    return False
                else:
                    # Start probe
                    self._bucket_probes.add(bucket_id)
                    logger.info(f"Starting cold start probe for bucket {bucket_id}")
                    # Proceed to reserve capacity for the probe

        if not await self._check_and_reserve_capacity_intelligent(
            request.metadata, schedule_watcher=False, bucket_id=bucket_id
        ):
            # If reservation failed, we must clear the probe flag if we set it
            if bucket_id and bucket_id in self._bucket_probes:
                self._bucket_probes.discard(bucket_id)

            logger.debug(
                f"Capacity check failed for request {request.metadata.request_id}"
            )
            # Failure Path: Analyze why and potentially retry
            if self.provider and self.state_manager:
                if bucket_id:
                    logger.debug(
                        f"Capacity check failed for {bucket_id}, attempting state refresh"
                    )

                    # 1. Force Refresh State
                    current_state = await self.state_manager.get_state(
                        bucket_id, force_refresh=True
                    )

                    # 2. Check if we should retry
                    if (
                        current_state
                        and current_state.remaining_requests is not None
                        and current_state.remaining_requests > 0
                    ):
                        logger.debug(
                            f"State shows capacity ({current_state.remaining_requests}), retrying reservation"
                        )
                        if await self._check_and_reserve_capacity_intelligent(
                            request.metadata,
                            schedule_watcher=False,
                            bucket_id=bucket_id,
                        ):
                            # Retry successful! Proceed to process request
                            pass
                        else:
                            # Retry failed - Schedule Watcher
                            if current_state.reset_at:
                                reset_ts = current_state.reset_at.timestamp()
                                await self._reset_watcher.schedule_watcher(
                                    bucket_id, reset_ts
                                )
                            return False
                    else:
                        # Still failed or no capacity - Schedule Watcher
                        if current_state and current_state.reset_at:
                            reset_ts = current_state.reset_at.timestamp()
                            await self._reset_watcher.schedule_watcher(
                                bucket_id, reset_ts
                            )
                        return False
                else:
                    return False
            else:
                return False

        # Remove request from queue
        request_to_process = queue.popleft()

        # Update metadata for safe priority tracking
        queue_info = self.queue_info.get(queue_key)
        if queue_info:
            await queue_info.update_on_dequeue()

        # Create tracked task with proper resource management
        task_id = f"{queue_key}:{request_to_process.metadata.request_id}"
        task = asyncio.create_task(
            self._execute_request_with_tracking(request_to_process, task_id, bucket_id)
        )

        # Track the task
        async with self._task_lock:
            self._active_tasks[task_id] = task
            self._active_request_count += 1

        return True

    # ===== CAPACITY CHECKING AND RESERVATION =====

    async def _check_and_reserve_capacity_intelligent(
        self,
        metadata: RequestMetadata,
        schedule_watcher: bool = True,
        bucket_id: str | None = None,
    ) -> bool:
        """
        Atomically check and reserve capacity for INTELLIGENT mode.

        Uses atomic backend operations to prevent TOCTOU race conditions.

        Args:
            metadata: Request metadata
            schedule_watcher: Whether to schedule a reset watcher on failure
            bucket_id: Optional pre-computed bucket_id to avoid redundant provider calls

        Returns:
            True if capacity was reserved, False otherwise
        """
        if not self.state_manager or not self.provider:
            return True  # No state manager or provider available

        # Use provided bucket_id or discover it (fallback for callers that don't have it)
        if bucket_id is None:
            bucket_id = await self.provider.get_bucket_for_model(
                metadata.model_id, metadata.resource_type
            )
        if not bucket_id:
            return True  # No bucket information available

        # Get bucket limits for atomic operation
        buckets = await self.provider.discover_limits()
        bucket = buckets.get(bucket_id)
        if not bucket:
            return True  # No bucket info available

        bucket_limits: dict[str, int] = {
            "rpm_limit": bucket.rpm_limit or 1000,  # Default if None
            "tpm_limit": bucket.tpm_limit or 10000,  # Default if None
        }

        # Estimate tokens for this request
        estimated_tokens = metadata.estimated_tokens or 0

        # Ensure bucket is initialized (Cold Start Protection)
        await self._ensure_bucket_initialized(bucket_id)

        # ATOMIC OPERATION - No race condition!
        (
            success,
            reservation_id,
        ) = await self.state_manager.backend.check_and_reserve_capacity(
            bucket_id,
            requests=1,
            tokens=estimated_tokens,
            bucket_limits=bucket_limits,
            safety_margin=self._safety_margin,
            request_id=metadata.request_id,
        )

        if success and reservation_id:
            # Store reservation context
            await self._store_reservation_context(
                metadata.request_id, bucket_id, reservation_id, estimated_tokens
            )

        if not success and schedule_watcher:
            # Schedule a watcher to wake up when capacity resets
            state = await self.state_manager.get_state(bucket_id)
            if state and state.reset_at:
                reset_ts = state.reset_at.timestamp()
                if bucket_id not in self._reset_watcher.buckets_waiting_for_reset:
                    await self._reset_watcher.schedule_watcher(bucket_id, reset_ts)

        return success

    # ===== RATE LIMIT STATE MANAGEMENT =====

    async def _update_rate_limit_state(
        self,
        metadata: RequestMetadata,
        result: Any | None = None,
        status_code: int | None = None,
        headers: dict[str, str] | None = None,
        bucket_id_override: str | None = None,
        clear_all_reservations: bool = False,
    ) -> None:
        """
        Update rate limit state from response.

        Supports distributed backend requirements including:
        - Reservation clearing (specific bucket or all for exception paths)
        - Header assessment and validation
        - Fallback handling when headers are insufficient

        Args:
            metadata: Request metadata (used for request_id lookup)
            result: API response (may be None for error cases)
            status_code: HTTP status code for script routing
            headers: Override headers (for exceptions with cached headers)
            bucket_id_override: Explicit bucket_id to use (skips discovery)
            clear_all_reservations: If True, clears ALL reservations for this request_id
        """
        if not self.state_manager:
            return

        # Determine bucket_id from override, or discovery
        bucket_id = bucket_id_override
        if bucket_id is None and self.provider:
            bucket_id = await self.provider.get_bucket_for_model(
                metadata.model_id, metadata.resource_type
            )

        # Clear reservations based on mode
        cleared_reservations: list[ReservationContext] = []
        if clear_all_reservations:
            # Exception path: clear ALL reservations for this request
            cleared_reservations = await self._clear_all_reservations_for_request(
                metadata.request_id
            )
            if len(cleared_reservations) > 1:
                logger.info(
                    f"Cleared {len(cleared_reservations)} reservations for request {metadata.request_id}"
                )
        else:
            # Success path: clear only the specific bucket's reservation
            reservation = await self._get_and_clear_reservation(
                metadata.request_id, bucket_id
            )
            if reservation:
                cleared_reservations = [reservation]

        # Use provided headers or extract from result
        if headers is None:
            headers = self.scheduler.extract_response_headers(result) if result else {}

        # Ensure headers is a dict
        safe_headers = headers or {}

        # Normalize headers to lowercase for case-insensitive matching
        headers = {k.lower(): v for k, v in safe_headers.items()}

        # Normalize reset headers (handle duration strings)
        for key in ["x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"]:
            if key in headers:
                val = headers[key]
                parsed = self._parse_duration_string(val)
                if parsed is not None:
                    headers[key] = str(parsed)

        # Determine header availability with value validation
        header_status = self._assess_header_availability(headers)

        if header_status == "full":
            # Full state sync path - all 6 headers present with valid values
            update_result = await self.state_manager.update_state_from_headers(
                metadata.model_id,
                metadata.resource_type,
                headers,
                request_id=metadata.request_id,
                status_code=status_code,
            )

            # Check result - if 0, Lua script rejected headers or mapping not found
            if update_result == 0:
                logger.warning(
                    f"update_rate_limits returned 0 for request {metadata.request_id}. "
                    f"Falling back to release_reservation for any cleared reservations."
                )
                self._reservation_metrics.record_update_fallback(
                    bucket_id, "lua_returned_zero"
                )

                # Release each cleared reservation as fallback
                for ctx in cleared_reservations:
                    try:
                        await asyncio.shield(
                            self.state_manager.backend.release_reservation(
                                ctx.bucket_id, ctx.reservation_id
                            )
                        )
                    except Exception as e:
                        logger.warning(
                            f"Fallback release failed for {ctx.reservation_id}: {e}"
                        )
            else:
                logger.debug(
                    f"Updated rate limit state for {bucket_id or metadata.model_id} from headers"
                )
        else:
            # Partial or no headers - use release-only mode
            reason = "partial_headers" if header_status == "partial" else "no_headers"
            self._reservation_metrics.record_update_fallback(bucket_id, reason)

            for ctx in cleared_reservations:
                logger.debug(
                    f"Release-only for request {metadata.request_id} ({reason}). "
                    f"Releasing reservation {ctx.reservation_id}."
                )
                try:
                    await asyncio.shield(
                        self.state_manager.backend.release_reservation(
                            ctx.bucket_id, ctx.reservation_id
                        )
                    )
                except Exception as e:
                    logger.warning(f"Release failed for {ctx.reservation_id}: {e}")

    def _parse_duration_string(self, value: str) -> float | None:
        """
        Parse a duration string into seconds.

        Handles formats like '500ms', '2s', '1m30s'.
        Returns None if parsing fails.
        """
        if not value:
            return None

        # First try direct float conversion
        try:
            return float(value)
        except (ValueError, TypeError):
            pass

        total_seconds = 0.0
        found_match = False

        # Regex to match number followed by unit
        # Units: ms (milliseconds), s (seconds), m (minutes), h (hours), d (days)
        pattern = r"(\d+(?:\.\d+)?)(ms|s|m|h|d)"

        matches = re.findall(pattern, value)
        if not matches:
            return None

        for amount_str, unit in matches:
            found_match = True
            try:
                amount = float(amount_str)
                if unit == "ms":
                    total_seconds += amount / 1000.0
                elif unit == "s":
                    total_seconds += amount
                elif unit == "m":
                    total_seconds += amount * 60.0
                elif unit == "h":
                    total_seconds += amount * 3600.0
                elif unit == "d":
                    total_seconds += amount * 86400.0
            except ValueError:
                continue

        return total_seconds if found_match else None

    def _assess_header_availability(self, headers: dict[str, str]) -> str:
        """
        Assess header availability for state sync.

        Validates both presence AND basic value sanity.

        Returns:
            "full" - All 6 required headers present with parseable values
            "partial" - Some headers present but not all 6 or some invalid
            "none" - No usable rate limit headers
        """
        full_set = [
            "x-ratelimit-remaining-requests",
            "x-ratelimit-remaining-tokens",
            "x-ratelimit-limit-requests",
            "x-ratelimit-limit-tokens",
            "x-ratelimit-reset-requests",
            "x-ratelimit-reset-tokens",
        ]

        valid_count = 0
        for h in full_set:
            value = headers.get(h)
            if value:
                # Basic sanity check - must be parseable as number
                try:
                    float(value)
                    valid_count += 1
                except (ValueError, TypeError):
                    pass  # Invalid value doesn't count

        if valid_count == 6:
            return "full"
        elif valid_count > 0:
            return "partial"
        else:
            return "none"

    # ===== REQUEST EXECUTION =====

    async def _can_accept_request(self) -> bool:
        """Check if we can accept a new request."""
        async with self._task_lock:
            return self._active_request_count < self.max_concurrent_requests

    async def _execute_request_with_tracking(
        self, request: QueuedRequest, task_id: str, bucket_id: str | None = None
    ) -> None:
        """
        Execute request with guaranteed cleanup and streaming detection.

        Uses asyncio.shield for all cleanup operations to ensure completion even
        during task cancellation. Handles all exception types with proper capacity
        release and state synchronization.

        For streaming responses, wraps the result before setting on the Future
        and delegates capacity release to the iterator wrapper.
        """
        try:
            async with self._request_semaphore:
                # Execute with timeout
                result = await asyncio.wait_for(
                    request.request_func(), timeout=self._request_timeout
                )

                # Detect and wrap streaming BEFORE set_result()
                is_streaming = self._streaming_handler.detect_streaming_response(result)
                self._reservation_metrics.record_request_type(is_streaming)

                if is_streaming and bucket_id:
                    # Get reservation context (don't clear - iterator owns it)
                    reservation = await self._streaming_handler.get_reservation(
                        request.metadata.request_id, bucket_id
                    )

                    if reservation:
                        # Wrap the result BEFORE setting on Future
                        result = await self._streaming_handler.wrap_streaming_response(
                            result, reservation, request.metadata
                        )
                        # The iterator wrapper owns the reservation and will release

                        # Set result ONCE with wrapped stream
                        if not request.future.cancelled():
                            request.future.set_result(result)

                        await self._update_metrics_for_request(
                            request.metadata, "requests_completed"
                        )
                        return  # Skip non-streaming release path

                # Non-streaming path: set result and release
                if not request.future.cancelled():
                    request.future.set_result(result)

                # Update success metrics
                await self._update_metrics_for_request(
                    request.metadata, "requests_completed"
                )

                # Non-streaming: standard release with header sync
                await self._update_rate_limit_state(
                    request.metadata,
                    result,
                    status_code=200,
                    bucket_id_override=bucket_id,
                    clear_all_reservations=False,
                )

        except asyncio.TimeoutError:
            # Timeout: release ALL reservations with shield
            try:
                await asyncio.shield(
                    self._update_rate_limit_state(
                        request.metadata,
                        None,
                        status_code=None,
                        bucket_id_override=bucket_id,
                        clear_all_reservations=True,
                    )
                )
            except Exception as e:
                logger.warning(f"Cleanup failed during timeout handling: {e}")

            if not request.future.cancelled():
                request.future.set_exception(
                    RateLimiterError(
                        f"Request timed out after {self._request_timeout}s"
                    )
                )
            await self._update_metrics_for_request(request.metadata, "requests_failed")

        except asyncio.CancelledError:
            # Cancellation: release ALL reservations with shield BEFORE re-raising
            try:
                await asyncio.shield(
                    self._update_rate_limit_state(
                        request.metadata,
                        None,
                        status_code=None,
                        bucket_id_override=bucket_id,
                        clear_all_reservations=True,
                    )
                )
            except Exception as e:
                logger.warning(f"Cleanup failed during cancellation handling: {e}")
            raise  # Always re-raise for graceful shutdown

        except Exception as e:
            # Check for rate limit errors
            is_rate_limit = self._is_rate_limit_error(e)

            if is_rate_limit:
                # 429 path: use cached headers from exception
                exc_headers = getattr(e, "cached_rate_limit_headers", {})

                try:
                    await asyncio.shield(
                        self._update_rate_limit_state(
                            request.metadata,
                            None,
                            status_code=429,
                            headers=exc_headers,
                            bucket_id_override=bucket_id,
                            clear_all_reservations=True,
                        )
                    )
                except Exception as cleanup_e:
                    logger.warning(f"Cleanup failed during 429 handling: {cleanup_e}")
            else:
                # Generic error path: release ALL reservations
                try:
                    await asyncio.shield(
                        self._update_rate_limit_state(
                            request.metadata,
                            None,
                            status_code=getattr(e, "status_code", None),
                            bucket_id_override=bucket_id,
                            clear_all_reservations=True,
                        )
                    )
                except Exception as cleanup_e:
                    logger.warning(
                        f"Cleanup failed during exception handling: {cleanup_e}"
                    )

            if not request.future.cancelled():
                request.future.set_exception(e)
            await self._update_metrics_for_request(request.metadata, "requests_failed")

        finally:
            # Clear probe flag if this was a probe
            if bucket_id and bucket_id in self._bucket_probes:
                self._bucket_probes.discard(bucket_id)
                logger.debug(f"Cold start probe finished for bucket {bucket_id}")
                # Wake up scheduler to process queued requests (fire-and-forget with exception handling)
                task = asyncio.create_task(self._safe_set_wakeup_event())
                task.add_done_callback(
                    lambda t: t.exception() if not t.cancelled() else None
                )

            # Always clean up tracking
            async with self._task_lock:
                self._active_tasks.pop(task_id, None)
                self._active_request_count = max(0, self._active_request_count - 1)

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if an exception is a rate limit error."""
        error_name = type(error).__name__.lower()
        if "ratelimit" in error_name:
            return True

        status_code = getattr(error, "status_code", None)
        return status_code == 429

    # ===== BACKGROUND TASKS =====

    async def _start_cleanup_task(self) -> None:
        """Start the background cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self) -> None:
        """Periodically clean up completed tasks."""
        while self._running:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_completed_tasks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")

    async def _cleanup_completed_tasks(self) -> None:
        """Remove completed tasks from tracking."""
        async with self._task_lock:
            completed = [
                task_id for task_id, task in self._active_tasks.items() if task.done()
            ]

            for task_id in completed:
                self._active_tasks.pop(task_id, None)

    # ===== RESERVATION TRACKING =====

    def _start_stale_cleanup_task(self) -> None:
        """Start the periodic stale reservation cleanup task."""
        if self._stale_cleanup_task is None or self._stale_cleanup_task.done():
            self._stale_cleanup_task = asyncio.create_task(
                self._cleanup_stale_reservations_loop(),
                name="stale_reservation_cleanup",
            )

    async def _cleanup_stale_reservations_loop(self) -> None:
        """Background task to clean up stale reservations."""
        while self._running:
            try:
                await asyncio.sleep(self.STALE_CLEANUP_INTERVAL)
                await self._cleanup_stale_reservations()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Stale reservation cleanup error: {e}")

    async def _cleanup_stale_reservations(self) -> int:
        """
        Clean up reservations older than MAX_RESERVATION_AGE.

        This method handles backend release calls for stale reservations.
        The ReservationTracker provides storage operations via its
        get_and_clear_stale() method, keeping internal details encapsulated.

        Returns:
            Number of reservations cleaned up.
        """
        cutoff = time.time() - self.MAX_RESERVATION_AGE

        # Use encapsulated API to get and clear stale reservations
        stale_contexts = await self._reservation_tracker.get_and_clear_stale(cutoff)

        # Release OUTSIDE lock to avoid holding lock during backend calls
        for ctx in stale_contexts:
            age = time.time() - ctx.created_at
            logger.warning(
                f"Cleaned up stale reservation {ctx.reservation_id} "
                f"(age: {age:.1f}s, bucket: {ctx.bucket_id})"
            )
            self._reservation_metrics.record_stale_cleanup(ctx.bucket_id)

            # Best-effort release to backend
            try:
                await asyncio.shield(
                    self.state_manager.backend.release_reservation(
                        ctx.bucket_id, ctx.reservation_id
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(
                    f"Failed to release stale reservation {ctx.reservation_id}: {e}"
                )

        return len(stale_contexts)

    async def _store_reservation_context(
        self,
        request_id: str,
        bucket_id: str,
        reservation_id: str,
        estimated_tokens: int,
    ) -> None:
        """
        Store reservation context with backpressure handling.

        Delegates to ReservationTracker.store().
        """
        await self._reservation_tracker.store(
            request_id, bucket_id, reservation_id, estimated_tokens
        )

    async def _get_and_clear_reservation(
        self, request_id: str, bucket_id: str | None = None
    ) -> ReservationContext | None:
        """
        Atomically retrieve and remove reservation context.

        This prevents double-release scenarios where error handling and
        success paths both try to release the same reservation.

        Delegates to ReservationTracker.get_and_clear().
        """
        return await self._reservation_tracker.get_and_clear(request_id, bucket_id)

    async def _clear_all_reservations_for_request(
        self, request_id: str
    ) -> list[ReservationContext]:
        """
        Clear ALL reservations for a request_id using secondary index.

        Returns list of all cleared reservations for logging/debugging.

        Delegates to ReservationTracker.clear_all_for_request().
        """
        return await self._reservation_tracker.clear_all_for_request(request_id)

    async def _get_earliest_reset_time(self) -> float | None:
        """Get the earliest rate limit reset time across all buckets."""
        return await self._reset_watcher.get_earliest_reset_time()


__all__ = [
    "IntelligentModeStrategy",
    "ReservationMetrics",
    # StreamingMetrics is now imported from observability.metrics
]
