# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
Base scheduling mode strategy for the adaptive rate limiter.

This module defines the abstract base class that all scheduling mode strategies
(BASIC, INTELLIGENT, ACCOUNT) must implement. It uses the Strategy Pattern to
separate scheduling logic into independent, testable classes.
"""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...protocols.client import ClientProtocol
    from ...scheduler.config import RateLimiterConfig

from ...types.request import RequestMetadata


class BaseSchedulingModeStrategy(ABC):
    """
    Abstract base class for scheduler mode strategies.

    Each mode (BASIC, INTELLIGENT, ACCOUNT) implements this interface to provide
    its own request handling, scheduling logic, and metrics collection.

    This replaces the mode-checking if/elif blocks in a monolithic Scheduler
    with a cleaner Strategy Pattern implementation.

    Attributes:
        scheduler: Reference to the main scheduler for shared functionality
        config: Scheduler configuration
        client: Client instance for API calls (follows ClientProtocol)
        _running: Flag indicating if the strategy is currently active

    Example:
        >>> class CustomModeStrategy(BaseSchedulingModeStrategy):
        ...     async def submit_request(self, metadata, request_func):
        ...         # Custom request handling logic
        ...         return await request_func()
        ...
        ...     async def run_scheduling_loop(self):
        ...         while self._running:
        ...             # Custom scheduling logic
        ...             await asyncio.sleep(0.01)
        ...
        ...     async def start(self):
        ...         self._running = True
        ...
        ...     async def stop(self):
        ...         self._running = False
        ...
        ...     def get_metrics(self) -> Dict[str, Any]:
        ...         return {"mode": "custom"}
    """

    def __init__(
        self,
        scheduler: Any,  # BaseScheduler - kept as Any to avoid circular imports
        config: "RateLimiterConfig",
        client: "ClientProtocol",
    ):
        """
        Initialize the scheduling mode strategy.

        Args:
            scheduler: Reference to the main scheduler for shared functionality.
                This provides access to circuit breakers, metrics, and other
                scheduler-wide services.
            config: Scheduler configuration containing rate limiting parameters,
                timeouts, and operational settings.
            client: Client instance following ClientProtocol for API integration.
                Provides base URL and timeout information for request handling.
        """
        self.scheduler = scheduler
        self.config = config
        self.client = client
        self._running = False

    @abstractmethod
    async def submit_request(
        self, metadata: RequestMetadata, request_func: Callable[[], Awaitable[Any]]
    ) -> Any:
        """
        Submit a request for execution according to this mode's logic.

        This is the primary entry point for request handling. Each mode implements
        its own logic for how requests are processed:
        - BASIC mode: Direct execution with retry logic
        - INTELLIGENT mode: Advanced queuing with tier-based scheduling
        - ACCOUNT mode: Account-level concurrency management

        Args:
            metadata: Request metadata containing routing information, priority,
                estimated tokens, and other classification data.
            request_func: Async callable that executes the actual API request.
                This function is provided by the caller and should return the
                API response.

        Returns:
            Result of request execution. The type varies by mode:
            - BASIC: Direct API response
            - INTELLIGENT: ScheduleResult containing queued request info
            - ACCOUNT: ScheduleResult or direct response

        Raises:
            Exception: Mode-specific exceptions (rate limit errors, queue overflow, etc.)
        """
        pass

    @abstractmethod
    async def run_scheduling_loop(self) -> None:
        """
        Main scheduling loop for this mode.

        This method runs continuously while the strategy is active, processing
        queued requests and managing rate limiting. The implementation varies
        significantly by mode:

        - BASIC mode: May be a simple no-op or lightweight polling loop
        - INTELLIGENT mode: Active queue processing with tier-based scheduling
        - ACCOUNT mode: Account-level request processing

        The loop should:
        1. Check for available capacity
        2. Select requests for processing (if applicable)
        3. Execute requests with proper rate limiting
        4. Handle errors and update metrics

        The loop continues until stop() is called, which sets _running to False.
        """
        pass

    @abstractmethod
    async def start(self) -> None:
        """
        Start any mode-specific resources or background tasks.

        This method is called when the scheduler begins operation. Implementations
        should:
        1. Set _running to True
        2. Start any background tasks (cleanup, monitoring, etc.)
        3. Initialize any mode-specific resources

        This method should be idempotent - calling it multiple times should be safe.
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """
        Stop and cleanup mode-specific resources.

        This method is called during scheduler shutdown. Implementations should:
        1. Set _running to False to stop the scheduling loop
        2. Cancel any background tasks
        3. Clean up resources (close connections, flush buffers, etc.)
        4. Wait for in-flight operations to complete (with timeout)

        This method should be idempotent - calling it multiple times should be safe.
        """
        pass

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]:
        """
        Get mode-specific metrics.

        Returns a dictionary of metrics relevant to this mode's operation.
        Common metrics include:
        - mode: The mode name (e.g., "basic", "intelligent", "account")
        - active_requests: Number of currently executing requests
        - queue_depth: Number of queued requests (for queued modes)
        - error_counts: Breakdown of errors by type

        Returns:
            Dictionary of metric names to values. Values can be primitives,
            nested dictionaries, or lists.

        Example:
            >>> strategy.get_metrics()
            {
                "mode": "intelligent",
                "total_queues": 5,
                "active_requests": 12,
                "idle_cycles": 0
            }
        """
        pass

    async def schedule_rate_limit_reset(  # noqa: B027
        self, bucket_id: str, reset_timestamp: int
    ) -> None:
        """
        Schedule a wake-up notification for when a bucket's rate limit resets.

        This is an optional method that can be overridden by mode strategies
        that support proactive rate limit reset watching (e.g., IntelligentModeStrategy).
        The default implementation is a no-op.

        This method is called by the scheduler when a 429 error is encountered,
        allowing the strategy to proactively wake up when rate limits reset
        rather than relying on polling.

        Args:
            bucket_id: The bucket identifier to watch
            reset_timestamp: Unix timestamp when the rate limit resets
        """
        pass

    def is_running(self) -> bool:
        """
        Check if this strategy is running.

        This method provides a thread-safe way to check the current state
        of the strategy without accessing the _running attribute directly.

        Returns:
            True if the strategy is currently running, False otherwise.
        """
        return self._running


__all__ = ["BaseSchedulingModeStrategy"]
