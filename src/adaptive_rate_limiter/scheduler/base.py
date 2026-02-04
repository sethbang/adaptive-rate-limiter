# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
Unified BaseScheduler abstract class for Adaptive Rate Limiter.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
)

from typing_extensions import Self

from ..exceptions import QueueOverflowError
from ..observability.collector import UnifiedMetricsCollector, get_metrics_collector
from ..observability.constants import (
    QUEUE_OVERFLOWS_TOTAL,
    legacy_to_prometheus_name,
)
from ..protocols.classifier import ClassifierProtocol
from ..protocols.client import ClientProtocol
from ..providers.base import ProviderInterface
from ..strategies.modes import create_mode_strategy
from ..types.request import RequestMetadata

if TYPE_CHECKING:
    from ..strategies.scheduling import BaseSchedulingStrategy
    from .config import RateLimiterConfig
    from .state import StateManager

logger = logging.getLogger(__name__)

# Legacy metric name constants (for backward compatibility)
METRIC_REQUESTS_SCHEDULED = "requests_scheduled"
METRIC_REQUESTS_COMPLETED = "requests_completed"
METRIC_REQUESTS_FAILED = "requests_failed"
METRIC_QUEUE_OVERFLOWS = "queue_overflows"
METRIC_SCHEDULER_LOOPS = "scheduler_loops"
METRIC_CIRCUIT_BREAKER_REJECTIONS = "circuit_breaker_rejections"
METRIC_REQUEST_TIMEOUTS = "request_timeouts"
BACKOFF_JITTER_FACTOR = 0.1


@dataclass
class FailedRequestCounter:
    """Tracks failed requests for the 20/30s limit."""

    count: int = 0
    window_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    max_failures: int = 20
    window_seconds: int = 30

    def increment(self) -> int:
        """Increment failed request count and return current count."""
        now = datetime.now(timezone.utc)

        # Reset window if expired
        if (now - self.window_start).total_seconds() > self.window_seconds:
            self.count = 0
            self.window_start = now

        self.count += 1
        return self.count

    def is_limit_exceeded(self) -> bool:
        """Check if failed request limit is exceeded."""
        now = datetime.now(timezone.utc)

        # Reset window if expired
        if (now - self.window_start).total_seconds() > self.window_seconds:
            self.count = 0
            self.window_start = now
            return False

        return self.count >= self.max_failures


class SchedulingStrategy(Enum):
    """Available scheduling strategies."""

    FIFO = "fifo"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    PRIORITY = "priority"
    FAIR_SHARE = "fair_share"


class BaseScheduler(ABC):
    """
    An abstract base class that defines the common interface and functionality
    for all scheduler implementations.

    This class provides a foundation for different scheduling strategies by
    offering a set of common features, including:
    - Integration with the unified configuration system.
    - Mode strategy management (Basic, Intelligent, Account).
    - A metrics collection system for monitoring performance.
    - Helper methods for parsing rate limit headers and calculating backoff
      delays.

    Subclasses are required to implement the `submit_request` method to
    define their specific request handling logic.
    """

    def __init__(
        self,
        client: ClientProtocol,
        config: "RateLimiterConfig",
        provider: ProviderInterface | None = None,
        classifier: ClassifierProtocol | None = None,
        state_manager: Optional["StateManager"] = None,
        scheduling_strategy: Optional["BaseSchedulingStrategy"] = None,
        test_rate_multiplier: float = 1.0,
        metrics_enabled: bool = False,
        default_rate_limits: dict[str, Any] | None = None,
    ):
        """
        Initialize the unified BaseScheduler.

        Args:
            client: Async client for API calls
            config: Unified scheduler configuration
            provider: Service for discovering rate limit buckets (optional for basic mode)
            classifier: Request classifier for routing (optional for basic mode)
            state_manager: Manager for rate limit state (optional for basic mode)
            scheduling_strategy: Optional custom scheduling strategy
            test_rate_multiplier: Rate multiplier for testing scenarios
            metrics_enabled: Whether to enable metrics collection
            default_rate_limits: Default rate limits for models
        """
        # Core dependencies
        self.client = client
        self.config = config
        self.provider = provider
        self.classifier = classifier
        self.state_manager = state_manager
        self.test_rate_multiplier = test_rate_multiplier
        self.default_rate_limits = default_rate_limits or {}

        # Setup mode strategy
        self._setup_mode_strategy(scheduling_strategy)

        # Setup execution control
        self._setup_execution_control()

        # Setup metrics
        self._setup_metrics(metrics_enabled)

        # Setup failure tracking
        self._setup_failure_tracking()

        logger.info(
            f"Initialized {self.__class__.__name__} with configuration mode={config.mode}"
        )

    def _setup_mode_strategy(
        self, scheduling_strategy: Optional["BaseSchedulingStrategy"]
    ) -> None:
        """Setup the mode strategy."""
        try:
            mode_value = (
                self.config.mode.value
                if hasattr(self.config.mode, "value")
                else str(self.config.mode)
            )
            self.mode_strategy = create_mode_strategy(
                mode=mode_value,
                scheduler=self,
                config=self.config,
                client=self.client,
                provider=self.provider,
                classifier=self.classifier,
                state_manager=self.state_manager,
                scheduling_strategy=scheduling_strategy,
            )
        except Exception as e:
            logger.error(f"Failed to create mode strategy: {e}")
            raise

    def _setup_execution_control(self) -> None:
        """Setup execution control and concurrency management."""
        max_concurrent = getattr(self.config, "max_concurrent_executions", 100)
        self.execution_semaphore = asyncio.Semaphore(max_concurrent)
        self._running = False
        self._scheduler_task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._request_counter = 0
        self._active_tasks: set[asyncio.Task[Any]] = set()
        self._shutdown_lock = asyncio.Lock()

    def _setup_metrics(self, metrics_enabled: bool) -> None:
        """Setup metrics collection if enabled."""
        self.metrics_enabled = metrics_enabled or getattr(
            self.config, "metrics_enabled", False
        )

        if self.metrics_enabled:
            self.metrics: dict[str, int] = defaultdict(int)
            self.metrics.update(
                {
                    METRIC_REQUESTS_SCHEDULED: 0,
                    METRIC_REQUESTS_COMPLETED: 0,
                    METRIC_REQUESTS_FAILED: 0,
                    METRIC_QUEUE_OVERFLOWS: 0,
                    METRIC_SCHEDULER_LOOPS: 0,
                    METRIC_CIRCUIT_BREAKER_REJECTIONS: 0,
                    METRIC_REQUEST_TIMEOUTS: 0,
                }
            )
            self.metrics_collector = self._create_metrics_collector()
        else:
            self.metrics = {}
            self.metrics_collector = None

    def _setup_failure_tracking(self) -> None:
        """Setup failure tracking for rate limiting protection."""
        failure_window = getattr(self.config, "failure_window", 30)
        max_failures = getattr(self.config, "max_failures", 20)

        self._failed_requests = FailedRequestCounter(
            max_failures=max_failures, window_seconds=failure_window
        )

    def _create_metrics_collector(self) -> UnifiedMetricsCollector | None:
        """
        Create metrics collector based on configuration.

        Returns the global UnifiedMetricsCollector singleton if metrics are enabled.
        Optionally starts the Prometheus HTTP server if configured.

        Returns:
            UnifiedMetricsCollector instance or None if metrics disabled

        Security Note:
            The default prometheus_host is "0.0.0.0" (all interfaces) for security.
            To expose metrics externally (e.g., in containerized environments), set
            config.prometheus_host = "0.0.0.0" and ensure network-level security
            controls are in place.
        """
        if not self.metrics_enabled:
            return None

        # Get Prometheus configuration from config
        prometheus_enabled = getattr(self.config, "prometheus_enabled", True)

        # Get the unified collector singleton
        collector = get_metrics_collector(enable_prometheus=prometheus_enabled)

        # Start Prometheus HTTP server if configured
        # Default to all interfaces (0.0.0.0) to match config.py default.
        prometheus_host = getattr(self.config, "prometheus_host", "0.0.0.0")  # noqa: S104 # nosec B104
        prometheus_port = getattr(self.config, "prometheus_port", 9090)

        # Only start if explicitly enabled and not already running
        if (
            prometheus_enabled
            and not collector.server_running
            and getattr(self.config, "start_prometheus_server", False)
        ):
            collector.start_http_server(prometheus_host, prometheus_port)

        return collector

    # Public interface methods
    async def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            return

        self._running = True
        self._loop = asyncio.get_running_loop()

        # Start StateManager if it exists (for WRITE_BACK batch processing)
        if self.state_manager and hasattr(self.state_manager, "start"):
            await self.state_manager.start()
            logger.debug("StateManager started")

        # Start mode strategy
        if hasattr(self.mode_strategy, "start"):
            await self.mode_strategy.start()

        # Only start scheduler loop for complex schedulers that need it
        if hasattr(self, "_scheduler_loop") and callable(self._scheduler_loop):
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())

        # Call subclass-specific startup logic
        await self._on_start()

        logger.info(f"{self.__class__.__name__} started")

    async def stop(self) -> None:
        """Stop the scheduler."""
        async with self._shutdown_lock:
            if not self._running:
                return

            self._running = False

            # Stop mode strategy
            if hasattr(self.mode_strategy, "stop"):
                await self.mode_strategy.stop()

            # Cancel scheduler task if it exists
            if self._scheduler_task:
                self._scheduler_task.cancel()
                try:
                    await self._scheduler_task
                except asyncio.CancelledError:
                    pass
                except (AttributeError, RuntimeError, OSError) as e:
                    logger.exception(f"Exception during scheduler task stop: {e}")

            # Stop StateManager if it exists (flush pending batches)
            if self.state_manager and hasattr(self.state_manager, "stop"):
                try:
                    await self.state_manager.stop()
                    logger.debug("StateManager stopped")
                except Exception as e:
                    logger.error(f"Error stopping StateManager: {e}")

            # Cancel all active tasks
            if self._active_tasks:
                for task in list(self._active_tasks):
                    if not task.done():
                        task.cancel()

                await asyncio.gather(*self._active_tasks, return_exceptions=True)

            self._active_tasks.clear()

            # Call subclass-specific cleanup logic
            await self._on_stop()

            logger.info(f"{self.__class__.__name__} stopped successfully")

    def is_running(self) -> bool:
        """Check if the scheduler is running."""
        return bool(self._running)

    async def __aenter__(self) -> Self:
        """
        Async context manager entry.

        Starts the scheduler and returns self for use in async with blocks.

        Returns:
            Self: The scheduler instance

        Example:
            async with scheduler:
                result = await scheduler.submit_request(metadata, request_func)
        """
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """
        Async context manager exit.

        Ensures the scheduler is stopped even if an exception occurred,
        guaranteeing proper cleanup of resources.

        Args:
            exc_type: Exception type if an exception was raised, None otherwise
            exc_val: Exception instance if an exception was raised, None otherwise
            exc_tb: Traceback if an exception was raised, None otherwise
        """
        await self.stop()

    @abstractmethod
    async def submit_request(
        self, metadata: RequestMetadata, request_func: Callable[[], Awaitable[Any]]
    ) -> Any:
        """
        Submit request for execution.

        Must be implemented by subclasses to handle their specific execution logic.

        Args:
            metadata: Request metadata containing routing information
            request_func: Async function to execute the request

        Returns:
            The result of the request execution
        """
        pass

    async def schedule_rate_limit_reset(
        self, bucket_id: str, reset_timestamp: int
    ) -> None:
        """
        Schedule a wake-up notification for when a bucket's rate limit resets.

        This method is called by middleware when a 429 error is encountered,
        allowing the scheduler to proactively wake up when rate limits reset
        rather than relying on polling.

        Args:
            bucket_id: The bucket identifier to watch
            reset_timestamp: Unix timestamp when the rate limit resets
        """
        # Delegate to mode strategy
        if hasattr(self.mode_strategy, "schedule_rate_limit_reset"):
            # Use Any cast to avoid static type checking errors since method is dynamic
            await self.mode_strategy.schedule_rate_limit_reset(
                bucket_id, reset_timestamp
            )
        else:
            logger.debug(
                f"{self.__class__.__name__} mode strategy does not support rate limit reset watching"
            )

    # Optional methods that subclasses can override
    async def _scheduler_loop(self) -> None:  # noqa: B027
        """
        Optional scheduler loop for active request processing.

        Override in subclasses that need background scheduling (e.g., queue processing).
        The default implementation is a no-op for basic schedulers that don't need
        an active loop.

        This method is called by start() if defined, and runs until the scheduler
        is stopped. Implementations should check self._running periodically and
        exit when it becomes False.
        """
        pass

    async def _on_start(self) -> None:  # noqa: B027
        """Hook for subclass-specific startup logic."""
        pass

    async def _on_stop(self) -> None:  # noqa: B027
        """Hook for subclass-specific cleanup logic."""
        pass

    # Helper methods that subclasses can use
    async def _get_queue_key(self, metadata: RequestMetadata) -> str:
        """
        Get the bucket-based queue key for a given request.

        This method is async to allow for dynamic bucket discovery.
        For basic mode without provider, returns a simple key.
        """
        if not self.provider or not metadata.requires_model:
            return f"default_{metadata.resource_type}"

        if not metadata.model_id or metadata.model_id == "unknown":
            return f"default_{metadata.resource_type}"

        bucket_id = await self.provider.get_bucket_for_model(
            metadata.model_id, metadata.resource_type
        )
        return f"{bucket_id}:{metadata.resource_type}"

    def should_throttle(self) -> bool:
        """Check if requests should be throttled due to failed request limits."""
        return bool(self._failed_requests.is_limit_exceeded())

    def track_failure(self) -> int:
        """Track a failed request and return current failure count."""
        result = self._failed_requests.increment()
        return int(result) if result is not None else 0

    def calculate_backoff(self, attempt: int, base_delay: float | None = None) -> float:
        """
        Calculate exponential backoff delay.

        Args:
            attempt: Current attempt number (0-based)
            base_delay: Optional base delay override

        Returns:
            Delay in seconds
        """
        if base_delay is None:
            base_delay = 1.0

        backoff_base = getattr(self.config, "backoff_base", 2.0)
        max_backoff = getattr(self.config, "max_backoff", 60.0)

        # Exponential backoff: base_delay * (backoff_base ^ attempt)
        delay = base_delay * (backoff_base**attempt)

        # Cap at max_backoff
        delay = min(delay, max_backoff)

        # Add small jitter to prevent thundering herd
        jitter = delay * BACKOFF_JITTER_FACTOR * (time.time() % 1)
        delay += jitter

        logger.debug(f"Calculated backoff for attempt {attempt}: {delay:.2f}s")
        return delay

    def extract_response_headers(self, result: Any) -> dict[str, str] | None:
        """
        Extract response headers from API result.

        Args:
            result: API response object

        Returns:
            Dict of headers or None if not extractable
        """
        # Strategy 1: Check for _response private attribute (most common)
        if (
            hasattr(result, "__pydantic_private__")
            and result.__pydantic_private__
            and "_response" in result.__pydantic_private__
        ):
            raw_response = result.__pydantic_private__["_response"]
            if hasattr(raw_response, "headers"):
                headers_dict: dict[str, str] = dict(raw_response.headers)
                return headers_dict

        # Strategy 2: Check for public response attribute
        if hasattr(result, "response") and hasattr(result.response, "headers"):
            headers_dict = dict(result.response.headers)
            return headers_dict

        # Strategy 3: Check for headers in dict format
        if isinstance(result, dict) and "_response_headers" in result:
            response_headers = result["_response_headers"]
            if isinstance(response_headers, dict):
                return response_headers

        # Strategy 4: Direct headers attribute (skip dict types)
        if (
            not isinstance(result, dict)
            and hasattr(result, "headers")
            and getattr(result, "headers", None) is not None
        ):
            headers_dict = dict(result.headers)
            return headers_dict

        # Strategy 5: Private headers attribute (skip dict types)
        if not isinstance(result, dict) and hasattr(result, "_headers"):
            headers_dict = dict(result._headers)
            return headers_dict

        logger.debug(f"Could not extract headers from result of type {type(result)}")
        return None

    def handle_rate_limit_headers(self, headers: dict[str, str]) -> dict[str, Any]:
        """
        Parse rate limit headers from API responses.

        Args:
            headers: HTTP response headers

        Returns:
            Dict containing parsed rate limit information
        """
        rate_limit_info = {}

        # Parse remaining requests
        remaining_requests = headers.get("x-ratelimit-remaining-requests")
        if remaining_requests is not None:
            try:
                rate_limit_info["remaining_requests"] = int(remaining_requests)
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid x-ratelimit-remaining-requests header: {remaining_requests}"
                )

        # Parse request reset time
        reset_requests = headers.get("x-ratelimit-reset-requests")
        if reset_requests is not None:
            try:
                rate_limit_info["reset_requests"] = int(reset_requests)
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid x-ratelimit-reset-requests header: {reset_requests}"
                )

        # Parse remaining tokens
        remaining_tokens = headers.get("x-ratelimit-remaining-tokens")
        if remaining_tokens is not None:
            try:
                rate_limit_info["remaining_tokens"] = int(remaining_tokens)
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid x-ratelimit-remaining-tokens header: {remaining_tokens}"
                )

        # Parse token reset time
        reset_tokens = headers.get("x-ratelimit-reset-tokens")
        if reset_tokens is not None:
            try:
                rate_limit_info["reset_tokens"] = int(reset_tokens)
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid x-ratelimit-reset-tokens header: {reset_tokens}"
                )

        logger.debug(f"Parsed rate limit info: {rate_limit_info}")
        return rate_limit_info

    def parse_rate_limit_headers(self, headers: dict[str, str]) -> float | None:
        """
        Parse rate limit headers to determine wait time.

        Args:
            headers: HTTP response headers from 429 error

        Returns:
            Wait time in seconds, or None if no wait needed
        """
        # Check for x-ratelimit-reset-requests (timestamp)
        reset_requests = headers.get("x-ratelimit-reset-requests")
        if reset_requests:
            try:
                reset_timestamp = int(reset_requests)
                current_timestamp = int(time.time())
                wait_time = max(0, reset_timestamp - current_timestamp)
                if wait_time > 0:
                    logger.info(
                        f"Rate limit reset in {wait_time}s (from x-ratelimit-reset-requests)"
                    )
                    return float(wait_time)
            except (ValueError, TypeError):
                logger.warning(f"Invalid x-ratelimit-reset-requests: {reset_requests}")

        # Check for x-ratelimit-reset-tokens (duration in seconds)
        reset_tokens = headers.get("x-ratelimit-reset-tokens")
        if reset_tokens:
            try:
                wait_time = int(reset_tokens)
                if wait_time > 0:
                    logger.info(
                        f"Rate limit reset in {wait_time}s (from x-ratelimit-reset-tokens)"
                    )
                    return float(wait_time)
            except (ValueError, TypeError):
                logger.warning(f"Invalid x-ratelimit-reset-tokens: {reset_tokens}")

        # If no specific reset time, use default backoff
        logger.info("No rate limit reset time found in headers, using default backoff")
        return None

    async def _check_queue_overflow(
        self,
        queue_size: int,
        queue_key: str,
        drop_callback: Callable[[], Awaitable[None]] | None = None,
    ) -> bool:
        """
        Check if queue would overflow and handle based on policy.

        This method checks if the queue has reached its maximum size and applies
        the configured overflow policy. For 'reject' policy, it raises an exception.
        For 'drop_oldest' policy, it calls the provided drop_callback to remove
        the oldest item from the queue.

        Args:
            queue_size: Current number of items in the queue
            queue_key: Identifier for the queue being checked
            drop_callback: Optional async callback that drops the oldest item
                          from the queue. Required for 'drop_oldest' policy.
                          If not provided and policy is 'drop_oldest', returns
                          True without any action (caller must handle the drop).

        Returns:
            bool: True if overflow occurred and was handled (drop_oldest policy),
                  False if no overflow. Never returns True for 'reject' policy
                  because it raises an exception instead.

        Raises:
            QueueOverflowError: If overflow_policy is 'reject' and queue is full
        """
        max_queue_size = getattr(self.config, "max_queue_size", 1000)

        if queue_size >= max_queue_size:
            # Update metrics
            if self.metrics_enabled:
                self.metrics[METRIC_QUEUE_OVERFLOWS] += 1
                if self.metrics_collector:
                    bucket_id = (
                        queue_key.split(":")[0] if ":" in queue_key else queue_key
                    )
                    self.metrics_collector.inc_counter(
                        QUEUE_OVERFLOWS_TOTAL, labels={"bucket_id": bucket_id}
                    )

            overflow_policy = getattr(self.config, "overflow_policy", "reject")
            if overflow_policy == "reject":
                raise QueueOverflowError(f"Queue {queue_key} is full")

            # drop_oldest policy
            if drop_callback is not None:
                # Execute the drop callback to remove the oldest item
                await drop_callback()
                logger.debug(
                    f"Dropped oldest request from queue {queue_key} due to overflow"
                )
                # Return False because we made room - queue is no longer overflowing
                return False

            # No callback provided - return True to signal overflow
            # (caller must handle dropping manually, as IntelligentModeStrategy does)
            return True
        return False

    async def _update_metrics_for_request(
        self, metadata: RequestMetadata, metric_name: str
    ) -> None:
        """
        Update metrics for a request event.

        Updates both the legacy dict-based metrics for backward compatibility
        and the unified collector metrics for Prometheus export.

        Args:
            metadata: Request metadata containing bucket_id, model_id, etc.
            metric_name: Legacy metric name (e.g., "requests_scheduled")
        """
        if self.metrics_enabled:
            # Update legacy dict-based metrics for backward compatibility
            self.metrics[metric_name] += 1

            # Update unified collector metrics
            if self.metrics_collector:
                # Convert legacy name to Prometheus-style name
                prom_name = legacy_to_prometheus_name(metric_name)

                # Build labels dict from available metadata attributes
                labels: dict[str, str] = {}
                if metadata.model_id:
                    labels["model_id"] = metadata.model_id
                    # Use model_id as bucket_id if not available elsewhere
                    labels["bucket_id"] = metadata.model_id
                if metadata.resource_type:
                    labels["bucket_id"] = (
                        f"{metadata.model_id}:{metadata.resource_type}"
                    )

                self.metrics_collector.inc_counter(
                    prom_name, labels=labels if labels else None
                )

    def get_metrics(self) -> dict[str, Any]:
        """
        Get current scheduler metrics.

        Returns a dictionary containing:
        - Standard scheduler state (running, active_tasks, etc.)
        - Legacy dict-based metrics (for backward compatibility)
        - Unified collector metrics (if enabled)

        Returns:
            Dictionary of metrics suitable for JSON serialization
        """
        base_metrics: dict[str, Any] = {
            "scheduler_type": self.__class__.__name__,
            "running": self._running,
            "active_tasks": len(self._active_tasks),
            "failed_requests": self._failed_requests.count,
            "total_requests": self._request_counter,
        }

        if self.metrics_enabled:
            # Add legacy dict-based metrics
            base_metrics.update(self.metrics)

            # Add unified collector metrics if available
            if self.metrics_collector:
                base_metrics["unified_metrics"] = (
                    self.metrics_collector.get_flat_metrics()
                )

        return base_metrics
