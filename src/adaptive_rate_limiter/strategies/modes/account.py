# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
Account mode strategy for the adaptive rate limiter.

This module implements the AccountModeStrategy, which manages requests at the
account level with per-account queues, concurrency limits, and conservative
rate limiting.
"""

import asyncio
import logging
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ...exceptions import QueueOverflowError, TooManyFailedRequestsError
from ...types.queue import QueuedRequest, ScheduleResult
from ...types.request import RequestMetadata
from .base import BaseSchedulingModeStrategy

if TYPE_CHECKING:
    from ...protocols.client import ClientProtocol
    from ...scheduler.config import RateLimiterConfig
    # BaseScheduler import removed to avoid circular dependency/missing file
    # Using Any for scheduler type hint as in BaseSchedulingModeStrategy

logger = logging.getLogger(__name__)

# Constants for metrics
METRIC_TOTAL_SCHEDULED = "total_scheduled"
METRIC_TOTAL_COMPLETED = "total_completed"
METRIC_TOTAL_FAILED = "total_failed"
METRIC_TOTAL_REJECTED = "total_rejected"
METRIC_CURRENT_QUEUE_SIZE = "current_queue_size"


class AccountModeStrategy(BaseSchedulingModeStrategy):
    """
    ACCOUNT mode strategy: Account-level request management.

    Manages requests at the account level with per-account queues,
    concurrency limits, and conservative rate limiting.
    """

    def __init__(
        self,
        scheduler: Any,
        config: "RateLimiterConfig",
        client: "ClientProtocol",
    ):
        super().__init__(scheduler, config, client)

        # Account-level queuing
        self.account_queues: dict[str, deque[QueuedRequest]] = defaultdict(deque)
        self.active_requests: dict[str, QueuedRequest] = {}
        self.active_count = 0

        # Account-specific configuration
        self.max_concurrent_requests = getattr(config, "max_concurrent_requests", 10)
        self.conservative_multiplier = getattr(config, "conservative_multiplier", 0.9)

        # Account metrics
        self.account_metrics = {
            METRIC_TOTAL_SCHEDULED: 0,
            METRIC_TOTAL_COMPLETED: 0,
            METRIC_TOTAL_FAILED: 0,
            METRIC_TOTAL_REJECTED: 0,
            METRIC_CURRENT_QUEUE_SIZE: 0,
        }

    async def submit_request(
        self, metadata: RequestMetadata, request_func: Callable[[], Awaitable[Any]]
    ) -> ScheduleResult:
        """Submit request in ACCOUNT mode - account-level queuing."""
        # Check circuit breaker first
        if (
            hasattr(self.scheduler, "circuit_breaker")
            and self.scheduler.circuit_breaker
            and not await self.scheduler.circuit_breaker.can_execute(metadata.model_id)
        ):
            raise TooManyFailedRequestsError("Circuit breaker active")

        # Create future and queued request
        future: asyncio.Future[Any] = asyncio.Future()
        queued_request = QueuedRequest(
            metadata=metadata,
            request_func=request_func,
            future=future,
            queue_entry_time=datetime.now(timezone.utc),
        )

        # Determine queue key (account-level)
        # Use model_id as proxy for account/resource in this context if not explicit
        queue_key = metadata.model_id or getattr(metadata, "resource_type", "default")

        # Check queue size limits
        total_size = sum(len(q) for q in self.account_queues.values())
        max_queue_size = getattr(self.config, "max_queue_size", 1000)

        if total_size >= max_queue_size:
            self.account_metrics[METRIC_TOTAL_REJECTED] += 1
            raise QueueOverflowError(
                f"Account queue full ({total_size}/{max_queue_size})"
            )

        # Add to account queue
        self.account_queues[queue_key].append(queued_request)
        self.account_metrics[METRIC_TOTAL_SCHEDULED] += 1
        self.account_metrics[METRIC_CURRENT_QUEUE_SIZE] = total_size + 1

        return ScheduleResult(request=queued_request, wait_time=0.0, should_retry=True)

    async def run_scheduling_loop(self) -> None:
        """Main scheduler loop for ACCOUNT mode."""
        while self._running:
            try:
                await self._loop_account_mode()

                # Small yield to prevent CPU hogging
                await asyncio.sleep(getattr(self.config, "scheduler_interval", 0.01))

            except asyncio.CancelledError:
                logger.info("Account scheduler loop cancelled")
                break
            except (
                AttributeError,
                ValueError,
                OSError,
                TypeError,
                RuntimeError,
                KeyError,
            ) as e:
                logger.exception(f"Account scheduler error: {e}")
                await asyncio.sleep(0.01)

    async def start(self) -> None:
        """Start account mode strategy."""
        self._running = True

    async def stop(self) -> None:
        """Stop account mode strategy."""
        self._running = False

    def get_metrics(self) -> dict[str, Any]:
        """Get account mode metrics."""
        metrics: dict[str, Any] = dict(self.account_metrics)
        metrics.update(
            {
                "mode": "account",
                "active_requests": self.active_count,
                "num_queues": len(self.account_queues),
            }
        )
        return metrics

    async def _loop_account_mode(self) -> None:
        """Scheduler loop for ACCOUNT mode."""
        # Check capacity
        if self.active_count >= self.max_concurrent_requests:
            return

        # Find eligible queues
        eligible_queues = await self._find_eligible_queues()

        # Process requests from eligible queues
        for queue_key in eligible_queues:
            if self.active_count >= self.max_concurrent_requests:
                break

            queue = self.account_queues.get(queue_key)
            if not queue:
                continue

            request = queue.popleft()
            self.account_metrics[METRIC_CURRENT_QUEUE_SIZE] -= 1

            # Clean up empty queue to prevent unbounded growth
            if not queue:
                del self.account_queues[queue_key]

            # Execute request
            _ = asyncio.create_task(self._execute_account_request(request))  # noqa: RUF006

    async def _find_eligible_queues(self) -> list[str]:
        """Find eligible queues for ACCOUNT mode."""
        eligible = []

        for queue_key, queue in self.account_queues.items():
            if queue:  # Has pending requests
                eligible.append(queue_key)

        return eligible

    async def _execute_account_request(self, request: QueuedRequest) -> None:
        """Execute request in ACCOUNT mode."""
        # Track as active
        self.active_requests[request.metadata.request_id] = request
        self.active_count += 1

        # Get timeout configuration before try block to ensure it's always bound
        request_timeout = getattr(self.config, "request_timeout", 30.0)

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                request.request_func(), timeout=request_timeout
            )

            # Update metrics
            self.account_metrics[METRIC_TOTAL_COMPLETED] += 1

            # Set result on future
            if not request.future.cancelled():
                request.future.set_result(result)

        except asyncio.CancelledError:
            raise  # Always re-raise for graceful shutdown
        except asyncio.TimeoutError:
            # Timeout: handle gracefully with proper future exception
            self.account_metrics[METRIC_TOTAL_FAILED] += 1
            logger.warning(
                f"Request {request.metadata.request_id} timed out after {request_timeout}s"
            )

            # Set timeout exception on future
            if not request.future.cancelled():
                request.future.set_exception(
                    TimeoutError(f"Request timed out after {request_timeout}s")
                )
        except Exception as e:
            # Catch all exceptions to ensure futures are never left unresolved.
            # CancelledError is handled separately above (line 208) and re-raised.
            self.account_metrics[METRIC_TOTAL_FAILED] += 1
            logger.exception(f"Account request execution error: {e}")

            # Set exception on future
            if not request.future.cancelled():
                request.future.set_exception(e)

        finally:
            # Remove from active requests
            self.active_requests.pop(request.metadata.request_id, None)
            self.active_count -= 1
