# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
Basic mode scheduling strategy for the adaptive rate limiter.

This module implements the BASIC mode strategy which provides simple direct
execution with retry logic. No complex queuing or state management - requests
are executed immediately with exponential backoff retry logic.
"""

import asyncio
import logging
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...protocols.client import ClientProtocol
    from ...scheduler.config import RateLimiterConfig

from ...exceptions import RateLimiterError
from ...types.request import RequestMetadata
from .base import BaseSchedulingModeStrategy

logger = logging.getLogger(__name__)

# Constants for state cleanup
DEFAULT_STALE_ENTRY_TTL = 3600.0  # 1 hour TTL for stale entries
DEFAULT_MAX_ENTRIES = 10000  # Maximum entries before LRU eviction


class BasicModeStrategy(BaseSchedulingModeStrategy):
    """
    BASIC mode strategy: Simple direct execution with retry logic.

    No complex queuing or state management. Requests are executed immediately
    with exponential backoff retry logic and basic rate limiting.

    This strategy is ideal for:
    - Simple use cases without complex rate limiting needs
    - Development and testing environments
    - Low-volume API access patterns

    Key Features:
    - Direct request execution (no queue)
    - Exponential backoff on failures
    - Request spacing to avoid bursts
    - Basic failure tracking

    Attributes:
        max_retries: Maximum number of retry attempts (default: 3)
        backoff_base: Base for exponential backoff calculation (default: 2.0)
        max_backoff: Maximum backoff time in seconds (default: 60.0)

    Example:
        >>> strategy = BasicModeStrategy(scheduler, config, client)
        >>> await strategy.start()
        >>> result = await strategy.submit_request(metadata, request_func)
        >>> await strategy.stop()
    """

    def __init__(
        self,
        scheduler: Any,  # BaseScheduler - kept as Any to avoid circular imports
        config: "RateLimiterConfig",
        client: "ClientProtocol",
    ):
        """
        Initialize the BasicModeStrategy.

        Args:
            scheduler: Reference to the main scheduler for shared functionality.
            config: Scheduler configuration containing rate limiting parameters.
            client: Client instance following ClientProtocol for API integration.
        """
        super().__init__(scheduler, config, client)

        # Simple failure tracking for basic mode with LRU eviction support
        # Using OrderedDict for LRU behavior (move_to_end on access)
        self._last_request_times: OrderedDict[str, float] = OrderedDict()

        # Basic mode configuration
        self.max_retries = getattr(config, "max_retries", 3)
        self.backoff_base = getattr(config, "backoff_base", 2.0)
        self.max_backoff = getattr(config, "max_backoff", 60.0)

        # Cleanup configuration - use hasattr to avoid Mock objects returning truthy Mocks
        stale_ttl = getattr(config, "stale_entry_ttl", None)
        self._stale_entry_ttl = (
            stale_ttl
            if isinstance(stale_ttl, (int, float))
            else DEFAULT_STALE_ENTRY_TTL
        )
        max_entries = getattr(config, "max_tracking_entries", None)
        self._max_entries = (
            max_entries if isinstance(max_entries, int) else DEFAULT_MAX_ENTRIES
        )

    async def submit_request(
        self, metadata: RequestMetadata, request_func: Callable[[], Awaitable[Any]]
    ) -> Any:
        """
        Submit request in BASIC mode - direct execution with retry logic.

        Checks global failure limits before executing. If too many failures
        have occurred recently, the request is rejected with an error.

        Args:
            metadata: Request metadata containing routing information
            request_func: Async function to execute the request

        Returns:
            Result of request execution (direct API response)

        Raises:
            RateLimiterError: If too many failed requests have occurred
        """
        # Check global failure limits
        if self.scheduler.should_throttle():
            failure_count = self.scheduler._failed_requests.count
            raise RateLimiterError(
                f"Too many failed requests ({failure_count}) - please wait and try again"
            )

        # Execute with retry logic
        return await self._execute_with_retry(metadata, request_func)

    async def run_scheduling_loop(self) -> None:
        """
        BASIC mode doesn't need a scheduling loop.

        This method runs a minimal idle loop to satisfy the interface.
        Since BASIC mode executes requests directly without queuing,
        there's no scheduling work to perform.
        """
        while self._running:  # noqa: ASYNC110
            # Basic mode doesn't need active scheduling
            await asyncio.sleep(1.0)

    async def start(self) -> None:
        """
        Start basic mode strategy.

        Sets the running flag to True. Basic mode requires no additional
        resources or background tasks.
        """
        self._running = True

    async def stop(self) -> None:
        """
        Stop basic mode strategy.

        Sets the running flag to False. Basic mode has no resources to clean up.
        """
        self._running = False

    def get_metrics(self) -> dict[str, Any]:
        """
        Get basic mode metrics.

        Returns:
            Dictionary containing:
            - mode: "basic"
            - last_request_times_count: Number of models with tracked request times
        """
        return {
            "mode": "basic",
            "last_request_times_count": len(self._last_request_times),
        }

    async def _execute_with_retry(
        self, metadata: RequestMetadata, request_func: Callable[[], Awaitable[Any]]
    ) -> Any:
        """
        Execute request with retry logic for BASIC mode.

        Implements exponential backoff with proper exception handling.
        The main benefit of exception handling is syncing authoritative 429
        headers to update remaining capacity state.

        Args:
            metadata: Request metadata
            request_func: Async function to execute the request

        Returns:
            Result of successful request execution

        Raises:
            Exception: The last error encountered after all retries exhausted
            RateLimiterError: If request failed after all retries with no specific error
        """
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                # Enforce minimum request spacing
                await self._enforce_request_spacing(metadata)

                # Execute the request
                result = await request_func()

                # Success path: sync state from headers with status_code=200
                await self._update_rate_limit_state(metadata, result, status_code=200)

                # Track successful request timing with LRU eviction
                self._update_last_request_time(metadata.model_id)

                return result

            except asyncio.CancelledError:
                raise  # Always re-raise for graceful shutdown

            except Exception as e:
                # Check if this is a rate limit error (429)
                is_rate_limit = self._is_rate_limit_error(e)

                if is_rate_limit:
                    # 429 path: sync state from cached headers (authoritative)
                    exc_headers = getattr(e, "cached_rate_limit_headers", {})
                    if (
                        exc_headers
                        and hasattr(self.scheduler, "state_manager")
                        and self.scheduler.state_manager
                    ):
                        try:
                            await (
                                self.scheduler.state_manager.update_state_from_headers(
                                    metadata.model_id,
                                    metadata.resource_type,
                                    exc_headers,
                                    request_id=metadata.request_id,
                                    status_code=429,
                                )
                            )
                        except Exception as sync_error:
                            logger.warning(
                                f"Failed to sync 429 headers for BasicMode: {sync_error}"
                            )

                last_error = e
                self.scheduler.track_failure()

                # Check if this is the last attempt
                if attempt >= self.max_retries:
                    break

                # Calculate backoff delay
                delay = self._calculate_retry_delay(e, attempt, metadata)
                if delay > 0:
                    await asyncio.sleep(delay)

        # All retries exhausted
        if last_error:
            raise last_error
        else:
            raise RateLimiterError("Request failed after all retries")

    async def _enforce_request_spacing(self, metadata: RequestMetadata) -> None:
        """
        Enforce request spacing for BASIC mode.

        Ensures minimum time between requests to the same model to avoid
        bursting behavior that could trigger rate limits.

        Args:
            metadata: Request metadata containing model_id
        """
        model_id = metadata.model_id
        current_time = time.time()

        last_request_time = self._last_request_times.get(model_id, 0)
        time_since_last = current_time - last_request_time

        # Use adaptive delay based on recent failures
        min_spacing = self._calculate_adaptive_delay(model_id)

        if time_since_last < min_spacing:
            wait_time = min_spacing - time_since_last
            await asyncio.sleep(wait_time)

    def _calculate_adaptive_delay(self, model_id: str) -> float:
        """
        Calculate adaptive delay for BASIC mode based on recent failure rate.

        Currently returns a simple default spacing. Can be enhanced to adjust
        based on failure tracking for more sophisticated rate limiting.

        Args:
            model_id: The model identifier

        Returns:
            Minimum delay in seconds between requests
        """
        # Simple implementation - can be enhanced based on failure tracking
        return 0.1  # 100ms default spacing

    async def _update_rate_limit_state(
        self,
        metadata: RequestMetadata,
        result: Any,
        status_code: int = 200,
    ) -> None:
        """
        Update rate limit state for BASIC mode.

        BasicModeStrategy doesn't use pending gauge tracking, but still syncs
        state from headers. The status_code is passed to the backend for
        proper script routing when using distributed backends.

        The main benefit for BasicMode is state synchronization from 429 headers,
        allowing accurate remaining capacity tracking even without pending gauges.

        Args:
            metadata: Request metadata
            result: API response
            status_code: HTTP status code (default 200 for backward compatibility)
        """
        headers = self.scheduler.extract_response_headers(result)
        if headers:
            # If we have a state_manager, use it with status_code support
            if (
                hasattr(self.scheduler, "state_manager")
                and self.scheduler.state_manager
            ):
                await self.scheduler.state_manager.update_state_from_headers(
                    metadata.model_id,
                    metadata.resource_type,
                    headers,
                    request_id=metadata.request_id,
                    status_code=status_code,
                )
            else:
                # Fallback to legacy handler
                rate_limit_info = self.scheduler.handle_rate_limit_headers(headers)
                logger.debug(
                    f"Updated rate limit state for {metadata.model_id}: {rate_limit_info}"
                )

    def _calculate_retry_delay(
        self, error: Exception, attempt: int, metadata: RequestMetadata
    ) -> float:
        """
        Calculate retry delay for BASIC mode.

        Uses the scheduler's backoff calculation, which implements exponential
        backoff with jitter.

        Args:
            error: The exception that triggered the retry
            attempt: The current attempt number (0-indexed)
            metadata: Request metadata

        Returns:
            Delay in seconds before next retry attempt
        """
        # Check if error has retry_after (common for rate limit errors)
        retry_after = getattr(error, "retry_after_seconds", None)
        if retry_after is not None and retry_after > 0:
            return float(min(retry_after, self.max_backoff))

        # Use base class backoff calculation
        return float(self.scheduler.calculate_backoff(attempt))

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """
        Check if an exception is a rate limit error.

        Args:
            error: The exception to check

        Returns:
            True if the error represents a 429 rate limit response
        """
        # Check common patterns for rate limit errors
        error_name = type(error).__name__.lower()
        if "ratelimit" in error_name:
            return True

        # Check for status_code attribute
        status_code = getattr(error, "status_code", None)
        return status_code == 429

    def _update_last_request_time(self, model_id: str) -> None:
        """
        Update last request time for a model with LRU eviction and TTL cleanup.

        This method:
        1. Updates the timestamp for the model
        2. Moves the entry to the end (most recently used)
        3. Evicts oldest entries if max_entries exceeded
        4. Cleans up stale entries older than TTL

        Args:
            model_id: The model identifier
        """
        current_time = time.time()

        # Update or add the entry
        self._last_request_times[model_id] = current_time
        # Move to end to mark as recently used
        self._last_request_times.move_to_end(model_id)

        # Evict oldest entries if we exceed max_entries
        while len(self._last_request_times) > self._max_entries:
            oldest_key, _ = self._last_request_times.popitem(last=False)
            logger.debug(f"LRU evicted stale entry for model: {oldest_key}")

        # Periodic TTL cleanup (every 100 updates to avoid overhead)
        # We use modulo on dict size as a simple trigger
        if len(self._last_request_times) % 100 == 0:
            self._cleanup_stale_entries()

    def _cleanup_stale_entries(self) -> None:
        """
        Remove entries older than TTL from tracking dictionaries.

        This method iterates through entries and removes those that haven't
        been accessed within the stale_entry_ttl period.
        """
        current_time = time.time()
        cutoff_time = current_time - self._stale_entry_ttl
        stale_keys = []

        # Find stale entries
        for model_id, last_time in self._last_request_times.items():
            if last_time < cutoff_time:
                stale_keys.append(model_id)

        # Remove stale entries
        for model_id in stale_keys:
            del self._last_request_times[model_id]

        if stale_keys:
            logger.debug(f"Cleaned up {len(stale_keys)} stale tracking entries")

    def cleanup_tracking_state(self) -> int:
        """
        Manually trigger cleanup of stale tracking state.

        This method can be called externally to force cleanup of stale entries.
        Useful for testing or when memory pressure is detected.

        Returns:
            Number of entries removed
        """
        initial_count = len(self._last_request_times)
        self._cleanup_stale_entries()
        removed = initial_count - len(self._last_request_times)
        return removed


__all__ = ["BasicModeStrategy"]
