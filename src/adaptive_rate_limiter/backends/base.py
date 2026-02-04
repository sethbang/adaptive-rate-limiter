# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
Base Backend for Adaptive Rate Limiter

This module provides the unified BaseBackend abstract class that defines
the common interface for all backend implementations.

Features:
- Unified state management for scheduler operations
- Account-level operations and failure tracking
- Rate limit management and capacity reservation
- Caching for provider and model information

"""

import abc
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """
    Structured health check result for backend monitoring.

    Attributes:
        healthy: Whether the backend is operational
        backend_type: Type of backend (e.g., 'redis', 'memory')
        namespace: Backend namespace
        error: Error message if unhealthy
        metadata: Additional backend-specific information
    """

    healthy: bool
    backend_type: str
    namespace: str
    error: str | None = None
    metadata: dict[str, Any] | None = None


def validate_safety_margin(safety_margin: float) -> None:
    """
    Validate that safety_margin is a valid number between 0.0 and 1.0.

    Args:
        safety_margin: The safety margin value to validate

    Raises:
        ValueError: If safety_margin is not a number or is outside the valid range
    """
    if not isinstance(safety_margin, (int, float)):
        raise ValueError(
            f"safety_margin must be a number, got {type(safety_margin).__name__}"
        )

    if not (0.0 <= safety_margin <= 1.0):
        raise ValueError(
            f"safety_margin must be between 0.0 and 1.0, got {safety_margin}"
        )


class BaseBackend(abc.ABC):
    """
    An abstract base class that defines the common interface for all backend
    implementations in the Adaptive Rate Limiter.

    This class provides a unified contract for various backend services,
    including state management, capacity and rate limit management, failure
    tracking, and caching. By consolidating multiple backend interfaces into
    a single, clean interface, it promotes a consistent architecture and
    allows for different backend implementations (e.g., Redis, in-memory) to
    be used interchangeably.

    Subclasses must implement all abstract methods to provide a concrete
    backend implementation.
    """

    def __init__(self, namespace: str = "rate_limiter"):
        """
        Initialize the backend with a namespace for isolation.

        Args:
            namespace: Namespace for isolating data across different instances
        """
        self.namespace = namespace

    # ==========================================================================
    # Core State Management
    # ==========================================================================

    @abc.abstractmethod
    async def get_state(self, key: str) -> dict[str, Any] | None:
        """
        Get the state for a given key.

        Args:
            key: The key to retrieve state for

        Returns:
            The state dictionary if it exists, None otherwise
        """
        pass

    @abc.abstractmethod
    async def set_state(self, key: str, state: dict[str, Any]) -> None:
        """
        Set the state for a given key.

        Args:
            key: The key to set state for
            state: The state dictionary to store
        """
        pass

    @abc.abstractmethod
    async def get_all_states(self) -> dict[str, dict[str, Any]]:
        """
        Get all stored states.

        Returns:
            A dictionary mapping keys to their state dictionaries
        """
        pass

    @abc.abstractmethod
    async def clear(self) -> None:
        """Clear all stored states."""
        pass

    # ==========================================================================
    # Capacity Management
    # ==========================================================================

    @abc.abstractmethod
    async def check_and_reserve_capacity(
        self,
        key: str,
        requests: int,
        tokens: int,
        bucket_limits: dict[str, int] | None = None,
        safety_margin: float = 1.0,
        request_id: str | None = None,
    ) -> tuple[bool, str | None]:
        """
        Atomically check and reserve capacity.

        This is the primary method for capacity reservation in scheduler-style
        rate limiting. It atomically checks if capacity is available and, if so,
        reserves the specified amount.

        Args:
            key: The key/bucket to check capacity for
            requests: Number of requests to reserve
            tokens: Number of tokens to reserve
            bucket_limits: Optional dictionary containing rpm_limit and tpm_limit
                          for initializing new keys
            safety_margin: Safety margin multiplier (0.0-1.0) applied atomically
                          to capacity. Default 1.0 = no margin. 0.9 = reserve only
                          when 90% capacity available. This helps prevent race
                          conditions in distributed systems.
            request_id: Optional request identifier for tracking

        Returns:
            Tuple of (can_proceed, reservation_id)
        """
        pass

    @abc.abstractmethod
    async def release_reservation(self, key: str, reservation_id: str) -> bool:
        """
        Release a capacity reservation with idempotent protection.

        This method should implement idempotent capacity release, preventing
        double-release corruption while maintaining the mathematical invariant:
        0 <= remaining_capacity <= limit.

        Args:
            key: The key the reservation was made for
            reservation_id: The reservation ID to release

        Returns:
            True if reservation was released or already released (idempotent)
            False if reservation or state not found, or on errors
        """
        pass

    @abc.abstractmethod
    async def release_streaming_reservation(
        self,
        key: str,
        reservation_id: str,
        reserved_tokens: int,
        actual_tokens: int,
    ) -> bool:
        """
        Release streaming reservation with refund-based accounting.

        Refund = reserved_tokens - actual_tokens

        This method is called when a streaming response completes. It uses
        refund-based accounting where:
        - rem_tok += (reserved_tokens - actual_tokens)
        - Result is clamped to [0, limit] to prevent overflow

        The implementation should be idempotent and handle concurrent calls.

        For non-streaming workloads, a simple delegation to release_reservation()
        is sufficient.

        Args:
            key: The rate limit bucket identifier
            reservation_id: The reservation identifier
            reserved_tokens: Tokens that were reserved at request start
            actual_tokens: Actual tokens consumed by the stream

        Returns:
            True if release succeeded, False on error
        """
        pass

    @abc.abstractmethod
    async def check_capacity(
        self, model: str, request_type: str = "default"
    ) -> tuple[bool, float]:
        """
        Check if there's capacity for a request.

        This is a non-reserving capacity check that can be used for preflight
        validation or status queries.

        Args:
            model: Model/resource identifier
            request_type: Type of request (default, streaming, etc.)

        Returns:
            Tuple of (can_proceed, wait_time_seconds)
        """
        pass

    # ==========================================================================
    # Request and Failure Management
    # ==========================================================================

    @abc.abstractmethod
    async def record_request(
        self,
        model: str,
        request_type: str = "default",
        tokens_used: int | None = None,
    ) -> None:
        """
        Record a successful request.

        This method is used for tracking request metrics and may be used
        by some rate limiting strategies.

        Args:
            model: Model/resource identifier
            request_type: Type of request
            tokens_used: Optional token count for the request
        """
        pass

    @abc.abstractmethod
    async def record_failure(self, error_type: str, error_message: str = "") -> None:
        """
        Record a failure for tracking and circuit breaking.

        Args:
            error_type: Type of error (e.g., "rate_limit", "timeout")
            error_message: Optional error message
        """
        pass

    @abc.abstractmethod
    async def get_failure_count(self, window_seconds: int = 30) -> int:
        """
        Get the number of failures within the specified window.

        Args:
            window_seconds: Time window in seconds (default 30)

        Returns:
            Number of failures in the window
        """
        pass

    @abc.abstractmethod
    async def is_circuit_broken(self) -> bool:
        """
        Check if the circuit breaker is triggered.

        Returns:
            True if circuit is broken, False otherwise
        """
        pass

    # ==========================================================================
    # Rate Limit Management
    # ==========================================================================

    @abc.abstractmethod
    async def update_rate_limits(
        self,
        model: str,
        headers: dict[str, str],
        bucket_id: str | None = None,
        request_id: str | None = None,
        status_code: int | None = None,
    ) -> int:
        """
        Update rate limits from API response headers.

        This method processes rate limit information from API response headers
        and updates the backend state accordingly. It returns an integer status
        to allow callers to detect backend failures.

        Args:
            model: Model/resource identifier
            headers: Response headers containing rate limit info
            bucket_id: Optional bucket ID to update state used by capacity reservation
            request_id: Optional request ID for drift correction
            status_code: Optional HTTP status code from the response. Used by
                        distributed backends to detect rate limit violations (429)
                        and adjust capacity accordingly. Ignored by in-memory backends.

        Returns:
            int: 1 if the update was successful, 0 if it failed. Callers should
                 check this return value to handle backend failures gracefully.
        """
        pass

    @abc.abstractmethod
    async def get_rate_limits(self, model: str) -> dict[str, Any]:
        """
        Get current rate limit state for a model/resource.

        Args:
            model: Model/resource identifier

        Returns:
            Dictionary with rate limit information
        """
        pass

    # ==========================================================================
    # Advanced Capacity Management
    # ==========================================================================

    @abc.abstractmethod
    async def reserve_capacity(
        self, model: str, request_id: str, tokens_estimated: int = 0
    ) -> bool:
        """
        Reserve capacity for an upcoming request.

        This is an alternative capacity reservation method that uses a request
        ID for tracking instead of returning a reservation ID.

        Args:
            model: Model/resource identifier
            request_id: Unique request identifier
            tokens_estimated: Estimated tokens for the request

        Returns:
            True if reservation successful, False otherwise
        """
        pass

    @abc.abstractmethod
    async def release_reservation_by_id(self, request_id: str) -> None:
        """
        Release a capacity reservation by request ID.

        Args:
            request_id: Unique request identifier
        """
        pass

    # ==========================================================================
    # Caching Operations
    # ==========================================================================

    @abc.abstractmethod
    async def cache_bucket_info(
        self, bucket_data: dict[str, Any], ttl_seconds: int = 3600
    ) -> None:
        """
        Cache bucket/tier discovery information.

        This method stores bucket configuration data that can be retrieved
        later to avoid repeated discovery operations.

        Args:
            bucket_data: Bucket information to cache
            ttl_seconds: Time-to-live in seconds
        """
        pass

    @abc.abstractmethod
    async def get_cached_bucket_info(self) -> dict[str, Any] | None:
        """
        Retrieve cached bucket information.

        Returns:
            Cached bucket data or None if not found/expired
        """
        pass

    @abc.abstractmethod
    async def cache_model_info(
        self, model: str, model_data: dict[str, Any], ttl_seconds: int = 3600
    ) -> None:
        """
        Cache model/resource information.

        Args:
            model: Model/resource identifier
            model_data: Model information to cache
            ttl_seconds: Time-to-live in seconds
        """
        pass

    @abc.abstractmethod
    async def get_cached_model_info(self, model: str) -> dict[str, Any] | None:
        """
        Retrieve cached model information.

        Args:
            model: Model/resource identifier

        Returns:
            Cached model data or None if not found/expired
        """
        pass

    # ==========================================================================
    # Health and Monitoring
    # ==========================================================================

    @abc.abstractmethod
    async def health_check(self) -> HealthCheckResult:
        """
        Perform a health check on the backend.

        Returns:
            HealthCheckResult containing:
                - healthy: bool - Whether the backend is operational
                - backend_type: str - Type of backend (e.g., 'redis', 'memory')
                - namespace: str - Backend namespace
                - error: Optional[str] - Error message if unhealthy
                - metadata: Optional[Dict] - Additional backend-specific info
        """
        pass

    @abc.abstractmethod
    async def get_all_stats(self) -> dict[str, Any]:
        """
        Get all statistics from the backend.

        Returns:
            Dictionary containing all backend statistics
        """
        pass

    # ==========================================================================
    # Cleanup and Maintenance
    # ==========================================================================

    @abc.abstractmethod
    async def cleanup(self) -> None:
        """Clean up backend resources."""
        pass

    @abc.abstractmethod
    async def clear_failures(self) -> None:
        """Clear all failure records from the backend."""
        pass

    @abc.abstractmethod
    async def force_circuit_break(self, duration: float) -> None:
        """
        Force a circuit break for the specified duration.

        Args:
            duration: Duration in seconds to maintain circuit break
        """
        pass

    # ==========================================================================
    # Helper Methods (Shared Implementation)
    # ==========================================================================

    def _parse_rate_limit_headers(self, headers: dict[str, str]) -> dict[str, Any]:
        """
        Parse rate limit information from response headers.

        This method expects headers to be normalized to lowercase keys by the caller.
        HTTP headers are case-insensitive per RFC 7230, but implementations vary,
        so normalization at a single authoritative point ensures consistency.

        Standard headers parsed:
        - x-ratelimit-limit-requests: Maximum requests per minute
        - x-ratelimit-remaining-requests: Remaining requests in current window
        - x-ratelimit-reset-requests: Time until request limit resets
        - x-ratelimit-limit-tokens: Maximum tokens per minute
        - x-ratelimit-remaining-tokens: Remaining tokens in current window
        - x-ratelimit-reset-tokens: Time until token limit resets
        - retry-after: Seconds to wait before retrying

        Args:
            headers: Response headers with lowercase keys (normalized by caller)

        Returns:
            Parsed rate limit data containing rpm_limit, rpm_remaining, rpm_reset,
            tpm_limit, tpm_remaining, tpm_reset, retry_after, and timestamp fields
            as available from the input headers. Malformed values are skipped with
            a warning logged.
        """
        result: dict[str, Any] = {}

        # RPM limits - headers expected to be lowercase
        if "x-ratelimit-limit-requests" in headers:
            try:
                result["rpm_limit"] = int(headers["x-ratelimit-limit-requests"])
            except ValueError:
                logger.warning(
                    f"Malformed x-ratelimit-limit-requests header: "
                    f"'{headers['x-ratelimit-limit-requests']}', skipping"
                )
        if "x-ratelimit-remaining-requests" in headers:
            try:
                result["rpm_remaining"] = int(headers["x-ratelimit-remaining-requests"])
            except ValueError:
                logger.warning(
                    f"Malformed x-ratelimit-remaining-requests header: "
                    f"'{headers['x-ratelimit-remaining-requests']}', skipping"
                )
        if "x-ratelimit-reset-requests" in headers:
            try:
                result["rpm_reset"] = int(headers["x-ratelimit-reset-requests"])
            except ValueError:
                logger.warning(
                    f"Malformed x-ratelimit-reset-requests header: "
                    f"'{headers['x-ratelimit-reset-requests']}', skipping"
                )

        # TPM limits - headers expected to be lowercase
        if "x-ratelimit-limit-tokens" in headers:
            try:
                result["tpm_limit"] = int(headers["x-ratelimit-limit-tokens"])
            except ValueError:
                logger.warning(
                    f"Malformed x-ratelimit-limit-tokens header: "
                    f"'{headers['x-ratelimit-limit-tokens']}', skipping"
                )
        if "x-ratelimit-remaining-tokens" in headers:
            try:
                result["tpm_remaining"] = int(headers["x-ratelimit-remaining-tokens"])
            except ValueError:
                logger.warning(
                    f"Malformed x-ratelimit-remaining-tokens header: "
                    f"'{headers['x-ratelimit-remaining-tokens']}', skipping"
                )
        if "x-ratelimit-reset-tokens" in headers:
            try:
                result["tpm_reset"] = int(headers["x-ratelimit-reset-tokens"])
            except ValueError:
                logger.warning(
                    f"Malformed x-ratelimit-reset-tokens header: "
                    f"'{headers['x-ratelimit-reset-tokens']}', skipping"
                )

        # Retry after - headers expected to be lowercase
        if "retry-after" in headers:
            try:
                result["retry_after"] = int(float(headers["retry-after"]))
            except ValueError:
                logger.warning(
                    f"Malformed retry-after header: "
                    f"'{headers['retry-after']}', skipping"
                )

        result["timestamp"] = int(time.time())
        return result

    def _calculate_wait_time(self, rate_limits: dict[str, Any]) -> float:
        """
        Calculate wait time based on rate limit state.

        This method determines how long to wait before the next request
        can be made based on the current rate limit state.

        Args:
            rate_limits: Current rate limit state dictionary

        Returns:
            Wait time in seconds (0.0 if no wait needed)
        """
        if not rate_limits:
            return 0.0

        wait_time = 0.0
        current_time = time.time()

        # Check RPM
        rpm_remaining = rate_limits.get("rpm_remaining", 1)
        if rpm_remaining <= 0:
            rpm_reset = rate_limits.get("rpm_reset")
            if rpm_reset:
                reset_time = self._parse_reset_time(rpm_reset)
                wait_time = max(wait_time, reset_time - current_time)

        # Check TPM
        tpm_remaining = rate_limits.get("tpm_remaining", 1)
        if tpm_remaining <= 0:
            tpm_reset = rate_limits.get("tpm_reset")
            if tpm_reset:
                reset_time = self._parse_reset_time(tpm_reset)
                wait_time = max(wait_time, reset_time - current_time)

        # Check retry-after
        retry_after = rate_limits.get("retry_after", 0)
        if retry_after > 0:
            wait_time = max(wait_time, retry_after)

        return max(0.0, wait_time)

    def _parse_reset_time(self, reset_str: str) -> float:
        """
        Parse reset time from header string.

        Handles multiple formats:
        - Milliseconds timestamp (e.g., 1764830040000)
        - Seconds timestamp (e.g., 1764830040)
        - Delta seconds (e.g., 60)
        - ISO datetime string (e.g., 2024-01-01T00:00:00Z)

        Args:
            reset_str: Reset time string in various formats

        Returns:
            Unix timestamp (seconds since epoch)
        """
        try:
            # Try parsing as number
            val = float(reset_str)

            # Heuristic to detect milliseconds timestamp (e.g. 1764830040000)
            # 1e11 is roughly year 5138, so anything larger is likely ms
            if val > 1e11:
                return val / 1000.0

            # Heuristic to detect seconds timestamp (e.g. 1764830040)
            # 1e9 is roughly year 2001, so anything larger is likely sec timestamp
            if val > 1e9:
                return val

            # Otherwise treat as delta seconds
            return time.time() + val

        except ValueError:
            try:
                # Try parsing as ISO datetime
                dt = datetime.fromisoformat(reset_str.replace("Z", "+00:00"))
                return dt.timestamp()
            except (ValueError, TypeError, AttributeError) as e:
                # Log the fallback for monitoring
                logger.warning(
                    f"Failed to parse reset time '{reset_str}', using 60s fallback. "
                    f"Error: {e}. This may indicate a provider API change."
                )
                # Default to 60 seconds from now
                return time.time() + 60
