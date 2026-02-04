# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
MemoryBackend for Adaptive Rate Limiter

This module provides an in-memory backend implementation that doesn't require Redis.
Perfect for testing, development, and single-process applications.
"""

import asyncio
import contextlib
import heapq
import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from .base import BaseBackend, HealthCheckResult, validate_safety_margin

logger = logging.getLogger(__name__)


class MemoryBackend(BaseBackend):
    """
    An in-memory backend implementation for the adaptive rate limiter.

    This backend provides a simple, Redis-free implementation suitable for:
    - Testing and development
    - Single-process applications
    - Scenarios where distributed state is not needed

    Key Features:
    - Pure in-memory dict-based storage
    - TTL (Time-To-Live) support for automatic expiration
    - Async-safe operations using asyncio.Lock
    - Automatic cleanup of expired entries
    - No external dependencies beyond Python stdlib

    Note:
        This backend is NOT suitable for:
        - Multi-process applications
        - Distributed systems
        - Production environments with high availability requirements
    """

    def __init__(
        self,
        namespace: str = "rate_limiter_memory",
        key_ttl: int = 3600,
        released_reservations_ttl: float = 3600.0,
        released_reservations_cleanup_interval: float = 1800.0,
    ) -> None:
        """
        Initialize the in-memory backend.

        Args:
            namespace: Namespace for key isolation (for compatibility)
            key_ttl: Default TTL for keys in seconds
            released_reservations_ttl: TTL for released reservation tracking in seconds
            released_reservations_cleanup_interval: Interval for cleaning up old released reservations
        """
        super().__init__(namespace)
        self.key_ttl = key_ttl
        self.released_reservations_ttl = released_reservations_ttl
        self.released_reservations_cleanup_interval = (
            released_reservations_cleanup_interval
        )

        # In-memory storage with TTL tracking
        # Format: Dict[key, Tuple[value, Optional[expiry_timestamp]]]
        self._states: dict[str, tuple[dict[str, Any], float | None]] = {}
        self._rate_limits: dict[str, tuple[dict[str, Any], float | None]] = {}
        self._reservations: dict[str, tuple[dict[str, Any], float | None]] = {}
        self._bucket_cache: tuple[dict[str, Any], float] | None = None
        self._model_cache: dict[str, tuple[dict[str, Any], float]] = {}

        # Expiration heap for O(log n) cleanup of expired entries
        # Format: List[Tuple[expiry_time, storage_name, key]]
        # This enables efficient cleanup by always processing entries with the earliest
        # expiration time first, rather than scanning all entries in each storage dict.
        self._expiration_heap: list[tuple[float, str, str]] = []

        # Idempotency tracking: track released reservation IDs to prevent double-release
        # Format: Dict[key, Set[reservation_id]]
        self._released_reservations: dict[str, set[str]] = defaultdict(set)
        # Track when each reservation was released for TTL-based cleanup
        self._released_reservation_timestamps: dict[str, dict[str, float]] = (
            defaultdict(dict)
        )

        # Failure tracking
        self._failures: list[tuple[float, str, str]] = []  # (timestamp, type, message)
        self._circuit_broken_until: float | None = None

        # Request tracking for capacity management
        self._request_counts: dict[str, list[float]] = defaultdict(
            list
        )  # model -> timestamps
        self._token_counts: dict[str, list[tuple[float, int]]] = defaultdict(
            list
        )  # model -> (timestamp, tokens)

        # Sequence tracking for drift correction
        self._sequences: dict[str, int] = defaultdict(int)  # key -> sequence
        self._request_sequences: dict[str, int] = {}  # request_id -> sequence
        # Track timestamps for request_sequences TTL cleanup
        self._request_sequence_timestamps: dict[
            str, float
        ] = {}  # request_id -> timestamp
        # Default TTL for orphaned request sequences (5 minutes)
        self._request_sequence_ttl = 300.0

        # Async lock for thread safety
        self._lock = asyncio.Lock()

        # Cleanup task
        self._cleanup_task: asyncio.Task[None] | None = None
        self._running = False

        logger.debug(f"Initialized MemoryBackend with namespace '{namespace}'")

    def _is_expired(self, expiry: float | None) -> bool:
        """Check if an entry has expired."""
        if expiry is None:
            return False
        return time.time() > expiry

    def _track_expiry(self, storage_name: str, key: str, expiry: float) -> None:
        """
        Track an entry in the expiration heap for efficient O(log n) cleanup.

        Args:
            storage_name: Name of the storage dict ("states", "rate_limits", "reservations", "model_cache")
            key: The key in the storage dict
            expiry: The expiration timestamp
        """
        heapq.heappush(self._expiration_heap, (expiry, storage_name, key))

    def _get_storage_by_name(
        self, storage_name: str
    ) -> dict[str, tuple[dict[str, Any], Any]] | None:
        """Get a storage dict by its name."""
        storage_map: dict[str, dict[str, tuple[dict[str, Any], Any]]] = {
            "states": self._states,
            "rate_limits": self._rate_limits,
            "reservations": self._reservations,
            "model_cache": self._model_cache,
        }
        return storage_map.get(storage_name)

    def _cleanup_expired_from_heap(self) -> int:
        """
        Remove expired entries using the expiration heap for O(log n) efficiency.

        Uses lazy deletion: heap entries may reference keys that have already been
        deleted or had their expiry updated. These stale entries are validated
        against the primary storage and skipped if no longer valid.

        Returns:
            Number of entries actually removed.
        """
        now = time.time()
        removed = 0

        while self._expiration_heap:
            expiry, storage_name, key = self._expiration_heap[0]

            # If the earliest entry hasn't expired yet, we're done
            if expiry > now:
                break

            # Pop the expired entry
            heapq.heappop(self._expiration_heap)

            # Lazy deletion: verify the entry still exists with the same expiry
            storage = self._get_storage_by_name(storage_name)
            if storage is None:
                continue

            if key not in storage:
                # Key already deleted, skip
                continue

            entry = storage[key]
            actual_expiry = entry[1] if len(entry) >= 2 else None

            # Check if expiry matches (entry wasn't updated with new expiry)
            if actual_expiry == expiry:
                del storage[key]
                removed += 1
                logger.debug(f"Cleaned up expired key: {storage_name}/{key}")

        return removed

    def _auto_cleanup_locked(self) -> None:
        """
        Automatically cleanup expired entries from all storage.

        Uses the O(log n) heap-based cleanup for efficient removal of expired entries.

        IMPORTANT: Must be called while holding self._lock.
        """
        # Use heap-based cleanup for O(log n) efficiency
        self._cleanup_expired_from_heap()

        # Clean up bucket cache if expired (not tracked in heap since it's a single entry)
        if self._bucket_cache and self._is_expired(self._bucket_cache[1]):
            self._bucket_cache = None

    # Core State Management

    async def get_state(self, key: str) -> dict[str, Any] | None:
        """Get the state for a given key."""
        async with self._lock:
            # O(log n) cleanup via heap instead of O(n) scan
            self._cleanup_expired_from_heap()
            if key in self._states:
                data, expiry = self._states[key]
                if not self._is_expired(expiry):
                    # Return a copy and ensure integer types for RateLimitState compatibility
                    result = data.copy()
                    if "remaining_requests" in result:
                        result["remaining_requests"] = int(result["remaining_requests"])
                    if "remaining_tokens" in result:
                        result["remaining_tokens"] = int(result["remaining_tokens"])
                    return result
                else:
                    del self._states[key]
            return None

    async def set_state(self, key: str, state: dict[str, Any]) -> None:
        """Set the state for a given key."""
        async with self._lock:
            expiry = time.time() + self.key_ttl if self.key_ttl > 0 else None
            self._states[key] = (state.copy(), expiry)
            if expiry is not None:
                self._track_expiry("states", key, expiry)

    async def get_all_states(self) -> dict[str, dict[str, Any]]:
        """Get all stored states."""
        async with self._lock:
            # O(log n) cleanup via heap instead of O(n) scan
            self._cleanup_expired_from_heap()
            results = {}
            for key, (data, expiry) in self._states.items():
                if not self._is_expired(expiry):
                    result = data.copy()
                    if "remaining_requests" in result:
                        result["remaining_requests"] = int(result["remaining_requests"])
                    if "remaining_tokens" in result:
                        result["remaining_tokens"] = int(result["remaining_tokens"])
                    results[key] = result
            return results

    async def clear(self) -> None:
        """Clear all stored states."""
        async with self._lock:
            self._states.clear()
            logger.debug("Cleared all states")

    # Capacity Management

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

        This is a simplified implementation that doesn't enforce strict limits
        but tracks reservations for testing purposes. It implements the same
        safety_margin logic as Redis backend for test parity.

        Args:
            key: The key/model to check capacity for
            requests: Number of requests to reserve
            tokens: Number of tokens to reserve
            bucket_limits: Optional dictionary containing rpm_limit and tpm_limit
            safety_margin: Safety margin multiplier (0.0-1.0) applied to capacity checks.
                          For memory backend, this is validated but not strictly enforced.
            request_id: Optional request ID for drift correction tracking
        """
        # Validate safety_margin
        validate_safety_margin(safety_margin)

        async with self._lock:
            # Get current state or create default
            state_data: dict[str, Any] | None
            if key in self._states:
                state_data, expiry = self._states[key]
                if self._is_expired(expiry):
                    del self._states[key]
                    state_data = None
                else:
                    state_data = state_data.copy()
            else:
                state_data = None

            # Create new state if needed
            if state_data is None:
                if not bucket_limits:
                    # No bucket limits provided for new state - allow for backward compatibility
                    reservation_id = str(uuid.uuid4())
                    expiry = time.time() + 60
                    self._reservations[reservation_id] = (
                        {
                            "key": key,
                            "requests": requests,
                            "tokens": tokens,
                            "bucket_limits": bucket_limits,
                            "safety_margin": safety_margin,
                        },
                        expiry,
                    )
                    self._track_expiry("reservations", reservation_id, expiry)
                    return True, reservation_id

                # Create new state with CONSERVATIVE initial values
                # We initialize with only 1 token to prevent cold-start race conditions.
                # When multiple workers hit a new bucket simultaneously, only the first
                # request proceeds; others queue until the first response updates state
                # with actual server-reported limits via _update_rate_limit_state.
                state_data = {
                    "model_id": key,  # Required by RateLimitState
                    "remaining_requests": 1,  # Conservative: only allow first request through
                    "remaining_tokens": 1,  # Conservative: only allow first request through
                    "request_limit": bucket_limits.get("rpm_limit", 100),
                    "token_limit": bucket_limits.get("tpm_limit", 10000),
                    "last_updated": datetime.now(
                        timezone.utc
                    ).isoformat(),  # Use ISO format for compatibility
                    "is_verified": False,  # Mark as unverified until server response
                }

            # Refill tokens based on time elapsed (Token Bucket simulation)
            # ALWAYS refill locally - the server sends remaining count per response,
            # but we need to model the continuous Token Bucket between requests.
            if bucket_limits and "last_updated" in state_data:
                # Update limits if they changed (e.g. bucket discovery refresh)
                if bucket_limits.get("rpm_limit") != state_data.get("request_limit"):
                    state_data["request_limit"] = bucket_limits.get("rpm_limit")

                if bucket_limits.get("tpm_limit") != state_data.get("token_limit"):
                    state_data["token_limit"] = bucket_limits.get("tpm_limit")

                now = time.time()
                try:
                    # Handle both float (legacy) and ISO string (RateLimitState) formats
                    last_updated_val = state_data["last_updated"]
                    if isinstance(last_updated_val, str):
                        last_updated = datetime.fromisoformat(
                            last_updated_val
                        ).timestamp()
                    else:
                        last_updated = float(last_updated_val)

                    elapsed = now - last_updated
                except (ValueError, TypeError):
                    elapsed = 0
                    # Don't reset last_updated here, wait for update at end

                if elapsed > 0:
                    # Calculate refill amounts
                    rpm_limit = state_data.get("request_limit") or 100
                    tpm_limit = state_data.get("token_limit") or 10000

                    # Refill rates (per second)
                    requests_per_sec = rpm_limit / 60.0
                    tokens_per_sec = tpm_limit / 60.0

                    # Add tokens
                    current_requests = state_data.get("remaining_requests")
                    if current_requests is None:
                        current_requests = 0

                    current_tokens = state_data.get("remaining_tokens")
                    if current_tokens is None:
                        current_tokens = 0

                    new_requests = current_requests + (requests_per_sec * elapsed)
                    new_tokens = current_tokens + (tokens_per_sec * elapsed)

                    # Cap at limits
                    state_data["remaining_requests"] = min(new_requests, rpm_limit)
                    state_data["remaining_tokens"] = min(new_tokens, tpm_limit)
                    # Update last_updated to now (ISO format)
                    state_data["last_updated"] = datetime.now(timezone.utc).isoformat()

            # Apply safety margin as a time-based buffer to handle client-server drift
            # Instead of a percentage of remaining capacity (which vanishes near zero),
            # we reserve a fixed amount of time (e.g., 100ms) worth of capacity.
            # safety_margin of 0.9 implies a 0.1s (100ms) buffer.
            time_buffer = max(0.0, 1.0 - safety_margin)

            limits = bucket_limits or {}
            rpm_limit = float(limits.get("rpm_limit", 0))
            tpm_limit = float(limits.get("tpm_limit", 0))

            rpm_buffer = (rpm_limit / 60.0) * time_buffer
            tpm_buffer = (tpm_limit / 60.0) * time_buffer

            current_requests = state_data.get("remaining_requests")
            if current_requests is None:
                current_requests = 0

            current_tokens = state_data.get("remaining_tokens")
            if current_tokens is None:
                current_tokens = 0

            effective_requests = current_requests - rpm_buffer
            effective_tokens = current_tokens - tpm_buffer

            # Calculate dynamic reset_at for Token Bucket BEFORE checking capacity
            # This ensures reset_at is always up-to-date, even when capacity check fails
            remaining_req_for_reset = state_data.get("remaining_requests")
            if remaining_req_for_reset is None:
                remaining_req_for_reset = 0
            if remaining_req_for_reset < 1:
                requests_per_sec = rpm_limit / 60.0 if rpm_limit else 0
                if requests_per_sec > 0:
                    needed = 1.0 - remaining_req_for_reset
                    time_to_refill = max(0, needed / requests_per_sec)
                    reset_at = datetime.now(timezone.utc) + timedelta(
                        seconds=time_to_refill
                    )
                    state_data["reset_at"] = reset_at.isoformat()
                else:
                    # No refill rate, reset to now + 1 minute default
                    state_data["reset_at"] = (
                        datetime.now(timezone.utc) + timedelta(minutes=1)
                    ).isoformat()
            else:
                # We have capacity, reset is effectively now
                state_data["reset_at"] = datetime.now(timezone.utc).isoformat()

            # Check capacity constraints against effective (post-margin) capacity
            if effective_requests < requests or effective_tokens < tokens:
                # FIX: Save the refilled state even when capacity check fails!
                # Without this, refill calculations are discarded and the state
                # stays stuck at 0, preventing recovery after rate limit exhaustion.
                state_expiry = time.time() + self.key_ttl if self.key_ttl > 0 else None
                self._states[key] = (state_data, state_expiry)
                if state_expiry is not None:
                    self._track_expiry("states", key, state_expiry)
                return False, None

            # Reserve capacity - decrement from ACTUAL values (not effective)
            state_data["remaining_requests"] = current_requests - requests
            state_data["remaining_tokens"] = current_tokens - tokens

            # Update state
            state_expiry = time.time() + self.key_ttl if self.key_ttl > 0 else None
            self._states[key] = (state_data, state_expiry)
            if state_expiry is not None:
                self._track_expiry("states", key, state_expiry)

            # Create reservation
            reservation_id = str(uuid.uuid4())
            reservation_expiry = time.time() + 60
            self._reservations[reservation_id] = (
                {
                    "key": key,
                    "requests": requests,
                    "tokens": tokens,
                    "bucket_limits": bucket_limits,
                    "safety_margin": safety_margin,
                },
                reservation_expiry,
            )
            self._track_expiry("reservations", reservation_id, reservation_expiry)

            # Track sequence number for drift correction with timestamp for TTL cleanup
            if request_id:
                self._sequences[key] += 1
                self._request_sequences[request_id] = self._sequences[key]
                self._request_sequence_timestamps[request_id] = time.time()

            logger.debug(
                f"Memory backend: reserved capacity for {key} "
                f"(remaining: {state_data['remaining_requests']} requests, {state_data['remaining_tokens']} tokens)"
            )

            return True, reservation_id

    async def release_reservation(self, key: str, reservation_id: str) -> bool:
        """
        Release a capacity reservation with idempotent protection.

        This implements idempotency semantics by tracking released reservation IDs
        to prevent double-release corruption. Released reservations are tracked
        with TTL and cleaned up periodically.

        Args:
            key: The key the reservation was made for
            reservation_id: The reservation ID to release

        Returns:
            True if reservation was released or already released (idempotent)
            False if reservation not found
        """
        async with self._lock:
            # Idempotency check: has this reservation already been released?
            if reservation_id in self._released_reservations[key]:
                logger.warning(
                    f"Reservation {reservation_id} for {key} was already released (idempotent)"
                )
                return True

            # Check if reservation exists
            if reservation_id not in self._reservations:
                logger.warning(
                    f"Cannot release reservation {reservation_id} for {key}: not found"
                )
                return False

            # Mark as released for idempotency tracking with timestamp
            self._released_reservations[key].add(reservation_id)
            self._released_reservation_timestamps[key][reservation_id] = time.time()

            # Delete the reservation
            del self._reservations[reservation_id]

            logger.debug(
                f"Successfully released reservation {reservation_id} for {key}"
            )
            return True

    async def release_streaming_reservation(
        self,
        key: str,
        reservation_id: str,
        reserved_tokens: int,
        actual_tokens: int,
    ) -> bool:
        """Release streaming reservation with refund-based accounting."""
        # Map key to bucket_id for internal logic compatibility
        bucket_id = key

        async with self._lock:
            # Idempotency check
            if reservation_id in self._released_reservations[bucket_id]:
                return True

            if bucket_id not in self._states:
                return False

            state_data, expiry = self._states[bucket_id]
            if self._is_expired(expiry):
                del self._states[bucket_id]
                return False

            refund = reserved_tokens - actual_tokens
            current = state_data.get("remaining_tokens", 0)
            limit = state_data.get("token_limit", float("inf"))

            # Apply refund
            state_data["remaining_tokens"] = max(0, min(current + refund, limit))

            # Mark as released
            self._released_reservations[bucket_id].add(reservation_id)
            self._released_reservation_timestamps[bucket_id][reservation_id] = (
                time.time()
            )

            # Cleanup reservation if it exists
            if reservation_id in self._reservations:
                del self._reservations[reservation_id]

            return True

    async def check_capacity(
        self, model: str, request_type: str = "default"
    ) -> tuple[bool, float]:
        """
        Check if there's capacity for a request.

        For memory backend, this is simplified and always returns True with 0 wait time.

        Args:
            model: Model identifier to check capacity for
            request_type: Type of request (default behavior)

        Returns:
            Tuple of (has_capacity, wait_time_seconds)
        """
        # Clean up old request timestamps (older than 1 minute)
        async with self._lock:
            current_time = time.time()
            cutoff = current_time - 60

            if model in self._request_counts:
                self._request_counts[model] = [
                    ts for ts in self._request_counts[model] if ts > cutoff
                ]

            # For memory backend, we don't enforce strict limits
            return True, 0.0

    # Request and Failure Management

    async def record_request(
        self,
        model: str,
        request_type: str = "default",
        tokens_used: int | None = None,
    ) -> None:
        """Record a successful request."""
        async with self._lock:
            current_time = time.time()
            self._request_counts[model].append(current_time)

            if tokens_used is not None:
                self._token_counts[model].append((current_time, tokens_used))

    async def record_failure(self, error_type: str, error_message: str = "") -> None:
        """Record a failure for tracking."""
        async with self._lock:
            self._failures.append((time.time(), error_type, error_message))

    async def get_failure_count(self, window_seconds: int = 30) -> int:
        """Get the number of failures within the specified window."""
        async with self._lock:
            cutoff = time.time() - window_seconds
            return sum(1 for ts, _, _ in self._failures if ts > cutoff)

    async def is_circuit_broken(self) -> bool:
        """Check if the circuit breaker is triggered."""
        async with self._lock:
            if self._circuit_broken_until is None:
                return False
            if time.time() < self._circuit_broken_until:
                return True
            # Circuit breaker expired, clear it
            self._circuit_broken_until = None
            return False

    # Rate Limit Management

    async def update_rate_limits(
        self,
        model: str,
        headers: dict[str, str],
        bucket_id: str | None = None,
        request_id: str | None = None,
        status_code: int | None = None,
    ) -> int:
        """
        Update rate limits from API response headers and sync with state.

        This in-memory implementation always succeeds. The status_code parameter
        is accepted for interface compatibility but ignored, as the memory backend
        does not need to handle distributed rate limit scenarios.

        Args:
            model: Model identifier
            headers: Response headers containing rate limit info
            bucket_id: Optional bucket ID to update state used by capacity reservation
            request_id: Optional request ID for drift correction
            status_code: Optional HTTP status code (ignored by memory backend)

        Returns:
            int: Always returns 1 (success) for the memory backend.
        """
        async with self._lock:
            parsed = self._parse_rate_limit_headers(headers)
            expiry = time.time() + self.key_ttl if self.key_ttl > 0 else None
            self._rate_limits[model] = (parsed, expiry)
            if expiry is not None:
                self._track_expiry("rate_limits", model, expiry)

            # Sync with state dict used by check_and_reserve_capacity
            key = bucket_id if bucket_id else model

            # Get existing state to perform conservative update
            existing_state = None
            if key in self._states:
                existing_state, _ = self._states[key]

            # Calculate new values using Sequence Numbers for Drift Correction
            new_remaining_requests = parsed.get("rpm_remaining", 0)
            new_remaining_tokens = parsed.get("tpm_remaining", 0)

            # Drift Correction Logic:
            # If we have a request_id, we can determine exactly how many requests were
            # "in-flight" (sent after this request) and adjust the header value accordingly.
            # This eliminates the "State Erasure" problem where updating from a header
            # ignores subsequent requests.
            in_flight_requests = 0
            if request_id and request_id in self._request_sequences:
                seq_resp = self._request_sequences.pop(request_id)
                # Also remove the timestamp tracking
                self._request_sequence_timestamps.pop(request_id, None)
                seq_curr = self._sequences[key]
                # In-flight = Total Sent (curr) - Sent up to this request (resp)
                in_flight_requests = max(0, seq_curr - seq_resp)

                # Adjust header value to account for in-flight requests
                # We subtract in-flight requests from the header's remaining count
                # to estimate what the server WOULD say if it saw all our requests.
                new_remaining_requests = max(
                    0, new_remaining_requests - in_flight_requests
                )

                # Note: We don't track token sequence, so we can't adjust tokens precisely.
                # But typically requests are the bottleneck for 429s in this context.

            if existing_state:
                current_remaining_requests = existing_state.get("remaining_requests", 0)
                current_remaining_tokens = existing_state.get("remaining_tokens", 0)

                # Use the minimum of local and (adjusted) server values
                if current_remaining_requests < new_remaining_requests:
                    new_remaining_requests = current_remaining_requests

                if current_remaining_tokens < new_remaining_tokens:
                    new_remaining_tokens = current_remaining_tokens

            # Determine the correct last_updated timestamp
            final_last_updated = time.time()

            if existing_state:
                used_local_requests = new_remaining_requests == existing_state.get(
                    "remaining_requests", 0
                )
                used_local_tokens = new_remaining_tokens == existing_state.get(
                    "remaining_tokens", 0
                )

                if (
                    used_local_requests or used_local_tokens
                ) and "last_updated" in existing_state:
                    # We are using local state, so we must respect its timestamp
                    final_last_updated = existing_state["last_updated"]

            state_data = {
                "remaining_requests": new_remaining_requests,
                "remaining_tokens": new_remaining_tokens,
                "request_limit": parsed.get("rpm_limit", 100),
                "token_limit": parsed.get("tpm_limit", 10000),
                "last_updated": final_last_updated,
                "is_verified": True,
            }
            self._states[key] = (state_data, expiry)
            if expiry is not None:
                self._track_expiry("states", key, expiry)

        return 1

    async def get_rate_limits(self, model: str) -> dict[str, Any]:
        """Get current rate limit state for a model."""
        async with self._lock:
            # O(log n) cleanup via heap instead of O(n) scan
            self._cleanup_expired_from_heap()
            if model in self._rate_limits:
                data, expiry = self._rate_limits[model]
                if not self._is_expired(expiry):
                    return data.copy()
            return {}

    # Advanced Capacity Management

    async def reserve_capacity(
        self, model: str, request_id: str, tokens_estimated: int = 0
    ) -> bool:
        """Reserve capacity for an upcoming request."""
        async with self._lock:
            # For memory backend, always succeed
            expiry = time.time() + 300  # 5 minute reservation
            self._reservations[request_id] = (
                {"model": model, "tokens_estimated": tokens_estimated},
                expiry,
            )
            self._track_expiry("reservations", request_id, expiry)
            return True

    async def release_reservation_by_id(self, request_id: str) -> None:
        """Release a capacity reservation by request ID."""
        async with self._lock:
            if request_id in self._reservations:
                del self._reservations[request_id]

    # Caching Operations

    async def cache_bucket_info(
        self, bucket_data: dict[str, Any], ttl_seconds: int = 3600
    ) -> None:
        """Cache bucket discovery information."""
        async with self._lock:
            expiry = time.time() + ttl_seconds
            self._bucket_cache = (bucket_data.copy(), expiry)

    async def get_cached_bucket_info(self) -> dict[str, Any] | None:
        """Retrieve cached bucket information."""
        async with self._lock:
            if self._bucket_cache is None:
                return None
            data, expiry = self._bucket_cache
            if self._is_expired(expiry):
                self._bucket_cache = None
                return None
            return data.copy()

    async def cache_model_info(
        self, model: str, model_data: dict[str, Any], ttl_seconds: int = 3600
    ) -> None:
        """Cache model information."""
        async with self._lock:
            expiry = time.time() + ttl_seconds
            self._model_cache[model] = (model_data.copy(), expiry)
            self._track_expiry("model_cache", model, expiry)

    async def get_cached_model_info(self, model: str) -> dict[str, Any] | None:
        """Retrieve cached model information."""
        async with self._lock:
            # O(log n) cleanup via heap instead of O(n) scan
            self._cleanup_expired_from_heap()
            if model in self._model_cache:
                data, expiry = self._model_cache[model]
                if not self._is_expired(expiry):
                    return data.copy()
                else:
                    del self._model_cache[model]
            return None

    # Health and Monitoring

    async def health_check(self) -> HealthCheckResult:
        """Perform a health check on the backend."""
        async with self._lock:
            # Perform auto-cleanup during health check
            self._auto_cleanup_locked()

            return HealthCheckResult(
                healthy=True,
                backend_type="memory",
                namespace=self.namespace,
                metadata={
                    "states_count": len(self._states),
                    "rate_limits_count": len(self._rate_limits),
                    "reservations_count": len(self._reservations),
                    "failures_count": len(self._failures),
                    "circuit_broken": self._circuit_broken_until is not None,
                },
            )

    async def get_all_stats(self) -> dict[str, Any]:
        """Get all statistics from the backend."""
        async with self._lock:
            # Calculate stats while holding lock (avoid reentrant lock issues)
            cutoff = time.time() - 30
            recent_failures = sum(1 for ts, _, _ in self._failures if ts > cutoff)

            circuit_broken = False
            if (
                self._circuit_broken_until is not None
                and time.time() < self._circuit_broken_until
            ):
                circuit_broken = True

            return {
                "states_count": len(self._states),
                "rate_limits_count": len(self._rate_limits),
                "reservations_count": len(self._reservations),
                "model_cache_count": len(self._model_cache),
                "bucket_cache_exists": self._bucket_cache is not None,
                "total_failures": len(self._failures),
                "recent_failures_30s": recent_failures,
                "circuit_broken": circuit_broken,
                "request_tracking": {
                    model: len(timestamps)
                    for model, timestamps in self._request_counts.items()
                },
            }

    # Cleanup and Maintenance

    async def cleanup(self) -> None:
        """Clean up backend resources."""
        async with self._lock:
            self._states.clear()
            self._rate_limits.clear()
            self._reservations.clear()
            self._failures.clear()
            self._request_counts.clear()
            self._token_counts.clear()
            self._bucket_cache = None
            self._model_cache.clear()
            self._circuit_broken_until = None
            self._released_reservations.clear()
            self._released_reservation_timestamps.clear()
            # Also clear request sequence tracking to prevent memory leaks
            self._request_sequences.clear()
            self._request_sequence_timestamps.clear()
            self._sequences.clear()
            # Clear the expiration heap
            self._expiration_heap.clear()
            logger.debug("MemoryBackend cleanup completed")

    async def clear_failures(self) -> None:
        """Clear all failure records."""
        async with self._lock:
            self._failures.clear()
            logger.debug("Cleared all failure records")

    async def force_circuit_break(self, duration: float) -> None:
        """Force a circuit break for the specified duration."""
        async with self._lock:
            self._circuit_broken_until = time.time() + duration

    async def start(self) -> None:
        """
        Start the background cleanup task for released reservations.

        This method should be called when the backend is initialized to enable
        automatic cleanup of old released reservation tracking data.
        """
        if self._running:
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("MemoryBackend cleanup task started")

    async def stop(self) -> None:
        """
        Stop the background cleanup task.

        This method should be called during shutdown to gracefully stop the
        cleanup background task.
        """
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task
            self._cleanup_task = None
        logger.info("MemoryBackend cleanup task stopped")

    async def _cleanup_loop(self) -> None:
        """
        Background task that periodically cleans up expired data.

        This loop runs continuously while the backend is active, sleeping for
        the configured cleanup interval between cleanup operations. It cleans up:
        - Expired released reservation tracking data
        - Orphaned request sequence entries (requests that failed before completion)
        """
        while self._running:
            try:
                await asyncio.sleep(self.released_reservations_cleanup_interval)
                await self._cleanup_released_reservations()
                await self._cleanup_orphaned_request_sequences()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}", exc_info=True)

    async def _cleanup_released_reservations(self) -> None:
        """
        Clean up expired released reservation tracking data.

        This method removes tracking entries for released reservations that are older
        than the configured TTL. This prevents unbounded growth of the tracking dict
        while maintaining idempotency protection for recent releases.

        Thread-safe: Uses asyncio lock to ensure atomic cleanup operations.
        """
        async with self._lock:
            current_time = time.time()
            cutoff_time = current_time - self.released_reservations_ttl

            keys_to_clean = []
            total_removed = 0

            # Find all keys with expired reservation tracking
            for (
                key,
                reservation_timestamps,
            ) in self._released_reservation_timestamps.items():
                expired_reservations = [
                    res_id
                    for res_id, timestamp in reservation_timestamps.items()
                    if timestamp < cutoff_time
                ]

                if expired_reservations:
                    # Remove expired reservation IDs from tracking
                    for res_id in expired_reservations:
                        del reservation_timestamps[res_id]
                        self._released_reservations[key].discard(res_id)
                        total_removed += 1

                    # If all reservations for this key have been cleaned, mark key for removal
                    if not reservation_timestamps:
                        keys_to_clean.append(key)

            # Clean up empty key entries
            for key in keys_to_clean:
                del self._released_reservation_timestamps[key]
                if (
                    key in self._released_reservations
                    and not self._released_reservations[key]
                ):
                    del self._released_reservations[key]

            if total_removed > 0:
                logger.debug(
                    f"Cleaned up {total_removed} expired released reservations "
                    f"across {len(keys_to_clean)} keys"
                )

    async def _cleanup_orphaned_request_sequences(self) -> None:
        """
        Clean up orphaned request sequence entries.

        Request sequences are created when capacity is reserved (check_and_reserve_capacity)
        and normally removed when rate limits are updated (update_rate_limits).
        However, if a request fails before completion (e.g., network error, timeout),
        the sequence entry becomes orphaned and can persist indefinitely.

        This method removes entries older than _request_sequence_ttl (default: 5 minutes)
        to prevent unbounded memory growth from failed requests.

        Thread-safe: Uses asyncio lock to ensure atomic cleanup operations.
        """
        async with self._lock:
            current_time = time.time()
            cutoff_time = current_time - self._request_sequence_ttl

            # Find orphaned entries (older than TTL)
            orphaned_request_ids = [
                request_id
                for request_id, timestamp in self._request_sequence_timestamps.items()
                if timestamp < cutoff_time
            ]

            # Remove orphaned entries
            for request_id in orphaned_request_ids:
                self._request_sequences.pop(request_id, None)
                self._request_sequence_timestamps.pop(request_id, None)

            if orphaned_request_ids:
                logger.debug(
                    f"Cleaned up {len(orphaned_request_ids)} orphaned request sequences"
                )
