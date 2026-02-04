# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""Exception classes for the adaptive rate limiter library.

This module defines the exception hierarchy used throughout the library.
All exceptions inherit from RateLimiterError, making it easy to catch
all rate limiter-related exceptions with a single except clause.
"""


class RateLimiterError(Exception):
    """Base exception for all rate limiter errors.

    This is the root exception class for the adaptive rate limiter library.
    Catch this exception to handle any error originating from the library.

    Example:
        try:
            await scheduler.execute(request)
        except RateLimiterError as e:
            logger.error(f"Rate limiter error: {e}")
    """

    pass


class CapacityExceededError(RateLimiterError):
    """Raised when a rate limit bucket has no available capacity.

    This exception is raised when attempting to acquire capacity from a
    rate limit bucket that is currently exhausted. The exception provides
    information about which bucket was exhausted and how long to wait
    before retrying.

    Attributes:
        bucket_id: The identifier of the exhausted rate limit bucket.
            May be None if the bucket context is not available.
        retry_after: Suggested wait time in seconds before retrying.
            May be None if retry timing cannot be determined.

    Example:
        try:
            await scheduler.execute(request)
        except CapacityExceededError as e:
            if e.retry_after is not None:
                await asyncio.sleep(e.retry_after)
                # Retry the request
            else:
                # Use exponential backoff
                await asyncio.sleep(1.0)
    """

    def __init__(
        self,
        message: str,
        bucket_id: str | None = None,
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.bucket_id = bucket_id
        self.retry_after = retry_after


class BucketNotFoundError(RateLimiterError):
    """Raised when a rate limit bucket does not exist.

    This exception is raised when attempting to access or modify a rate
    limit bucket that has not been created or has been removed. This
    typically indicates a configuration issue or that the bucket was
    never initialized.

    Attributes:
        bucket_id: The identifier of the bucket that was not found.

    Example:
        try:
            bucket = await backend.get_bucket("api-calls")
        except BucketNotFoundError as e:
            logger.warning(f"Bucket '{e.bucket_id}' not found, creating...")
            await backend.create_bucket(e.bucket_id, default_config)
    """

    def __init__(self, bucket_id: str):
        super().__init__(f"Rate limit bucket not found: {bucket_id}")
        self.bucket_id = bucket_id


class ReservationCapacityError(RateLimiterError):
    """Raised when the reservation tracker is at capacity.

    This exception is raised when attempting to create a new reservation
    but the reservation tracker has reached its maximum number of tracked
    reservations. This is a safeguard to prevent unbounded memory growth.

    This typically indicates either:
    - Too many concurrent requests are in flight
    - Reservations are not being properly released after completion
    - The max_reservations configuration is too low for the workload

    Example:
        try:
            async with tracker.reserve(request_id, capacity=1):
                await process_request()
        except ReservationCapacityError:
            # Shed load or queue the request externally
            await external_queue.push(request)
    """

    pass


class BackendConnectionError(RateLimiterError):
    """Raised when connection to the storage backend fails.

    This exception is raised when the rate limiter cannot establish or
    maintain a connection to its storage backend (e.g., Redis, in-memory).
    This may be a transient network issue or indicate that the backend
    service is unavailable.

    Example:
        try:
            backend = await RedisBackend.connect(redis_url)
        except BackendConnectionError:
            logger.warning("Redis unavailable, falling back to memory backend")
            backend = MemoryBackend()
    """

    pass


class BackendOperationError(RateLimiterError):
    """Raised when a backend operation fails.

    This exception is raised when a specific operation on the storage
    backend fails after the connection has been established. This could
    be due to data corruption, serialization issues, or backend-specific
    errors.

    Example:
        try:
            await backend.update_bucket(bucket_id, new_limits)
        except BackendOperationError as e:
            logger.error(f"Failed to update bucket: {e}")
            # Consider retry with backoff or fail gracefully
    """

    pass


class ConfigurationError(RateLimiterError):
    """Raised when configuration is invalid.

    This exception is raised during initialization or reconfiguration
    when the provided configuration values are invalid, incompatible,
    or missing required fields.

    Common causes include:
    - Invalid rate limit values (negative or zero)
    - Incompatible backend and strategy combinations
    - Missing required configuration fields
    - Type mismatches in configuration values

    Example:
        try:
            scheduler = AdaptiveScheduler(config)
        except ConfigurationError as e:
            logger.error(f"Invalid configuration: {e}")
            raise SystemExit(1)
    """

    pass


class QueueOverflowError(RateLimiterError):
    """Raised when a request queue is full and cannot accept more requests.

    This exception is raised when attempting to enqueue a request but the
    queue has reached its maximum capacity. This is a backpressure mechanism
    to prevent unbounded queue growth under heavy load.

    Attributes:
        queue_key: The identifier of the queue that overflowed.
            May be None if the queue context is not available.

    Example:
        try:
            await scheduler.schedule(request)
        except QueueOverflowError as e:
            if e.queue_key:
                logger.warning(f"Queue '{e.queue_key}' is full")
            # Return 503 Service Unavailable or apply backpressure
            raise HTTPException(status_code=503, detail="Service overloaded")
    """

    def __init__(self, message: str, queue_key: str | None = None):
        super().__init__(message)
        self.queue_key = queue_key


class TooManyFailedRequestsError(RateLimiterError):
    """Raised when too many requests have failed within a time window.

    This exception is raised when the failure rate exceeds a threshold,
    indicating potential service degradation or upstream issues. This is
    a circuit-breaker mechanism to prevent cascading failures.

    Attributes:
        failure_count: The number of failed requests in the current window.
        window_seconds: The time window in seconds over which failures are tracked.
        threshold: The failure threshold that was exceeded.

    Example:
        try:
            await scheduler.execute(request)
        except TooManyFailedRequestsError as e:
            logger.error(
                f"Circuit breaker tripped: {e.failure_count} failures "
                f"in {e.window_seconds}s (threshold: {e.threshold})"
            )
            # Back off or fail fast
            raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    """

    def __init__(
        self,
        message: str = "Too many failed requests",
        failure_count: int | None = None,
        window_seconds: float | None = None,
        threshold: int | None = None,
    ):
        super().__init__(message)
        self.failure_count = failure_count
        self.window_seconds = window_seconds
        self.threshold = threshold
