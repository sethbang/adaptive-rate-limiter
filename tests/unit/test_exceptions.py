"""Unit tests for the exceptions module.

Tests all exception classes defined in adaptive_rate_limiter.exceptions.
"""

import pytest

from adaptive_rate_limiter.exceptions import (
    BackendConnectionError,
    BackendOperationError,
    BucketNotFoundError,
    CapacityExceededError,
    ConfigurationError,
    QueueOverflowError,
    RateLimiterError,
    ReservationCapacityError,
    TooManyFailedRequestsError,
)


class TestRateLimiterError:
    """Tests for the base RateLimiterError exception."""

    def test_can_be_caught_as_exception(self):
        """RateLimiterError can be caught as a standard Exception."""
        with pytest.raises(Exception):  # noqa: B017
            raise RateLimiterError("test error")

    def test_message_preserved(self):
        """RateLimiterError preserves its message."""
        error = RateLimiterError("test message")
        assert str(error) == "test message"

    def test_can_be_raised_without_message(self):
        """RateLimiterError can be raised without a message."""
        error = RateLimiterError()
        assert str(error) == ""


class TestCapacityExceededError:
    """Tests for CapacityExceededError."""

    def test_can_be_caught_as_rate_limiter_error(self):
        """CapacityExceededError can be caught as RateLimiterError."""
        with pytest.raises(RateLimiterError):
            raise CapacityExceededError("capacity exceeded")

    def test_stores_bucket_id(self):
        """CapacityExceededError stores bucket_id attribute."""
        error = CapacityExceededError("capacity exceeded", bucket_id="api-bucket-1")
        assert error.bucket_id == "api-bucket-1"

    def test_stores_retry_after(self):
        """CapacityExceededError stores retry_after attribute."""
        error = CapacityExceededError("capacity exceeded", retry_after=5.5)
        assert error.retry_after == 5.5

    def test_stores_both_attributes(self):
        """CapacityExceededError stores both bucket_id and retry_after."""
        error = CapacityExceededError(
            "capacity exceeded", bucket_id="bucket-123", retry_after=10.0
        )
        assert error.bucket_id == "bucket-123"
        assert error.retry_after == 10.0

    def test_defaults_to_none(self):
        """CapacityExceededError defaults bucket_id and retry_after to None."""
        error = CapacityExceededError("capacity exceeded")
        assert error.bucket_id is None
        assert error.retry_after is None

    def test_message_preserved(self):
        """CapacityExceededError preserves its message."""
        error = CapacityExceededError("custom message", bucket_id="b1")
        assert str(error) == "custom message"


class TestBucketNotFoundError:
    """Tests for BucketNotFoundError."""

    def test_can_be_caught_as_rate_limiter_error(self):
        """BucketNotFoundError can be caught as RateLimiterError."""
        with pytest.raises(RateLimiterError):
            raise BucketNotFoundError("missing-bucket")

    def test_stores_bucket_id(self):
        """BucketNotFoundError stores bucket_id attribute."""
        error = BucketNotFoundError("api-bucket-2")
        assert error.bucket_id == "api-bucket-2"

    def test_message_includes_bucket_id(self):
        """BucketNotFoundError message includes the bucket_id."""
        error = BucketNotFoundError("my-bucket")
        assert "my-bucket" in str(error)
        assert "not found" in str(error).lower()


class TestReservationCapacityError:
    """Tests for ReservationCapacityError."""

    def test_can_be_caught_as_rate_limiter_error(self):
        """ReservationCapacityError can be caught as RateLimiterError."""
        with pytest.raises(RateLimiterError):
            raise ReservationCapacityError("reservation limit reached")

    def test_message_preserved(self):
        """ReservationCapacityError preserves its message."""
        error = ReservationCapacityError("too many reservations")
        assert str(error) == "too many reservations"


class TestBackendConnectionError:
    """Tests for BackendConnectionError."""

    def test_can_be_caught_as_rate_limiter_error(self):
        """BackendConnectionError can be caught as RateLimiterError."""
        with pytest.raises(RateLimiterError):
            raise BackendConnectionError("connection failed")

    def test_can_be_instantiated(self):
        """BackendConnectionError can be instantiated with a message."""
        error = BackendConnectionError("Redis connection refused")
        assert str(error) == "Redis connection refused"

    def test_can_be_instantiated_without_message(self):
        """BackendConnectionError can be instantiated without a message."""
        error = BackendConnectionError()
        assert str(error) == ""


class TestBackendOperationError:
    """Tests for BackendOperationError."""

    def test_can_be_caught_as_rate_limiter_error(self):
        """BackendOperationError can be caught as RateLimiterError."""
        with pytest.raises(RateLimiterError):
            raise BackendOperationError("operation failed")

    def test_can_be_instantiated(self):
        """BackendOperationError can be instantiated with a message."""
        error = BackendOperationError("Failed to serialize data")
        assert str(error) == "Failed to serialize data"

    def test_can_be_instantiated_without_message(self):
        """BackendOperationError can be instantiated without a message."""
        error = BackendOperationError()
        assert str(error) == ""


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_can_be_caught_as_rate_limiter_error(self):
        """ConfigurationError can be caught as RateLimiterError."""
        with pytest.raises(RateLimiterError):
            raise ConfigurationError("invalid config")

    def test_can_be_instantiated(self):
        """ConfigurationError can be instantiated with a message."""
        error = ConfigurationError("Invalid rate limit value: -1")
        assert str(error) == "Invalid rate limit value: -1"

    def test_can_be_instantiated_without_message(self):
        """ConfigurationError can be instantiated without a message."""
        error = ConfigurationError()
        assert str(error) == ""


class TestQueueOverflowError:
    """Tests for QueueOverflowError."""

    def test_can_be_caught_as_rate_limiter_error(self):
        """QueueOverflowError can be caught as RateLimiterError."""
        with pytest.raises(RateLimiterError):
            raise QueueOverflowError("queue full")

    def test_stores_queue_key(self):
        """QueueOverflowError stores queue_key attribute."""
        error = QueueOverflowError("queue overflow", queue_key="requests-queue")
        assert error.queue_key == "requests-queue"

    def test_defaults_queue_key_to_none(self):
        """QueueOverflowError defaults queue_key to None."""
        error = QueueOverflowError("queue overflow")
        assert error.queue_key is None

    def test_message_preserved(self):
        """QueueOverflowError preserves its message."""
        error = QueueOverflowError("custom overflow message", queue_key="q1")
        assert str(error) == "custom overflow message"


class TestTooManyFailedRequestsError:
    """Tests for TooManyFailedRequestsError."""

    def test_can_be_caught_as_rate_limiter_error(self):
        """TooManyFailedRequestsError can be caught as RateLimiterError."""
        with pytest.raises(RateLimiterError):
            raise TooManyFailedRequestsError("too many failures")

    def test_stores_failure_count(self):
        """TooManyFailedRequestsError stores failure_count attribute."""
        error = TooManyFailedRequestsError("failures exceeded", failure_count=15)
        assert error.failure_count == 15

    def test_stores_window_seconds(self):
        """TooManyFailedRequestsError stores window_seconds attribute."""
        error = TooManyFailedRequestsError("failures exceeded", window_seconds=60.0)
        assert error.window_seconds == 60.0

    def test_stores_threshold(self):
        """TooManyFailedRequestsError stores threshold attribute."""
        error = TooManyFailedRequestsError("failures exceeded", threshold=10)
        assert error.threshold == 10

    def test_stores_all_attributes(self):
        """TooManyFailedRequestsError stores all attributes."""
        error = TooManyFailedRequestsError(
            "circuit breaker tripped",
            failure_count=20,
            window_seconds=30.0,
            threshold=15,
        )
        assert error.failure_count == 20
        assert error.window_seconds == 30.0
        assert error.threshold == 15
        assert str(error) == "circuit breaker tripped"

    def test_defaults_all_to_none(self):
        """TooManyFailedRequestsError defaults all attributes to None."""
        error = TooManyFailedRequestsError("failures")
        assert error.failure_count is None
        assert error.window_seconds is None
        assert error.threshold is None

    def test_default_message(self):
        """TooManyFailedRequestsError has a default message."""
        error = TooManyFailedRequestsError()
        assert str(error) == "Too many failed requests"


class TestExceptionHierarchy:
    """Tests for the exception inheritance hierarchy."""

    def test_all_exceptions_inherit_from_rate_limiter_error(self):
        """All custom exceptions inherit from RateLimiterError."""
        exceptions = [
            CapacityExceededError("test"),
            BucketNotFoundError("bucket"),
            ReservationCapacityError("test"),
            BackendConnectionError("test"),
            BackendOperationError("test"),
            ConfigurationError("test"),
            QueueOverflowError("test"),
            TooManyFailedRequestsError("test"),
        ]
        for exc in exceptions:
            assert isinstance(exc, RateLimiterError)
            assert isinstance(exc, Exception)

    def test_catching_base_exception_catches_all(self):
        """Catching RateLimiterError catches all subclass exceptions."""
        error_classes = [
            lambda: CapacityExceededError("test"),
            lambda: BucketNotFoundError("bucket"),
            lambda: ReservationCapacityError("test"),
            lambda: BackendConnectionError("test"),
            lambda: BackendOperationError("test"),
            lambda: ConfigurationError("test"),
            lambda: QueueOverflowError("test"),
            lambda: TooManyFailedRequestsError("test"),
        ]

        for error_factory in error_classes:
            try:
                raise error_factory()
            except RateLimiterError:
                pass  # Expected - all should be caught
            else:
                pytest.fail(
                    f"Exception {error_factory()} was not caught as RateLimiterError"
                )
