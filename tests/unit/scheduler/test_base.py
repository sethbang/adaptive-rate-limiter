import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from adaptive_rate_limiter.exceptions import QueueOverflowError
from adaptive_rate_limiter.scheduler.base import (
    METRIC_QUEUE_OVERFLOWS,
    BaseScheduler,
    FailedRequestCounter,
)
from adaptive_rate_limiter.scheduler.config import RateLimiterConfig, SchedulerMode
from adaptive_rate_limiter.types.request import RequestMetadata


class TestFailedRequestCounter:
    def test_increment_and_count(self):
        """Test incrementing and counting failures."""
        counter = FailedRequestCounter(max_failures=5, window_seconds=10)
        assert counter.count == 0

        assert counter.increment() == 1
        assert counter.count == 1

        assert counter.increment() == 2
        assert counter.count == 2

    def test_window_reset(self):
        """Test that count resets after window expiration."""
        counter = FailedRequestCounter(max_failures=5, window_seconds=0.1)  # type: ignore
        counter.increment()
        assert counter.count == 1

        # Wait for window to expire
        time.sleep(0.15)

        # Next increment should reset count
        assert counter.increment() == 1
        assert counter.count == 1

    def test_is_limit_exceeded(self):
        """Test limit checking."""
        counter = FailedRequestCounter(max_failures=2, window_seconds=10)

        counter.increment()
        assert not counter.is_limit_exceeded()

        counter.increment()
        assert counter.is_limit_exceeded()

        counter.increment()
        assert counter.is_limit_exceeded()

    def test_is_limit_exceeded_window_reset(self):
        """Test limit check resets after window expiration."""
        counter = FailedRequestCounter(max_failures=1, window_seconds=0.1)  # type: ignore
        counter.increment()
        assert counter.is_limit_exceeded()

        time.sleep(0.15)
        assert not counter.is_limit_exceeded()
        assert counter.count == 0  # Should be reset


class ConcreteScheduler(BaseScheduler):
    """Concrete implementation of BaseScheduler for testing."""

    async def submit_request(
        self, metadata: RequestMetadata, request_func: Callable[[], Awaitable[Any]]
    ) -> Any:
        return await request_func()


class TestBaseScheduler:
    @pytest.fixture
    def mock_client(self):
        return Mock()

    @pytest.fixture
    def mock_config(self):
        # Use BASIC mode to avoid needing provider/classifier mocks for base tests
        return RateLimiterConfig(mode=SchedulerMode.BASIC)

    @pytest.fixture
    def scheduler(self, mock_client, mock_config):
        return ConcreteScheduler(client=mock_client, config=mock_config)

    @pytest.mark.asyncio
    async def test_init(self, mock_client, mock_config):
        """Test initialization."""
        scheduler = ConcreteScheduler(client=mock_client, config=mock_config)
        assert scheduler.client == mock_client
        assert scheduler.config == mock_config
        assert not scheduler.is_running()
        # Metrics enabled by default
        assert scheduler.metrics is not None

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, scheduler):
        """Test start and stop lifecycle."""
        # Mock mode strategy
        scheduler.mode_strategy = AsyncMock()

        await scheduler.start()
        assert scheduler.is_running()
        scheduler.mode_strategy.start.assert_called_once()

        await scheduler.stop()
        assert not scheduler.is_running()
        scheduler.mode_strategy.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_idempotency(self, scheduler):
        """Test that start is idempotent."""
        scheduler.mode_strategy = AsyncMock()

        await scheduler.start()
        assert scheduler.is_running()

        # Call start again
        await scheduler.start()
        assert scheduler.is_running()
        # Should still be called once
        scheduler.mode_strategy.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_idempotency(self, scheduler):
        """Test that stop is idempotent."""
        scheduler.mode_strategy = AsyncMock()
        await scheduler.start()

        await scheduler.stop()
        assert not scheduler.is_running()

        # Call stop again
        await scheduler.stop()
        assert not scheduler.is_running()

    # ===== Async Context Manager Tests =====
    @pytest.mark.asyncio
    async def test_async_context_manager_basic(self, scheduler):
        """Test that async with scheduler: works correctly."""
        scheduler.mode_strategy = AsyncMock()

        async with scheduler:
            assert scheduler.is_running()

        assert not scheduler.is_running()
        scheduler.mode_strategy.start.assert_called_once()
        scheduler.mode_strategy.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_context_manager_returns_self(self, scheduler):
        """Test that __aenter__ returns the scheduler instance."""
        scheduler.mode_strategy = AsyncMock()

        async with scheduler as ctx:
            assert ctx is scheduler
            assert ctx.is_running()

        await scheduler.stop()  # Ensure cleanup

    @pytest.mark.asyncio
    async def test_async_context_manager_calls_start_on_entry(self, scheduler):
        """Test that start() is called on entry."""
        scheduler.mode_strategy = AsyncMock()

        # Verify start hasn't been called yet
        scheduler.mode_strategy.start.assert_not_called()

        async with scheduler:
            # start() should have been called on entry
            scheduler.mode_strategy.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_context_manager_calls_stop_on_exit(self, scheduler):
        """Test that stop() is called on exit (even without exceptions)."""
        scheduler.mode_strategy = AsyncMock()

        async with scheduler:
            # stop() not called yet
            scheduler.mode_strategy.stop.assert_not_called()

        # stop() should have been called on exit
        scheduler.mode_strategy.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_context_manager_cleanup_on_exception(self, scheduler):
        """Test that stop() is called on exit even when exception occurs."""
        scheduler.mode_strategy = AsyncMock()

        with pytest.raises(ValueError, match="test error"):
            async with scheduler:
                assert scheduler.is_running()
                raise ValueError("test error")

        # stop() should still have been called despite the exception
        assert not scheduler.is_running()
        scheduler.mode_strategy.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_context_manager_nested_usage(self, scheduler):
        """Test that context manager handles double-start gracefully."""
        scheduler.mode_strategy = AsyncMock()

        async with scheduler:
            # Already running, start should be idempotent
            await scheduler.start()
            assert scheduler.is_running()
            # start() called twice but should only log/return early the second time
            assert scheduler.mode_strategy.start.call_count == 1

    def test_calculate_backoff(self, scheduler):
        """Test exponential backoff calculation."""
        # Mock time to control jitter
        with patch("time.time", return_value=0.0):
            delay_0 = scheduler.calculate_backoff(0, base_delay=1.0)
            assert delay_0 == 1.0

            delay_1 = scheduler.calculate_backoff(1, base_delay=1.0)
            assert delay_1 == 2.0

            delay_2 = scheduler.calculate_backoff(2, base_delay=1.0)
            assert delay_2 == 4.0

    def test_calculate_backoff_max_cap(self, scheduler):
        """Test backoff max cap."""
        scheduler.config.max_backoff = 10.0
        with patch("time.time", return_value=0.0):
            delay = scheduler.calculate_backoff(10, base_delay=1.0)
            assert delay == 10.0

    def test_extract_response_headers_pydantic_private(self, scheduler):
        """Test extracting headers from pydantic private attribute."""
        mock_response = Mock()
        mock_response.headers = {"x-test": "value"}

        result = Mock()
        result.__pydantic_private__ = {"_response": mock_response}

        headers = scheduler.extract_response_headers(result)
        assert headers == {"x-test": "value"}

    def test_extract_response_headers_public_attr(self, scheduler):
        """Test extracting headers from public response attribute."""
        mock_response = Mock()
        mock_response.headers = {"x-test": "value"}

        result = Mock()
        result.response = mock_response

        headers = scheduler.extract_response_headers(result)
        assert headers == {"x-test": "value"}

    def test_extract_response_headers_dict(self, scheduler):
        """Test extracting headers from dict."""
        result = {"_response_headers": {"x-test": "value"}}

        headers = scheduler.extract_response_headers(result)
        assert headers == {"x-test": "value"}

    def test_extract_response_headers_direct(self, scheduler):
        """Test extracting headers directly from object."""
        # Use spec to prevent auto-creation of 'response' attribute which triggers Strategy 2
        result = Mock(spec=["headers"])
        result.headers = {"x-test": "value"}

        headers = scheduler.extract_response_headers(result)
        assert headers == {"x-test": "value"}

    def test_extract_response_headers_none(self, scheduler):
        """Test failure to extract headers."""
        result = "string result"
        headers = scheduler.extract_response_headers(result)
        assert headers is None

    def test_handle_rate_limit_headers(self, scheduler):
        """Test parsing rate limit headers."""
        headers = {
            "x-ratelimit-remaining-requests": "10",
            "x-ratelimit-reset-requests": "1234567890",
            "x-ratelimit-remaining-tokens": "1000",
            "x-ratelimit-reset-tokens": "60",
        }

        info = scheduler.handle_rate_limit_headers(headers)

        assert info["remaining_requests"] == 10
        assert info["reset_requests"] == 1234567890
        assert info["remaining_tokens"] == 1000
        assert info["reset_tokens"] == 60

    def test_handle_rate_limit_headers_invalid(self, scheduler):
        """Test parsing invalid rate limit headers."""
        headers = {
            "x-ratelimit-remaining-requests": "invalid",
        }

        info = scheduler.handle_rate_limit_headers(headers)
        assert "remaining_requests" not in info

    def test_parse_rate_limit_headers_reset_requests(self, scheduler):
        """Test parsing wait time from reset timestamp."""
        current_time = 1000
        reset_time = 1010

        headers = {"x-ratelimit-reset-requests": str(reset_time)}

        with patch("time.time", return_value=current_time):
            wait_time = scheduler.parse_rate_limit_headers(headers)
            assert wait_time == 10.0

    def test_parse_rate_limit_headers_reset_tokens(self, scheduler):
        """Test parsing wait time from reset duration."""
        headers = {"x-ratelimit-reset-tokens": "5"}

        wait_time = scheduler.parse_rate_limit_headers(headers)
        assert wait_time == 5.0

    def test_parse_rate_limit_headers_none(self, scheduler):
        """Test parsing when no headers present."""
        headers = {}
        wait_time = scheduler.parse_rate_limit_headers(headers)
        assert wait_time is None

    @pytest.mark.asyncio
    async def test_get_queue_key_default(self, scheduler):
        """Test queue key generation without provider."""
        metadata = RequestMetadata(
            resource_type="chat", model_id="test-model", request_id="req-1"
        )
        key = await scheduler._get_queue_key(metadata)
        assert key == "default_chat"

    @pytest.mark.asyncio
    async def test_get_queue_key_with_provider(self, mock_client, mock_config):
        """Test queue key generation with provider."""
        mock_provider = AsyncMock()
        mock_provider.get_bucket_for_model.return_value = "bucket-123"

        scheduler = ConcreteScheduler(
            client=mock_client, config=mock_config, provider=mock_provider
        )

        metadata = RequestMetadata(
            resource_type="chat", model_id="test-model", request_id="req-1"
        )
        key = await scheduler._get_queue_key(metadata)

        assert key == "bucket-123:chat"
        mock_provider.get_bucket_for_model.assert_called_with("test-model", "chat")

    def test_should_throttle(self, scheduler):
        """Test should_throttle based on failures."""
        assert not scheduler.should_throttle()

        # Simulate failures
        scheduler._failed_requests.count = 100
        assert scheduler.should_throttle()

    def test_track_failure(self, scheduler):
        """Test failure tracking."""
        count = scheduler.track_failure()
        assert count == 1
        assert scheduler._failed_requests.count == 1

    @pytest.mark.asyncio
    async def test_check_queue_overflow_ok(self, scheduler):
        """Test queue overflow check when OK."""
        scheduler.config.max_queue_size = 10
        result = await scheduler._check_queue_overflow(5, "test_queue")
        assert result is False

    @pytest.mark.asyncio
    async def test_check_queue_overflow_reject(self, scheduler):
        """Test queue overflow rejection."""
        scheduler.config.max_queue_size = 10
        scheduler.config.overflow_policy = "reject"

        with pytest.raises(QueueOverflowError):
            await scheduler._check_queue_overflow(10, "test_queue")

    @pytest.mark.asyncio
    async def test_check_queue_overflow_metrics(self, scheduler):
        """Test metrics update on overflow."""
        scheduler.config.max_queue_size = 10
        scheduler.config.metrics_enabled = True
        scheduler._setup_metrics(True)

        with contextlib.suppress(QueueOverflowError):
            await scheduler._check_queue_overflow(10, "test_queue")

        assert scheduler.metrics[METRIC_QUEUE_OVERFLOWS] == 1

    def test_get_metrics(self, scheduler):
        """Test metrics retrieval."""
        scheduler._running = True
        metrics = scheduler.get_metrics()

        assert metrics["scheduler_type"] == "ConcreteScheduler"
        assert metrics["running"] is True
        assert "failed_requests" in metrics
        assert "total_requests" in metrics

    @pytest.mark.asyncio
    async def test_schedule_rate_limit_reset(self, scheduler):
        """Test scheduling rate limit reset delegation."""
        scheduler.mode_strategy = AsyncMock()

        await scheduler.schedule_rate_limit_reset("bucket-1", 12345)

        scheduler.mode_strategy.schedule_rate_limit_reset.assert_called_with(
            "bucket-1", 12345
        )

    @pytest.mark.asyncio
    async def test_stop_error_handling(self, scheduler):
        """Test error handling during stop."""
        scheduler._running = True

        # Create a real task that we can cancel
        async def dummy_task():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                raise  # Re-raise to allow proper cancellation

        scheduler._scheduler_task = asyncio.create_task(dummy_task())

        # Give the task a moment to start
        await asyncio.sleep(0.01)

        # Should not raise even if cancellation handling has issues
        await scheduler.stop()
        assert not scheduler.is_running()

    @pytest.mark.asyncio
    async def test_stop_state_manager_error(self, scheduler):
        """Test error handling when stopping state manager."""
        scheduler._running = True
        scheduler.state_manager = AsyncMock()
        scheduler.state_manager.stop.side_effect = Exception("State manager error")

        # Should not raise
        await scheduler.stop()
        assert not scheduler.is_running()


class SchedulerWithLoop(BaseScheduler):
    """Scheduler that has a _scheduler_loop method for testing."""

    async def submit_request(
        self, metadata: RequestMetadata, request_func: Callable[[], Awaitable[Any]]
    ) -> Any:
        return await request_func()

    async def _scheduler_loop(self):
        while self._running:  # noqa: ASYNC110
            await asyncio.sleep(0.01)


class TestBaseSchedulerCoverageExpansion:
    """Additional tests to expand coverage for scheduler/base.py."""

    @pytest.fixture
    def mock_client(self):
        return Mock()

    @pytest.fixture
    def mock_config(self):
        return RateLimiterConfig(mode=SchedulerMode.BASIC)

    @pytest.fixture
    def scheduler(self, mock_client, mock_config):
        return ConcreteScheduler(client=mock_client, config=mock_config)

    # ===== Lines 179-181: Mode Strategy Creation Failure =====
    def test_mode_strategy_creation_failure(self, mock_client):
        """Test handling of mode strategy creation failure."""
        with patch(
            "adaptive_rate_limiter.scheduler.base.create_mode_strategy"
        ) as mock_create:
            mock_create.side_effect = ValueError("Invalid mode configuration")

            with pytest.raises(ValueError, match="Invalid mode configuration"):
                ConcreteScheduler(
                    client=mock_client,
                    config=RateLimiterConfig(mode=SchedulerMode.BASIC),
                )

    # ===== Lines 215-216: Metrics Disabled Path =====
    def test_metrics_disabled(self, mock_client):
        """Test scheduler with metrics explicitly disabled."""
        config = RateLimiterConfig(mode=SchedulerMode.BASIC, metrics_enabled=False)
        scheduler = ConcreteScheduler(
            client=mock_client, config=config, metrics_enabled=False
        )

        assert scheduler.metrics == {}
        assert scheduler.metrics_collector is None
        assert not scheduler.metrics_enabled

    # ===== Lines 244-245: StateManager Start =====
    @pytest.mark.asyncio
    async def test_start_with_state_manager(self, scheduler):
        """Test start() initializes state manager."""
        scheduler.state_manager = AsyncMock()
        scheduler.mode_strategy = AsyncMock()

        await scheduler.start()

        scheduler.state_manager.start.assert_called_once()
        await scheduler.stop()

    # ===== Lines 252-255: Scheduler Loop Task Creation =====
    @pytest.mark.asyncio
    async def test_start_with_scheduler_loop(self, mock_client, mock_config):
        """Test start() creates scheduler task when _scheduler_loop exists."""
        scheduler = SchedulerWithLoop(client=mock_client, config=mock_config)
        scheduler.mode_strategy = AsyncMock()

        await scheduler.start()

        assert scheduler._scheduler_task is not None
        assert not scheduler._scheduler_task.done()

        await scheduler.stop()

    # ===== Lines 283-284: Exception During Scheduler Task Stop =====
    @pytest.mark.asyncio
    async def test_stop_with_scheduler_task_runtime_error(self, scheduler):
        """Test stop handles RuntimeError from scheduler task gracefully."""
        scheduler._running = True
        scheduler.mode_strategy = AsyncMock()

        # Create a real task that raises RuntimeError when cancelled
        async def faulty_loop():
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                # Simulate RuntimeError occurring during cancellation handling
                raise RuntimeError(
                    "Simulated runtime error during cancellation"
                ) from None

        scheduler._scheduler_task = asyncio.create_task(faulty_loop())
        await asyncio.sleep(0.01)  # Let task start

        # Should not raise - the RuntimeError should be caught and logged
        await scheduler.stop()
        assert not scheduler.is_running()

    # ===== Lines 296-300: Active Tasks Cancellation =====
    @pytest.mark.asyncio
    async def test_stop_cancels_active_tasks(self, scheduler):
        """Test stop() cancels all active tasks."""
        scheduler._running = True
        scheduler.mode_strategy = AsyncMock()

        async def long_running_task():
            await asyncio.sleep(100)

        task = asyncio.create_task(long_running_task())
        scheduler._active_tasks.add(task)

        await scheduler.stop()

        assert task.cancelled() or task.done()
        assert len(scheduler._active_tasks) == 0

    # ===== Line 352: Schedule Rate Limit Reset Fallback =====
    @pytest.mark.asyncio
    async def test_schedule_rate_limit_reset_no_support(self, scheduler):
        """Test schedule_rate_limit_reset when mode strategy doesn't support it."""
        scheduler.mode_strategy = Mock(spec=[])  # No schedule_rate_limit_reset

        # Should not raise, just log
        await scheduler.schedule_rate_limit_reset("bucket-1", 12345)

    # ===== Line 377: Queue Key Unknown Model =====
    @pytest.mark.asyncio
    async def test_get_queue_key_unknown_model(self, mock_client, mock_config):
        """Test queue key for unknown model ID."""
        mock_provider = AsyncMock()
        scheduler = ConcreteScheduler(
            client=mock_client, config=mock_config, provider=mock_provider
        )

        metadata = RequestMetadata(
            resource_type="chat", model_id="unknown", request_id="req-1"
        )
        key = await scheduler._get_queue_key(metadata)

        assert key == "default_chat"
        mock_provider.get_bucket_for_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_queue_key_none_model(self, mock_client, mock_config):
        """Test queue key when model_id is None."""
        mock_provider = AsyncMock()
        scheduler = ConcreteScheduler(
            client=mock_client, config=mock_config, provider=mock_provider
        )

        metadata = RequestMetadata(
            resource_type="embeddings",
            model_id=None,  # type: ignore
            request_id="req-2",
        )
        key = await scheduler._get_queue_key(metadata)

        assert key == "default_embeddings"
        mock_provider.get_bucket_for_model.assert_not_called()

    # ===== Line 407: Backoff Default Delay =====
    def test_calculate_backoff_default_delay(self, scheduler):
        """Test backoff with no base_delay provided."""
        with patch("time.time", return_value=0.0):
            delay = scheduler.calculate_backoff(0)  # No base_delay arg
            assert delay == 1.0  # Uses default 1.0

    # ===== Lines 468-469: Private _headers Attribute =====
    def test_extract_response_headers_private_attr(self, scheduler):
        """Test extracting headers from _headers private attribute."""
        result = Mock(spec=["_headers"])
        result._headers = {"x-private": "value"}

        headers = scheduler.extract_response_headers(result)
        assert headers == {"x-private": "value"}

    # ===== Lines 501-502: Invalid Reset Requests Header =====
    def test_handle_rate_limit_headers_invalid_reset_requests(self, scheduler):
        """Test handling invalid reset-requests header."""
        headers = {"x-ratelimit-reset-requests": "not-a-number"}
        info = scheduler.handle_rate_limit_headers(headers)
        assert "reset_requests" not in info

    # ===== Lines 511-512: Invalid Remaining Tokens Header =====
    def test_handle_rate_limit_headers_invalid_remaining_tokens(self, scheduler):
        """Test handling invalid remaining-tokens header."""
        headers = {"x-ratelimit-remaining-tokens": "invalid"}
        info = scheduler.handle_rate_limit_headers(headers)
        assert "remaining_tokens" not in info

    # ===== Lines 521-522: Invalid Reset Tokens Header =====
    def test_handle_rate_limit_headers_invalid_reset_tokens(self, scheduler):
        """Test handling invalid reset-tokens header."""
        headers = {"x-ratelimit-reset-tokens": "bad"}
        info = scheduler.handle_rate_limit_headers(headers)
        assert "reset_tokens" not in info

    # ===== Lines 551-552: Parse Invalid Reset Requests =====
    def test_parse_rate_limit_headers_invalid_reset_requests(self, scheduler):
        """Test parsing invalid reset-requests timestamp."""
        headers = {"x-ratelimit-reset-requests": "not-a-timestamp"}
        wait_time = scheduler.parse_rate_limit_headers(headers)
        assert wait_time is None  # Falls through to return None

    # ===== Lines 564-565: Parse Invalid Reset Tokens =====
    def test_parse_rate_limit_headers_invalid_reset_tokens(self, scheduler):
        """Test parsing invalid reset-tokens duration."""
        headers = {"x-ratelimit-reset-tokens": "invalid"}
        wait_time = scheduler.parse_rate_limit_headers(headers)
        assert wait_time is None

    # ===== Parse Zero Wait Time (x-ratelimit-reset-requests in past) =====
    def test_parse_rate_limit_headers_zero_wait_time(self, scheduler):
        """Test reset time in the past (zero wait)."""
        headers = {"x-ratelimit-reset-requests": "0"}
        with patch("time.time", return_value=1000):
            wait_time = scheduler.parse_rate_limit_headers(headers)
            # wait_time = max(0, 0 - 1000) = 0, so condition `if wait_time > 0` fails
            assert wait_time is None  # Falls through

    # ===== Lines 580, 588: Queue Overflow Non-Reject Policy =====
    @pytest.mark.asyncio
    async def test_check_queue_overflow_non_reject_policy(self, scheduler):
        """Test overflow with non-reject policy returns True."""
        scheduler.config.max_queue_size = 5
        scheduler.config.overflow_policy = "drop"  # Not "reject"

        result = await scheduler._check_queue_overflow(10, "test:queue")
        assert result is True  # Overflow occurred but no exception

    @pytest.mark.asyncio
    async def test_check_queue_overflow_with_metrics_collector(self, scheduler):
        """Test overflow records to metrics collector if present."""
        scheduler.config.max_queue_size = 5
        scheduler.config.overflow_policy = "drop"
        scheduler.metrics_enabled = True
        scheduler.metrics = {METRIC_QUEUE_OVERFLOWS: 0}
        scheduler.metrics_collector = Mock()  # Non-None collector

        await scheduler._check_queue_overflow(10, "bucket-1:chat")
        # Line 580 extracts model_id from queue_key
        assert scheduler.metrics[METRIC_QUEUE_OVERFLOWS] == 1

    # ===== Lines 595-596: Update Metrics for Request =====
    @pytest.mark.asyncio
    async def test_update_metrics_for_request(self, scheduler):
        """Test metrics update for request events."""
        scheduler._setup_metrics(True)

        metadata = RequestMetadata(
            resource_type="chat", model_id="test", request_id="1"
        )
        await scheduler._update_metrics_for_request(metadata, "requests_scheduled")

        assert scheduler.metrics["requests_scheduled"] == 1

    @pytest.mark.asyncio
    async def test_update_metrics_for_request_disabled(self, scheduler):
        """Test metrics update is skipped when metrics disabled."""
        scheduler.metrics_enabled = False
        scheduler.metrics = {}

        metadata = RequestMetadata(
            resource_type="chat", model_id="test", request_id="1"
        )
        await scheduler._update_metrics_for_request(metadata, "requests_scheduled")

        assert "requests_scheduled" not in scheduler.metrics

    # ===== Line 609: Get Metrics With Metrics Enabled =====
    def test_get_metrics_with_metrics_enabled(self, scheduler):
        """Test metrics retrieval includes metric counters when enabled."""
        scheduler._setup_metrics(True)
        scheduler._running = True
        scheduler.metrics["requests_scheduled"] = 5
        scheduler.metrics["requests_completed"] = 3

        metrics = scheduler.get_metrics()

        assert metrics["scheduler_type"] == "ConcreteScheduler"
        assert metrics["running"] is True
        assert metrics["requests_scheduled"] == 5
        assert metrics["requests_completed"] == 3

    # ===== StateManager Stop with Success Logging (Line 290) =====
    @pytest.mark.asyncio
    async def test_stop_with_state_manager_success(self, scheduler):
        """Test successful state manager stop logs debug message."""
        scheduler._running = True
        scheduler.mode_strategy = AsyncMock()
        scheduler.state_manager = AsyncMock()

        await scheduler.stop()

        scheduler.state_manager.stop.assert_called_once()
        assert not scheduler.is_running()
