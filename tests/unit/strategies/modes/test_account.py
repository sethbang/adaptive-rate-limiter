"""
Unit tests for AccountModeStrategy.

Tests the ACCOUNT mode strategy - account-level request management.
"""

import asyncio
import contextlib
from unittest.mock import AsyncMock, Mock

import pytest

from adaptive_rate_limiter.exceptions import QueueOverflowError
from adaptive_rate_limiter.strategies.modes.account import (
    METRIC_CURRENT_QUEUE_SIZE,
    METRIC_TOTAL_COMPLETED,
    METRIC_TOTAL_FAILED,
    METRIC_TOTAL_REJECTED,
    METRIC_TOTAL_SCHEDULED,
    AccountModeStrategy,
)
from adaptive_rate_limiter.types.request import RequestMetadata


class TestAccountModeStrategyInit:
    """Tests for AccountModeStrategy initialization."""

    @pytest.fixture
    def mock_scheduler(self):
        """Create a mock scheduler."""
        scheduler = Mock()
        scheduler.circuit_breaker = None
        return scheduler

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = Mock()
        config.max_concurrent_requests = 10
        config.conservative_multiplier = 0.9
        config.scheduler_interval = 0.01
        config.max_queue_size = 1000
        config.request_timeout = 30.0
        return config

    @pytest.fixture
    def mock_client(self):
        """Create a mock client."""
        return Mock()

    @pytest.fixture
    def strategy(self, mock_scheduler, mock_config, mock_client):
        """Create an AccountModeStrategy instance."""
        return AccountModeStrategy(mock_scheduler, mock_config, mock_client)

    def test_init_stores_scheduler(self, strategy, mock_scheduler):
        """Verify scheduler is stored."""
        assert strategy.scheduler is mock_scheduler

    def test_init_stores_config(self, strategy, mock_config):
        """Verify config is stored."""
        assert strategy.config is mock_config

    def test_init_stores_client(self, strategy, mock_client):
        """Verify client is stored."""
        assert strategy.client is mock_client

    def test_init_max_concurrent_requests(self, strategy):
        """Verify max_concurrent_requests from config."""
        assert strategy.max_concurrent_requests == 10

    def test_init_conservative_multiplier(self, strategy):
        """Verify conservative_multiplier from config."""
        assert strategy.conservative_multiplier == 0.9

    def test_init_empty_queues(self, strategy):
        """Verify account_queues starts empty."""
        assert len(strategy.account_queues) == 0

    def test_init_empty_active_requests(self, strategy):
        """Verify active_requests starts empty."""
        assert len(strategy.active_requests) == 0

    def test_init_zero_active_count(self, strategy):
        """Verify active_count starts at zero."""
        assert strategy.active_count == 0

    def test_init_metrics(self, strategy):
        """Verify account_metrics are initialized."""
        assert strategy.account_metrics[METRIC_TOTAL_SCHEDULED] == 0
        assert strategy.account_metrics[METRIC_TOTAL_COMPLETED] == 0
        assert strategy.account_metrics[METRIC_TOTAL_FAILED] == 0
        assert strategy.account_metrics[METRIC_TOTAL_REJECTED] == 0
        assert strategy.account_metrics[METRIC_CURRENT_QUEUE_SIZE] == 0


class TestAccountModeStrategySubmitRequest:
    """Tests for AccountModeStrategy.submit_request()."""

    @pytest.fixture
    def mock_scheduler(self):
        """Create a mock scheduler."""
        scheduler = Mock()
        scheduler.circuit_breaker = None
        return scheduler

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = Mock()
        config.max_concurrent_requests = 10
        config.conservative_multiplier = 0.9
        config.scheduler_interval = 0.01
        config.max_queue_size = 100
        config.request_timeout = 30.0
        return config

    @pytest.fixture
    def strategy(self, mock_scheduler, mock_config):
        """Create an AccountModeStrategy instance."""
        return AccountModeStrategy(mock_scheduler, mock_config, Mock())

    @pytest.fixture
    def metadata(self):
        """Create request metadata."""
        return RequestMetadata(
            request_id="req-123",
            model_id="test-model",
            resource_type="chat",
        )

    @pytest.mark.asyncio
    async def test_submit_request_returns_schedule_result(self, strategy, metadata):
        """Test submit_request returns ScheduleResult."""
        request_func = AsyncMock(return_value="success")

        result = await strategy.submit_request(metadata, request_func)

        assert result.request is not None
        assert result.wait_time == 0.0
        assert result.should_retry is True

    @pytest.mark.asyncio
    async def test_submit_request_adds_to_queue(self, strategy, metadata):
        """Test request is added to appropriate queue."""
        request_func = AsyncMock(return_value="success")

        await strategy.submit_request(metadata, request_func)

        queue_key = metadata.model_id
        assert queue_key in strategy.account_queues
        assert len(strategy.account_queues[queue_key]) == 1

    @pytest.mark.asyncio
    async def test_submit_request_increments_scheduled_metric(self, strategy, metadata):
        """Test scheduled metric is incremented."""
        request_func = AsyncMock(return_value="success")

        await strategy.submit_request(metadata, request_func)

        assert strategy.account_metrics[METRIC_TOTAL_SCHEDULED] == 1

    @pytest.mark.asyncio
    async def test_submit_request_updates_queue_size_metric(self, strategy, metadata):
        """Test queue size metric is updated."""
        request_func = AsyncMock(return_value="success")

        await strategy.submit_request(metadata, request_func)

        assert strategy.account_metrics[METRIC_CURRENT_QUEUE_SIZE] == 1

    @pytest.mark.asyncio
    async def test_submit_request_circuit_breaker_active(self, strategy, metadata):
        """Test request is rejected when circuit breaker is active."""
        strategy.scheduler.circuit_breaker = Mock()
        strategy.scheduler.circuit_breaker.can_execute = AsyncMock(return_value=False)

        request_func = AsyncMock(return_value="success")

        with pytest.raises(Exception) as exc_info:
            await strategy.submit_request(metadata, request_func)

        assert "Circuit breaker active" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_submit_request_queue_overflow(self, strategy, metadata, mock_config):
        """Test QueueOverflowError when queue is full."""
        mock_config.max_queue_size = 2

        request_func = AsyncMock(return_value="success")

        # Fill the queue
        for i in range(2):
            meta = RequestMetadata(
                request_id=f"req-{i}",
                model_id="test-model",
                resource_type="chat",
            )
            await strategy.submit_request(meta, request_func)

        # This should overflow
        with pytest.raises(QueueOverflowError) as exc_info:
            await strategy.submit_request(metadata, request_func)

        assert "full" in str(exc_info.value).lower()
        assert strategy.account_metrics[METRIC_TOTAL_REJECTED] == 1


class TestAccountModeStrategySchedulingLoop:
    """Tests for AccountModeStrategy scheduling loop."""

    @pytest.fixture
    def mock_scheduler(self):
        """Create a mock scheduler."""
        scheduler = Mock()
        scheduler.circuit_breaker = None
        return scheduler

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = Mock()
        config.max_concurrent_requests = 2
        config.conservative_multiplier = 0.9
        config.scheduler_interval = 0.001  # Fast interval for tests
        config.max_queue_size = 100
        config.request_timeout = 30.0
        return config

    @pytest.fixture
    def strategy(self, mock_scheduler, mock_config):
        """Create an AccountModeStrategy instance."""
        return AccountModeStrategy(mock_scheduler, mock_config, Mock())

    @pytest.fixture
    def metadata(self):
        """Create request metadata."""
        return RequestMetadata(
            request_id="req-123",
            model_id="test-model",
            resource_type="chat",
        )

    @pytest.mark.asyncio
    async def test_loop_respects_concurrency_limit(self, strategy):
        """Test loop respects max_concurrent_requests."""
        # Simulate reaching max concurrent requests
        strategy.active_count = strategy.max_concurrent_requests

        # Add requests to queue
        metadata = RequestMetadata(
            request_id="req-1", model_id="test", resource_type="chat"
        )
        await strategy.submit_request(metadata, AsyncMock(return_value="success"))

        # Run one iteration
        await strategy._loop_account_mode()

        # Queue should not be processed
        assert len(strategy.account_queues["test"]) == 1

    @pytest.mark.asyncio
    async def test_find_eligible_queues_returns_non_empty(self, strategy, metadata):
        """Test _find_eligible_queues returns queues with items."""
        request_func = AsyncMock(return_value="success")
        await strategy.submit_request(metadata, request_func)

        eligible = await strategy._find_eligible_queues()

        assert metadata.model_id in eligible

    @pytest.mark.asyncio
    async def test_find_eligible_queues_excludes_empty(self, strategy, metadata):
        """Test _find_eligible_queues excludes empty queues."""
        # Create an empty queue entry
        strategy.account_queues["empty-queue"]  # Access to create defaultdict entry

        eligible = await strategy._find_eligible_queues()

        assert "empty-queue" not in eligible


class TestAccountModeStrategyExecution:
    """Tests for AccountModeStrategy request execution."""

    @pytest.fixture
    def mock_scheduler(self):
        """Create a mock scheduler."""
        scheduler = Mock()
        scheduler.circuit_breaker = None
        return scheduler

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = Mock()
        config.max_concurrent_requests = 10
        config.conservative_multiplier = 0.9
        config.scheduler_interval = 0.001
        config.max_queue_size = 100
        config.request_timeout = 1.0  # Short timeout for tests
        return config

    @pytest.fixture
    def strategy(self, mock_scheduler, mock_config):
        """Create an AccountModeStrategy instance."""
        return AccountModeStrategy(mock_scheduler, mock_config, Mock())

    @pytest.fixture
    def metadata(self):
        """Create request metadata."""
        return RequestMetadata(
            request_id="req-execute-test",
            model_id="test-model",
            resource_type="chat",
        )

    @pytest.mark.asyncio
    async def test_execute_request_success(self, strategy, metadata):
        """Test successful request execution."""
        request_func = AsyncMock(return_value="success")

        _result = await strategy.submit_request(metadata, request_func)

        # Get the queued request
        queue_key = metadata.model_id
        queued_request = strategy.account_queues[queue_key].popleft()

        # Execute the request
        await strategy._execute_account_request(queued_request)

        # Verify future is resolved
        assert queued_request.future.result() == "success"
        assert strategy.account_metrics[METRIC_TOTAL_COMPLETED] == 1

    @pytest.mark.asyncio
    async def test_execute_request_failure(self, strategy, metadata):
        """Test failed request execution."""
        error = ValueError("test error")
        request_func = AsyncMock(side_effect=error)

        _result = await strategy.submit_request(metadata, request_func)

        # Get the queued request
        queue_key = metadata.model_id
        queued_request = strategy.account_queues[queue_key].popleft()

        # Execute the request
        await strategy._execute_account_request(queued_request)

        # Verify future has exception
        with pytest.raises(ValueError):
            queued_request.future.result()

        assert strategy.account_metrics[METRIC_TOTAL_FAILED] == 1

    @pytest.mark.asyncio
    async def test_execute_request_tracks_active(self, strategy, metadata):
        """Test active request tracking during execution."""
        event = asyncio.Event()

        async def slow_request():
            await event.wait()
            return "success"

        request_func = AsyncMock(side_effect=slow_request)
        _result = await strategy.submit_request(metadata, request_func)

        # Get the queued request
        queue_key = metadata.model_id
        queued_request = strategy.account_queues[queue_key].popleft()

        # Start execution in background
        task = asyncio.create_task(strategy._execute_account_request(queued_request))

        # Allow task to start
        await asyncio.sleep(0.01)

        # Should be tracked as active
        assert strategy.active_count == 1
        assert metadata.request_id in strategy.active_requests

        # Complete the request
        event.set()
        await task

        # Should be removed from tracking
        assert strategy.active_count == 0
        assert metadata.request_id not in strategy.active_requests

    @pytest.mark.asyncio
    async def test_execute_request_timeout(self, strategy, metadata, mock_config):
        """Test request timeout handling."""
        mock_config.request_timeout = 0.01  # Very short timeout

        async def slow_request():
            await asyncio.sleep(1.0)
            return "success"

        request_func = AsyncMock(side_effect=slow_request)
        _result = await strategy.submit_request(metadata, request_func)

        # Get the queued request
        queue_key = metadata.model_id
        queued_request = strategy.account_queues[queue_key].popleft()

        # Execute the request
        await strategy._execute_account_request(queued_request)

        # Verify failed due to timeout
        assert strategy.account_metrics[METRIC_TOTAL_FAILED] == 1


class TestAccountModeStrategyLifecycle:
    """Tests for AccountModeStrategy lifecycle methods."""

    @pytest.fixture
    def strategy(self):
        """Create an AccountModeStrategy instance."""
        mock_scheduler = Mock()
        mock_scheduler.circuit_breaker = None
        mock_config = Mock()
        mock_config.max_concurrent_requests = 10
        mock_config.conservative_multiplier = 0.9
        mock_config.scheduler_interval = 0.01
        mock_config.max_queue_size = 100
        return AccountModeStrategy(mock_scheduler, mock_config, Mock())

    @pytest.mark.asyncio
    async def test_start_sets_running(self, strategy):
        """Test start sets _running to True."""
        assert strategy._running is False
        await strategy.start()
        assert strategy._running is True

    @pytest.mark.asyncio
    async def test_stop_clears_running(self, strategy):
        """Test stop sets _running to False."""
        await strategy.start()
        await strategy.stop()
        assert strategy._running is False

    @pytest.mark.asyncio
    async def test_run_scheduling_loop_processes_requests(self, strategy):
        """Test scheduling loop processes queued requests."""
        await strategy.start()

        # Add a request
        metadata = RequestMetadata(
            request_id="req-loop-test",
            model_id="test",
            resource_type="chat",
        )
        completed = asyncio.Event()

        async def completing_request():
            completed.set()
            return "done"

        request_func = AsyncMock(side_effect=completing_request)
        await strategy.submit_request(metadata, request_func)

        # Run loop in background
        loop_task = asyncio.create_task(strategy.run_scheduling_loop())

        # Wait for request to complete
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(completed.wait(), timeout=1.0)

        # Stop the loop
        await strategy.stop()
        loop_task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await loop_task


class TestAccountModeStrategyMetrics:
    """Tests for AccountModeStrategy metrics."""

    @pytest.fixture
    def strategy(self):
        """Create an AccountModeStrategy instance."""
        mock_scheduler = Mock()
        mock_scheduler.circuit_breaker = None
        mock_config = Mock()
        mock_config.max_concurrent_requests = 10
        mock_config.conservative_multiplier = 0.9
        mock_config.scheduler_interval = 0.01
        mock_config.max_queue_size = 100
        return AccountModeStrategy(mock_scheduler, mock_config, Mock())

    def test_get_metrics_includes_mode(self, strategy):
        """Test metrics includes mode name."""
        metrics = strategy.get_metrics()
        assert metrics["mode"] == "account"

    def test_get_metrics_includes_active_requests(self, strategy):
        """Test metrics includes active_requests count."""
        strategy.active_count = 5
        metrics = strategy.get_metrics()
        assert metrics["active_requests"] == 5

    def test_get_metrics_includes_num_queues(self, strategy):
        """Test metrics includes number of queues."""
        strategy.account_queues["queue1"]  # Create queue
        strategy.account_queues["queue2"]  # Create queue
        metrics = strategy.get_metrics()
        assert metrics["num_queues"] == 2

    @pytest.mark.asyncio
    async def test_get_metrics_includes_account_metrics(self, strategy):
        """Test metrics includes account-specific metrics."""
        # Simulate some activity
        metadata = RequestMetadata(
            request_id="req-1", model_id="test", resource_type="chat"
        )
        await strategy.submit_request(metadata, AsyncMock(return_value="success"))

        metrics = strategy.get_metrics()

        assert METRIC_TOTAL_SCHEDULED in metrics
        assert metrics[METRIC_TOTAL_SCHEDULED] == 1
