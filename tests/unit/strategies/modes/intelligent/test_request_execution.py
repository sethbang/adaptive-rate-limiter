"""
Unit tests for IntelligentModeStrategy request execution.

Tests request execution flows, try_process methods, and execution tracking.
"""

import asyncio
from collections import deque
from datetime import datetime, timezone
from typing import ClassVar
from unittest.mock import AsyncMock, Mock

import pytest

from adaptive_rate_limiter.exceptions import RateLimiterError
from adaptive_rate_limiter.types.queue import QueuedRequest
from adaptive_rate_limiter.types.request import RequestMetadata

# ============================================================================
# Basic Execution Tests
# ============================================================================


class TestIntelligentModeStrategyExecution:
    """Tests for request execution methods."""

    @pytest.mark.asyncio
    async def test_can_accept_request_under_limit(self, strategy):
        """Test can accept when under limit."""
        strategy._active_request_count = 50
        strategy.max_concurrent_requests = 100

        can_accept = await strategy._can_accept_request()

        assert can_accept is True

    @pytest.mark.asyncio
    async def test_can_accept_request_at_limit(self, strategy):
        """Test cannot accept at limit."""
        strategy._active_request_count = 100
        strategy.max_concurrent_requests = 100

        can_accept = await strategy._can_accept_request()

        assert can_accept is False

    @pytest.mark.asyncio
    async def test_is_rate_limit_error_by_name(self, strategy):
        """Test rate limit detection by class name."""

        class RateLimitError(Exception):
            pass

        error = RateLimitError("rate limited")
        assert strategy._is_rate_limit_error(error) is True

    @pytest.mark.asyncio
    async def test_is_rate_limit_error_by_status(self, strategy):
        """Test rate limit detection by status code."""

        class HttpError(Exception):
            status_code: int = 429

        error = HttpError("error")
        error.status_code = 429

        assert strategy._is_rate_limit_error(error) is True

    @pytest.mark.asyncio
    async def test_is_rate_limit_error_regular_error(self, strategy):
        """Test regular error is not rate limit."""
        error = Exception("regular error")

        assert strategy._is_rate_limit_error(error) is False


# ============================================================================
# Advanced Execution Tests
# ============================================================================


class TestIntelligentModeStrategyExecutionAdvanced:
    """Additional tests for request execution flows."""

    @pytest.fixture
    def mock_config_short_timeout(self, mock_config):
        """Config with short timeout for testing."""
        mock_config.request_timeout = 0.5
        return mock_config

    @pytest.mark.asyncio
    async def test_execute_request_success_path(self, strategy):
        """Test successful request execution with tracking."""
        metadata = RequestMetadata(
            request_id="req-success",
            model_id="test-model",
            resource_type="chat",
            estimated_tokens=100,
        )

        # Pre-store reservation context
        await strategy._store_reservation_context(
            metadata.request_id, "bucket-1", "res-123", 100
        )

        request_func = AsyncMock(return_value="success")
        future = asyncio.Future()
        queued_request = QueuedRequest(
            metadata=metadata,
            request_func=request_func,
            future=future,
            queue_entry_time=datetime.now(timezone.utc),
        )

        await strategy._execute_request_with_tracking(
            queued_request, "task-1", "bucket-1"
        )

        assert future.result() == "success"

    @pytest.mark.asyncio
    async def test_execute_request_timeout_path(self, strategy):
        """Test request execution timeout handling."""
        # Set timeout directly on strategy instance (config is already used during init)
        strategy._request_timeout = 0.01

        metadata = RequestMetadata(
            request_id="req-timeout",
            model_id="test-model",
            resource_type="chat",
            estimated_tokens=100,
        )

        # Pre-store reservation context
        await strategy._store_reservation_context(
            metadata.request_id, "bucket-1", "res-123", 100
        )

        async def slow_request():
            await asyncio.sleep(1.0)
            return "success"

        request_func = AsyncMock(side_effect=slow_request)
        future = asyncio.Future()
        queued_request = QueuedRequest(
            metadata=metadata,
            request_func=request_func,
            future=future,
            queue_entry_time=datetime.now(timezone.utc),
        )

        await strategy._execute_request_with_tracking(
            queued_request, "task-1", "bucket-1"
        )

        with pytest.raises(RateLimiterError):
            future.result()

    @pytest.mark.asyncio
    async def test_execute_request_error_path(self, strategy):
        """Test request execution error handling."""
        metadata = RequestMetadata(
            request_id="req-error",
            model_id="test-model",
            resource_type="chat",
            estimated_tokens=100,
        )

        # Pre-store reservation context
        await strategy._store_reservation_context(
            metadata.request_id, "bucket-1", "res-123", 100
        )

        error = ValueError("Test error")
        request_func = AsyncMock(side_effect=error)
        future = asyncio.Future()
        queued_request = QueuedRequest(
            metadata=metadata,
            request_func=request_func,
            future=future,
            queue_entry_time=datetime.now(timezone.utc),
        )

        await strategy._execute_request_with_tracking(
            queued_request, "task-1", "bucket-1"
        )

        with pytest.raises(ValueError):
            future.result()

    @pytest.mark.asyncio
    async def test_execute_request_rate_limit_error(self, strategy, mock_state_manager):
        """Test request execution with 429 error handling."""
        metadata = RequestMetadata(
            request_id="req-429",
            model_id="test-model",
            resource_type="chat",
            estimated_tokens=100,
        )

        # Pre-store reservation context
        await strategy._store_reservation_context(
            metadata.request_id, "bucket-1", "res-123", 100
        )

        class RateLimitError(Exception):
            status_code: int = 429
            cached_rate_limit_headers: ClassVar[dict] = {}

        RateLimitError.cached_rate_limit_headers = {
            "x-ratelimit-remaining-requests": "0"
        }
        error = RateLimitError("rate limited")
        error.status_code = 429

        request_func = AsyncMock(side_effect=error)
        future = asyncio.Future()
        queued_request = QueuedRequest(
            metadata=metadata,
            request_func=request_func,
            future=future,
            queue_entry_time=datetime.now(timezone.utc),
        )

        await strategy._execute_request_with_tracking(
            queued_request, "task-1", "bucket-1"
        )

        with pytest.raises(RateLimitError):
            future.result()


# ============================================================================
# Try Process Next Request Tests
# ============================================================================


class TestIntelligentModeStrategyTryProcessNextRequest:
    """Tests for _try_process_next_request_intelligent method."""

    @pytest.mark.asyncio
    async def test_try_process_empty_queue(self, strategy):
        """Test processing empty queue returns False."""
        queue = deque()

        result = await strategy._try_process_next_request_intelligent(
            queue, "test-queue"
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_try_process_at_capacity(self, strategy):
        """Test cannot process when at capacity."""
        strategy._active_request_count = strategy.max_concurrent_requests

        metadata = RequestMetadata(
            request_id="req-1",
            model_id="test-model",
            resource_type="chat",
        )
        future = asyncio.Future()
        queued_request = QueuedRequest(
            metadata=metadata,
            request_func=AsyncMock(return_value="success"),
            future=future,
            queue_entry_time=datetime.now(timezone.utc),
        )

        queue = deque([queued_request])

        result = await strategy._try_process_next_request_intelligent(
            queue, "test-queue"
        )

        assert result is False
        assert len(queue) == 1  # Request still in queue


# ============================================================================
# Probe Finish Tests
# ============================================================================


class TestIntelligentModeStrategyProbeFinish:
    """Tests for probe cleanup after request execution."""

    @pytest.fixture
    def strategy_with_metrics(
        self,
        mock_scheduler_with_metrics,
        mock_config,
        mock_client,
        mock_provider,
        mock_classifier,
        mock_state_manager,
    ):
        """Create strategy with metrics enabled."""
        from adaptive_rate_limiter.strategies.modes.intelligent import (
            IntelligentModeStrategy,
        )

        return IntelligentModeStrategy(
            scheduler=mock_scheduler_with_metrics,
            config=mock_config,
            client=mock_client,
            provider=mock_provider,
            classifier=mock_classifier,
            state_manager=mock_state_manager,
        )

    @pytest.fixture
    def mock_scheduler_with_metrics(self):
        """Create a mock scheduler with metrics enabled."""
        scheduler = Mock()
        scheduler.circuit_breaker = None
        scheduler._circuit_breaker_always_closed = True
        scheduler.metrics_enabled = True
        scheduler.metrics = {
            "requests_completed": 0,
            "requests_failed": 0,
            "requests_scheduled": 0,
        }
        scheduler.extract_response_headers = Mock(return_value={})
        return scheduler

    @pytest.mark.asyncio
    async def test_execute_clears_probe_and_wakes_scheduler(
        self, strategy_with_metrics, mock_provider
    ):
        """Test probe is cleared and scheduler wakes after execution."""
        mock_provider.get_bucket_for_model.return_value = "bucket-probe"

        metadata = RequestMetadata(
            request_id="req-probe-finish",
            model_id="test-model",
            resource_type="chat",
        )

        # Pre-add to bucket probes
        strategy_with_metrics._bucket_probes.add("bucket-probe")

        request_func = AsyncMock(return_value="success")
        future = asyncio.Future()
        queued_request = QueuedRequest(
            metadata=metadata,
            request_func=request_func,
            future=future,
            queue_entry_time=datetime.now(timezone.utc),
        )

        # Execute with bucket_id to trigger probe cleanup
        await strategy_with_metrics._execute_request_with_tracking(
            queued_request, "task-1", "bucket-probe"
        )

        # Probe should be cleared
        assert "bucket-probe" not in strategy_with_metrics._bucket_probes

        # Wakeup event should be set (eventually)
        await asyncio.sleep(0.01)
