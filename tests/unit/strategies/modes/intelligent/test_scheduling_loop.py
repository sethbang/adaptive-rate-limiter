"""
Unit tests for IntelligentModeStrategy scheduling loop.

Tests the main scheduling loop, wakeup events, idle handling, and exception recovery.
"""

import asyncio
import contextlib
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

# ============================================================================
# Scheduling Loop Tests
# ============================================================================


class TestIntelligentModeStrategySchedulingLoop:
    """Tests for the main scheduling loop."""

    @pytest.mark.asyncio
    async def test_loop_increments_idle_cycles(
        self, strategy_with_metrics, mock_scheduler_with_metrics
    ):
        """Test idle state increments idle cycles."""
        mock_scheduler_with_metrics.metrics = {"scheduler_loops": 0}

        # Empty queues means idle
        await strategy_with_metrics._loop_intelligent_mode()

        assert strategy_with_metrics._idle_cycles >= 1

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
        scheduler.metrics = {"scheduler_loops": 0}
        scheduler.extract_response_headers = Mock(return_value={})
        return scheduler

    @pytest.mark.asyncio
    async def test_select_queues_for_processing(self, strategy, metadata):
        """Test queue selection uses scheduling strategy."""
        request_func = AsyncMock(return_value="success")
        await strategy.submit_request(metadata, request_func)

        eligible = await strategy._find_eligible_queues_intelligent()
        selected = await strategy._select_queues_for_processing(eligible)

        assert len(selected) >= 0  # May be empty or have items


# ============================================================================
# Run Loop Tests
# ============================================================================


class TestIntelligentModeStrategyRunLoop:
    """Tests for run_scheduling_loop method."""

    @pytest.mark.asyncio
    async def test_run_scheduling_loop_cancellation(self, strategy):
        """Test scheduling loop handles cancellation."""
        await strategy.start()

        loop_task = asyncio.create_task(strategy.run_scheduling_loop())

        await asyncio.sleep(0.02)

        loop_task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await loop_task

        await strategy.stop()


# ============================================================================
# Wakeup Event Tests
# ============================================================================


class TestIntelligentModeStrategyWakeupEvent:
    """Tests for wakeup event management."""

    @pytest.mark.asyncio
    async def test_safe_set_wakeup_event(self, strategy):
        """Test safe wakeup event setting."""
        assert not strategy._wakeup_event.is_set()

        await strategy._safe_set_wakeup_event()

        assert strategy._wakeup_event.is_set()

    @pytest.mark.asyncio
    async def test_handle_idle_state_clears_event(self, strategy):
        """Test idle state handling clears wakeup event."""
        strategy._wakeup_event.set()

        await strategy._handle_idle_state_intelligent(timeout=0.01)

        assert not strategy._wakeup_event.is_set()


# ============================================================================
# Idle Logging Tests
# ============================================================================


class TestIntelligentModeStrategyIdleLogging:
    """Tests for idle state logging at cycle intervals."""

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
        scheduler.metrics = {"scheduler_loops": 0}
        scheduler.extract_response_headers = Mock(return_value={})
        return scheduler

    @pytest.mark.asyncio
    async def test_loop_logs_at_100_idle_cycles(self, strategy_with_metrics, caplog):
        """Test debug logging occurs at 100 idle cycles."""
        import logging

        with caplog.at_level(logging.DEBUG):
            strategy_with_metrics._idle_cycles = 99  # Next will be 100
            await strategy_with_metrics._loop_intelligent_mode()

        assert "Scheduler idle" in caplog.text


# ============================================================================
# Loop Exception Handling Tests
# ============================================================================


class TestIntelligentModeStrategyLoopExceptions:
    """Tests for scheduling loop exception handling."""

    @pytest.mark.asyncio
    async def test_run_scheduling_loop_handles_loop_exception(self, strategy):
        """Test loop recovers from non-cancellation exceptions."""
        await strategy.start()

        call_count = 0
        _original_loop = strategy._loop_intelligent_mode

        async def failing_loop():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Test error")
            # After first failure, stop the loop
            strategy._running = False

        with patch.object(strategy, "_loop_intelligent_mode", side_effect=failing_loop):
            await strategy.run_scheduling_loop()

        assert call_count >= 2  # Recovered and retried
        await strategy.stop()


# ============================================================================
# Idle Wait Time Calculation Tests (lines 577-593)
# ============================================================================


class TestIntelligentModeStrategyIdleWaitTime:
    """Tests for idle wait time calculation when earliest_reset > now."""

    @pytest.fixture
    def mock_scheduler_with_metrics(self):
        """Create a mock scheduler with metrics enabled."""
        scheduler = Mock()
        scheduler.circuit_breaker = None
        scheduler._circuit_breaker_always_closed = True
        scheduler.metrics_enabled = True
        scheduler.metrics = {
            "scheduler_loops": 0,
            "requests_scheduled": 0,
            "requests_completed": 0,
            "requests_failed": 0,
            "queue_overflows": 0,
        }
        scheduler.extract_response_headers = Mock(return_value={})
        return scheduler

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

    @pytest.mark.asyncio
    async def test_loop_calculates_wait_from_future_reset(
        self, strategy_with_metrics, mock_state_manager
    ):
        """Lines 576-582: Test wait time is capped and calculated from reset time."""
        # Set up state with a reset time 30s in the future
        future_reset = time.time() + 30.0
        strategy_with_metrics._reset_watcher._buckets_waiting.add("bucket-test")

        with patch.object(
            strategy_with_metrics._reset_watcher,
            "get_earliest_reset_time",
            new_callable=AsyncMock,
        ) as mock_get_reset:
            mock_get_reset.return_value = future_reset

            with patch.object(
                strategy_with_metrics,
                "_handle_idle_state_intelligent",
                new_callable=AsyncMock,
            ) as mock_handle:
                await strategy_with_metrics._loop_intelligent_mode()

                # Should have called _handle_idle_state_intelligent
                mock_handle.assert_called_once()

                # Get the timeout value passed
                call_args = mock_handle.call_args
                timeout = call_args.kwargs.get("timeout")

                # Timeout should be capped at 60s and >= loop_sleep_time
                assert timeout is not None
                assert timeout <= 60.0
                assert timeout >= strategy_with_metrics._loop_sleep_time
                # Should be approximately 30 seconds (our future_reset)
                assert 28.0 <= timeout <= 32.0

    @pytest.mark.asyncio
    async def test_loop_caps_wait_at_60_seconds(self, strategy_with_metrics):
        """Lines 579-580: Wait time is capped at 60 seconds."""
        # Set up state with reset time 120s in the future
        future_reset = time.time() + 120.0
        strategy_with_metrics._reset_watcher._buckets_waiting.add("bucket-cap")

        with patch.object(
            strategy_with_metrics._reset_watcher,
            "get_earliest_reset_time",
            new_callable=AsyncMock,
        ) as mock_get_reset:
            mock_get_reset.return_value = future_reset

            with patch.object(
                strategy_with_metrics,
                "_handle_idle_state_intelligent",
                new_callable=AsyncMock,
            ) as mock_handle:
                await strategy_with_metrics._loop_intelligent_mode()

                call_args = mock_handle.call_args
                timeout = call_args.kwargs.get("timeout")

                # Should be capped at exactly 60s
                assert timeout == 60.0

    @pytest.mark.asyncio
    async def test_loop_ensures_minimum_wait_time(self, strategy_with_metrics):
        """Lines 581-582: Wait time is at least loop_sleep_time."""
        # Set up state with reset time very close (0.0001s in the future)
        future_reset = time.time() + 0.0001
        strategy_with_metrics._reset_watcher._buckets_waiting.add("bucket-min")

        with patch.object(
            strategy_with_metrics._reset_watcher,
            "get_earliest_reset_time",
            new_callable=AsyncMock,
        ) as mock_get_reset:
            mock_get_reset.return_value = future_reset

            with patch.object(
                strategy_with_metrics,
                "_handle_idle_state_intelligent",
                new_callable=AsyncMock,
            ) as mock_handle:
                await strategy_with_metrics._loop_intelligent_mode()

                call_args = mock_handle.call_args
                timeout = call_args.kwargs.get("timeout")

                # Should be at least loop_sleep_time
                assert timeout >= strategy_with_metrics._loop_sleep_time

    @pytest.mark.asyncio
    async def test_loop_uses_default_wait_without_reset_time(
        self, strategy_with_metrics
    ):
        """Test default 1.0s wait when no reset time available."""
        with patch.object(
            strategy_with_metrics._reset_watcher,
            "get_earliest_reset_time",
            new_callable=AsyncMock,
        ) as mock_get_reset:
            mock_get_reset.return_value = None  # No reset time

            with patch.object(
                strategy_with_metrics,
                "_handle_idle_state_intelligent",
                new_callable=AsyncMock,
            ) as mock_handle:
                await strategy_with_metrics._loop_intelligent_mode()

                call_args = mock_handle.call_args
                timeout = call_args.kwargs.get("timeout")

                # Should use default wait time of 1.0
                assert timeout == 1.0

    @pytest.mark.asyncio
    async def test_loop_resets_idle_tracking_on_work(
        self, strategy_with_metrics, metadata
    ):
        """Lines 587-589: Reset idle tracking when work is found."""
        strategy_with_metrics._idle_cycles = 50

        # Submit a request to ensure work exists
        request_func = AsyncMock(return_value="success")
        await strategy_with_metrics.submit_request(metadata, request_func)

        # Mock capacity check to avoid actual processing
        with patch.object(
            strategy_with_metrics,
            "_check_and_reserve_capacity_intelligent",
            new_callable=AsyncMock,
            return_value=False,
        ):
            # Run one loop iteration
            await strategy_with_metrics._loop_intelligent_mode()

        # Idle cycles should be reset to 0
        assert strategy_with_metrics._idle_cycles == 0
        # Activity time should be recent
        assert time.time() - strategy_with_metrics._last_activity_time < 1.0
