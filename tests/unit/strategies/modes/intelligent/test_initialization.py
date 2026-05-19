"""
Unit tests for IntelligentModeStrategy initialization, lifecycle, and metrics.

Tests the setup, startup, shutdown, and metrics collection of the strategy.
"""

import pytest

from adaptive_rate_limiter.strategies.modes.intelligent import (
    ReservationMetrics,
    StreamingMetrics,
)

# ============================================================================
# Initialization Tests
# ============================================================================


class TestIntelligentModeStrategyInit:
    """Tests for IntelligentModeStrategy initialization."""

    def test_init_stores_dependencies(
        self, strategy, mock_provider, mock_classifier, mock_state_manager
    ):
        """Test dependencies are stored."""
        assert strategy.provider is mock_provider
        assert strategy.classifier is mock_classifier
        assert strategy.state_manager is mock_state_manager

    def test_init_creates_empty_queues(self, strategy):
        """Test queues are initialized empty."""
        assert len(strategy.fast_queues) == 0
        assert len(strategy.queue_info) == 0

    def test_init_running_is_false(self, strategy):
        """Test _running starts False."""
        assert strategy._running is False

    def test_init_creates_reservation_metrics(self, strategy):
        """Test reservation metrics are created."""
        assert isinstance(strategy._reservation_metrics, ReservationMetrics)

    def test_init_creates_streaming_metrics(self, strategy):
        """Test streaming metrics are created."""
        assert isinstance(strategy._streaming_metrics, StreamingMetrics)

    def test_init_class_variables(self, strategy):
        """Test class-level configuration variables."""
        assert strategy.MAX_RESERVATION_AGE == 240
        assert strategy.MAX_RESERVATIONS == 10000
        assert strategy.STALE_CLEANUP_INTERVAL == 60


# ============================================================================
# Lifecycle Tests
# ============================================================================


class TestIntelligentModeStrategyLifecycle:
    """Tests for lifecycle methods."""

    @pytest.mark.asyncio
    async def test_start_sets_running(self, strategy):
        """Test start sets _running."""
        await strategy.start()

        assert strategy._running is True

        await strategy.stop()

    @pytest.mark.asyncio
    async def test_stop_clears_running(self, strategy):
        """Test stop clears _running."""
        await strategy.start()
        await strategy.stop()

        assert strategy._running is False

    @pytest.mark.asyncio
    async def test_stop_cancels_reset_watchers(self, strategy):
        """Test stop cancels rate limit reset watchers."""
        import time

        await strategy.start()

        # Create a watcher
        await strategy._reset_watcher.schedule_watcher("bucket-1", time.time() + 100)

        assert len(strategy._reset_watcher._reset_tasks) == 1

        await strategy.stop()

        assert len(strategy._reset_watcher._reset_tasks) == 0


# ============================================================================
# Background Tasks Tests
# ============================================================================


class TestIntelligentModeStrategyBackgroundTasks:
    """Tests for background cleanup tasks."""

    @pytest.mark.asyncio
    async def test_start_creates_cleanup_tasks(self, strategy):
        """Test start creates background tasks."""
        await strategy.start()

        assert strategy._running is True
        assert strategy._cleanup_task is not None
        assert strategy._stale_cleanup_task is not None
        assert strategy._streaming_cleanup_manager._cleanup_task is not None

        await strategy.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_cleanup_tasks(self, strategy):
        """Test stop cancels background tasks."""
        await strategy.start()
        await strategy.stop()

        assert strategy._running is False


# ============================================================================
# Metrics Tests
# ============================================================================


class TestIntelligentModeStrategyMetrics:
    """Tests for metrics collection."""

    def test_get_metrics_includes_mode(self, strategy):
        """Test metrics includes mode name."""
        metrics = strategy.get_metrics()

        assert metrics["mode"] == "intelligent"

    def test_get_metrics_includes_queue_counts(self, strategy):
        """Test metrics includes queue counts."""
        metrics = strategy.get_metrics()

        assert "total_queues" in metrics
        assert "queues_with_items" in metrics

    def test_get_metrics_includes_reservation_metrics(self, strategy):
        """Test metrics includes reservation metrics."""
        metrics = strategy.get_metrics()

        assert "reservation_metrics" in metrics

    def test_get_metrics_includes_streaming_metrics(self, strategy):
        """Test metrics includes streaming metrics."""
        metrics = strategy.get_metrics()

        assert "streaming_metrics" in metrics

    def test_get_streaming_metrics(self, strategy):
        """Test get_streaming_metrics returns instance."""
        streaming_metrics = strategy.get_streaming_metrics()

        assert isinstance(streaming_metrics, StreamingMetrics)


class TestIntelligentModeReservationAge:
    """Tests for the effective stale-reservation age invariant."""

    def _make_strategy(self, request_timeout):
        from unittest.mock import AsyncMock, Mock

        from adaptive_rate_limiter.strategies.modes.intelligent import (
            IntelligentModeStrategy,
        )

        config = Mock()
        config.request_timeout = request_timeout
        config.batch_size = 50
        config.scheduler_interval = 0.001
        config.rate_limit_buffer_ratio = 0.9
        config.max_queue_size = 1000
        config.overflow_policy = "reject"
        config.max_concurrent_executions = 100

        scheduler = Mock()
        scheduler.metrics_enabled = False
        state_manager = Mock()
        state_manager.backend = Mock()
        state_manager.get_state = AsyncMock(return_value=None)

        return IntelligentModeStrategy(
            scheduler=scheduler,
            config=config,
            client=Mock(),
            provider=Mock(),
            classifier=Mock(),
            state_manager=state_manager,
        )

    def test_effective_age_uses_default_for_short_timeout(self):
        """A short request_timeout keeps the default reservation age."""
        strategy = self._make_strategy(request_timeout=30.0)
        assert strategy._max_reservation_age == strategy.MAX_RESERVATION_AGE

    def test_effective_age_exceeds_long_request_timeout(self):
        """A request_timeout larger than MAX_RESERVATION_AGE must raise the
        effective reservation age above it.

        Regression: MAX_RESERVATION_AGE was a fixed 240s ClassVar, so a
        configured request_timeout > 240s caused the stale-reservation
        cleanup to reclaim live, in-flight reservations.
        """
        request_timeout = 600.0
        strategy = self._make_strategy(request_timeout=request_timeout)
        assert strategy._max_reservation_age > request_timeout
        # And the reservation tracker is configured with the same value.
        assert (
            strategy._reservation_tracker._max_reservation_age
            == strategy._max_reservation_age
        )
