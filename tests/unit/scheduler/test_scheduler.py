from unittest.mock import AsyncMock, Mock

import pytest

from adaptive_rate_limiter.exceptions import RateLimiterError
from adaptive_rate_limiter.scheduler.config import RateLimiterConfig, SchedulerMode
from adaptive_rate_limiter.scheduler.scheduler import Scheduler, create_scheduler
from adaptive_rate_limiter.types.request import RequestMetadata


class TestScheduler:
    @pytest.fixture
    def mock_client(self):
        return Mock()

    @pytest.fixture
    def mock_config(self):
        return RateLimiterConfig(mode=SchedulerMode.BASIC)

    @pytest.fixture
    def scheduler(self, mock_client, mock_config):
        return Scheduler(client=mock_client, config=mock_config)

    @pytest.mark.asyncio
    async def test_submit_request_delegation(self, scheduler):
        """Test that submit_request delegates to mode strategy."""
        # Mock mode strategy
        scheduler.mode_strategy = AsyncMock()
        scheduler.mode_strategy.submit_request.return_value = "result"

        # Must be running
        scheduler._running = True

        metadata = RequestMetadata(
            resource_type="chat", model_id="test", request_id="req-1"
        )
        request_func = AsyncMock()

        result = await scheduler.submit_request(metadata, request_func)

        assert result == "result"
        scheduler.mode_strategy.submit_request.assert_called_once_with(
            metadata, request_func
        )

    @pytest.mark.asyncio
    async def test_submit_request_not_running(self, scheduler):
        """Test submit_request raises error if not running."""
        scheduler._running = False

        metadata = RequestMetadata(
            resource_type="chat", model_id="test", request_id="req-1"
        )
        request_func = AsyncMock()

        with pytest.raises(RateLimiterError, match="Scheduler is not running"):
            await scheduler.submit_request(metadata, request_func)

    @pytest.mark.asyncio
    async def test_scheduler_loop_delegation(self, scheduler):
        """Test that scheduler loop delegates to mode strategy."""
        scheduler.mode_strategy = AsyncMock()

        # Should delegate for INTELLIGENT mode
        scheduler.config.mode = SchedulerMode.INTELLIGENT
        await scheduler._scheduler_loop()
        scheduler.mode_strategy.run_scheduling_loop.assert_called_once()

    @pytest.mark.asyncio
    async def test_scheduler_loop_basic_mode(self, scheduler):
        """Test that scheduler loop does nothing in BASIC mode."""
        scheduler.mode_strategy = AsyncMock()

        # Should NOT delegate for BASIC mode
        scheduler.config.mode = SchedulerMode.BASIC
        await scheduler._scheduler_loop()
        scheduler.mode_strategy.run_scheduling_loop.assert_not_called()

    def test_get_metrics_combination(self, scheduler):
        """Test combining base metrics with mode metrics."""
        scheduler._running = True

        # Mock mode strategy metrics
        scheduler.mode_strategy = Mock()
        scheduler.mode_strategy.get_metrics.return_value = {
            "mode_metric": 123,
            "scheduler_type": "OverriddenType",  # Should override base
        }

        metrics = scheduler.get_metrics()

        assert metrics["mode_metric"] == 123
        assert metrics["scheduler_type"] == "OverriddenType"
        assert metrics["running"] is True
        # Config is BASIC from fixture
        assert metrics["scheduler_mode"] == "basic"


class TestCreateScheduler:
    @pytest.fixture
    def mock_client(self):
        return Mock()

    def test_create_scheduler_default(self, mock_client):
        """Test creating scheduler with defaults."""
        # Default is INTELLIGENT, so we need provider, classifier, and state_manager
        mock_provider = Mock()
        mock_classifier = Mock()
        mock_state_manager = Mock()

        scheduler = create_scheduler(
            client=mock_client,
            provider=mock_provider,
            classifier=mock_classifier,
            state_manager=mock_state_manager,
        )
        assert isinstance(scheduler, Scheduler)
        assert scheduler.config.mode == SchedulerMode.INTELLIGENT

    def test_create_scheduler_explicit_mode(self, mock_client):
        """Test creating scheduler with explicit mode."""
        scheduler = create_scheduler(client=mock_client, mode="basic")
        assert scheduler.config.mode == SchedulerMode.BASIC

    def test_create_scheduler_invalid_mode(self, mock_client):
        """Test creating scheduler with invalid mode."""
        with pytest.raises(ValueError, match="Unknown scheduler mode"):
            create_scheduler(client=mock_client, mode="invalid_mode")

    def test_create_scheduler_with_config(self, mock_client):
        """Test creating scheduler with provided config."""
        # We must pass mode="account" because create_scheduler overrides config.mode with default "intelligent"
        # And we need dependencies for ACCOUNT mode (likely similar to INTELLIGENT)

        config = RateLimiterConfig(mode=SchedulerMode.ACCOUNT)
        mock_provider = Mock()
        mock_classifier = Mock()
        mock_state_manager = Mock()

        scheduler = create_scheduler(
            client=mock_client,
            config=config,
            mode="account",
            provider=mock_provider,
            classifier=mock_classifier,
            state_manager=mock_state_manager,
        )
        assert scheduler.config.mode == SchedulerMode.ACCOUNT

    def test_create_scheduler_config_override(self, mock_client):
        """Test that mode param overrides config mode."""
        config = RateLimiterConfig(mode=SchedulerMode.ACCOUNT)
        scheduler = create_scheduler(client=mock_client, config=config, mode="basic")
        assert scheduler.config.mode == SchedulerMode.BASIC
