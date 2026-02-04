"""
Unit tests for BaseSchedulingModeStrategy.

Tests the abstract base class and its concrete implementation requirements.
"""

from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import Mock

import pytest

from adaptive_rate_limiter.strategies.modes.base import BaseSchedulingModeStrategy
from adaptive_rate_limiter.types.request import RequestMetadata


class ConcreteTestStrategy(BaseSchedulingModeStrategy):
    """Concrete implementation for testing BaseSchedulingModeStrategy."""

    def __init__(self, scheduler: Any, config: Any, client: Any):
        super().__init__(scheduler, config, client)
        self._submit_result = None
        self._loop_called = False
        self._start_called = False
        self._stop_called = False

    async def submit_request(
        self, metadata: RequestMetadata, request_func: Callable[[], Awaitable[Any]]
    ) -> Any:
        """Test implementation of submit_request."""
        return self._submit_result

    async def run_scheduling_loop(self) -> None:
        """Test implementation of run_scheduling_loop."""
        self._loop_called = True

    async def start(self) -> None:
        """Test implementation of start."""
        self._start_called = True
        self._running = True

    async def stop(self) -> None:
        """Test implementation of stop."""
        self._stop_called = True
        self._running = False

    def get_metrics(self) -> dict[str, Any]:
        """Test implementation of get_metrics."""
        return {"mode": "test"}


class TestBaseSchedulingModeStrategy:
    """Tests for BaseSchedulingModeStrategy abstract base class."""

    @pytest.fixture
    def mock_scheduler(self):
        """Create a mock scheduler."""
        scheduler = Mock()
        scheduler.should_throttle = Mock(return_value=False)
        scheduler.track_failure = Mock()
        scheduler.calculate_backoff = Mock(return_value=1.0)
        scheduler.extract_response_headers = Mock(return_value={})
        scheduler.metrics_enabled = False
        scheduler.metrics = {}
        return scheduler

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = Mock()
        config.max_retries = 3
        config.backoff_base = 2.0
        config.max_backoff = 60.0
        return config

    @pytest.fixture
    def mock_client(self):
        """Create a mock client."""
        client = Mock()
        client.base_url = "https://api.example.com"
        client.timeout = 30.0
        return client

    @pytest.fixture
    def strategy(self, mock_scheduler, mock_config, mock_client):
        """Create a concrete test strategy instance."""
        return ConcreteTestStrategy(mock_scheduler, mock_config, mock_client)

    def test_init_sets_scheduler(self, strategy, mock_scheduler):
        """Verify scheduler is stored correctly."""
        assert strategy.scheduler is mock_scheduler

    def test_init_sets_config(self, strategy, mock_config):
        """Verify config is stored correctly."""
        assert strategy.config is mock_config

    def test_init_sets_client(self, strategy, mock_client):
        """Verify client is stored correctly."""
        assert strategy.client is mock_client

    def test_init_running_is_false(self, strategy):
        """Verify _running starts as False."""
        assert strategy._running is False

    def test_is_running_returns_false_initially(self, strategy):
        """Verify is_running() returns False before start."""
        assert strategy.is_running() is False

    @pytest.mark.asyncio
    async def test_is_running_returns_true_after_start(self, strategy):
        """Verify is_running() returns True after start()."""
        await strategy.start()
        assert strategy.is_running() is True

    @pytest.mark.asyncio
    async def test_is_running_returns_false_after_stop(self, strategy):
        """Verify is_running() returns False after stop()."""
        await strategy.start()
        await strategy.stop()
        assert strategy.is_running() is False

    @pytest.mark.asyncio
    async def test_start_sets_running_flag(self, strategy):
        """Verify start() sets _running to True."""
        assert strategy._running is False
        await strategy.start()
        assert strategy._running is True
        assert strategy._start_called is True

    @pytest.mark.asyncio
    async def test_stop_clears_running_flag(self, strategy):
        """Verify stop() sets _running to False."""
        await strategy.start()
        assert strategy._running is True
        await strategy.stop()
        assert strategy._running is False
        assert strategy._stop_called is True


class TestAbstractMethodEnforcement:
    """Tests verifying abstract methods require implementation."""

    def test_cannot_instantiate_abstract_class(self):
        """Verify BaseSchedulingModeStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            BaseSchedulingModeStrategy(Mock(), Mock(), Mock())  # type: ignore
        assert "abstract" in str(exc_info.value).lower()

    def test_missing_submit_request_raises_error(self):
        """Verify missing submit_request implementation raises TypeError."""

        class IncompleteStrategy(BaseSchedulingModeStrategy):
            async def run_scheduling_loop(self) -> None:
                pass

            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                pass

            def get_metrics(self) -> dict[str, Any]:
                return {}

        with pytest.raises(TypeError) as exc_info:
            IncompleteStrategy(Mock(), Mock(), Mock())  # type: ignore
        assert "submit_request" in str(exc_info.value)

    def test_missing_run_scheduling_loop_raises_error(self):
        """Verify missing run_scheduling_loop implementation raises TypeError."""

        class IncompleteStrategy(BaseSchedulingModeStrategy):
            async def submit_request(
                self,
                metadata: RequestMetadata,
                request_func: Callable[[], Awaitable[Any]],
            ) -> Any:
                pass

            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                pass

            def get_metrics(self) -> dict[str, Any]:
                return {}

        with pytest.raises(TypeError) as exc_info:
            IncompleteStrategy(Mock(), Mock(), Mock())  # type: ignore
        assert "run_scheduling_loop" in str(exc_info.value)

    def test_missing_start_raises_error(self):
        """Verify missing start implementation raises TypeError."""

        class IncompleteStrategy(BaseSchedulingModeStrategy):
            async def submit_request(
                self,
                metadata: RequestMetadata,
                request_func: Callable[[], Awaitable[Any]],
            ) -> Any:
                pass

            async def run_scheduling_loop(self) -> None:
                pass

            async def stop(self) -> None:
                pass

            def get_metrics(self) -> dict[str, Any]:
                return {}

        with pytest.raises(TypeError) as exc_info:
            IncompleteStrategy(Mock(), Mock(), Mock())  # type: ignore
        assert "start" in str(exc_info.value)

    def test_missing_stop_raises_error(self):
        """Verify missing stop implementation raises TypeError."""

        class IncompleteStrategy(BaseSchedulingModeStrategy):
            async def submit_request(
                self,
                metadata: RequestMetadata,
                request_func: Callable[[], Awaitable[Any]],
            ) -> Any:
                pass

            async def run_scheduling_loop(self) -> None:
                pass

            async def start(self) -> None:
                pass

            def get_metrics(self) -> dict[str, Any]:
                return {}

        with pytest.raises(TypeError) as exc_info:
            IncompleteStrategy(None, None, None)  # type: ignore
        assert "stop" in str(exc_info.value)

    def test_missing_get_metrics_raises_error(self):
        """Verify missing get_metrics implementation raises TypeError."""

        class IncompleteStrategy(BaseSchedulingModeStrategy):
            async def submit_request(
                self,
                metadata: RequestMetadata,
                request_func: Callable[[], Awaitable[Any]],
            ) -> Any:
                pass

            async def run_scheduling_loop(self) -> None:
                pass

            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                pass

        with pytest.raises(TypeError) as exc_info:
            IncompleteStrategy(Mock(), Mock(), Mock())  # type: ignore
        assert "get_metrics" in str(exc_info.value)
