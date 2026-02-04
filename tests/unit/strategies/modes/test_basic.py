"""
Unit tests for BasicModeStrategy.

Tests the BASIC mode strategy - simple direct execution with retry logic.
"""

import asyncio
import contextlib
import time
from typing import ClassVar
from unittest.mock import AsyncMock, Mock

import pytest

from adaptive_rate_limiter.exceptions import RateLimiterError
from adaptive_rate_limiter.strategies.modes.basic import BasicModeStrategy
from adaptive_rate_limiter.types.request import RequestMetadata


class TestBasicModeStrategyInitDefaults:
    """Tests for BasicModeStrategy initialization defaults (no fixtures needed)."""

    def test_init_default_max_retries(self):
        """Verify max_retries comes from config."""

        class Config:
            max_retries = 3
            backoff_base = 2.0
            max_backoff = 60.0

        strategy = BasicModeStrategy(None, Config(), None)  # type: ignore
        assert strategy.max_retries == 3

    def test_init_default_backoff_base(self):
        """Verify backoff_base comes from config."""

        class Config:
            max_retries = 3
            backoff_base = 2.0
            max_backoff = 60.0

        strategy = BasicModeStrategy(None, Config(), None)  # type: ignore
        assert strategy.backoff_base == 2.0

    def test_init_default_max_backoff(self):
        """Verify max_backoff comes from config."""

        class Config:
            max_retries = 3
            backoff_base = 2.0
            max_backoff = 60.0

        strategy = BasicModeStrategy(None, Config(), None)  # type: ignore
        assert strategy.max_backoff == 60.0

    def test_init_fallback_defaults(self):
        """Verify fallback defaults when config has no attributes."""

        # Use a plain object with no attributes instead of Mock
        class EmptyConfig:
            pass

        strategy = BasicModeStrategy(None, EmptyConfig(), None)  # type: ignore
        assert strategy.max_retries == 3
        assert strategy.backoff_base == 2.0
        assert strategy.max_backoff == 60.0


class TestBasicModeStrategyInit:
    """Tests for BasicModeStrategy initialization with mocks."""

    @pytest.fixture
    def mock_scheduler(self):
        """Create a mock scheduler."""
        scheduler = Mock()
        scheduler.should_throttle = Mock(return_value=False)
        scheduler.track_failure = Mock()
        scheduler.calculate_backoff = Mock(return_value=1.0)
        scheduler.extract_response_headers = Mock(return_value={})
        scheduler._failed_requests = Mock()
        scheduler._failed_requests.count = 0
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
        """Create a BasicModeStrategy instance."""
        return BasicModeStrategy(mock_scheduler, mock_config, mock_client)

    def test_init_stores_scheduler(self, strategy, mock_scheduler):
        """Verify scheduler is stored."""
        assert strategy.scheduler is mock_scheduler

    def test_init_stores_config(self, strategy, mock_config):
        """Verify config is stored."""
        assert strategy.config is mock_config

    def test_init_stores_client(self, strategy, mock_client):
        """Verify client is stored."""
        assert strategy.client is mock_client

    def test_init_running_is_false(self, strategy):
        """Verify _running starts as False."""
        assert strategy._running is False

    def test_init_empty_tracking_dicts(self, strategy):
        """Verify tracking dictionaries are initialized empty."""
        assert len(strategy._last_request_times) == 0


class TestBasicModeStrategySubmitRequest:
    """Tests for BasicModeStrategy.submit_request()."""

    @pytest.fixture
    def mock_scheduler(self):
        """Create a mock scheduler."""
        scheduler = Mock()
        scheduler.should_throttle = Mock(return_value=False)
        scheduler.track_failure = Mock()
        scheduler.calculate_backoff = Mock(return_value=0.001)
        scheduler.extract_response_headers = Mock(return_value={})
        scheduler._failed_requests = Mock()
        scheduler._failed_requests.count = 10
        scheduler.state_manager = None
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
        return Mock()

    @pytest.fixture
    def strategy(self, mock_scheduler, mock_config, mock_client):
        """Create a BasicModeStrategy instance."""
        return BasicModeStrategy(mock_scheduler, mock_config, mock_client)

    @pytest.fixture
    def metadata(self):
        """Create request metadata."""
        return RequestMetadata(
            request_id="req-123",
            model_id="test-model",
            resource_type="chat",
            priority=0,
        )

    @pytest.mark.asyncio
    async def test_submit_request_success(self, strategy, metadata):
        """Test successful request execution."""
        request_func = AsyncMock(return_value="success")

        result = await strategy.submit_request(metadata, request_func)

        assert result == "success"
        assert request_func.call_count == 1

    @pytest.mark.asyncio
    async def test_submit_request_throttled(self, strategy, metadata, mock_scheduler):
        """Test request is rejected when throttled."""
        mock_scheduler.should_throttle.return_value = True
        request_func = AsyncMock(return_value="success")

        with pytest.raises(RateLimiterError) as exc_info:
            await strategy.submit_request(metadata, request_func)

        assert "Too many failed requests" in str(exc_info.value)
        assert request_func.call_count == 0

    @pytest.mark.asyncio
    async def test_submit_request_retry_success_after_failures(
        self, strategy, metadata, mock_scheduler
    ):
        """Test request succeeds after retries."""
        call_count = 0

        async def failing_then_succeeding():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"

        request_func = AsyncMock(side_effect=failing_then_succeeding)

        result = await strategy.submit_request(metadata, request_func)

        assert result == "success"
        assert call_count == 3
        assert mock_scheduler.track_failure.call_count == 2

    @pytest.mark.asyncio
    async def test_submit_request_max_retries_exhausted(
        self, strategy, metadata, mock_scheduler
    ):
        """Test request fails after max retries."""
        request_func = AsyncMock(side_effect=Exception("Persistent failure"))

        with pytest.raises(Exception) as exc_info:
            await strategy.submit_request(metadata, request_func)

        assert "Persistent failure" in str(exc_info.value)
        assert request_func.call_count == 4  # 1 initial + 3 retries
        assert mock_scheduler.track_failure.call_count == 4

    @pytest.mark.asyncio
    async def test_submit_request_tracks_request_time(self, strategy, metadata):
        """Test that successful requests update last request time."""
        request_func = AsyncMock(return_value="success")

        await strategy.submit_request(metadata, request_func)

        assert metadata.model_id in strategy._last_request_times
        assert strategy._last_request_times[metadata.model_id] <= time.time()

    @pytest.mark.asyncio
    async def test_submit_request_cancellation_reraises(self, strategy, metadata):
        """Test that CancelledError is re-raised."""
        request_func = AsyncMock(side_effect=asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await strategy.submit_request(metadata, request_func)


class TestBasicModeStrategyRetryDelay:
    """Tests for retry delay calculation."""

    @pytest.fixture
    def mock_scheduler(self):
        """Create a mock scheduler."""
        scheduler = Mock()
        scheduler.should_throttle = Mock(return_value=False)
        scheduler.track_failure = Mock()
        scheduler.calculate_backoff = Mock(return_value=1.5)
        scheduler.extract_response_headers = Mock(return_value={})
        scheduler.state_manager = None
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
    def strategy(self, mock_scheduler, mock_config):
        """Create a BasicModeStrategy instance."""
        return BasicModeStrategy(mock_scheduler, mock_config, Mock())

    @pytest.fixture
    def metadata(self):
        """Create request metadata."""
        return RequestMetadata(
            request_id="req-123",
            model_id="test-model",
            resource_type="chat",
        )

    def test_calculate_retry_delay_uses_scheduler_backoff(self, strategy, metadata):
        """Test that retry delay uses scheduler's backoff calculation."""
        error = Exception("test error")

        delay = strategy._calculate_retry_delay(error, attempt=1, metadata=metadata)

        assert delay == 1.5
        strategy.scheduler.calculate_backoff.assert_called_once_with(1)

    def test_calculate_retry_delay_uses_retry_after(self, strategy, metadata):
        """Test that retry_after from error takes precedence."""

        class RetryableError(Exception):
            retry_after_seconds: float = 5.0

        error = RetryableError("rate limit")
        error.retry_after_seconds = 5.0

        delay = strategy._calculate_retry_delay(error, attempt=1, metadata=metadata)

        assert delay == 5.0

    def test_calculate_retry_delay_caps_retry_after(self, strategy, metadata):
        """Test that retry_after is capped at max_backoff."""

        class RetryableError(Exception):
            retry_after_seconds: float = 120.0

        error = RetryableError("rate limit")
        error.retry_after_seconds = 120.0  # More than max_backoff

        delay = strategy._calculate_retry_delay(error, attempt=1, metadata=metadata)

        assert delay == 60.0  # max_backoff

    def test_calculate_retry_delay_ignores_zero_retry_after(self, strategy, metadata):
        """Test that zero retry_after falls back to scheduler backoff."""

        class RetryableError(Exception):
            retry_after_seconds: float = 0.0

        error = RetryableError("rate limit")
        error.retry_after_seconds = 0

        delay = strategy._calculate_retry_delay(error, attempt=1, metadata=metadata)

        assert delay == 1.5  # From scheduler.calculate_backoff

    def test_calculate_retry_delay_ignores_negative_retry_after(
        self, strategy, metadata
    ):
        """Test that negative retry_after falls back to scheduler backoff."""

        class RetryableError(Exception):
            retry_after_seconds: float = -1.0

        error = RetryableError("rate limit")
        error.retry_after_seconds = -1.0

        delay = strategy._calculate_retry_delay(error, attempt=1, metadata=metadata)

        assert delay == 1.5  # From scheduler.calculate_backoff


class TestBasicModeStrategyRateLimitDetection:
    """Tests for rate limit error detection."""

    @pytest.fixture
    def strategy(self):
        """Create a BasicModeStrategy instance."""
        mock_scheduler = Mock()
        mock_scheduler.should_throttle = Mock(return_value=False)
        mock_scheduler.track_failure = Mock()
        mock_scheduler.calculate_backoff = Mock(return_value=1.0)
        mock_scheduler.extract_response_headers = Mock(return_value={})
        mock_config = Mock()
        mock_config.max_retries = 3
        mock_config.backoff_base = 2.0
        mock_config.max_backoff = 60.0
        return BasicModeStrategy(mock_scheduler, mock_config, Mock())

    def test_is_rate_limit_error_by_name(self, strategy):
        """Test detection by error class name."""

        class RateLimitError(Exception):
            pass

        error = RateLimitError("rate limited")
        assert strategy._is_rate_limit_error(error) is True

    def test_is_rate_limit_error_by_status_code(self, strategy):
        """Test detection by status_code attribute."""

        class HttpError(Exception):
            status_code: int = 429

        error = HttpError("error")
        error.status_code = 429
        assert strategy._is_rate_limit_error(error) is True

    def test_is_rate_limit_error_non_429(self, strategy):
        """Test non-429 status code is not rate limit."""

        class HttpError(Exception):
            status_code: int = 500

        error = HttpError("error")
        error.status_code = 500
        assert strategy._is_rate_limit_error(error) is False

    def test_is_rate_limit_error_regular_error(self, strategy):
        """Test regular exception is not rate limit."""
        error = Exception("regular error")
        assert strategy._is_rate_limit_error(error) is False


class TestBasicModeStrategyRateLimitState:
    """Tests for rate limit state updates."""

    @pytest.fixture
    def mock_state_manager(self):
        """Create a mock state manager."""
        state_manager = Mock()
        state_manager.update_state_from_headers = AsyncMock()
        return state_manager

    @pytest.fixture
    def mock_scheduler(self, mock_state_manager):
        """Create a mock scheduler with state manager."""
        scheduler = Mock()
        scheduler.should_throttle = Mock(return_value=False)
        scheduler.track_failure = Mock()
        scheduler.calculate_backoff = Mock(return_value=0.001)
        scheduler.extract_response_headers = Mock(
            return_value={
                "x-ratelimit-remaining-requests": "99",
                "x-ratelimit-limit-requests": "100",
            }
        )
        scheduler.state_manager = mock_state_manager
        scheduler._failed_requests = Mock()
        scheduler._failed_requests.count = 0
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
    def strategy(self, mock_scheduler, mock_config):
        """Create a BasicModeStrategy instance."""
        return BasicModeStrategy(mock_scheduler, mock_config, Mock())

    @pytest.fixture
    def metadata(self):
        """Create request metadata."""
        return RequestMetadata(
            request_id="req-123",
            model_id="test-model",
            resource_type="chat",
        )

    @pytest.mark.asyncio
    async def test_update_rate_limit_state_with_state_manager(
        self, strategy, metadata, mock_state_manager
    ):
        """Test state update uses state_manager when available."""
        result = Mock()
        strategy.scheduler.extract_response_headers.return_value = {
            "x-ratelimit-remaining-requests": "50"
        }

        await strategy._update_rate_limit_state(metadata, result, status_code=200)

        mock_state_manager.update_state_from_headers.assert_called_once()
        call_kwargs = mock_state_manager.update_state_from_headers.call_args[1]
        assert call_kwargs["status_code"] == 200

    @pytest.mark.asyncio
    async def test_update_rate_limit_state_without_state_manager(
        self, strategy, metadata
    ):
        """Test state update uses legacy handler without state_manager."""
        strategy.scheduler.state_manager = None
        strategy.scheduler.handle_rate_limit_headers = Mock(return_value={})
        result = Mock()

        await strategy._update_rate_limit_state(metadata, result)

        strategy.scheduler.handle_rate_limit_headers.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_rate_limit_state_no_headers(self, strategy, metadata):
        """Test state update handles no headers gracefully."""
        strategy.scheduler.extract_response_headers.return_value = {}
        result = Mock()

        # Should not raise
        await strategy._update_rate_limit_state(metadata, result)


class TestBasicModeStrategyLifecycle:
    """Tests for lifecycle methods."""

    @pytest.fixture
    def strategy(self):
        """Create a BasicModeStrategy instance."""
        mock_scheduler = Mock()
        mock_scheduler.should_throttle = Mock(return_value=False)
        mock_config = Mock()
        mock_config.max_retries = 3
        mock_config.backoff_base = 2.0
        mock_config.max_backoff = 60.0
        return BasicModeStrategy(mock_scheduler, mock_config, Mock())

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
    async def test_run_scheduling_loop_minimal(self, strategy):
        """Test scheduling loop runs until stopped."""
        await strategy.start()

        # Run loop in background
        loop_task = asyncio.create_task(strategy.run_scheduling_loop())

        # Let it run briefly
        await asyncio.sleep(0.05)

        # Stop and wait for loop to exit
        await strategy.stop()

        # Give the loop time to exit
        await asyncio.sleep(0.05)

        # Cancel if still running
        if not loop_task.done():
            loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await loop_task


class TestBasicModeStrategyMetrics:
    """Tests for metrics collection."""

    @pytest.fixture
    def strategy(self):
        """Create a BasicModeStrategy instance."""
        mock_scheduler = Mock()
        mock_scheduler.should_throttle = Mock(return_value=False)
        mock_config = Mock()
        mock_config.max_retries = 3
        mock_config.backoff_base = 2.0
        mock_config.max_backoff = 60.0
        return BasicModeStrategy(mock_scheduler, mock_config, Mock())

    def test_get_metrics_mode_name(self, strategy):
        """Test metrics includes mode name."""
        metrics = strategy.get_metrics()
        assert metrics["mode"] == "basic"

    def test_get_metrics_has_tracking_counts(self, strategy):
        """Test metrics includes tracking counts."""
        metrics = strategy.get_metrics()
        assert "last_request_times_count" in metrics

    def test_get_metrics_reflects_state(self, strategy):
        """Test metrics reflects internal state."""
        strategy._last_request_times["model1"] = time.time()
        strategy._last_request_times["model2"] = time.time()

        metrics = strategy.get_metrics()

        assert metrics["last_request_times_count"] == 2


class TestBasicModeStrategyRequestSpacing:
    """Tests for request spacing logic."""

    @pytest.fixture
    def strategy(self):
        """Create a BasicModeStrategy instance."""
        mock_scheduler = Mock()
        mock_scheduler.should_throttle = Mock(return_value=False)
        mock_scheduler.track_failure = Mock()
        mock_scheduler.calculate_backoff = Mock(return_value=0.001)
        mock_scheduler.extract_response_headers = Mock(return_value={})
        mock_scheduler.state_manager = None
        mock_config = Mock()
        mock_config.max_retries = 3
        mock_config.backoff_base = 2.0
        mock_config.max_backoff = 60.0
        return BasicModeStrategy(mock_scheduler, mock_config, Mock())

    @pytest.fixture
    def metadata(self):
        """Create request metadata."""
        return RequestMetadata(
            request_id="req-123",
            model_id="test-model",
            resource_type="chat",
        )

    @pytest.mark.asyncio
    async def test_enforce_request_spacing_no_prior_request(self, strategy, metadata):
        """Test spacing passes when no prior request."""
        start_time = time.time()

        await strategy._enforce_request_spacing(metadata)

        elapsed = time.time() - start_time
        assert elapsed < 0.1  # Should be nearly instant

    @pytest.mark.asyncio
    async def test_enforce_request_spacing_after_recent_request(
        self, strategy, metadata
    ):
        """Test spacing waits after recent request."""
        strategy._last_request_times[metadata.model_id] = time.time()

        start_time = time.time()
        await strategy._enforce_request_spacing(metadata)
        elapsed = time.time() - start_time

        # Should wait approximately _calculate_adaptive_delay() seconds
        assert elapsed >= 0.05  # Allow some margin

    def test_calculate_adaptive_delay_default(self, strategy):
        """Test default adaptive delay is 0.1 seconds."""
        delay = strategy._calculate_adaptive_delay("any-model")
        assert delay == 0.1


class TestBasicModeStrategy429Handling:
    """Tests for 429 rate limit error handling."""

    @pytest.fixture
    def mock_state_manager(self):
        """Create a mock state manager."""
        state_manager = Mock()
        state_manager.update_state_from_headers = AsyncMock()
        return state_manager

    @pytest.fixture
    def mock_scheduler(self, mock_state_manager):
        """Create a mock scheduler with state manager."""
        scheduler = Mock()
        scheduler.should_throttle = Mock(return_value=False)
        scheduler.track_failure = Mock()
        scheduler.calculate_backoff = Mock(return_value=0.001)
        scheduler.extract_response_headers = Mock(return_value={})
        scheduler.state_manager = mock_state_manager
        scheduler._failed_requests = Mock()
        scheduler._failed_requests.count = 0
        return scheduler

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = Mock()
        config.max_retries = 1  # Fail faster for testing
        config.backoff_base = 2.0
        config.max_backoff = 60.0
        return config

    @pytest.fixture
    def strategy(self, mock_scheduler, mock_config):
        """Create a BasicModeStrategy instance."""
        return BasicModeStrategy(mock_scheduler, mock_config, Mock())

    @pytest.fixture
    def metadata(self):
        """Create request metadata."""
        return RequestMetadata(
            request_id="req-123",
            model_id="test-model",
            resource_type="chat",
        )

    @pytest.mark.asyncio
    async def test_429_syncs_cached_headers(
        self, strategy, metadata, mock_state_manager
    ):
        """Test 429 errors sync cached rate limit headers."""

        class RateLimitError(Exception):
            status_code: int = 429
            cached_rate_limit_headers: ClassVar[dict] = {}

        error = RateLimitError("rate limited")
        error.status_code = 429
        error.cached_rate_limit_headers = {  # type: ignore[misc]
            "x-ratelimit-remaining-requests": "0",
            "x-ratelimit-reset-requests": "60",
        }

        request_func = AsyncMock(side_effect=error)

        with pytest.raises(RateLimitError):
            await strategy.submit_request(metadata, request_func)


class TestBasicModeStrategyCleanup:
    """Tests for LRU eviction and TTL cleanup in BasicModeStrategy (Issue other_strategies_004)."""

    @pytest.fixture
    def strategy_with_small_max_entries(self):
        """Create a BasicModeStrategy with small max_entries for testing LRU eviction."""
        mock_scheduler = Mock()
        mock_scheduler.should_throttle = Mock(return_value=False)

        # Use a real config class with max_tracking_entries set
        class TestConfig:
            max_retries = 3
            backoff_base = 2.0
            max_backoff = 60.0
            max_tracking_entries = 5  # Small value for testing LRU eviction
            stale_entry_ttl = 3600.0

        return BasicModeStrategy(mock_scheduler, TestConfig(), Mock())  # type: ignore[arg-type]

    @pytest.fixture
    def strategy_with_short_ttl(self):
        """Create a BasicModeStrategy with short TTL for testing stale entry cleanup."""
        mock_scheduler = Mock()
        mock_scheduler.should_throttle = Mock(return_value=False)

        class TestConfig:
            max_retries = 3
            backoff_base = 2.0
            max_backoff = 60.0
            max_tracking_entries = 10000
            stale_entry_ttl = 0.1  # 100ms for fast testing

        return BasicModeStrategy(mock_scheduler, TestConfig(), Mock())  # type: ignore[arg-type]

    def test_lru_eviction_when_max_entries_exceeded(
        self, strategy_with_small_max_entries
    ):
        """Test that oldest entries are evicted when max_entries is exceeded."""
        strategy = strategy_with_small_max_entries

        # Add 5 entries (at max)
        for i in range(5):
            strategy._update_last_request_time(f"model-{i}")

        assert len(strategy._last_request_times) == 5

        # Add 6th entry - should evict model-0
        strategy._update_last_request_time("model-5")

        assert len(strategy._last_request_times) == 5
        assert "model-0" not in strategy._last_request_times
        assert "model-5" in strategy._last_request_times

    def test_lru_moves_accessed_entry_to_end(self, strategy_with_small_max_entries):
        """Test that accessing an entry moves it to the end (most recently used)."""
        strategy = strategy_with_small_max_entries

        # Add 5 entries
        for i in range(5):
            strategy._update_last_request_time(f"model-{i}")

        # Access model-0 again - should move to end
        strategy._update_last_request_time("model-0")

        # Add model-5 - should evict model-1 (now oldest), not model-0
        strategy._update_last_request_time("model-5")

        assert "model-0" in strategy._last_request_times  # Was accessed, so not evicted
        assert "model-1" not in strategy._last_request_times  # Now oldest, evicted

    def test_stale_entry_cleanup(self, strategy_with_short_ttl):
        """Test that stale entries older than TTL are cleaned up."""
        strategy = strategy_with_short_ttl

        # Add an entry with old timestamp (simulating stale entry)
        old_time = time.time() - 1.0  # 1 second ago (older than 0.1s TTL)
        strategy._last_request_times["old-model"] = old_time

        # Run cleanup
        removed = strategy.cleanup_tracking_state()

        assert removed == 1
        assert "old-model" not in strategy._last_request_times

    def test_cleanup_tracking_state_returns_count(self, strategy_with_short_ttl):
        """Test cleanup_tracking_state returns the count of removed entries."""
        strategy = strategy_with_short_ttl

        # Add entries with old timestamps
        old_time = time.time() - 1.0
        for i in range(3):
            strategy._last_request_times[f"old-model-{i}"] = old_time

        # Add one fresh entry
        strategy._last_request_times["fresh-model"] = time.time()

        removed = strategy.cleanup_tracking_state()

        assert removed == 3
        assert len(strategy._last_request_times) == 1
        assert "fresh-model" in strategy._last_request_times

    def test_init_uses_default_max_entries_with_mock_config(self):
        """Test that Mock config falls back to default max_entries."""
        from adaptive_rate_limiter.strategies.modes.basic import DEFAULT_MAX_ENTRIES

        mock_scheduler = Mock()
        mock_config = Mock()  # Mock returns Mock for any attribute
        strategy = BasicModeStrategy(mock_scheduler, mock_config, Mock())

        # Should use default, not a Mock object
        assert strategy._max_entries == DEFAULT_MAX_ENTRIES
        assert isinstance(strategy._max_entries, int)

    def test_init_uses_default_stale_entry_ttl_with_mock_config(self):
        """Test that Mock config falls back to default stale_entry_ttl."""
        from adaptive_rate_limiter.strategies.modes.basic import DEFAULT_STALE_ENTRY_TTL

        mock_scheduler = Mock()
        mock_config = Mock()  # Mock returns Mock for any attribute
        strategy = BasicModeStrategy(mock_scheduler, mock_config, Mock())

        # Should use default, not a Mock object
        assert strategy._stale_entry_ttl == DEFAULT_STALE_ENTRY_TTL
        assert isinstance(strategy._stale_entry_ttl, float)
