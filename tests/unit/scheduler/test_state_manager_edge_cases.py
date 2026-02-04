"""
Tests for StateManager edge cases and error handling.

This module covers:
- Production warnings
- Failed counter edge cases
- Check and reserve capacity edge cases
- Release reservation edge cases
- Get next reset time edge cases
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest

from adaptive_rate_limiter.backends.base import BaseBackend
from adaptive_rate_limiter.scheduler.config import CachePolicy, StateConfig
from adaptive_rate_limiter.scheduler.state import (
    RateLimitState,
    StateManager,
    StateType,
)


@pytest.fixture
def mock_backend():
    """Create a mock backend."""
    backend = AsyncMock(spec=BaseBackend)
    backend.namespace = "default"
    return backend


@pytest.fixture
def manager(mock_backend):
    """Create a StateManager with mock backend."""
    config = StateConfig(cache_policy=CachePolicy.WRITE_THROUGH)
    return StateManager(backend=mock_backend, config=config)


class TestStateManagerProductionWarning:
    """Tests for production warning and error on WRITE_BACK policy."""

    def test_write_back_production_raises_without_acknowledgment(self, mock_backend):
        """Test ValueError raised for WRITE_BACK in production without acknowledgment."""
        config = StateConfig(
            cache_policy=CachePolicy.WRITE_BACK,
            is_production=True,
            warn_write_back_production=True,
            acknowledge_write_back_risk=False,  # Default
        )

        with pytest.raises(
            ValueError,
            match="WRITE_BACK cache policy in production requires explicit acknowledgment",
        ):
            StateManager(backend=mock_backend, config=config)

    def test_write_back_production_warning_with_acknowledgment(
        self, mock_backend, caplog
    ):
        """Test warning is logged when WRITE_BACK acknowledged in production."""
        config = StateConfig(
            cache_policy=CachePolicy.WRITE_BACK,
            is_production=True,
            warn_write_back_production=True,
            acknowledge_write_back_risk=True,
        )

        with caplog.at_level(logging.WARNING):
            _manager = StateManager(backend=mock_backend, config=config)

        assert (
            "WRITE_BACK cache policy in production - data loss possible" in caplog.text
        )

    def test_write_back_production_no_warning_when_disabled_but_acknowledged(
        self, mock_backend, caplog
    ):
        """Test no warning when warn_write_back_production is False (but still acknowledged)."""
        config = StateConfig(
            cache_policy=CachePolicy.WRITE_BACK,
            is_production=True,
            warn_write_back_production=False,
            acknowledge_write_back_risk=True,
        )

        with caplog.at_level(logging.WARNING):
            _manager = StateManager(backend=mock_backend, config=config)

        assert "WRITE_BACK cache policy" not in caplog.text

    def test_write_back_non_production_no_warning(self, mock_backend, caplog):
        """Test no warning when not in production (acknowledgment not required)."""
        config = StateConfig(
            cache_policy=CachePolicy.WRITE_BACK,
            is_production=False,
            warn_write_back_production=True,
        )

        with caplog.at_level(logging.WARNING):
            _manager = StateManager(backend=mock_backend, config=config)

        assert "WRITE_BACK cache policy" not in caplog.text


class TestFailedCounterEdgeCases:
    """Tests for failed request counter edge cases."""

    def test_get_count_triggers_cleanup_when_stale(self, manager):
        """Test get_count triggers cleanup when stale (lines 814-818)."""
        counter = manager.failed_request_counter

        # Set last cleanup to be old
        counter._last_cleanup = time.time() - 1.0
        counter.failure_times.append(time.time() - 100)  # Old failure
        counter._count = 1

        # get_count should trigger cleanup since > 0.1s since check
        count = counter.get_count()
        assert count == 0  # Old failure cleaned up

    def test_get_count_returns_cached_when_fresh(self, manager):
        """Test get_count returns cached count when fresh (lines 812-813)."""
        counter = manager.failed_request_counter

        # Set last cleanup to very recent
        counter._last_cleanup = time.time()
        counter._count = 5

        # Should return cached count without cleanup
        count = counter.get_count()
        assert count == 5

    def test_increment_triggers_cleanup_throttled(self, manager):
        """Test increment throttles cleanup to 1s intervals."""
        counter = manager.failed_request_counter

        # Add old failure
        old_time = time.time() - counter.window_seconds - 10
        counter.failure_times.append(old_time)
        counter._count = 1

        # Set last cleanup to very recent (within 1s)
        counter._last_cleanup = time.time()

        # Increment - should NOT trigger cleanup due to throttle
        count = counter.increment()

        # Should be 2 (old failure not cleaned up due to throttle)
        assert count == 2


class TestCheckAndReserveCapacityEdgeCases:
    """Tests for check_and_reserve_capacity edge cases."""

    @pytest.mark.asyncio
    async def test_bucket_not_in_discover_limits(self, manager):
        """Test check_and_reserve when bucket not in discover_limits (line 1012-1013)."""
        manager.provider = AsyncMock()
        manager.provider.get_bucket_for_model.return_value = "bucket-1"
        manager.provider.discover_limits.return_value = {}  # Empty

        with pytest.raises(ValueError, match="Bucket bucket-1 not found"):
            await manager.check_and_reserve_capacity("model", "chat")

    @pytest.mark.asyncio
    async def test_invalid_rpm_limit_negative(self, manager):
        """Test check_and_reserve with negative rpm_limit (line 1020-1021).

        Note: rpm_limit=0 triggers `or 60` fallback, so we need negative value
        to trigger validation. Line 1016 uses `bucket.rpm_limit or 60`.
        """
        manager.provider = AsyncMock()
        manager.provider.get_bucket_for_model.return_value = "bucket-1"
        # Use a mock that returns -1 for rpm_limit (bypasses `or 60` since -1 is truthy)
        bucket = Mock()
        bucket.rpm_limit = -1
        bucket.tpm_limit = 1000
        manager.provider.discover_limits.return_value = {"bucket-1": bucket}

        with pytest.raises(ValueError, match="Invalid rpm_limit"):
            await manager.check_and_reserve_capacity("model", "chat")

    @pytest.mark.asyncio
    async def test_invalid_tpm_limit_negative(self, manager):
        """Test check_and_reserve with negative tpm_limit (line 1022-1023).

        Note: tpm_limit=0 triggers `or 10000` fallback, so we need negative value.
        """
        manager.provider = AsyncMock()
        manager.provider.get_bucket_for_model.return_value = "bucket-1"
        bucket = Mock()
        bucket.rpm_limit = 60
        bucket.tpm_limit = -1  # Negative bypasses `or 10000`
        manager.provider.discover_limits.return_value = {"bucket-1": bucket}

        with pytest.raises(ValueError, match="Invalid tpm_limit"):
            await manager.check_and_reserve_capacity("model", "chat")

    @pytest.mark.asyncio
    async def test_cache_sync_error_handled(self, manager):
        """Test check_and_reserve handles cache sync error (line 1039-1041)."""
        manager.provider = AsyncMock()
        manager.provider.get_bucket_for_model.return_value = "bucket-1"
        manager.provider.discover_limits.return_value = {
            "bucket-1": Mock(rpm_limit=60, tpm_limit=1000)
        }
        manager.backend.check_and_reserve_capacity.return_value = (True, "res-1")
        manager.backend.get_state.return_value = {"model_id": "bucket-1"}

        # Make cache set fail
        with patch.object(manager.cache, "set", side_effect=Exception("cache error")):
            success, res_id = await manager.check_and_reserve_capacity("model", "chat")

        # Should still succeed (error is logged but not raised)
        assert success is True
        assert res_id == "res-1"

    @pytest.mark.asyncio
    async def test_provider_attribute_error(self, manager):
        """Test check_and_reserve handles AttributeError (lines 1044-1048)."""
        manager.provider = AsyncMock()
        manager.provider.get_bucket_for_model.side_effect = AttributeError(
            "Bad attribute"
        )

        success, res_id = await manager.check_and_reserve_capacity("model", "chat")

        assert success is False
        assert res_id is None

    @pytest.mark.asyncio
    async def test_provider_type_error(self, manager):
        """Test check_and_reserve handles TypeError (lines 1044-1048)."""
        manager.provider = AsyncMock()
        manager.provider.get_bucket_for_model.side_effect = TypeError("Type issue")

        success, res_id = await manager.check_and_reserve_capacity("model", "chat")

        assert success is False
        assert res_id is None

    @pytest.mark.asyncio
    async def test_provider_os_error(self, manager):
        """Test check_and_reserve handles OSError (lines 1044-1048)."""
        manager.provider = AsyncMock()
        manager.provider.get_bucket_for_model.side_effect = OSError("Network error")

        success, res_id = await manager.check_and_reserve_capacity("model", "chat")

        assert success is False
        assert res_id is None

    @pytest.mark.asyncio
    async def test_backend_state_sync_success(self, manager):
        """Test successful cache sync from backend state (lines 1030-1039)."""
        manager.provider = AsyncMock()
        manager.provider.get_bucket_for_model.return_value = "bucket-1"
        manager.provider.discover_limits.return_value = {
            "bucket-1": Mock(rpm_limit=60, tpm_limit=1000)
        }
        manager.backend.check_and_reserve_capacity.return_value = (True, "res-1")
        manager.backend.get_state.return_value = {
            "model_id": "bucket-1",
            "remaining_requests": 50,
        }

        success, res_id = await manager.check_and_reserve_capacity("model", "chat")

        assert success is True
        assert res_id == "res-1"

        # Verify cache was updated
        cached = await manager.cache.get("bucket-1")
        assert cached is not None
        assert cached.data["remaining_requests"] == 50


class TestReleaseReservationEdgeCases:
    """Tests for release_reservation edge cases."""

    @pytest.mark.asyncio
    async def test_release_reservation_no_bucket(self, manager):
        """Test release_reservation when bucket not found (lines 1059-1061)."""
        manager.provider = AsyncMock()
        manager.provider.get_bucket_for_model.return_value = None

        # Should log warning but not raise
        await manager.release_reservation("res-1", "model", "chat")

        manager.backend.release_reservation.assert_not_called()

    @pytest.mark.asyncio
    async def test_release_reservation_success(self, manager):
        """Test release_reservation success path."""
        manager.provider = AsyncMock()
        manager.provider.get_bucket_for_model.return_value = "bucket-1"

        await manager.release_reservation("res-1", "model", "chat")

        manager.backend.release_reservation.assert_called_once_with("bucket-1", "res-1")


class TestGetNextResetTimeEdgeCases:
    """Tests for get_next_reset_time edge cases."""

    @pytest.mark.asyncio
    async def test_get_next_reset_time_no_state(self, manager):
        """Test get_next_reset_time when state not found (line 1084-1085)."""
        manager.backend.get_state.return_value = None

        result = await manager.get_next_reset_time("unknown-bucket")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_next_reset_time_daily_exhausted(self, manager):
        """Test get_next_reset_time with daily exhausted (lines 1097-1103)."""
        future_daily = datetime.now(timezone.utc) + timedelta(hours=12)
        future_regular = datetime.now(timezone.utc) + timedelta(minutes=1)

        state = RateLimitState(
            model_id="bucket-1",
            remaining_requests=10,  # Not exhausted
            remaining_requests_daily=0,  # Daily exhausted
            reset_at=future_regular,
            reset_at_daily=future_daily,
        )
        await manager.set_state("bucket-1", state)

        reset_time = await manager.get_next_reset_time("bucket-1")
        assert reset_time == future_daily

    @pytest.mark.asyncio
    async def test_get_next_reset_time_both_exhausted_takes_earlier(self, manager):
        """Test get_next_reset_time returns earlier reset when both exhausted."""
        now = datetime.now(timezone.utc)
        future_regular = now + timedelta(minutes=1)
        future_daily = now + timedelta(hours=12)

        state = RateLimitState(
            model_id="bucket-1",
            remaining_requests=0,  # Exhausted
            remaining_requests_daily=0,  # Also exhausted
            reset_at=future_regular,
            reset_at_daily=future_daily,
        )
        await manager.set_state("bucket-1", state)

        reset_time = await manager.get_next_reset_time("bucket-1")
        # Should return the earlier one
        assert reset_time == future_regular

    @pytest.mark.asyncio
    async def test_get_next_reset_time_daily_earlier(self, manager):
        """Test get_next_reset_time when daily reset is earlier."""
        now = datetime.now(timezone.utc)
        future_regular = now + timedelta(hours=2)
        future_daily = now + timedelta(minutes=30)  # Earlier

        state = RateLimitState(
            model_id="bucket-1",
            remaining_requests=0,  # Exhausted
            remaining_requests_daily=0,  # Also exhausted
            reset_at=future_regular,
            reset_at_daily=future_daily,
        )
        await manager.set_state("bucket-1", state)

        reset_time = await manager.get_next_reset_time("bucket-1")
        # Should return the daily one (earlier)
        assert reset_time == future_daily

    @pytest.mark.asyncio
    async def test_get_next_reset_time_reset_in_past(self, manager):
        """Test get_next_reset_time when reset is in the past."""
        past = datetime.now(timezone.utc) - timedelta(minutes=5)

        state = RateLimitState(
            model_id="bucket-1",
            remaining_requests=0,  # Exhausted
            reset_at=past,  # But reset is in the past
            reset_at_daily=past,
        )
        await manager.set_state("bucket-1", state)

        reset_time = await manager.get_next_reset_time("bucket-1")
        # Should return None since both resets are in the past
        assert reset_time is None


class TestGetStateEdgeCases:
    """Tests for get_state edge cases."""

    @pytest.mark.asyncio
    async def test_get_state_non_rate_limit_from_backend(self, manager):
        """Test get_state returns raw dict for non-RATE_LIMIT type (line 897-899)."""
        manager.backend.get_state.return_value = {"config": "value"}

        state = await manager.get_state(
            "test", StateType.MODEL_CONFIG, force_refresh=True
        )

        # Should return dict, not RateLimitState
        assert isinstance(state, dict)
        assert state["config"] == "value"

    @pytest.mark.asyncio
    async def test_get_state_force_refresh_rate_limit(self, manager):
        """Test get_state with force_refresh returns RateLimitState."""
        manager.backend.get_state.return_value = {
            "model_id": "test",
            "remaining_requests": 50,
        }

        state = await manager.get_state(
            "test", StateType.RATE_LIMIT, force_refresh=True
        )

        assert isinstance(state, RateLimitState)
        assert state.remaining_requests == 50


class TestUpdateStateFromHeadersEdgeCases:
    """Tests for update_state_from_headers edge cases."""

    @pytest.mark.asyncio
    async def test_update_state_result_not_one(self, manager):
        """Test update_state_from_headers when result != 1 (line 979-990)."""
        manager.provider = AsyncMock()
        manager.provider.get_bucket_for_model.return_value = "bucket-1"
        manager.backend.update_rate_limits.return_value = 0  # Failed

        result = await manager.update_state_from_headers(
            "model", "chat", {"x-ratelimit-remaining-requests": "10"}
        )

        assert result == 0
        # Should not update local state
        cached = await manager.cache.get("bucket-1")
        assert cached is None

    @pytest.mark.asyncio
    async def test_update_state_creates_fallback_if_no_state(self, manager):
        """Test update creates fallback state if none exists (line 982-985)."""
        manager.provider = AsyncMock()
        manager.provider.get_bucket_for_model.return_value = "bucket-1"
        manager.backend.update_rate_limits.return_value = 1

        # No existing state
        manager.backend.get_state.return_value = None

        result = await manager.update_state_from_headers(
            "model", "chat", {"x-ratelimit-remaining-requests": "10"}
        )

        assert result == 1

        # Should have created state with header values
        cached = await manager.cache.get("bucket-1")
        assert cached is not None
