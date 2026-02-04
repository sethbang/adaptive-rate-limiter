"""
Tests for StateEntry and RateLimitState model edge cases.

This module covers:
- StateEntry property and serialization edge cases
- RateLimitState property edge cases
- Header parsing error handling
- Token conservative update logic
"""

import time
from datetime import datetime, timedelta, timezone

from adaptive_rate_limiter.scheduler.state import (
    RateLimitState,
    StateEntry,
    StateType,
)


class TestStateEntryProperties:
    """Tests for StateEntry property edge cases."""

    def test_age_seconds(self):
        """Test StateEntry.age_seconds property (line 105)."""
        past = datetime.now(timezone.utc) - timedelta(seconds=10)
        entry = StateEntry(key="test", data={}, created_at=past)

        assert 9.5 < entry.age_seconds < 11.0

    def test_age_seconds_new_entry(self):
        """Test age_seconds for newly created entry."""
        entry = StateEntry(key="test", data={})

        # Should be very close to 0
        assert entry.age_seconds < 1.0

    def test_is_expired_none_expires_at(self):
        """Test is_expired when expires_at is None."""
        entry = StateEntry(key="test", data={})
        assert entry.expires_at is None
        assert not entry.is_expired

    def test_to_dict_with_all_fields(self):
        """Test to_dict includes all fields properly."""
        now = datetime.now(timezone.utc)
        future = now + timedelta(hours=1)

        entry = StateEntry(
            key="test",
            data={"val": 1},
            state_type=StateType.BUCKET_INFO,
            version=5,
            created_at=now,
            updated_at=now,
            expires_at=future,
            metadata={"source": "test"},
            namespace="custom",
        )

        data = entry.to_dict()

        assert data["key"] == "test"
        assert data["data"]["val"] == 1
        assert data["state_type"] == "bucket_info"
        assert data["version"] == 5
        assert data["expires_at"] is not None
        assert data["metadata"] == {"source": "test"}
        assert data["namespace"] == "custom"

    def test_from_dict_minimal(self):
        """Test from_dict with minimal data."""
        data = {
            "key": "test",
            "data": {"val": 1},
        }

        entry = StateEntry.from_dict(data)

        assert entry.key == "test"
        assert entry.data["val"] == 1
        assert entry.version == 1
        assert entry.state_type == StateType.RATE_LIMIT
        assert entry.namespace == "default"


class TestRateLimitStateUsagePercentage:
    """Tests for RateLimitState.usage_percentage edge cases."""

    def test_usage_percentage_no_limits(self):
        """Test usage_percentage when limits are None (lines 610-611)."""
        state = RateLimitState(model_id="test")
        # Both None
        assert state.usage_percentage == 0.0

    def test_usage_percentage_only_limit_set(self):
        """Test usage_percentage when only request_limit is set."""
        state = RateLimitState(model_id="test", request_limit=100)
        # remaining_requests is None
        assert state.usage_percentage == 0.0

    def test_usage_percentage_only_remaining_set(self):
        """Test usage_percentage when only remaining_requests is set."""
        state = RateLimitState(model_id="test", remaining_requests=50)
        # request_limit is None
        assert state.usage_percentage == 0.0

    def test_usage_percentage_zero_limit(self):
        """Test usage_percentage when limit is 0."""
        state = RateLimitState(model_id="test", request_limit=0, remaining_requests=0)
        assert state.usage_percentage == 0.0

    def test_usage_percentage_full_capacity(self):
        """Test usage_percentage at full capacity."""
        state = RateLimitState(
            model_id="test", request_limit=100, remaining_requests=100
        )
        assert state.usage_percentage == 0.0  # 0% used

    def test_usage_percentage_exhausted(self):
        """Test usage_percentage when exhausted."""
        state = RateLimitState(model_id="test", request_limit=100, remaining_requests=0)
        assert state.usage_percentage == 100.0


class TestRateLimitStateHeaderParsing:
    """Tests for header parsing error handling."""

    def test_update_from_headers_invalid_tokens(self):
        """Test header update with invalid token values (lines 634-635)."""
        state = RateLimitState(model_id="test")

        headers = {"x-ratelimit-remaining-tokens": "not-a-number"}
        state.update_from_headers(headers)

        # Should remain unchanged
        assert state.remaining_tokens is None

    def test_update_from_headers_invalid_request_limit(self):
        """Test header update with invalid request limit (lines 641-642)."""
        state = RateLimitState(model_id="test")

        headers = {"x-ratelimit-limit-requests": "invalid"}
        state.update_from_headers(headers)

        # Should remain unchanged
        assert state.request_limit is None

    def test_update_from_headers_invalid_token_limit(self):
        """Test header update with invalid token limit."""
        state = RateLimitState(model_id="test")
        original_limit = state.token_limit

        headers = {"x-ratelimit-limit-tokens": "not-an-int"}
        state.update_from_headers(headers)

        assert state.token_limit == original_limit

    def test_update_from_headers_invalid_reset_time(self):
        """Test header update with invalid reset timestamp (lines 656-657)."""
        state = RateLimitState(model_id="test")
        original_reset = state.reset_at

        headers = {"x-ratelimit-reset-requests": "bad-timestamp"}
        state.update_from_headers(headers)

        # Should remain unchanged
        assert state.reset_at == original_reset

    def test_update_from_headers_all_invalid(self):
        """Test all header values being invalid."""
        state = RateLimitState(model_id="test")
        original_reset = state.reset_at

        headers = {
            "x-ratelimit-remaining-requests": "invalid",
            "x-ratelimit-remaining-tokens": "not-int",
            "x-ratelimit-limit-requests": "bad",
            "x-ratelimit-limit-tokens": "nope",
            "x-ratelimit-reset-requests": "bad-timestamp",
        }
        state.update_from_headers(headers)

        # Should remain unchanged
        assert state.remaining_requests is None
        assert state.remaining_tokens is None
        assert state.request_limit is None
        assert state.token_limit is None
        assert state.reset_at == original_reset


class TestRateLimitStateTokenConservativeUpdate:
    """Tests for token conservative update logic."""

    def test_update_tokens_verified_local_exhausted(self):
        """Test token update when verified and local exhausted (line 677-678)."""
        state = RateLimitState(
            model_id="test",
            remaining_tokens=None,  # None = will be treated as exhausted
            is_verified=True,
        )

        headers = {"x-ratelimit-remaining-tokens": "500"}
        state.update_from_headers(headers)

        # Should accept server value because local is None (exhausted)
        assert state.remaining_tokens == 500

    def test_update_tokens_verified_local_zero(self):
        """Test token update when verified and local is 0."""
        state = RateLimitState(
            model_id="test",
            remaining_tokens=0,  # Exhausted
            is_verified=True,
        )

        headers = {"x-ratelimit-remaining-tokens": "500"}
        state.update_from_headers(headers)

        # Should accept server value because local is exhausted
        assert state.remaining_tokens == 500

    def test_update_tokens_verified_server_at_limit(self):
        """Test token update when server is at/near limit."""
        state = RateLimitState(
            model_id="test",
            remaining_tokens=500,
            token_limit=1000,
            is_verified=True,
        )

        # Server at 95% of limit (900/1000 = 90%+)
        headers = {"x-ratelimit-remaining-tokens": "950"}
        state.update_from_headers(headers)

        # Should accept server value because it's near limit
        assert state.remaining_tokens == 950

    def test_update_tokens_verified_local_lower(self):
        """Test token update when local is lower (conservative)."""
        state = RateLimitState(
            model_id="test",
            remaining_tokens=500,
            token_limit=1000,
            is_verified=True,
        )

        # Server says 800, local says 500 -> keep 500 (conservative)
        headers = {"x-ratelimit-remaining-tokens": "800"}
        state.update_from_headers(headers)

        assert state.remaining_tokens == 500

    def test_update_tokens_verified_server_lower(self):
        """Test token update when server is lower."""
        state = RateLimitState(
            model_id="test",
            remaining_tokens=500,
            token_limit=1000,
            is_verified=True,
        )

        # Server says 300, local says 500 -> take 300
        headers = {"x-ratelimit-remaining-tokens": "300"}
        state.update_from_headers(headers)

        assert state.remaining_tokens == 300

    def test_update_requests_verified_local_lower(self):
        """Test request update when local is lower (conservative)."""
        state = RateLimitState(
            model_id="test",
            remaining_requests=10,
            request_limit=100,
            is_verified=True,
        )

        # Server says 50, local says 10 -> keep 10
        headers = {"x-ratelimit-remaining-requests": "50"}
        state.update_from_headers(headers)

        assert state.remaining_requests == 10

    def test_update_last_updated_with_local_value(self):
        """Test last_updated not updated when using local value."""
        state = RateLimitState(
            model_id="test",
            remaining_requests=10,
            request_limit=100,
            is_verified=True,
        )
        original_updated = state.last_updated

        # Wait a tiny bit
        time.sleep(0.001)

        # Server says 50, local says 10 -> keep 10 (used local value)
        headers = {"x-ratelimit-remaining-requests": "50"}
        state.update_from_headers(headers)

        # last_updated should still be updated even when using local value
        # because the flag only prevents update when `not used_local_value`
        # Actually, re-reading code: if used_local_value is True, last_updated is NOT updated
        # So last_updated should remain the same
        assert state.last_updated == original_updated


class TestRateLimitStateTimeUntilReset:
    """Tests for time_until_reset edge cases."""

    def test_time_until_reset_both_in_past(self):
        """Test time_until_reset when both resets are in past."""
        past = datetime.now(timezone.utc) - timedelta(seconds=10)
        state = RateLimitState(model_id="test", reset_at=past, reset_at_daily=past)

        assert state.time_until_reset == 0.0

    def test_time_until_reset_only_regular_in_future(self):
        """Test time_until_reset with only regular reset in future."""
        now = datetime.now(timezone.utc)
        past = now - timedelta(seconds=10)
        future = now + timedelta(seconds=30)

        state = RateLimitState(model_id="test", reset_at=future, reset_at_daily=past)

        assert 29.0 < state.time_until_reset < 31.0

    def test_time_until_reset_only_daily_in_future(self):
        """Test time_until_reset with only daily reset in future."""
        now = datetime.now(timezone.utc)
        past = now - timedelta(seconds=10)
        future = now + timedelta(seconds=60)

        state = RateLimitState(model_id="test", reset_at=past, reset_at_daily=future)

        assert 59.0 < state.time_until_reset < 61.0

    def test_time_until_reset_takes_minimum(self):
        """Test time_until_reset returns minimum of future resets."""
        now = datetime.now(timezone.utc)
        near_future = now + timedelta(seconds=10)
        far_future = now + timedelta(seconds=100)

        state = RateLimitState(
            model_id="test", reset_at=near_future, reset_at_daily=far_future
        )

        # Should return the nearer reset (10 seconds)
        assert 9.0 < state.time_until_reset < 11.0


class TestRateLimitStateIsExhausted:
    """Tests for is_exhausted property edge cases."""

    def test_is_exhausted_negative_requests(self):
        """Test is_exhausted with negative remaining."""
        state = RateLimitState(model_id="test", remaining_requests=-1)
        assert state.is_exhausted

    def test_is_exhausted_negative_tokens(self):
        """Test is_exhausted with negative tokens."""
        state = RateLimitState(
            model_id="test", remaining_requests=10, remaining_tokens=-5
        )
        assert state.is_exhausted

    def test_is_exhausted_both_positive(self):
        """Test is_exhausted with both positive."""
        state = RateLimitState(
            model_id="test", remaining_requests=10, remaining_tokens=100
        )
        assert not state.is_exhausted

    def test_is_exhausted_both_none(self):
        """Test is_exhausted with both None."""
        state = RateLimitState(model_id="test")
        assert not state.is_exhausted
