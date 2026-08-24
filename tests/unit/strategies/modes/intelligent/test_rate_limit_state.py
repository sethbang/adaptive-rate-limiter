"""
Unit tests for IntelligentModeStrategy rate limit state management.

Tests rate limit state updates, reset watchers, and header assessment.
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from adaptive_rate_limiter.types.queue import QueuedRequest
from adaptive_rate_limiter.types.request import RequestMetadata

# ============================================================================
# Rate Limit State Update Tests
# ============================================================================


class TestIntelligentModeStrategyRateLimitState:
    """Tests for rate limit state update methods."""

    @pytest.mark.asyncio
    async def test_assess_header_full(self, strategy):
        """Test full header assessment."""
        headers = {
            "x-ratelimit-remaining-requests": "99",
            "x-ratelimit-remaining-tokens": "9900",
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-limit-tokens": "10000",
            "x-ratelimit-reset-requests": "60",
            "x-ratelimit-reset-tokens": "60",
        }

        status = strategy._assess_header_availability(headers)

        assert status == "full"

    @pytest.mark.asyncio
    async def test_assess_header_partial(self, strategy):
        """Test partial header assessment."""
        headers = {
            "x-ratelimit-remaining-requests": "99",
            "x-ratelimit-limit-requests": "100",
        }

        status = strategy._assess_header_availability(headers)

        assert status == "partial"

    @pytest.mark.asyncio
    async def test_assess_header_none(self, strategy):
        """Test no headers assessment."""
        headers = {}

        status = strategy._assess_header_availability(headers)

        assert status == "none"

    @pytest.mark.asyncio
    async def test_assess_header_invalid_values(self, strategy):
        """Test headers with invalid values."""
        headers = {
            "x-ratelimit-remaining-requests": "invalid",
            "x-ratelimit-limit-requests": "100",
        }

        status = strategy._assess_header_availability(headers)

        assert status == "partial"

    @pytest.mark.asyncio
    async def test_update_rate_limit_state_no_state_manager(self, strategy, metadata):
        """Test update when no state manager."""
        strategy.state_manager = None

        # Should not raise
        await strategy._update_rate_limit_state(metadata, None)

    @pytest.mark.asyncio
    async def test_assess_header_with_duration_strings(
        self, strategy, mock_scheduler, mock_state_manager
    ):
        """Test header assessment with duration strings (via normalization logic).

        Drives the real ``_update_rate_limit_state`` rather than re-implementing
        its normalization loop: a copy of the loop would keep passing after the
        production code changed, while no longer testing it.
        """
        mock_scheduler.extract_response_headers.return_value = {
            "x-ratelimit-remaining-requests": "99",
            "x-ratelimit-remaining-tokens": "9900",
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-limit-tokens": "10000",
            "x-ratelimit-reset-requests": "2s",
            "x-ratelimit-reset-tokens": "500ms",
        }

        metadata = RequestMetadata(
            request_id="req-duration",
            model_id="test-model",
            resource_type="chat",
        )

        await strategy._update_rate_limit_state(
            metadata, result=Mock(), status_code=200
        )

        call = mock_state_manager.update_state_from_headers.call_args
        headers = call.args[2] if len(call.args) > 2 else call.kwargs["headers"]

        assert strategy._assess_header_availability(headers) == "full"
        assert headers["x-ratelimit-reset-requests"] == "2.0"
        assert headers["x-ratelimit-reset-tokens"] == "0.5"

    @pytest.mark.asyncio
    async def test_parse_duration_string(self, strategy):
        """Test parsing of duration strings."""
        assert strategy._parse_duration_string("2s") == 2.0
        assert strategy._parse_duration_string("500ms") == 0.5
        assert strategy._parse_duration_string("1m") == 60.0
        assert strategy._parse_duration_string("1m30s") == 90.0
        assert strategy._parse_duration_string("1.5s") == 1.5
        assert strategy._parse_duration_string("invalid") is None


# ============================================================================
# Update State Tests
# ============================================================================


class TestIntelligentModeStrategyUpdateState:
    """Tests for _update_rate_limit_state method."""

    @pytest.fixture
    def mock_scheduler_with_headers(self, mock_scheduler):
        """Create a mock scheduler with full rate limit headers."""
        mock_scheduler.extract_response_headers = Mock(
            return_value={
                "x-ratelimit-remaining-requests": "99",
                "x-ratelimit-remaining-tokens": "9900",
                "x-ratelimit-limit-requests": "100",
                "x-ratelimit-limit-tokens": "10000",
                "x-ratelimit-reset-requests": "60",
                "x-ratelimit-reset-tokens": "60",
            }
        )
        return mock_scheduler

    @pytest.mark.asyncio
    async def test_update_with_full_headers(
        self, strategy, mock_state_manager, mock_scheduler
    ):
        """Test update with full header set."""
        mock_scheduler.extract_response_headers.return_value = {
            "x-ratelimit-remaining-requests": "99",
            "x-ratelimit-remaining-tokens": "9900",
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-limit-tokens": "10000",
            "x-ratelimit-reset-requests": "60",
            "x-ratelimit-reset-tokens": "60",
        }

        metadata = RequestMetadata(
            request_id="req-1",
            model_id="test-model",
            resource_type="chat",
        )

        await strategy._update_rate_limit_state(
            metadata, result=Mock(), status_code=200
        )

        mock_state_manager.update_state_from_headers.assert_called()

    @pytest.mark.asyncio
    async def test_update_with_clear_all_reservations(self, strategy, mock_backend):
        """Test update clears all reservations."""
        metadata = RequestMetadata(
            request_id="req-1",
            model_id="test-model",
            resource_type="chat",
        )

        # Store multiple reservations
        await strategy._store_reservation_context("req-1", "bucket-1", "res-1", 100)
        await strategy._store_reservation_context("req-1", "bucket-2", "res-2", 100)

        await strategy._update_rate_limit_state(
            metadata,
            result=None,
            status_code=None,
            clear_all_reservations=True,
        )

        # Both reservations should be cleared - verify via tracker's internal storage
        async with strategy._reservation_tracker._lock:
            assert len(strategy._reservation_tracker._reservation_contexts) == 0

    @pytest.mark.asyncio
    async def test_update_with_lua_zero_result(
        self, strategy, mock_state_manager, mock_backend, mock_scheduler
    ):
        """Test update handles Lua script returning zero."""
        mock_scheduler.extract_response_headers.return_value = {
            "x-ratelimit-remaining-requests": "99",
            "x-ratelimit-remaining-tokens": "9900",
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-limit-tokens": "10000",
            "x-ratelimit-reset-requests": "60",
            "x-ratelimit-reset-tokens": "60",
        }

        metadata = RequestMetadata(
            request_id="req-1",
            model_id="test-model",
            resource_type="chat",
        )

        # Store a reservation
        await strategy._store_reservation_context("req-1", "bucket-1", "res-1", 100)

        # Lua returns 0
        mock_state_manager.update_state_from_headers.return_value = 0

        await strategy._update_rate_limit_state(
            metadata, result=Mock(), status_code=200, bucket_id_override="bucket-1"
        )

        # Should fall back to release_reservation
        mock_backend.release_reservation.assert_called()

    @pytest.mark.asyncio
    async def test_update_with_partial_headers(
        self, strategy, mock_scheduler, mock_backend
    ):
        """Test update with partial headers uses release-only mode."""
        mock_scheduler.extract_response_headers.return_value = {
            "x-ratelimit-remaining-requests": "99",
        }

        metadata = RequestMetadata(
            request_id="req-1",
            model_id="test-model",
            resource_type="chat",
        )

        # Store a reservation
        await strategy._store_reservation_context("req-1", "bucket-1", "res-1", 100)

        await strategy._update_rate_limit_state(
            metadata, result=Mock(), status_code=200, bucket_id_override="bucket-1"
        )

        # Should use release-only
        mock_backend.release_reservation.assert_called()


# ============================================================================
# Reset Watcher Tests
# ============================================================================


class TestIntelligentModeStrategyResetWatcher:
    """Tests for rate limit reset watcher."""

    @pytest.mark.asyncio
    async def test_schedule_reset_watcher_immediate(self, strategy):
        """Test immediate reset triggers wakeup."""
        # Reset time in the past
        reset_ts = time.time() - 1.0

        await strategy._reset_watcher.schedule_watcher("bucket-1", reset_ts)

        # Wait deterministically for the fire-and-forget wakeup task to run
        # instead of racing a fixed sleep against a coarse timer.
        await asyncio.wait_for(strategy._wakeup_event.wait(), timeout=5.0)

        # Should trigger wakeup immediately
        assert strategy._wakeup_event.is_set()

    @pytest.mark.asyncio
    async def test_schedule_reset_watcher_future(self, strategy):
        """Test future reset schedules task."""
        reset_ts = time.time() + 10.0

        await strategy._reset_watcher.schedule_watcher("bucket-1", reset_ts)

        assert "bucket-1" in strategy._reset_watcher._buckets_waiting
        assert len(strategy._reset_watcher._reset_tasks) == 1

        # Cleanup
        for task in strategy._reset_watcher._reset_tasks:
            task.cancel()

    @pytest.mark.asyncio
    async def test_schedule_reset_watcher_deduplication(self, strategy):
        """Test duplicate watchers are not created."""
        reset_ts = time.time() + 10.0

        await strategy._reset_watcher.schedule_watcher("bucket-1", reset_ts)
        await strategy._reset_watcher.schedule_watcher("bucket-1", reset_ts)

        assert len(strategy._reset_watcher._reset_tasks) == 1

        # Cleanup
        for task in strategy._reset_watcher._reset_tasks:
            task.cancel()


# ============================================================================
# Earliest Reset Time Tests
# ============================================================================


class TestIntelligentModeStrategyEarliestReset:
    """Tests for _get_earliest_reset_time method."""

    @pytest.mark.asyncio
    async def test_get_earliest_reset_no_state_manager(self, strategy):
        """Test earliest reset with no state manager."""
        strategy.state_manager = None

        result = await strategy._get_earliest_reset_time()

        assert result is None

    @pytest.mark.asyncio
    async def test_get_earliest_reset_no_buckets(self, strategy):
        """Test earliest reset with no waiting buckets."""
        result = await strategy._get_earliest_reset_time()

        assert result is None


# ============================================================================
# Reset Time Calculation Tests
# ============================================================================


class TestIntelligentModeStrategyResetTimeCalculation:
    """Tests for adaptive wait time from earliest reset."""

    @pytest.mark.asyncio
    async def test_loop_calculates_adaptive_wait_time(
        self, strategy, mock_state_manager
    ):
        """Test wait time from earliest reset."""
        state = Mock()
        state.reset_at = datetime.now(timezone.utc) + timedelta(seconds=5)
        state.remaining_requests = 0

        mock_state_manager.get_state.return_value = state
        strategy._reset_watcher._buckets_waiting.add("bucket-1")

        result = await strategy._get_earliest_reset_time()
        assert result is not None
        assert result > time.time()

    @pytest.mark.asyncio
    async def test_get_earliest_reset_selects_minimum(
        self, strategy, mock_state_manager
    ):
        """Test earliest reset selects the minimum across buckets."""
        now = datetime.now(timezone.utc)

        # Set up multiple buckets with different reset times
        async def get_state_by_bucket(bucket_id, **kwargs):
            if bucket_id == "bucket-early":
                state = Mock()
                state.reset_at = now + timedelta(seconds=5)
                return state
            elif bucket_id == "bucket-late":
                state = Mock()
                state.reset_at = now + timedelta(seconds=30)
                return state
            return None

        mock_state_manager.get_state.side_effect = get_state_by_bucket
        strategy._reset_watcher._buckets_waiting.add("bucket-early")
        strategy._reset_watcher._buckets_waiting.add("bucket-late")

        result = await strategy._get_earliest_reset_time()

        # Should return the earlier time (about 5s from now)
        assert result is not None
        expected_min = (now + timedelta(seconds=5)).timestamp()
        assert abs(result - expected_min) < 1  # Within 1 second tolerance


# ============================================================================
# Rate Limit Branches Tests
# ============================================================================


class TestIntelligentModeStrategyRateLimitBranches:
    """Tests for rate limit checking branches in _find_eligible_queues_intelligent."""

    @pytest.mark.asyncio
    async def test_find_eligible_skips_exhausted_bucket_schedules_watcher(
        self, strategy, mock_state_manager
    ):
        """Test queue with exhausted rate limit schedules reset watcher."""
        state = Mock()
        state.remaining_requests = 0
        state.reset_at = datetime.now(timezone.utc) + timedelta(
            seconds=30
        )  # Not imminent
        mock_state_manager.get_state.return_value = state

        # Create a queue manually
        metadata = RequestMetadata(
            request_id="req-1",
            model_id="test-model",
            resource_type="chat",
        )
        await strategy._create_fast_queue("bucket-1:chat", metadata)
        strategy.fast_queues["bucket-1:chat"].append(
            QueuedRequest(
                metadata=metadata,
                request_func=AsyncMock(return_value="success"),
                future=asyncio.Future(),
                queue_entry_time=datetime.now(timezone.utc),
            )
        )
        strategy._queue_has_items["bucket-1:chat"] = True

        eligible = await strategy._find_eligible_queues_intelligent()

        assert len(eligible) == 0
        assert "bucket-1" in strategy._reset_watcher._buckets_waiting

        # Cleanup watchers
        for task in strategy._reset_watcher._reset_tasks:
            task.cancel()

    @pytest.mark.asyncio
    async def test_find_eligible_allows_imminent_reset(
        self, strategy, mock_state_manager
    ):
        """Test queue with imminent reset is included."""
        state = Mock()
        state.remaining_requests = 0
        state.reset_at = datetime.now(timezone.utc) + timedelta(
            milliseconds=1
        )  # Imminent
        mock_state_manager.get_state.return_value = state

        metadata = RequestMetadata(
            request_id="req-1",
            model_id="test-model",
            resource_type="chat",
        )
        await strategy._create_fast_queue("bucket-1:chat", metadata)
        strategy.fast_queues["bucket-1:chat"].append(
            QueuedRequest(
                metadata=metadata,
                request_func=AsyncMock(return_value="success"),
                future=asyncio.Future(),
                queue_entry_time=datetime.now(timezone.utc),
            )
        )
        strategy._queue_has_items["bucket-1:chat"] = True

        eligible = await strategy._find_eligible_queues_intelligent()

        # Should be included due to imminent reset
        assert len(eligible) >= 1

    @pytest.mark.asyncio
    async def test_find_eligible_skips_no_reset_time(
        self, strategy, mock_state_manager
    ):
        """Test queue with no reset time is skipped."""
        state = Mock()
        state.remaining_requests = 0
        state.reset_at = None  # No reset time
        mock_state_manager.get_state.return_value = state

        metadata = RequestMetadata(
            request_id="req-1",
            model_id="test-model",
            resource_type="chat",
        )
        await strategy._create_fast_queue("bucket-1:chat", metadata)
        strategy.fast_queues["bucket-1:chat"].append(
            QueuedRequest(
                metadata=metadata,
                request_func=AsyncMock(return_value="success"),
                future=asyncio.Future(),
                queue_entry_time=datetime.now(timezone.utc),
            )
        )
        strategy._queue_has_items["bucket-1:chat"] = True

        eligible = await strategy._find_eligible_queues_intelligent()

        # Should be skipped - no reset time available
        assert len(eligible) == 0


class TestResetHeaderPassthrough:
    """The reset-header normalization must not corrupt already-numeric values.

    Regression coverage for the venice-py addendum (2026-08-23). Venice sends a
    clean integer epoch (``'1767570180000'``). ``_parse_duration_string`` takes
    its ``float(value)`` fast path on that, and the result was written back with
    ``str()`` -- turning ``'1767570180000'`` into ``'1767570180000.0'``. The
    library corrupted the header and every downstream ``int()`` consumer then
    rejected the library's own output.

    The duration translation itself is still needed: OpenAI-style APIs send
    ``'6m0s'``. Only the already-numeric case must pass through untouched.

    These tests drive the real ``_update_rate_limit_state`` and assert on the
    header dict actually handed downstream -- not on a copy of the loop.
    """

    @staticmethod
    def _metadata():
        return RequestMetadata(
            request_id="req-passthrough",
            model_id="test-model",
            resource_type="chat",
        )

    @staticmethod
    def _headers(reset_req, reset_tok):
        return {
            "x-ratelimit-remaining-requests": "499",
            "x-ratelimit-remaining-tokens": "999000",
            "x-ratelimit-limit-requests": "500",
            "x-ratelimit-limit-tokens": "1000000",
            "x-ratelimit-reset-requests": reset_req,
            "x-ratelimit-reset-tokens": reset_tok,
        }

    @staticmethod
    def _sent_headers(mock_state_manager):
        """The headers dict actually passed to the state manager."""
        call = mock_state_manager.update_state_from_headers.call_args
        return call.args[2] if len(call.args) > 2 else call.kwargs["headers"]

    @pytest.mark.asyncio
    async def test_integer_epoch_header_is_not_rewritten(
        self, strategy, mock_scheduler, mock_state_manager
    ):
        """Venice's clean integer epoch must reach the backend unchanged."""
        mock_scheduler.extract_response_headers.return_value = self._headers(
            "1767570180000", "1767570180000"
        )

        await strategy._update_rate_limit_state(
            self._metadata(), result=Mock(), status_code=200
        )

        sent = self._sent_headers(mock_state_manager)
        assert sent["x-ratelimit-reset-requests"] == "1767570180000"
        assert sent["x-ratelimit-reset-tokens"] == "1767570180000"

    @pytest.mark.asyncio
    async def test_plain_numeric_headers_are_not_rewritten(
        self, strategy, mock_scheduler, mock_state_manager
    ):
        """Any already-numeric value passes through verbatim."""
        mock_scheduler.extract_response_headers.return_value = self._headers("60", "30")

        await strategy._update_rate_limit_state(
            self._metadata(), result=Mock(), status_code=200
        )

        sent = self._sent_headers(mock_state_manager)
        assert sent["x-ratelimit-reset-requests"] == "60"
        assert sent["x-ratelimit-reset-tokens"] == "30"

    @pytest.mark.asyncio
    async def test_duration_strings_are_still_translated(
        self, strategy, mock_scheduler, mock_state_manager
    ):
        """OpenAI-style duration strings must still become seconds."""
        mock_scheduler.extract_response_headers.return_value = self._headers(
            "6m0s", "500ms"
        )

        await strategy._update_rate_limit_state(
            self._metadata(), result=Mock(), status_code=200
        )

        sent = self._sent_headers(mock_state_manager)
        assert sent["x-ratelimit-reset-requests"] == "360.0"
        assert sent["x-ratelimit-reset-tokens"] == "0.5"

    @pytest.mark.asyncio
    async def test_rewritten_headers_survive_backend_parsing(
        self, strategy, mock_scheduler, mock_state_manager
    ):
        """Whatever this layer emits must be parseable by the backend parser.

        This is the assertion that would have caught the original defect: the
        producer and the consumer are checked against each other.
        """
        from adaptive_rate_limiter.backends.base import BaseBackend

        for reset_req, reset_tok in [
            ("1767570180000", "1767570180000"),  # Venice: epoch milliseconds
            ("6m0s", "500ms"),  # OpenAI: duration strings
            ("60", "30"),  # plain delta seconds
        ]:
            mock_scheduler.extract_response_headers.return_value = self._headers(
                reset_req, reset_tok
            )
            await strategy._update_rate_limit_state(
                self._metadata(), result=Mock(), status_code=200
            )
            sent = self._sent_headers(mock_state_manager)

            parsed = BaseBackend._parse_rate_limit_headers(BaseBackend, sent)
            assert "rpm_reset" in parsed, f"backend dropped {sent!r}"
            assert "tpm_reset" in parsed, f"backend dropped {sent!r}"
