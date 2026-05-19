"""Unit tests for RateLimitResetWatcher."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from adaptive_rate_limiter.strategies.modes.reset_watcher import RateLimitResetWatcher


class TestGetEarliestResetTime:
    """Tests for RateLimitResetWatcher.get_earliest_reset_time."""

    @pytest.mark.asyncio
    async def test_survives_concurrent_buckets_waiting_mutation(self):
        """get_earliest_reset_time must not crash when _buckets_waiting is
        mutated (by a reset watcher firing) while it iterates.

        Regression: the method iterated the live set directly and awaited
        inside the loop, so a concurrent discard raised
        ``RuntimeError: Set changed size during iteration``.
        """
        state_manager = Mock()
        watcher = RateLimitResetWatcher(
            state_manager=state_manager,
            wakeup_callback=Mock(),
        )

        for i in range(5):
            watcher._buckets_waiting.add(f"bucket-{i}")

        reset_at = datetime.now(timezone.utc) + timedelta(seconds=30)
        call_count = 0

        async def get_state(bucket_id):
            nonlocal call_count
            call_count += 1
            # Simulate a reset watcher task firing mid-iteration.
            if call_count == 1:
                watcher._buckets_waiting.discard("bucket-4")
            return Mock(reset_at=reset_at)

        state_manager.get_state = get_state

        result = await watcher.get_earliest_reset_time()

        assert result == reset_at.timestamp()

    @pytest.mark.asyncio
    async def test_returns_earliest_across_buckets(self):
        """Returns the smallest reset timestamp across all watched buckets."""
        state_manager = Mock()
        watcher = RateLimitResetWatcher(
            state_manager=state_manager,
            wakeup_callback=Mock(),
        )

        now = datetime.now(timezone.utc)
        resets = {
            "bucket-a": now + timedelta(seconds=90),
            "bucket-b": now + timedelta(seconds=10),
            "bucket-c": now + timedelta(seconds=60),
        }
        for bucket_id in resets:
            watcher._buckets_waiting.add(bucket_id)

        async def get_state(bucket_id):
            return Mock(reset_at=resets[bucket_id])

        state_manager.get_state = get_state

        result = await watcher.get_earliest_reset_time()

        assert result == resets["bucket-b"].timestamp()

    @pytest.mark.asyncio
    async def test_returns_none_when_no_buckets(self):
        """Returns None when no buckets are being watched."""
        state_manager = Mock()
        state_manager.get_state = Mock(side_effect=AssertionError("unexpected"))
        watcher = RateLimitResetWatcher(
            state_manager=state_manager,
            wakeup_callback=Mock(),
        )

        assert await watcher.get_earliest_reset_time() is None
