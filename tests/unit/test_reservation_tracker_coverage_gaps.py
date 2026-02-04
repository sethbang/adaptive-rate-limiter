"""Tests targeting specific coverage gaps in ReservationTracker.

This module focuses on:
- Lines 222-224, 230-233: Concurrency logic for empty keys set and secondary index cleanup
- Lines 334-342: Cleanup paths for stale/orphaned heap entries
"""

import asyncio
import time

import pytest

from adaptive_rate_limiter.reservation import ReservationTracker


class TestEmptyKeysSetEdgeCase:
    """Tests for lines 222-224: handling empty keys set in secondary index.

    This edge case can occur under concurrent modification when another task
    clears the last key from a request's set between the index lookup and
    the keys check.
    """

    @pytest.mark.asyncio
    async def test_get_and_clear_when_keys_set_is_empty(self):
        """Test get_and_clear returns None when keys set exists but is empty.

        Targets lines 222-224:
            if not keys:
                del self._request_id_index[request_id]
                return None
        """
        tracker = ReservationTracker()

        # Store a reservation
        await tracker.store("req-1", "bucket-a", "res-1", 50)

        # Manually create the edge case: request_id exists but keys set is empty
        # This simulates a race condition where another task cleared the last key
        async with tracker._lock:
            # Clear the set but leave the key in the index
            tracker._request_id_index["req-1"].clear()
            # Also remove from primary storage to match the state
            tracker._reservation_contexts.pop(("req-1", "bucket-a"), None)

        # Now get_and_clear without bucket_id should hit lines 222-224
        result = await tracker.get_and_clear("req-1", bucket_id=None)
        assert result is None

        # The empty index entry should have been cleaned up
        assert "req-1" not in tracker._request_id_index

    @pytest.mark.asyncio
    async def test_concurrent_clear_and_get_race_condition(self):
        """Test race condition where clear_all_for_request races with get_and_clear.

        This simulates the scenario where:
        1. Task A calls get_and_clear(request_id) without bucket_id
        2. Task B concurrently calls clear_all_for_request(request_id)
        3. Task A finds the request_id in index but set might be emptied by B
        """
        tracker = ReservationTracker()

        # Store multiple reservations for same request
        await tracker.store("req-shared", "bucket-a", "res-a", 50)
        await tracker.store("req-shared", "bucket-b", "res-b", 75)

        results: list = []
        clear_results: list = []

        async def task_get_and_clear():
            for _ in range(5):
                ctx = await tracker.get_and_clear("req-shared", bucket_id=None)
                if ctx:
                    results.append(ctx)
                await asyncio.sleep(0.001)

        async def task_clear_all():
            await asyncio.sleep(0.002)  # Slight delay
            cleared = await tracker.clear_all_for_request("req-shared")
            clear_results.extend(cleared)

        # Run both concurrently
        await asyncio.gather(task_get_and_clear(), task_clear_all())

        # All reservations should be accounted for between both operations
        total_cleared = len(results) + len(clear_results)
        assert total_cleared == 2  # Both reservations should be cleared exactly once

        # Tracker should be empty
        assert tracker.reservation_count == 0
        assert tracker.request_count == 0


class TestSecondaryIndexCleanupAfterPop:
    """Tests for lines 230-233: cleanup of secondary index after successful pop.

    This tests the normal path where context is found and removed via
    secondary index lookup (without bucket_id).
    """

    @pytest.mark.asyncio
    async def test_get_and_clear_without_bucket_cleans_index(self):
        """Test that get_and_clear without bucket_id properly cleans secondary index.

        Targets lines 230-233:
            if context is not None:
                keys.discard(key)
                if not keys:
                    del self._request_id_index[request_id]
        """
        tracker = ReservationTracker()

        # Store a single reservation
        await tracker.store("req-solo", "bucket-only", "res-solo", 100)

        assert "req-solo" in tracker._request_id_index
        assert len(tracker._request_id_index["req-solo"]) == 1

        # get_and_clear without bucket_id should remove context and clean index
        ctx = await tracker.get_and_clear("req-solo", bucket_id=None)

        assert ctx is not None
        assert ctx.reservation_id == "res-solo"

        # Secondary index should be cleaned up since it was the last key
        assert "req-solo" not in tracker._request_id_index

    @pytest.mark.asyncio
    async def test_get_and_clear_without_bucket_partial_cleanup(self):
        """Test that get_and_clear without bucket_id only removes one key from set.

        This tests the partial cleanup path where keys set still has entries
        after discarding one.
        """
        tracker = ReservationTracker()

        # Store multiple reservations for same request
        await tracker.store("req-multi", "bucket-a", "res-a", 50)
        await tracker.store("req-multi", "bucket-b", "res-b", 75)
        await tracker.store("req-multi", "bucket-c", "res-c", 100)

        assert "req-multi" in tracker._request_id_index
        assert len(tracker._request_id_index["req-multi"]) == 3

        # First get_and_clear should remove one entry
        ctx1 = await tracker.get_and_clear("req-multi", bucket_id=None)
        assert ctx1 is not None
        assert "req-multi" in tracker._request_id_index
        assert len(tracker._request_id_index["req-multi"]) == 2

        # Second get_and_clear should remove another
        ctx2 = await tracker.get_and_clear("req-multi", bucket_id=None)
        assert ctx2 is not None
        assert "req-multi" in tracker._request_id_index
        assert len(tracker._request_id_index["req-multi"]) == 1

        # Third should remove last and clean up index
        ctx3 = await tracker.get_and_clear("req-multi", bucket_id=None)
        assert ctx3 is not None
        assert "req-multi" not in tracker._request_id_index

    @pytest.mark.asyncio
    async def test_context_none_after_pop_from_concurrent_removal(self):
        """Test edge case where context is None after pop due to concurrent removal.

        This tests line 230 (if context is not None) by simulating a scenario
        where the key is found in the index but the context was already removed.
        """
        tracker = ReservationTracker()

        await tracker.store("req-race", "bucket-x", "res-x", 50)

        # Create a scenario where key exists in index but not in primary storage
        # This simulates another operation removing it between finding the key
        # and popping the context
        async with tracker._lock:
            # Remove from primary storage but leave in index
            tracker._reservation_contexts.pop(("req-race", "bucket-x"), None)

        # get_and_clear should handle this gracefully
        ctx = await tracker.get_and_clear("req-race", bucket_id=None)

        # Context was already removed, so None is returned
        # The key should still be in the index but with the key removed
        # Actually, since pop returned None, lines 230-233 won't execute,
        # and the function returns None with stale index entry
        assert ctx is None


class TestStaleHeapEntryCleanup:
    """Tests for lines 334-342: cleanup of stale/orphaned heap entries.

    The min-heap may contain stale references to:
    1. Keys that were already removed (context is None)
    2. Keys that were removed and re-added (created_at mismatch)
    """

    @pytest.mark.asyncio
    async def test_cleanup_skips_already_removed_entries(self):
        """Test cleanup skips heap entries for already-removed keys.

        Targets lines 334-336:
            if context is None:
                # Key was already removed (e.g., via get_and_clear)
                continue
        """
        tracker = ReservationTracker(
            max_reservations=100,
            max_reservation_age=1,  # 1 second
            stale_cleanup_interval=3600,  # Don't auto-cleanup
        )

        # Store and immediately clear a reservation
        await tracker.store("req-gone", "bucket-a", "res-gone", 50)
        await tracker.get_and_clear("req-gone", "bucket-a")

        # Heap still has the entry (per design - lazy cleanup)
        assert len(tracker._time_heap) == 1
        assert tracker.reservation_count == 0

        # Make the heap entry stale by manipulating the time
        # The entry's timestamp is from when it was created
        old_entry = tracker._time_heap[0]
        # Replace with an older timestamp to trigger cleanup
        tracker._time_heap[0] = (time.time() - 2, old_entry[1])

        # Run cleanup - should skip the orphaned entry
        cleaned = await tracker._cleanup_stale()
        assert cleaned == 0  # Nothing to clean, context was already gone

        # Heap should be empty after processing the stale entry
        assert len(tracker._time_heap) == 0

    @pytest.mark.asyncio
    async def test_cleanup_skips_replaced_entries_with_mismatched_timestamp(self):
        """Test cleanup skips heap entries where key was removed and re-added.

        Targets lines 340-342:
            if context.created_at != created_at:
                # Different context at this key, skip
                continue
        """
        tracker = ReservationTracker(
            max_reservations=100,
            max_reservation_age=1,  # 1 second
            stale_cleanup_interval=3600,  # Don't auto-cleanup
        )

        # Store an old reservation
        old_time = time.time() - 5  # 5 seconds ago
        await tracker.store("req-replaced", "bucket-a", "res-old", 50)

        # Manually age it
        key = ("req-replaced", "bucket-a")
        tracker._reservation_contexts[key].created_at = old_time

        # Rebuild heap to have the correct old timestamp
        tracker._rebuild_time_heap()

        # Now replace with a new reservation (same key, new context)
        await tracker.store("req-replaced", "bucket-a", "res-new", 100)

        # Now the heap has TWO entries for the same key:
        # 1. Old entry with old_time timestamp (stale reference)
        # 2. New entry with current timestamp
        assert len(tracker._time_heap) == 2

        # Run cleanup - the old stale entry should be skipped because
        # context.created_at != heap entry's created_at
        await tracker._cleanup_stale()

        # The old heap entry was processed and skipped (timestamp mismatch)
        # Only one reservation remains
        assert tracker.reservation_count == 1

        # The new reservation should still be there
        ctx = await tracker.get("req-replaced", "bucket-a")
        assert ctx is not None
        assert ctx.reservation_id == "res-new"

    @pytest.mark.asyncio
    async def test_cleanup_handles_mixed_stale_scenarios(self):
        """Test cleanup correctly handles a mix of valid, removed, and replaced entries."""
        tracker = ReservationTracker(
            max_reservations=100,
            max_reservation_age=2,
            stale_cleanup_interval=3600,
        )

        now = time.time()
        old_time = now - 5  # 5 seconds ago (stale)
        fresh_time = now + 10  # Future (not stale)

        # Scenario 1: Valid stale entry (will be cleaned)
        await tracker.store("req-valid-stale", "bucket-a", "res-valid-stale", 50)
        tracker._reservation_contexts[
            ("req-valid-stale", "bucket-a")
        ].created_at = old_time

        # Scenario 2: Removed entry (will be skipped - context is None)
        await tracker.store("req-removed", "bucket-a", "res-removed", 50)
        tracker._reservation_contexts[("req-removed", "bucket-a")].created_at = old_time
        await tracker.get_and_clear("req-removed", "bucket-a")

        # Scenario 3: Replaced entry (will be skipped - timestamp mismatch)
        await tracker.store("req-replaced", "bucket-a", "res-old", 50)
        tracker._reservation_contexts[
            ("req-replaced", "bucket-a")
        ].created_at = old_time
        # Now replace it with same key
        await tracker.store("req-replaced", "bucket-a", "res-new", 100)

        # Scenario 4: Fresh entry (not stale, won't be processed)
        await tracker.store("req-fresh", "bucket-a", "res-fresh", 50)
        tracker._reservation_contexts[("req-fresh", "bucket-a")].created_at = fresh_time

        # Rebuild heap to sync timestamps
        tracker._rebuild_time_heap()

        # We should have:
        # Primary storage: 3 contexts (valid-stale, replaced-new, fresh)
        # Heap: entries for all including the orphaned/stale ones
        assert tracker.reservation_count == 3

        # Run cleanup
        cleaned = await tracker._cleanup_stale()

        # Only the valid stale entry should be cleaned
        assert cleaned == 1

        # Should have 2 remaining: replaced (new) and fresh
        assert tracker.reservation_count == 2

        # Verify the right ones remain
        ctx_new = await tracker.get("req-replaced", "bucket-a")
        assert ctx_new is not None
        assert ctx_new.reservation_id == "res-new"

        ctx_fresh = await tracker.get("req-fresh", "bucket-a")
        assert ctx_fresh is not None
        assert ctx_fresh.reservation_id == "res-fresh"


class TestHighConcurrencyScenarios:
    """Tests for thread-safety under high concurrency."""

    @pytest.mark.asyncio
    async def test_concurrent_store_get_and_clear_same_request(self):
        """Test rapid concurrent store/clear operations on same request_id."""
        tracker = ReservationTracker(max_reservations=1000)

        async def rapid_store_and_clear(iteration: int):
            request_id = f"req-concurrent-{iteration % 10}"  # Shared among iterations
            bucket_id = f"bucket-{iteration}"

            await tracker.store(request_id, bucket_id, f"res-{iteration}", 10)
            await asyncio.sleep(0.001)  # Small delay
            await tracker.get_and_clear(request_id, bucket_id)

        # Run 100 concurrent operations
        tasks = [rapid_store_and_clear(i) for i in range(100)]
        await asyncio.gather(*tasks)

        # All should be cleaned up
        assert tracker.reservation_count == 0

    @pytest.mark.asyncio
    async def test_concurrent_cleanup_and_get_and_clear(self):
        """Test concurrent cleanup operation races with get_and_clear."""
        tracker = ReservationTracker(
            max_reservations=100,
            max_reservation_age=1,
            stale_cleanup_interval=3600,
        )

        # Store multiple reservations
        for i in range(20):
            await tracker.store(f"req-{i}", "bucket", f"res-{i}", 10)
            tracker._reservation_contexts[(f"req-{i}", "bucket")].created_at = (
                time.time() - 5
            )

        tracker._rebuild_time_heap()

        results: list = []

        async def task_cleanup():
            for _ in range(5):
                await tracker._cleanup_stale()
                await asyncio.sleep(0.01)

        async def task_get_and_clear():
            for i in range(20):
                ctx = await tracker.get_and_clear(f"req-{i}", "bucket")
                if ctx:
                    results.append(ctx)
                await asyncio.sleep(0.005)

        # Race cleanup against get_and_clear
        await asyncio.gather(task_cleanup(), task_get_and_clear())

        # All reservations should be accounted for (either cleaned or returned)
        # Due to race conditions, not all may be returned via get_and_clear
        assert tracker.reservation_count == 0

    @pytest.mark.asyncio
    async def test_parallel_store_at_capacity(self):
        """Test parallel stores when near capacity boundary."""
        tracker = ReservationTracker(
            max_reservations=10,
            max_reservation_age=60,
            stale_cleanup_interval=3600,
        )

        # Fill to near capacity
        for i in range(8):
            await tracker.store(f"req-fill-{i}", "bucket", f"res-fill-{i}", 10)

        errors = []
        success_count = 0

        async def try_store(idx: int):
            nonlocal success_count
            try:
                await tracker.store(f"req-new-{idx}", "bucket", f"res-new-{idx}", 10)
                success_count += 1
            except Exception as e:
                errors.append(e)

        # Try to add 5 more (only 2 slots available)
        tasks = [try_store(i) for i in range(5)]
        await asyncio.gather(*tasks)

        # At least some should succeed, and some may hit capacity
        assert success_count + len(errors) == 5
        assert tracker.reservation_count <= 10


class TestGetAndClearStaleHeapEdgeCases:
    """Tests for edge cases in get_and_clear_stale heap processing."""

    @pytest.mark.asyncio
    async def test_get_and_clear_stale_skips_orphaned_entries(self):
        """Test that get_and_clear_stale properly handles orphaned heap entries."""
        tracker = ReservationTracker(
            max_reservations=100,
            max_reservation_age=5,
            stale_cleanup_interval=3600,
        )

        # Store and immediately remove (creates orphan in heap)
        await tracker.store("req-orphan", "bucket-a", "res-orphan", 50)
        old_time = time.time() - 10
        tracker._reservation_contexts[("req-orphan", "bucket-a")].created_at = old_time
        tracker._rebuild_time_heap()
        await tracker.get_and_clear("req-orphan", "bucket-a")

        # Store fresh one
        await tracker.store("req-fresh", "bucket-a", "res-fresh", 50)

        # Heap has 2 entries: orphan (old) and fresh (new)
        assert len(tracker._time_heap) == 2
        assert tracker.reservation_count == 1

        # get_and_clear_stale should skip the orphan
        cutoff = time.time() - 1
        stale = await tracker.get_and_clear_stale(cutoff)

        # No stale contexts returned (orphan skipped, fresh not stale)
        assert len(stale) == 0
        assert tracker.reservation_count == 1

    @pytest.mark.asyncio
    async def test_get_and_clear_stale_timestamp_mismatch(self):
        """Test get_and_clear_stale handles replaced entries correctly."""
        tracker = ReservationTracker(
            max_reservations=100,
            max_reservation_age=5,
            stale_cleanup_interval=3600,
        )

        # Store old entry
        await tracker.store("req-x", "bucket-a", "res-old", 50)
        old_time = time.time() - 10
        tracker._reservation_contexts[("req-x", "bucket-a")].created_at = old_time
        tracker._rebuild_time_heap()

        # Replace with new entry (same key)
        await tracker.store("req-x", "bucket-a", "res-new", 100)

        # Heap now has old entry (with old timestamp) and new entry
        assert len(tracker._time_heap) == 2

        # get_and_clear_stale should skip old entry (timestamp mismatch with current context)
        cutoff = time.time() - 1
        stale = await tracker.get_and_clear_stale(cutoff)

        # No stale returned - old heap entry was skipped due to mismatch
        assert len(stale) == 0

        # Context should still exist (the new one)
        ctx = await tracker.get("req-x", "bucket-a")
        assert ctx is not None
        assert ctx.reservation_id == "res-new"
