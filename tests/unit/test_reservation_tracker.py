"""Tests for ReservationTracker and ReservationContext."""

import asyncio
import time

import pytest

from adaptive_rate_limiter.exceptions import ReservationCapacityError
from adaptive_rate_limiter.reservation import ReservationContext, ReservationTracker


class TestReservationContext:
    """Tests for the ReservationContext dataclass."""

    def test_creation_with_defaults(self):
        """Test creating a ReservationContext with default created_at."""
        before = time.time()
        ctx = ReservationContext(
            reservation_id="res-123",
            bucket_id="bucket-abc",
            estimated_tokens=100,
        )
        after = time.time()

        assert ctx.reservation_id == "res-123"
        assert ctx.bucket_id == "bucket-abc"
        assert ctx.estimated_tokens == 100
        assert before <= ctx.created_at <= after

    def test_creation_with_explicit_timestamp(self):
        """Test creating a ReservationContext with explicit created_at."""
        timestamp = 1234567890.0
        ctx = ReservationContext(
            reservation_id="res-123",
            bucket_id="bucket-abc",
            estimated_tokens=100,
            created_at=timestamp,
        )

        assert ctx.created_at == timestamp


class TestReservationTracker:
    """Tests for the ReservationTracker class."""

    @pytest.fixture
    def tracker(self):
        """Create a fresh tracker for each test."""
        return ReservationTracker(
            max_reservations=100,
            max_reservation_age=60,
            stale_cleanup_interval=10,
        )

    @pytest.mark.asyncio
    async def test_store_and_get(self, tracker):
        """Test storing and retrieving a reservation."""
        await tracker.store(
            request_id="req-1",
            bucket_id="bucket-a",
            reservation_id="res-1",
            estimated_tokens=50,
        )

        ctx = await tracker.get("req-1", "bucket-a")
        assert ctx is not None
        assert ctx.reservation_id == "res-1"
        assert ctx.bucket_id == "bucket-a"
        assert ctx.estimated_tokens == 50

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, tracker):
        """Test getting a non-existent reservation returns None."""
        ctx = await tracker.get("nonexistent", "bucket")
        assert ctx is None

    @pytest.mark.asyncio
    async def test_compound_key_lookup(self, tracker):
        """Test compound key (request_id, bucket_id) lookup."""
        # Store multiple reservations for same request with different buckets
        await tracker.store("req-1", "bucket-a", "res-1", 50)
        await tracker.store("req-1", "bucket-b", "res-2", 75)
        await tracker.store("req-2", "bucket-a", "res-3", 100)

        # Each compound key should return the correct reservation
        ctx1 = await tracker.get("req-1", "bucket-a")
        assert ctx1 is not None
        assert ctx1.reservation_id == "res-1"

        ctx2 = await tracker.get("req-1", "bucket-b")
        assert ctx2 is not None
        assert ctx2.reservation_id == "res-2"

        ctx3 = await tracker.get("req-2", "bucket-a")
        assert ctx3 is not None
        assert ctx3.reservation_id == "res-3"

    @pytest.mark.asyncio
    async def test_secondary_index_lookup(self, tracker):
        """Test O(1) lookup by request_id without bucket_id."""
        await tracker.store("req-1", "bucket-a", "res-1", 50)
        await tracker.store("req-1", "bucket-b", "res-2", 75)

        # get_and_clear without bucket_id should return one of the reservations
        ctx = await tracker.get_and_clear("req-1", bucket_id=None)
        assert ctx is not None
        assert ctx.reservation_id in ["res-1", "res-2"]

        # Should still have one reservation left
        assert tracker.reservation_count == 1

    @pytest.mark.asyncio
    async def test_get_and_clear_specific(self, tracker):
        """Test get_and_clear with specific bucket_id."""
        await tracker.store("req-1", "bucket-a", "res-1", 50)
        await tracker.store("req-1", "bucket-b", "res-2", 75)

        ctx = await tracker.get_and_clear("req-1", "bucket-a")
        assert ctx is not None
        assert ctx.reservation_id == "res-1"

        # Should not be able to get it again
        ctx2 = await tracker.get_and_clear("req-1", "bucket-a")
        assert ctx2 is None

        # Other reservation should still exist
        ctx3 = await tracker.get("req-1", "bucket-b")
        assert ctx3 is not None
        assert ctx3.reservation_id == "res-2"

    @pytest.mark.asyncio
    async def test_idempotent_get_and_clear(self, tracker):
        """Test that get_and_clear is idempotent."""
        await tracker.store("req-1", "bucket-a", "res-1", 50)

        # First call returns the context
        ctx1 = await tracker.get_and_clear("req-1", "bucket-a")
        assert ctx1 is not None
        assert ctx1.reservation_id == "res-1"

        # Second call returns None (idempotent)
        ctx2 = await tracker.get_and_clear("req-1", "bucket-a")
        assert ctx2 is None

        # Third call also returns None
        ctx3 = await tracker.get_and_clear("req-1", "bucket-a")
        assert ctx3 is None

    @pytest.mark.asyncio
    async def test_clear_all_for_request(self, tracker):
        """Test clearing all reservations for a request."""
        await tracker.store("req-1", "bucket-a", "res-1", 50)
        await tracker.store("req-1", "bucket-b", "res-2", 75)
        await tracker.store("req-1", "bucket-c", "res-3", 100)
        await tracker.store("req-2", "bucket-a", "res-4", 25)

        assert tracker.reservation_count == 4
        assert tracker.request_count == 2

        # Clear all for req-1
        cleared = await tracker.clear_all_for_request("req-1")
        assert len(cleared) == 3
        reservation_ids = {c.reservation_id for c in cleared}
        assert reservation_ids == {"res-1", "res-2", "res-3"}

        # req-1 should have no reservations left
        assert tracker.reservation_count == 1
        assert tracker.request_count == 1

        # req-2 should still have its reservation
        ctx = await tracker.get("req-2", "bucket-a")
        assert ctx is not None
        assert ctx.reservation_id == "res-4"

    @pytest.mark.asyncio
    async def test_clear_all_for_fallback(self, tracker):
        """Test clearing all reservations for a request (fallback scenario)."""
        # Simulate a fallback scenario where request made reservations on multiple buckets
        await tracker.store("req-fallback", "tier1:bucket", "res-tier1", 100)
        await tracker.store("req-fallback", "tier2:bucket", "res-tier2", 100)
        await tracker.store("req-fallback", "tier3:bucket", "res-tier3", 100)

        # When fallback completes, clear all reservations
        cleared = await tracker.clear_all_for_request("req-fallback")
        assert len(cleared) == 3

        # All should be cleared
        assert tracker.reservation_count == 0
        assert tracker.request_count == 0

    @pytest.mark.asyncio
    async def test_clear_all_for_nonexistent_request(self, tracker):
        """Test clearing reservations for a non-existent request returns empty list."""
        cleared = await tracker.clear_all_for_request("nonexistent")
        assert cleared == []

    @pytest.mark.asyncio
    async def test_capacity_limit(self, tracker):
        """Test that ReservationCapacityError is raised when at max capacity."""
        # Fill to capacity (100 for this test tracker)
        for i in range(100):
            await tracker.store(f"req-{i}", "bucket", f"res-{i}", 10)

        assert tracker.reservation_count == 100

        # Next store should raise ReservationCapacityError
        with pytest.raises(ReservationCapacityError) as exc_info:
            await tracker.store("req-overflow", "bucket", "res-overflow", 10)

        assert "at capacity" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_capacity_limit_cleans_stale_first(self):
        """Test that hitting capacity triggers cleanup before raising error."""
        tracker = ReservationTracker(
            max_reservations=10,
            max_reservation_age=1,  # 1 second max age
            stale_cleanup_interval=3600,  # Don't auto-cleanup during test
        )

        # Add reservations with old timestamps by manipulating created_at directly
        old_time = time.time() - 2  # 2 seconds ago (older than max_age)
        for i in range(10):
            await tracker.store(f"req-{i}", "bucket", f"res-{i}", 10)
            # Manually set the created_at to be stale
            key = (f"req-{i}", "bucket")
            tracker._reservation_contexts[key].created_at = old_time

        # Rebuild heap after modifying timestamps (required for heap-based cleanup)
        tracker._rebuild_time_heap()

        assert tracker.reservation_count == 10

        # Store should succeed because it cleans stale reservations first
        await tracker.store("req-new", "bucket", "res-new", 10)
        assert tracker.reservation_count == 1  # Only the new one remains

    @pytest.mark.asyncio
    async def test_stale_cleanup(self):
        """Test cleanup of reservations older than max_age."""
        tracker = ReservationTracker(
            max_reservations=100,
            max_reservation_age=2,  # 2 seconds
            stale_cleanup_interval=3600,  # Don't auto-cleanup during test
        )

        # Add a stale reservation and manually set its created_at
        await tracker.store("req-old", "bucket", "res-old", 50)
        old_time = time.time() - 3  # 3 seconds ago
        tracker._reservation_contexts[("req-old", "bucket")].created_at = old_time

        # Add a fresh reservation
        await tracker.store("req-new", "bucket", "res-new", 50)

        # Rebuild heap after modifying timestamps (required for heap-based cleanup)
        tracker._rebuild_time_heap()

        assert tracker.reservation_count == 2

        # Run cleanup
        cleaned = await tracker._cleanup_stale()
        assert cleaned == 1

        # Only fresh reservation should remain
        assert tracker.reservation_count == 1
        ctx = await tracker.get("req-new", "bucket")
        assert ctx is not None
        assert ctx.reservation_id == "res-new"

        # Old reservation should be gone
        ctx_old = await tracker.get("req-old", "bucket")
        assert ctx_old is None

    @pytest.mark.asyncio
    async def test_start_and_stop(self, tracker):
        """Test starting and stopping the cleanup task."""
        assert tracker._cleanup_task is None
        assert tracker._running is False

        await tracker.start()
        assert tracker._cleanup_task is not None
        assert tracker._running is True

        # Starting again should be a no-op
        task = tracker._cleanup_task
        await tracker.start()
        assert tracker._cleanup_task is task

        await tracker.stop()
        assert tracker._cleanup_task is None
        assert tracker._running is False

        # Stopping again should be a no-op
        await tracker.stop()
        assert tracker._cleanup_task is None

    @pytest.mark.asyncio
    async def test_cleanup_loop_runs(self):
        """Test that the cleanup loop runs periodically."""
        tracker = ReservationTracker(
            max_reservations=100,
            max_reservation_age=1,
            stale_cleanup_interval=0.1,  # Very short interval for test
        )

        # Add a stale reservation and manually set its created_at
        await tracker.store("req-old", "bucket", "res-old", 50)
        old_time = time.time() - 2
        tracker._reservation_contexts[("req-old", "bucket")].created_at = old_time

        # Rebuild heap after modifying timestamps (required for heap-based cleanup)
        tracker._rebuild_time_heap()

        assert tracker.reservation_count == 1

        await tracker.start()
        try:
            # Wait for cleanup to run
            await asyncio.sleep(0.3)
            assert tracker.reservation_count == 0
        finally:
            await tracker.stop()

    @pytest.mark.asyncio
    async def test_overwrite_reservation(self, tracker):
        """Test that storing same key overwrites existing reservation."""
        await tracker.store("req-1", "bucket-a", "res-1", 50)
        ctx1 = await tracker.get("req-1", "bucket-a")
        assert ctx1.reservation_id == "res-1"
        assert ctx1.estimated_tokens == 50

        # Store with same key
        await tracker.store("req-1", "bucket-a", "res-2", 100)
        ctx2 = await tracker.get("req-1", "bucket-a")
        assert ctx2.reservation_id == "res-2"
        assert ctx2.estimated_tokens == 100

        # Should still only have 1 reservation
        assert tracker.reservation_count == 1

    @pytest.mark.asyncio
    async def test_reservation_count_property(self, tracker):
        """Test the reservation_count property."""
        assert tracker.reservation_count == 0

        await tracker.store("req-1", "bucket-a", "res-1", 50)
        assert tracker.reservation_count == 1

        await tracker.store("req-1", "bucket-b", "res-2", 75)
        assert tracker.reservation_count == 2

        await tracker.get_and_clear("req-1", "bucket-a")
        assert tracker.reservation_count == 1

    @pytest.mark.asyncio
    async def test_request_count_property(self, tracker):
        """Test the request_count property."""
        assert tracker.request_count == 0

        await tracker.store("req-1", "bucket-a", "res-1", 50)
        assert tracker.request_count == 1

        await tracker.store("req-1", "bucket-b", "res-2", 75)
        assert tracker.request_count == 1  # Same request

        await tracker.store("req-2", "bucket-a", "res-3", 100)
        assert tracker.request_count == 2

        await tracker.clear_all_for_request("req-1")
        assert tracker.request_count == 1

    @pytest.mark.asyncio
    async def test_concurrent_access(self, tracker):
        """Test thread-safe concurrent access."""

        async def store_and_clear(i: int):
            await tracker.store(f"req-{i}", "bucket", f"res-{i}", 10)
            await asyncio.sleep(0.01)
            await tracker.get_and_clear(f"req-{i}", "bucket")

        # Run many concurrent operations
        tasks = [store_and_clear(i) for i in range(50)]
        await asyncio.gather(*tasks)

        # Should be back to 0
        assert tracker.reservation_count == 0

    @pytest.mark.asyncio
    async def test_get_and_clear_without_bucket_nonexistent(self, tracker):
        """Test get_and_clear without bucket_id for non-existent request."""
        ctx = await tracker.get_and_clear("nonexistent", bucket_id=None)
        assert ctx is None

    @pytest.mark.asyncio
    async def test_secondary_index_cleanup_on_last_reservation(self, tracker):
        """Test that secondary index is cleaned up when last reservation is removed."""
        await tracker.store("req-1", "bucket-a", "res-1", 50)
        assert "req-1" in tracker._request_id_index

        await tracker.get_and_clear("req-1", "bucket-a")
        assert "req-1" not in tracker._request_id_index

    @pytest.mark.asyncio
    async def test_get_and_clear_stale(self):
        """Test get_and_clear_stale returns and removes stale reservations."""
        tracker = ReservationTracker(
            max_reservations=100,
            max_reservation_age=2,
            stale_cleanup_interval=3600,  # Don't auto-cleanup
        )

        # Add reservations and manually set created_at for some
        await tracker.store("req-old-1", "bucket-a", "res-old-1", 50)
        await tracker.store("req-old-2", "bucket-a", "res-old-2", 75)
        await tracker.store("req-new", "bucket-a", "res-new", 100)

        # Make first two reservations stale
        old_time = time.time() - 3  # 3 seconds ago
        tracker._reservation_contexts[("req-old-1", "bucket-a")].created_at = old_time
        tracker._reservation_contexts[("req-old-2", "bucket-a")].created_at = old_time

        # Rebuild heap after modifying timestamps (required for heap-based cleanup)
        tracker._rebuild_time_heap()

        assert tracker.reservation_count == 3

        # Get stale reservations with cutoff 2 seconds ago
        cutoff = time.time() - 2
        stale_contexts = await tracker.get_and_clear_stale(cutoff)

        # Should return the 2 stale reservations
        assert len(stale_contexts) == 2
        reservation_ids = {ctx.reservation_id for ctx in stale_contexts}
        assert reservation_ids == {"res-old-1", "res-old-2"}

        # Only fresh reservation should remain
        assert tracker.reservation_count == 1
        ctx = await tracker.get("req-new", "bucket-a")
        assert ctx is not None
        assert ctx.reservation_id == "res-new"

        # Secondary index should be cleaned up for removed requests
        assert "req-old-1" not in tracker._request_id_index
        assert "req-old-2" not in tracker._request_id_index
        assert "req-new" in tracker._request_id_index

    @pytest.mark.asyncio
    async def test_get_and_clear_stale_none_stale(self, tracker):
        """Test get_and_clear_stale returns empty list when no stale reservations."""
        await tracker.store("req-1", "bucket-a", "res-1", 50)
        await tracker.store("req-2", "bucket-a", "res-2", 75)

        # Use cutoff in the past - no reservations should be stale
        cutoff = time.time() - 1000  # 1000 seconds ago
        stale_contexts = await tracker.get_and_clear_stale(cutoff)

        assert stale_contexts == []
        assert tracker.reservation_count == 2

    @pytest.mark.asyncio
    async def test_get_and_clear_stale_updates_secondary_index(self):
        """Test get_and_clear_stale properly updates secondary index for requests with multiple buckets."""
        tracker = ReservationTracker(
            max_reservations=100,
            max_reservation_age=2,
            stale_cleanup_interval=3600,
        )

        # Same request, multiple buckets, one stale
        await tracker.store("req-1", "bucket-a", "res-1a", 50)
        await tracker.store("req-1", "bucket-b", "res-1b", 75)

        # Make only one stale
        old_time = time.time() - 3
        tracker._reservation_contexts[("req-1", "bucket-a")].created_at = old_time

        # Rebuild heap after modifying timestamps (required for heap-based cleanup)
        tracker._rebuild_time_heap()

        cutoff = time.time() - 2
        stale_contexts = await tracker.get_and_clear_stale(cutoff)

        assert len(stale_contexts) == 1
        assert stale_contexts[0].reservation_id == "res-1a"

        # Secondary index should still have req-1 (for bucket-b)
        assert "req-1" in tracker._request_id_index
        assert tracker.reservation_count == 1

        # Fresh reservation should still be accessible
        ctx = await tracker.get("req-1", "bucket-b")
        assert ctx is not None
        assert ctx.reservation_id == "res-1b"
