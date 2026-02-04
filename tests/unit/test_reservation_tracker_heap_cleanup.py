import pytest

from adaptive_rate_limiter.reservation.tracker import ReservationTracker


@pytest.mark.asyncio
async def test_heap_stale_entries_and_compaction():
    tracker = ReservationTracker()

    # 1. Add reservations
    await tracker.store("req1", "bucket1", "res1", 10)
    await tracker.store("req2", "bucket2", "res2", 10)

    assert len(tracker._time_heap) == 2
    assert tracker.reservation_count == 2

    # 2. Clear one reservation using get_and_clear
    await tracker.get_and_clear("req1", "bucket1")

    # Verify it's gone from primary storage
    assert tracker.reservation_count == 1

    # Verify it's STILL in the heap (current behavior)
    assert len(tracker._time_heap) == 2

    # 3. Check stale_entry_ratio
    ratio = tracker.stale_entry_ratio
    assert ratio == 0.5  # 1 stale / 2 total

    # 4. Compact heap
    removed = tracker.compact_heap()
    assert removed == 1
    assert len(tracker._time_heap) == 1
    assert tracker.stale_entry_ratio == 0.0


@pytest.mark.asyncio
async def test_stale_entry_ratio_calculation():
    tracker = ReservationTracker()

    # Add 10 reservations
    for i in range(10):
        await tracker.store(f"req{i}", "bucket1", f"res{i}", 1)

    # Clear 8 of them
    for i in range(8):
        await tracker.get_and_clear(f"req{i}", "bucket1")

    # Should have 2 valid, 10 in heap (8 stale)
    assert tracker.reservation_count == 2
    assert len(tracker._time_heap) == 10

    # Check ratio
    assert tracker.stale_entry_ratio == 0.8

    # Compact
    tracker.compact_heap()
    assert len(tracker._time_heap) == 2
    assert tracker.stale_entry_ratio == 0.0
