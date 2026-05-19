"""
Unit tests for StreamingInFlightEntry.

Tests cover:
- StreamingInFlightEntry dataclass initialization
"""

from __future__ import annotations

import gc
import time
import weakref
from unittest.mock import Mock

from adaptive_rate_limiter.streaming.tracker import (
    StreamingInFlightEntry,
)


class TestStreamingInFlightEntry:
    """Tests for StreamingInFlightEntry dataclass."""

    def test_init_with_all_fields(self) -> None:
        """Verify initialization with all required fields."""
        wrapper = Mock()
        now = time.monotonic()

        entry = StreamingInFlightEntry(
            reservation_id="res-1",
            bucket_id="bucket-1",
            reserved_tokens=1000,
            started_at=now,
            last_activity_at=now,
            wrapper_ref=weakref.ref(wrapper),
        )

        assert entry.reservation_id == "res-1"
        assert entry.bucket_id == "bucket-1"
        assert entry.reserved_tokens == 1000
        assert entry.started_at == now
        assert entry.last_activity_at == now
        assert entry.wrapper_ref() is wrapper

    def test_wrapper_ref_is_weak_reference(self) -> None:
        """Verify wrapper_ref is a weak reference."""
        wrapper = Mock()
        now = time.monotonic()

        entry = StreamingInFlightEntry(
            reservation_id="res-1",
            bucket_id="bucket-1",
            reserved_tokens=1000,
            started_at=now,
            last_activity_at=now,
            wrapper_ref=weakref.ref(wrapper),
        )

        assert entry.wrapper_ref() is wrapper

        # Delete wrapper and force GC
        del wrapper
        gc.collect()

        # Weak reference should now return None
        assert entry.wrapper_ref() is None
