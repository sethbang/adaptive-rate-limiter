"""Tests for the public API surface (top-level and scheduler __init__)."""

import adaptive_rate_limiter
from adaptive_rate_limiter import scheduler as scheduler_pkg


class TestScheduleResultIsPublic:
    """ScheduleResult is the INTELLIGENT/ACCOUNT submit_request return type
    and must be importable from the public API.

    Regression: submit_request returned a ScheduleResult in INTELLIGENT and
    ACCOUNT modes, but the type was not exported from any public __init__,
    leaving callers unable to import the type they were handed.
    """

    def test_schedule_result_importable_from_top_level(self):
        from adaptive_rate_limiter import ScheduleResult

        assert ScheduleResult.__name__ == "ScheduleResult"
        assert "ScheduleResult" in adaptive_rate_limiter.__all__

    def test_schedule_result_importable_from_scheduler(self):
        from adaptive_rate_limiter.scheduler import ScheduleResult

        assert "ScheduleResult" in scheduler_pkg.__all__
        assert ScheduleResult is adaptive_rate_limiter.ScheduleResult

    def test_queued_request_importable_from_top_level(self):
        """QueuedRequest is reachable via ScheduleResult.request, so it must
        also be part of the public surface."""
        from adaptive_rate_limiter import QueuedRequest

        assert QueuedRequest.__name__ == "QueuedRequest"
        assert "QueuedRequest" in adaptive_rate_limiter.__all__

    def test_schedule_result_is_the_intelligent_submit_return_type(self):
        """The exported type is the same object the strategy constructs."""
        from adaptive_rate_limiter import ScheduleResult
        from adaptive_rate_limiter.types.queue import (
            ScheduleResult as InternalScheduleResult,
        )

        assert ScheduleResult is InternalScheduleResult
