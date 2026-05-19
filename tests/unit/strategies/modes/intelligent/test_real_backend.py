"""IntelligentModeStrategy exercised against a real MemoryBackend.

Every other intelligent-mode test uses a Mock() backend hardcoded to
return ``(True, "res-123")``, so it can never observe a real rejection,
exhaustion, or refund. These tests wire the strategy to a real
MemoryBackend + real StateManager so the reservation logic - the exact
seam where the audit's concurrency bugs live - is genuinely exercised.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from adaptive_rate_limiter.backends.memory import MemoryBackend
from adaptive_rate_limiter.scheduler.config import RateLimiterConfig, SchedulerMode
from adaptive_rate_limiter.scheduler.state import StateManager
from adaptive_rate_limiter.strategies.modes.intelligent import IntelligentModeStrategy
from adaptive_rate_limiter.types.rate_limit import RateLimitBucket
from adaptive_rate_limiter.types.request import RequestMetadata

BUCKET_ID = "bucket-real"


@pytest.fixture(autouse=True)
def _frozen_wall_clock():
    """Freeze ``time.time()`` for every test in this module.

    MemoryBackend runs a token-bucket refill keyed on wall-clock elapsed
    time (``tpm_limit / 60`` tokens per second — ~167/s here). These tests
    assert exact post-reservation token/request counts, so even the few
    milliseconds between seeding state and reserving capacity would refill
    a token or two and flake the assertions (``assert 901 == 900``).
    Freezing the clock makes elapsed time exactly zero, so the refill
    contributes nothing and the decrements are exact.
    """
    with patch("time.time", return_value=1_700_000_000.0):
        yield


@pytest.fixture
def backend():
    """A real in-memory backend."""
    return MemoryBackend(namespace="test-real", key_ttl=3600)


@pytest.fixture
def state_manager(backend):
    """A real StateManager over the real backend."""
    return StateManager(backend=backend)


@pytest.fixture
def real_provider():
    """Provider stub supplying real bucket metadata.

    Only the backend is the system under test here; the provider just
    advertises one bucket with concrete limits.
    """
    provider = Mock()
    provider.get_bucket_for_model = AsyncMock(return_value=BUCKET_ID)
    provider.discover_limits = AsyncMock(
        return_value={
            BUCKET_ID: RateLimitBucket(
                model_id="test-model",
                resource_type="chat",
                rpm_limit=100,
                tpm_limit=10_000,
            )
        }
    )
    return provider


@pytest.fixture
def strategy(state_manager, real_provider):
    """IntelligentModeStrategy wired to the real backend/state manager."""
    scheduler = Mock()
    scheduler.metrics_enabled = False
    scheduler.metrics = {}
    scheduler.circuit_breaker = None
    scheduler._circuit_breaker_always_closed = True
    scheduler.extract_response_headers = Mock(return_value={})

    config = RateLimiterConfig(mode=SchedulerMode.INTELLIGENT)

    return IntelligentModeStrategy(
        scheduler=scheduler,
        config=config,
        client=Mock(),
        provider=real_provider,
        classifier=Mock(),
        state_manager=state_manager,
    )


def _metadata(request_id="req-1", estimated_tokens=100):
    return RequestMetadata(
        request_id=request_id,
        model_id="test-model",
        resource_type="chat",
        estimated_tokens=estimated_tokens,
        priority=0,
    )


async def _seed_capacity(backend, *, remaining_requests, remaining_tokens):
    """Seed verified bucket state with concrete remaining capacity."""
    import time

    await backend.set_state(
        BUCKET_ID,
        {
            "model_id": BUCKET_ID,
            "remaining_requests": remaining_requests,
            "remaining_tokens": remaining_tokens,
            "request_limit": 100,
            "token_limit": 10_000,
            "last_updated": time.time(),
            "is_verified": True,
        },
    )


class TestRealBackendReservationSucceeds:
    """A reservation against a backend with capacity really succeeds."""

    @pytest.mark.asyncio
    async def test_reservation_decrements_real_backend_state(self, strategy, backend):
        await _seed_capacity(backend, remaining_requests=10, remaining_tokens=1_000)

        ok = await strategy._check_and_reserve_capacity_intelligent(
            _metadata(estimated_tokens=100), bucket_id=BUCKET_ID
        )

        assert ok is True
        state = await backend.get_state(BUCKET_ID)
        # Real decrement: 1 request and 100 tokens consumed.
        assert state["remaining_requests"] == 9
        assert state["remaining_tokens"] == 900

    @pytest.mark.asyncio
    async def test_reservation_is_tracked_for_the_request(self, strategy, backend):
        await _seed_capacity(backend, remaining_requests=10, remaining_tokens=1_000)
        metadata = _metadata(request_id="req-tracked")

        ok = await strategy._check_and_reserve_capacity_intelligent(
            metadata, bucket_id=BUCKET_ID
        )

        assert ok is True
        # The strategy stored the reservation context for later release.
        ctx = await strategy._reservation_tracker.get("req-tracked", BUCKET_ID)
        assert ctx is not None


class TestRealBackendReservationRejected:
    """A reservation against an exhausted backend really fails."""

    @pytest.mark.asyncio
    async def test_token_exhaustion_rejects_reservation(self, strategy, backend):
        # Only 10 tokens left but the request needs 100.
        await _seed_capacity(backend, remaining_requests=10, remaining_tokens=10)

        ok = await strategy._check_and_reserve_capacity_intelligent(
            _metadata(estimated_tokens=100), bucket_id=BUCKET_ID
        )

        assert ok is False
        # No reservation was tracked for a rejected request.
        ctx = await strategy._reservation_tracker.get("req-1", BUCKET_ID)
        assert ctx is None

    @pytest.mark.asyncio
    async def test_request_exhaustion_rejects_reservation(self, strategy, backend):
        # No request capacity left at all.
        await _seed_capacity(backend, remaining_requests=0, remaining_tokens=10_000)

        ok = await strategy._check_and_reserve_capacity_intelligent(
            _metadata(estimated_tokens=100), bucket_id=BUCKET_ID
        )

        assert ok is False

    @pytest.mark.asyncio
    async def test_concurrent_reservations_cannot_oversubscribe(
        self, strategy, backend
    ):
        """Capacity for exactly 5 requests must admit exactly 5 of 20
        concurrent reservation attempts against the real backend."""
        import asyncio

        # Disable the safety-margin time buffer so capacity maps 1:1 to
        # requests and the anti-oversubscription assertion is exact.
        strategy._safety_margin = 1.0
        await _seed_capacity(backend, remaining_requests=5, remaining_tokens=10_000)

        results = await asyncio.gather(
            *(
                strategy._check_and_reserve_capacity_intelligent(
                    _metadata(request_id=f"req-{i}", estimated_tokens=100),
                    bucket_id=BUCKET_ID,
                )
                for i in range(20)
            )
        )

        # The backend's atomic check-and-reserve must not over-admit.
        assert sum(1 for r in results if r) == 5
        state = await backend.get_state(BUCKET_ID)
        assert state["remaining_requests"] == 0


class TestRealBackendStreamingRefund:
    """A streaming refund really returns tokens to the backend."""

    @pytest.mark.asyncio
    async def test_streaming_refund_returns_unused_tokens(self, strategy, backend):
        await _seed_capacity(backend, remaining_requests=10, remaining_tokens=1_000)

        ok = await strategy._check_and_reserve_capacity_intelligent(
            _metadata(estimated_tokens=500), bucket_id=BUCKET_ID
        )
        assert ok is True
        assert (await backend.get_state(BUCKET_ID))["remaining_tokens"] == 500

        ctx = await strategy._reservation_tracker.get("req-1", BUCKET_ID)
        assert ctx is not None

        # Stream completes having used only 100 of the 500 reserved tokens.
        released = await backend.release_streaming_reservation(
            key=BUCKET_ID,
            reservation_id=ctx.reservation_id,
            reserved_tokens=500,
            actual_tokens=100,
        )

        assert released is True
        # 500 + (500 - 100) refund = 900.
        assert (await backend.get_state(BUCKET_ID))["remaining_tokens"] == 900
