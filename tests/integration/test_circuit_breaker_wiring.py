# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0

"""Circuit breaker wiring for RedisBackend.

``record_failure`` had no production call sites: it was declared on
``BaseBackend``, implemented on both backends, and called only by tests and by
``force_circuit_break``. ``_failure_timestamps`` therefore stayed empty forever,
``is_circuit_broken()`` was always False, and the MemoryBackend fallback at
``check_and_reserve_capacity`` was unreachable.

The consequence was the opposite of the intended design: with Redis unavailable
the backend denied all capacity indefinitely instead of degrading gracefully.
"""

import asyncio

import pytest

from adaptive_rate_limiter.backends.redis import ModelLimits, RedisBackend

# Nothing listens here, so every connection attempt fails immediately with
# ECONNREFUSED. That exercises the real connection paths without a container.
DEAD_URL = "redis://127.0.0.1:1"

LIMITS = {"rpm_limit": 100, "tpm_limit": 100_000}


@pytest.fixture
async def dead_backend():
    b = RedisBackend(redis_url=DEAD_URL, namespace="cb-wiring")
    yield b
    await b.cleanup()


class TestFailuresAreRecorded:
    """The regression: real failures must reach the breaker."""

    @pytest.mark.asyncio
    async def test_connection_failure_increments_the_counter(self, dead_backend):
        assert await dead_backend.get_failure_count(30) == 0

        await dead_backend.check_and_reserve_capacity("m", 1, 100, bucket_limits=LIMITS)

        assert await dead_backend.get_failure_count(30) > 0, (
            "a failed Redis call must be recorded, or the breaker can never trip"
        )

    @pytest.mark.asyncio
    async def test_get_model_limits_failure_is_recorded(self, dead_backend):
        """This path swallows its exception, so it must report the failure itself.

        It also runs *before* the circuit-breaker gate, so it is the call that
        burns the most time against a dead Redis.
        """
        await dead_backend.get_model_limits("m")

        assert await dead_backend.get_failure_count(30) > 0

    @pytest.mark.asyncio
    async def test_update_rate_limits_failure_is_recorded(self, dead_backend):
        await dead_backend.update_rate_limits(
            "m",
            {"x-ratelimit-remaining-requests": "5"},
            request_id="r1",
            status_code=200,
        )

        assert await dead_backend.get_failure_count(30) > 0


class TestBreakerOpensAndDegrades:
    """Sustained failure must degrade to the fallback, not deny everything."""

    @pytest.mark.asyncio
    async def test_breaker_opens_under_sustained_failure(self, dead_backend):
        for _ in range(12):
            await dead_backend.check_and_reserve_capacity(
                "m", 1, 100, bucket_limits=LIMITS
            )
            if await dead_backend.is_circuit_broken():
                break

        assert await dead_backend.is_circuit_broken() is True

    @pytest.mark.asyncio
    async def test_fallback_engages_and_eventually_grants(self, dead_backend):
        """The point of the breaker: degrade gracefully instead of denying.

        The fallback applies 1/20th of the real limits through a token bucket
        that cold-starts near empty, so a large request waits for refill rather
        than being granted immediately. What must not happen - and did happen
        before the breaker was wired - is denial that never ends.
        """
        for _ in range(20):
            await dead_backend.check_and_reserve_capacity(
                "m", 1, 100, bucket_limits=LIMITS
            )
            if dead_backend.is_in_fallback_mode():
                break
        assert dead_backend.is_in_fallback_mode() is True

        deadline = asyncio.get_running_loop().time() + 5.0
        granted = False
        while asyncio.get_running_loop().time() < deadline:
            ok, _ = await dead_backend.check_and_reserve_capacity(
                "m", 1, 100, bucket_limits=LIMITS
            )
            if ok:
                granted = True
                break
            await asyncio.sleep(0.05)

        assert granted is True, (
            "with the breaker open the fallback must serve requests once its "
            "conservative bucket refills, not deny capacity indefinitely"
        )

    @pytest.mark.asyncio
    async def test_fallback_grants_a_small_request_immediately(self, dead_backend):
        """A request within the conservative cold-start budget is served at once."""
        for _ in range(20):
            ok, _ = await dead_backend.check_and_reserve_capacity(
                "m", 1, 1, bucket_limits=LIMITS
            )
            if dead_backend.is_in_fallback_mode() and ok:
                break

        assert dead_backend.is_in_fallback_mode() is True
        assert ok is True


class TestBreakerRecovers:
    """The breaker is a rolling window, so it must close on its own."""

    @pytest.mark.asyncio
    async def test_window_ages_out(self, dead_backend):
        for _ in range(20):
            await dead_backend.check_and_reserve_capacity(
                "m", 1, 100, bucket_limits=LIMITS
            )
            if await dead_backend.is_circuit_broken():
                break
        assert await dead_backend.is_circuit_broken() is True

        # Age the rolling window rather than sleeping 30s - same arithmetic.
        async with dead_backend._failure_lock:
            dead_backend._failure_timestamps = [
                ts - 31 for ts in dead_backend._failure_timestamps
            ]

        assert await dead_backend.is_circuit_broken() is False

    @pytest.mark.asyncio
    async def test_recovery_is_reachable_with_model_limits_cached(self, dead_backend):
        """The path that could have stranded the fallback forever.

        ``get_model_limits`` returns from its in-memory cache without calling
        ``_ensure_connected``, and the fallback block returns before reaching it
        too - so while the circuit is open nothing runs the teardown. Recovery
        therefore depends on the window closing on its own.
        """
        async with dead_backend._model_limits_lock:
            dead_backend._model_limits["m"] = ModelLimits(rpm=100, tpm=100_000)

        for _ in range(20):
            await dead_backend.check_and_reserve_capacity(
                "m", 1, 100, bucket_limits=LIMITS
            )
            if dead_backend.is_in_fallback_mode():
                break
        assert dead_backend.is_in_fallback_mode() is True

        async with dead_backend._failure_lock:
            dead_backend._failure_timestamps = [
                ts - 31 for ts in dead_backend._failure_timestamps
            ]

        assert await dead_backend.is_circuit_broken() is False, (
            "the circuit must close on its own, or the fallback is permanent"
        )


class TestNoFailuresWhenHealthy:
    """A healthy backend must not accumulate phantom failures."""

    @pytest.mark.asyncio
    async def test_successful_calls_record_nothing(self):
        import fakeredis.aioredis as fr

        client = fr.FakeRedis(decode_responses=True)
        b = RedisBackend(redis_url="redis://localhost:6379", namespace="cb-ok")
        b._redis = client
        b._connected = True
        await b._load_scripts()

        for _ in range(5):
            ok, _ = await b.check_and_reserve_capacity(
                "m", 1, 100, bucket_limits=LIMITS
            )
            assert ok

        assert await b.get_failure_count(30) == 0
        assert await b.is_circuit_broken() is False
        assert b.is_in_fallback_mode() is False
        await client.aclose()


class TestSuccessfulConnectDoesNotEraseHistory:
    """Clearing failures is tied to fallback teardown, not to every connect.

    The cluster ping loop retries up to CLUSTER_PING_ATTEMPTS times, so clearing
    on every successful connect would erase the failures recorded by that very
    call. A flapping cluster - one that always connects eventually, after tens
    of seconds of retries - could then never trip the breaker no matter how
    degraded it became.
    """

    @pytest.fixture
    async def live_backend(self):
        """A backend whose connects always succeed, against fakeredis."""
        from unittest.mock import MagicMock, patch

        import fakeredis.aioredis as fr

        client = fr.FakeRedis(decode_responses=True)
        # _ensure_connected builds the client via Redis(connection_pool=...) when
        # a pool exists and Redis.from_url otherwise, so patch the name itself.
        fake_cls = MagicMock(return_value=client)
        fake_cls.from_url = MagicMock(return_value=client)
        with patch("adaptive_rate_limiter.backends.redis.Redis", fake_cls):
            b = RedisBackend(redis_url="redis://localhost:6379", namespace="cb-keep")
            await b._ensure_connected()
            yield b
        await client.aclose()

    @pytest.mark.asyncio
    async def test_failures_survive_a_successful_connect(self, live_backend):
        for _ in range(5):
            await live_backend.record_failure("connection", "flap")
        assert await live_backend.get_failure_count() == 5

        live_backend._connected = False
        await live_backend._ensure_connected()

        assert await live_backend.get_failure_count() == 5, (
            "a successful connect must not erase the failures that a retrying "
            "connect just recorded"
        )

    @pytest.mark.asyncio
    async def test_fallback_teardown_does_clear(self, live_backend):
        from adaptive_rate_limiter.backends.memory import MemoryBackend

        for _ in range(5):
            await live_backend.record_failure("connection", "down")
        live_backend._fallback_backend = MemoryBackend(namespace=live_backend.namespace)
        live_backend._fallback_start_time = 0.0

        live_backend._connected = False
        await live_backend._ensure_connected()

        assert live_backend.is_in_fallback_mode() is False
        assert await live_backend.get_failure_count() == 0, (
            "tearing down the fallback must clear the history that opened it"
        )


class TestForcedBreakHonoursConfiguration:
    """``force_circuit_break`` must not depend on a hardcoded failure count.

    It used to append exactly 25 synthetic failures, which silently stopped
    working the moment the threshold became configurable, and capped the break
    at ``failure_window_seconds`` regardless of the duration asked for.
    """

    @pytest.mark.asyncio
    async def test_forced_break_opens_above_a_raised_threshold(self):
        b = RedisBackend(
            redis_url=DEAD_URL, namespace="cb-forced", failure_threshold=30
        )
        try:
            await b.force_circuit_break(60)
            assert await b.is_circuit_broken(), (
                "a forced break must open the circuit at any configured threshold"
            )
        finally:
            await b.cleanup()

    @pytest.mark.asyncio
    async def test_forced_break_outlives_the_failure_window(self):
        """A break longer than the rolling window must still be in force."""
        b = RedisBackend(
            redis_url=DEAD_URL, namespace="cb-forced-win", failure_window_seconds=0.05
        )
        try:
            await b.force_circuit_break(30)
            await asyncio.sleep(0.2)  # longer than the window, shorter than the break
            assert await b.is_circuit_broken(), (
                "the forced break must not age out of the failure window"
            )
        finally:
            await b.cleanup()

    @pytest.mark.asyncio
    async def test_clear_failures_lifts_a_forced_break(self):
        b = RedisBackend(redis_url=DEAD_URL, namespace="cb-forced-clear")
        try:
            await b.force_circuit_break(60)
            assert await b.is_circuit_broken()
            await b.clear_failures()
            assert not await b.is_circuit_broken(), "clear_failures must lift the break"
        finally:
            await b.cleanup()


class TestOnlyConnectionFailuresFeedTheBreaker:
    """A healthy Redis must not be forced into fallback by bad cached data.

    ``get_model_limits`` swallows everything and returns defaults. It only
    writes its in-memory cache on the hit path, so a corrupt cache entry is
    re-read on every call -- recording a failure each time would drive a
    perfectly healthy backend to the threshold within seconds.
    """

    @pytest.mark.asyncio
    async def test_corrupt_cache_entry_does_not_record_failures(self):
        import fakeredis.aioredis as fakeredis

        client = fakeredis.FakeRedis(decode_responses=True)
        b = RedisBackend(redis_url="redis://localhost:6379", namespace="cb-corrupt")
        b._redis = client
        b._connected = True
        try:
            await client.set(b.model_limits_key, "{not valid json")

            for _ in range(30):  # comfortably past DEFAULT_FAILURE_THRESHOLD
                rpm, tpm = await b.get_model_limits("m")

            assert (rpm, tpm) == (b.DEFAULT_RPM_LIMIT, b.DEFAULT_TPM_LIMIT)
            assert await b.get_failure_count() == 0, (
                "a decode error is not a connection failure and must not trip "
                "the breaker on a healthy Redis"
            )
            assert not await b.is_circuit_broken()
        finally:
            await client.aclose()
