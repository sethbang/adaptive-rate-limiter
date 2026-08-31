# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0

"""Per-dimension rate-limit header sync.

Regression tests for a family of defects that all shared one cause: an absent
header was rendered as a fabricated default before it reached the Lua scripts,
so the scripts could not tell "the server did not say" from a real value.

The reporter's case is Venice's ``gemini-3-5-flash``, which meters requests but
not tokens and therefore sends three of the six headers.
"""

import asyncio
import os as _os
import time

import fakeredis.aioredis as fakeredis
import pytest

from adaptive_rate_limiter.backends.memory import MemoryBackend
from adaptive_rate_limiter.strategies.modes.intelligent import IntelligentModeStrategy

# gemini-3-5-flash, verbatim from a live response. The reset is epoch
# MILLISECONDS, which also exercises the reset-header coercion.
GEMINI_HEADERS = {
    "x-ratelimit-limit-requests": "1000",
    "x-ratelimit-remaining-requests": "999",
    "x-ratelimit-reset-requests": "1787812992834",
}

# The mirror shape: a provider that meters tokens but not requests.
TOKEN_ONLY_HEADERS = {
    "x-ratelimit-limit-tokens": "3000000",
    "x-ratelimit-remaining-tokens": "2900000",
    "x-ratelimit-reset-tokens": "30",
}


@pytest.fixture
async def backend():
    """A real RedisBackend over fakeredis."""
    from unittest.mock import patch

    from adaptive_rate_limiter.backends.redis import RedisBackend

    redis_url = _os.environ.get("REDIS_URL")
    if redis_url:
        import redis.asyncio as aioredis

        client = aioredis.Redis.from_url(redis_url, decode_responses=True)
        await client.flushdb()
    else:
        client = fakeredis.FakeRedis(decode_responses=True)

    with patch(
        "adaptive_rate_limiter.backends.redis.Redis.from_url", return_value=client
    ):
        b = RedisBackend(
            redis_url=redis_url or "redis://localhost:6379",
            namespace="partial",
            account_id="acct",
        )
        b._redis = client
        b._connected = True
        await b._load_scripts()
        yield b

    if redis_url:
        await client.flushdb()
    await client.aclose()


async def _reserve_and_update(backend, headers, *, status_code=200, tokens=1000):
    ok, request_id = await backend.check_and_reserve_capacity(
        "m", 1, tokens, bucket_limits={"rpm_limit": 1000, "tpm_limit": 3_000_000}
    )
    assert ok
    rc = await backend.update_rate_limits(
        "m", headers, request_id=request_id, status_code=status_code
    )
    state = await backend._redis.hgetall(backend._get_state_key("m"))
    return rc, state


def _full(now):
    return {
        "x-ratelimit-limit-requests": "1000",
        "x-ratelimit-remaining-requests": "999",
        "x-ratelimit-reset-requests": str(now + 30),
        "x-ratelimit-limit-tokens": "3000000",
        "x-ratelimit-remaining-tokens": "2900000",
        "x-ratelimit-reset-tokens": "30",
    }


class TestRequestOnlyProvider:
    """A provider that meters requests but not tokens must still sync."""

    @pytest.mark.asyncio
    async def test_gemini_headers_verify_the_request_dimension(self, backend):
        """The reporter's exact case: 3 of 6 headers must sync, not be discarded."""
        rc, state = await _reserve_and_update(backend, GEMINI_HEADERS)

        assert rc == 1
        assert int(state["rem_req"]) == 999, "server's remaining must be adopted"
        assert int(state["lim_req"]) == 1000
        assert int(state["vrf_req"]) == 1, "request dimension must verify"

    @pytest.mark.asyncio
    async def test_absent_token_dimension_stays_unverified(self, backend):
        """An unreported dimension must not be marked verified."""
        _, state = await _reserve_and_update(backend, GEMINI_HEADERS)

        assert int(state["vrf_tok"]) == 0, (
            "token dimension was never reported, so it cannot be verified"
        )

    @pytest.mark.asyncio
    async def test_absent_token_headers_do_not_zero_remaining(self, backend):
        """Absent != zero.

        Reading absent as 0 made the bucket believe it had no token capacity.
        """
        _, state = await _reserve_and_update(backend, GEMINI_HEADERS)

        assert int(state["rem_tok"]) > 0
        assert int(state["lim_tok"]) != 500_000, (
            "must not stamp the DEFAULT_TPM_LIMIT fallback as observed"
        )

    @pytest.mark.asyncio
    async def test_absent_token_reset_does_not_expire_the_window(self, backend):
        """The oscillation regression.

        An absent token reset used to be sent as 0, which the script read as
        "resets now": it adopted an already-expired window, marked it verified,
        and the next reservation rotated the bucket to a fabricated full limit.
        """
        _, state = await _reserve_and_update(backend, GEMINI_HEADERS)
        now = int(time.time())

        assert int(state["rst_tok"]) > now, "token window must not be born expired"

        before = int(state["gen_tok"])
        await backend.check_and_reserve_capacity(
            "m", 1, 1000, bucket_limits={"rpm_limit": 1000, "tpm_limit": 3_000_000}
        )
        after = await backend._redis.hgetall(backend._get_state_key("m"))
        assert int(after["gen_tok"]) == before, "must not rotate to a fabricated limit"


class TestPartialDimensionsAreIndependent:
    """Neither dimension may reject the other."""

    @pytest.mark.asyncio
    async def test_token_only_headers_are_applied(self, backend):
        """The mirror case: absent request headers used to reject everything."""
        now = int(time.time())
        headers = {k: v for k, v in _full(now).items() if "requests" not in k}

        rc, state = await _reserve_and_update(backend, headers)

        assert rc == 1, "an absent request dimension must not reject the update"
        assert int(state["rem_tok"]) == 2_900_000
        assert int(state["vrf_tok"]) == 1
        assert int(state["vrf_req"]) == 0

    @pytest.mark.asyncio
    async def test_incomplete_dimension_does_not_fabricate_a_limit(self, backend):
        """Missing limit-requests must skip the dimension, not invent RPM=20."""
        now = int(time.time())
        headers = {
            k: v for k, v in _full(now).items() if k != "x-ratelimit-limit-requests"
        }

        _, state = await _reserve_and_update(backend, headers)

        assert int(state["lim_req"]) != 20, (
            "DEFAULT_RPM_LIMIT must never be stored as an observed value"
        )
        assert int(state["vrf_req"]) == 0
        assert int(state["vrf_tok"]) == 1, "the complete dimension still lands"

    @pytest.mark.asyncio
    async def test_no_usable_dimension_rejects(self, backend):
        """With nothing complete, the caller must be told to release."""
        rc, _ = await _reserve_and_update(
            backend, {"x-ratelimit-remaining-requests": "999"}
        )
        assert rc == 0


class TestRealZeroLimit:
    """A reported limit of 0 is a real value, distinct from absence."""

    @pytest.mark.asyncio
    async def test_zero_token_limit_is_applied_not_rejected(self, backend):
        now = int(time.time())
        headers = dict(
            _full(now),
            **{
                "x-ratelimit-limit-tokens": "0",
                "x-ratelimit-remaining-tokens": "0",
            },
        )

        rc, state = await _reserve_and_update(backend, headers)

        assert rc == 1
        assert int(state["lim_tok"]) == 0, "a real 0 must be stored as reported"

    @pytest.mark.asyncio
    async def test_zero_token_limit_does_not_discard_the_request_side(self, backend):
        """The original report: valid request fields died with the token field."""
        now = int(time.time())
        headers = dict(
            _full(now),
            **{
                "x-ratelimit-limit-tokens": "0",
                "x-ratelimit-remaining-tokens": "0",
            },
        )

        _, state = await _reserve_and_update(backend, headers)

        assert int(state["rem_req"]) == 999
        assert int(state["vrf_req"]) == 1

    @pytest.mark.asyncio
    async def test_429_keeps_the_request_side(self, backend):
        """A 429 states scarcity; a token field must not discard that."""
        now = int(time.time())
        headers = dict(
            _full(now),
            **{
                "x-ratelimit-remaining-requests": "7",
                "x-ratelimit-limit-tokens": "0",
                "x-ratelimit-remaining-tokens": "0",
            },
        )

        rc, state = await _reserve_and_update(backend, headers, status_code=429)

        assert rc == 1
        assert int(state["rem_req"]) == 7, "the server's scarcity signal must survive"
        assert int(state["vrf_req"]) == 1


class TestHeaderAvailabilityGate:
    """The mode strategy gates on a complete dimension, not on all six."""

    @pytest.mark.parametrize(
        ("label", "headers", "expected"),
        [
            ("request-only (gemini)", GEMINI_HEADERS, "full"),
            (
                "token-only",
                {
                    "x-ratelimit-limit-tokens": "3000000",
                    "x-ratelimit-remaining-tokens": "2900000",
                    "x-ratelimit-reset-tokens": "30",
                },
                "full",
            ),
            (
                "no complete dimension",
                {
                    "x-ratelimit-remaining-requests": "99",
                    "x-ratelimit-limit-requests": "100",
                },
                "partial",
            ),
            ("nothing", {}, "none"),
        ],
    )
    def test_assessment_is_per_dimension(self, label, headers, expected):
        assert (
            IntelligentModeStrategy._assess_header_availability(None, headers)
            == expected
        ), label


class TestMemoryBackendParity:
    """MemoryBackend shares the contract, so it shares the fix."""

    @pytest.mark.asyncio
    async def test_absent_token_headers_do_not_zero_remaining(self):
        b = MemoryBackend()
        await b.check_and_reserve_capacity(
            "m", 1, 1000, bucket_limits={"rpm_limit": 1000, "tpm_limit": 3_000_000}
        )
        before = b._states["m"][0]["remaining_tokens"]

        await b.update_rate_limits("m", GEMINI_HEADERS)

        assert b._states["m"][0]["remaining_tokens"] == before, (
            "an unreported dimension must be carried forward, not zeroed"
        )

    @pytest.mark.asyncio
    async def test_no_headers_is_not_verified(self):
        b = MemoryBackend()
        await b.check_and_reserve_capacity(
            "m", 1, 1000, bucket_limits={"rpm_limit": 1000, "tpm_limit": 3_000_000}
        )

        await b.update_rate_limits("m", {})

        assert b._states["m"][0]["is_verified"] is False

    @pytest.mark.asyncio
    async def test_partial_sync_advances_last_updated(self):
        """An unreported dimension must not pin the refill clock.

        ``used_local_*`` means "we kept our local value because the server's
        was higher", and it holds last_updated back so the refill in
        check_and_reserve does not double-credit. Carrying an unreported
        dimension forward makes that comparison trivially true, so a
        request-only provider would freeze last_updated on every sync and the
        refill -- rate * (now - last_updated) -- would grow without bound.
        """
        b = MemoryBackend()
        await b.check_and_reserve_capacity(
            "m", 1, 1000, bucket_limits={"rpm_limit": 1000, "tpm_limit": 3_000_000}
        )
        # The reported dimension must NOT itself trigger the local-wins rule,
        # or it pins the clock for a legitimate reason and the unreported
        # dimension's behaviour is invisible. A cold-start bucket holds ~1
        # request, so report a value at or below that.
        low_request_headers = {
            "x-ratelimit-limit-requests": "1000",
            "x-ratelimit-remaining-requests": "1",
            "x-ratelimit-reset-requests": "1787812992834",
        }
        await b.update_rate_limits("m", low_request_headers)
        first = b._states["m"][0]["last_updated"]

        await asyncio.sleep(0.05)
        await b.update_rate_limits("m", low_request_headers)
        second = b._states["m"][0]["last_updated"]

        assert second > first, (
            "the unreported token dimension must not pin the refill clock"
        )


class TestMissingStatusCodeIsDiagnosed:
    """Omitting status_code silently discarded headers; it must now warn."""

    @pytest.mark.asyncio
    async def test_warns_when_headers_present_but_status_code_none(
        self, backend, caplog
    ):
        ok, request_id = await backend.check_and_reserve_capacity(
            "m", 1, 1000, bucket_limits={"rpm_limit": 1000, "tpm_limit": 3_000_000}
        )
        assert ok

        with caplog.at_level("WARNING"):
            await backend.update_rate_limits("m", GEMINI_HEADERS, request_id=request_id)

        assert any("status_code=None" in r.message for r in caplog.records)


class TestHasHeadersIsDimensionAware:
    """A token-only provider reports headers too."""

    def test_5xx_with_token_only_headers_syncs(self, backend):
        """The 5xx branch must not route valid token data to release-only."""
        parsed = backend._parse_rate_limit_headers(TOKEN_ONLY_HEADERS)
        has_headers = (
            parsed.get("rpm_remaining") is not None
            or parsed.get("tpm_remaining") is not None
        )

        assert has_headers is True
        assert (
            backend._select_script(503, has_headers) == "distributed_update_rate_limits"
        )

    @pytest.mark.asyncio
    async def test_warns_for_token_only_caller_omitting_status_code(
        self, backend, caplog
    ):
        """The diagnostic must not itself be silent for half the providers."""
        ok, request_id = await backend.check_and_reserve_capacity(
            "m", 1, 1000, bucket_limits={"rpm_limit": 1000, "tpm_limit": 3_000_000}
        )
        assert ok

        with caplog.at_level("WARNING"):
            await backend.update_rate_limits(
                "m", TOKEN_ONLY_HEADERS, request_id=request_id
            )

        assert any("status_code=None" in r.message for r in caplog.records)


class TestRequestOnlyAcrossWindowRotation:
    """A request-only bucket must survive its own window rotating.

    Fixing the availability gate made this path reachable in production for the
    first time: before, a request-only provider never wrote state at all, so
    nothing here ever ran. The reporter's live run covered six responses inside
    a single window, which cannot show what rotation does. These tests do.
    """

    async def _sync_then_expire_request_window(self, backend):
        """Sync gemini's headers, then force the request window to roll over."""
        rc, state = await _reserve_and_update(backend, GEMINI_HEADERS)
        assert rc == 1
        assert int(state["vrf_req"]) == 1
        await backend._redis.hset(
            backend._get_state_key("m"), "rst_req", str(int(time.time()) - 1)
        )
        return state

    @pytest.mark.asyncio
    async def test_request_rotation_leaves_the_token_dimension_alone(self, backend):
        """Rotating the request window must not touch token state.

        The two dimensions rotate on independent clocks. If rotation were joint,
        a request-only provider would have its fabricated token window reset on
        every request-window roll, and the reporter's steady ``rem_tok`` would
        start moving under sustained traffic.
        """
        before = await self._sync_then_expire_request_window(backend)

        await backend.check_and_reserve_capacity(
            "m", 1, 1000, bucket_limits={"rpm_limit": 1000, "tpm_limit": 3_000_000}
        )
        after = await backend._redis.hgetall(backend._get_state_key("m"))

        assert int(after["gen_req"]) == int(before["gen_req"]) + 1, "request must roll"
        assert int(after["gen_tok"]) == int(before["gen_tok"]), "token must not roll"
        assert after["rst_tok"] == before["rst_tok"], "token window must not move"
        assert after["lim_tok"] == before["lim_tok"], "token limit must not move"

    @pytest.mark.asyncio
    async def test_request_side_recovers_its_observed_limit_after_rotation(
        self, backend
    ):
        """After rotation the next response must re-adopt the server's limit.

        Rotation deliberately re-fabricates from the caller's fallback and drops
        ``vrf_req``. That is only safe if the next observed header restores the
        real limit -- otherwise a request-only provider would decay toward the
        conservative default one window at a time.
        """
        await self._sync_then_expire_request_window(backend)

        await backend.check_and_reserve_capacity(
            "m", 1, 1000, bucket_limits={"rpm_limit": 1000, "tpm_limit": 3_000_000}
        )
        rotated = await backend._redis.hgetall(backend._get_state_key("m"))
        assert int(rotated["vrf_req"]) == 0, "rotation re-fabricates the window"

        _, resynced = await _reserve_and_update(backend, GEMINI_HEADERS)

        assert int(resynced["lim_req"]) == 1000, "must re-adopt the reported limit"
        # 999 reported, minus the post-rotation reservation above that is still
        # in flight. The header is authoritative for what the server has seen;
        # capacity this process already holds has to come off on top of it.
        assert int(resynced["rem_req"]) == 998, "reported value net of in-flight"
        assert int(resynced["vrf_req"]) == 1, "observed again after rotation"
        assert int(resynced["vrf_tok"]) == 0, "token dimension still never observed"

    @pytest.mark.asyncio
    async def test_no_drift_across_repeated_rotations(self, backend):
        """Sustained request-only traffic must not drift over many windows.

        This is the shape six live responses could not reach: the reporter saw
        no oscillation, but only inside one window. Three forced rotations pin
        that the request limit returns to the reported value every time and the
        untouched token dimension never moves.
        """
        _, first = await _reserve_and_update(backend, GEMINI_HEADERS)
        token_window = (first["rst_tok"], first["lim_tok"], first["rem_tok"])

        for cycle in range(3):
            await backend._redis.hset(
                backend._get_state_key("m"), "rst_req", str(int(time.time()) - 1)
            )
            await backend.check_and_reserve_capacity(
                "m", 1, 1000, bucket_limits={"rpm_limit": 1000, "tpm_limit": 3_000_000}
            )
            _, state = await _reserve_and_update(backend, GEMINI_HEADERS)

            assert int(state["lim_req"]) == 1000, f"limit drifted on cycle {cycle}"
            assert int(state["vrf_req"]) == 1, f"lost verification on cycle {cycle}"
            assert int(state["vrf_tok"]) == 0, f"fabricated a token window ({cycle})"
            assert (
                state["rst_tok"],
                state["lim_tok"],
                state["rem_tok"],
            ) == token_window, f"token dimension moved on cycle {cycle}"


class TestIncompleteDimensionOnErrorReleases:
    """A 5xx carrying a partial dimension must free its reservation.

    Routing on ``remaining`` alone sent such a response to the update script,
    which rejects both dimensions and returns 0 *before* the pending
    decrement -- so the reservation leaked until orphan recovery, where the
    release-only script would have freed it immediately.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "label,headers",
        [
            ("token remaining only", {"x-ratelimit-remaining-tokens": "2900000"}),
            ("request remaining only", {"x-ratelimit-remaining-requests": "999"}),
        ],
    )
    async def test_partial_dimension_on_5xx_releases_pending(
        self, backend, label, headers
    ):
        ok, request_id = await backend.check_and_reserve_capacity(
            "m", 1, 1000, bucket_limits={"rpm_limit": 1000, "tpm_limit": 3_000_000}
        )
        assert ok

        await backend.update_rate_limits(
            "m", headers, request_id=request_id, status_code=500
        )

        pend_req = await backend._redis.get(backend._get_pending_req_key("m"))
        pend_tok = await backend._redis.get(backend._get_pending_tok_key("m"))
        assert int(pend_req or 0) == 0, f"request capacity leaked ({label})"
        assert int(pend_tok or 0) == 0, f"token capacity leaked ({label})"
