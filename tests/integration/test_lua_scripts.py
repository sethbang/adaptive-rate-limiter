"""
Integration tests for Redis Lua scripts using fakeredis.

These tests verify the actual Lua script logic for the distributed rate limiting
system. Unlike the unit tests which mock evalsha() to return expected values,
these tests execute the real Lua scripts against a fake Redis instance.

Prerequisites:
    - fakeredis>=2.26.0
    - lupa>=2.0 (required for Lua script execution in fakeredis)

Key Scenarios Tested:
    - Cold start initialization
    - Window rotation with generation tracking
    - Generation mismatch handling
    - Collision detection (duplicate request IDs)
    - Refund calculations and clamping
    - Rate limiting when capacity exhausted

Scripts Tested:
    1. distributed_check_and_reserve.lua - Reserve capacity atomically
    2. distributed_release_capacity.lua - Release on client failure
    3. distributed_release_streaming.lua - Release streaming with refund
    4. distributed_update_rate_limits.lua - Update from 2xx response headers
    5. distributed_update_rate_limits_429.lua - Update from 429 response headers
    6. distributed_recover_orphan.lua - Recover orphaned reservations
"""

import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    pass

try:
    import fakeredis.aioredis as fakeredis
except ImportError:
    fakeredis = None

try:
    import lupa
except ImportError:
    lupa = None


import os as _os

_using_real_redis = bool(_os.environ.get("REDIS_URL"))

# Skip all tests if the fakeredis/lupa dependencies are missing AND no real
# Redis is available via REDIS_URL.
pytestmark = [
    pytest.mark.skipif(
        not _using_real_redis and fakeredis is None,
        reason="fakeredis not installed (and REDIS_URL not set)",
    ),
    pytest.mark.skipif(
        not _using_real_redis and lupa is None,
        reason="lupa not installed (required for Lua in fakeredis; and REDIS_URL not set)",
    ),
]


# Load Lua scripts at module level
LUA_DIR = Path(__file__).parent.parent.parent / "src/adaptive_rate_limiter/backends/lua"
LUA_SCRIPTS: dict[str, str] = {}

for script_name in [
    "distributed_check_and_reserve",
    "distributed_release_capacity",
    "distributed_release_streaming",
    "distributed_update_rate_limits",
    "distributed_update_rate_limits_429",
    "distributed_recover_orphan",
]:
    script_path = LUA_DIR / f"{script_name}.lua"
    if script_path.exists():
        LUA_SCRIPTS[script_name] = script_path.read_text()


@pytest.fixture
async def script_shas(redis):
    """Load all Lua scripts and return their SHAs."""
    shas = {}
    for name, script in LUA_SCRIPTS.items():
        shas[name] = await redis.script_load(script)
    return shas


def get_keys(model: str = "test-model", req_id: str = "req-123") -> tuple:
    """Generate Redis keys for a model and request ID."""
    state_key = f"rl:test:{model}:state"
    pend_req_key = f"rl:test:{model}:pend_req"
    pend_tok_key = f"rl:test:{model}:pend_tok"
    req_map_key = f"rl:test:{model}:req:{req_id}"
    return state_key, pend_req_key, pend_tok_key, req_map_key


class TestCheckAndReserve:
    """Tests for distributed_check_and_reserve.lua."""

    @pytest.mark.asyncio
    async def test_cold_start_success(self, redis, script_shas):
        """Test successful reservation on cold start (no existing state)."""
        sha = script_shas["distributed_check_and_reserve"]
        keys = get_keys(req_id="cold-start-req")

        result = await redis.evalsha(
            sha,
            4,  # num keys
            *keys,
            1,  # cost_req
            100,  # cost_tok
            20,  # fb_lim_req
            500000,  # fb_lim_tok
            60,  # fb_win_req
            60,  # fb_win_tok
            "cold-start-req",  # req_id
            1800,  # req_map_ttl
        )

        # Verify successful reservation
        assert result[0] == 1, "Should return status 1 (allowed)"
        assert result[1] == 0, "Wait time should be 0"
        assert result[2] == 19, "Remaining req should be 20 - 1 = 19"
        assert result[3] == 499900, "Remaining tok should be 500000 - 100 = 499900"
        assert result[4] == 1, "Gen req should be 1"
        assert result[5] == 1, "Gen tok should be 1"

        # Verify state was initialized
        state = await redis.hgetall(keys[0])
        assert state[b"v"] == b"1", "Version should be 1"
        assert state[b"lim_req"] == b"20", "Request limit should be set"
        assert state[b"lim_tok"] == b"500000", "Token limit should be set"

        # Verify pending gauges were incremented
        pend_req = await redis.get(keys[1])
        pend_tok = await redis.get(keys[2])
        assert pend_req == b"1", "Pending requests should be 1"
        assert pend_tok == b"100", "Pending tokens should be 100"

        # Verify request mapping was created
        req_map = await redis.get(keys[3])
        assert req_map == b"1:1:1:100", (
            "Request map format: gen_req:gen_tok:cost_req:cost_tok"
        )

    @pytest.mark.asyncio
    async def test_rate_limited_when_capacity_exhausted(self, redis, script_shas):
        """Test rate limiting when no capacity available."""
        sha = script_shas["distributed_check_and_reserve"]
        state_key, _pend_req_key, _pend_tok_key, _ = get_keys()

        # Pre-set state with no remaining capacity
        now = int(time.time())
        await redis.hset(
            state_key,
            mapping={
                "v": "1",
                "rem_req": "0",
                "rem_tok": "0",
                "lim_req": "20",
                "lim_tok": "500000",
                "rst_req": str(now + 30),  # Reset in 30 seconds
                "rst_tok": str(now + 30),
                "gen_req": "1",
                "gen_tok": "1",
            },
        )

        keys = get_keys(req_id="limited-req")
        result = await redis.evalsha(
            sha,
            4,
            *keys,
            1,  # cost_req
            100,  # cost_tok
            20,  # fb_lim_req
            500000,  # fb_lim_tok
            60,  # fb_win_req
            60,  # fb_win_tok
            "limited-req",  # req_id
            1800,  # req_map_ttl
        )

        # Should be rate limited
        assert result[0] == 0, "Should return status 0 (rate limited)"
        assert result[1] > 0, "Wait time should be positive"
        # No request mapping should be created
        req_map = await redis.get(keys[3])
        assert req_map is None, "No mapping should be created when rate limited"

    @pytest.mark.asyncio
    async def test_collision_detection(self, redis, script_shas):
        """Test collision detection for duplicate request IDs."""
        sha = script_shas["distributed_check_and_reserve"]
        keys = get_keys(req_id="dup-req")

        # First reservation should succeed
        result1 = await redis.evalsha(
            sha,
            4,
            *keys,
            1,
            100,
            20,
            500000,
            60,
            60,
            "dup-req",
            1800,
        )
        assert result1[0] == 1, "First reservation should succeed"

        # Second reservation with same req_id should fail with collision
        keys2 = get_keys(req_id="dup-req")
        result2 = await redis.evalsha(
            sha,
            4,
            *keys2,
            1,
            100,
            20,
            500000,
            60,
            60,
            "dup-req",
            1800,
        )
        assert result2[0] == -2, "Should return -2 (collision)"

    @pytest.mark.asyncio
    async def test_cost_exceeds_limit(self, redis, script_shas):
        """Test when requested cost exceeds limits."""
        sha = script_shas["distributed_check_and_reserve"]
        keys = get_keys(req_id="big-req")

        result = await redis.evalsha(
            sha,
            4,
            *keys,
            100,  # cost_req > fb_lim_req (20)
            100,
            20,  # fb_lim_req
            500000,
            60,
            60,
            "big-req",
            1800,
        )

        assert result[0] == -3, "Should return -3 (cost exceeds limit)"
        assert result[2] == 20, "Should return request limit"
        assert result[3] == 500000, "Should return token limit"

    @pytest.mark.asyncio
    async def test_window_rotation(self, redis, script_shas):
        """Test window rotation resets capacity and increments generation."""
        sha = script_shas["distributed_check_and_reserve"]
        state_key, pend_req_key, pend_tok_key, _ = get_keys()

        # Pre-set expired state
        now = int(time.time())
        await redis.hset(
            state_key,
            mapping={
                "v": "1",
                "rem_req": "5",
                "rem_tok": "1000",
                "lim_req": "20",
                "lim_tok": "500000",
                "rst_req": str(now - 10),  # Expired 10 seconds ago
                "rst_tok": str(now - 10),
                "gen_req": "1",
                "gen_tok": "1",
            },
        )
        await redis.set(pend_req_key, "10")
        await redis.set(pend_tok_key, "5000")

        keys = get_keys(req_id="rotation-req")
        result = await redis.evalsha(
            sha,
            4,
            *keys,
            1,
            100,
            20,
            500000,
            60,
            60,
            "rotation-req",
            1800,
        )

        # Should succeed after rotation
        assert result[0] == 1, "Should succeed after window rotation"
        assert result[4] == 2, "Gen req should be incremented to 2"
        assert result[5] == 2, "Gen tok should be incremented to 2"

        # Pending should be reset (then incremented by new request)
        pend_req = await redis.get(pend_req_key)
        pend_tok = await redis.get(pend_tok_key)
        assert pend_req == b"1", "Pending req should be reset then incremented to 1"
        assert pend_tok == b"100", "Pending tok should be reset then incremented to 100"

    @pytest.mark.asyncio
    async def test_invalid_input_negative_cost(self, redis, script_shas):
        """Test rejection of negative cost values."""
        sha = script_shas["distributed_check_and_reserve"]
        keys = get_keys(req_id="neg-req")

        result = await redis.evalsha(
            sha,
            4,
            *keys,
            -1,  # Negative cost_req
            100,
            20,
            500000,
            60,
            60,
            "neg-req",
            1800,
        )

        assert result[0] == -1, "Should return -1 (invalid input)"

    @pytest.mark.asyncio
    async def test_invalid_input_empty_req_id(self, redis, script_shas):
        """Test rejection of empty request ID."""
        sha = script_shas["distributed_check_and_reserve"]
        keys = get_keys(req_id="empty-req")

        result = await redis.evalsha(
            sha,
            4,
            *keys,
            1,
            100,
            20,
            500000,
            60,
            60,
            "",  # Empty req_id
            1800,
        )

        assert result[0] == -1, "Should return -1 (invalid input)"


class TestReleaseCapacity:
    """Tests for distributed_release_capacity.lua."""

    @pytest.mark.asyncio
    async def test_release_success(self, redis, script_shas):
        """Test successful capacity release."""
        # First reserve capacity
        reserve_sha = script_shas["distributed_check_and_reserve"]
        keys = get_keys(req_id="release-test")

        await redis.evalsha(
            reserve_sha,
            4,
            *keys,
            1,
            100,
            20,
            500000,
            60,
            60,
            "release-test",
            1800,
        )

        # Verify pending before release
        pend_req_before = await redis.get(keys[1])
        assert pend_req_before == b"1"

        # Now release
        release_sha = script_shas["distributed_release_capacity"]
        result = await redis.evalsha(release_sha, 4, *keys)

        assert result == 1, "Should return 1 (success)"

        # Verify pending was decremented
        pend_req_after = await redis.get(keys[1])
        assert pend_req_after == b"0", "Pending should be decremented to 0"

        # Verify mapping was deleted
        req_map = await redis.get(keys[3])
        assert req_map is None, "Mapping should be deleted"

    @pytest.mark.asyncio
    async def test_release_idempotent(self, redis, script_shas):
        """Test release is idempotent (succeeds even if already released)."""
        release_sha = script_shas["distributed_release_capacity"]
        keys = get_keys(req_id="idempotent-test")

        # Release without any prior reservation
        result = await redis.evalsha(release_sha, 4, *keys)

        assert result == 1, "Should return 1 (idempotent success)"

    @pytest.mark.asyncio
    async def test_release_with_generation_mismatch(self, redis, script_shas):
        """Test release skips decrement when generation mismatches."""
        reserve_sha = script_shas["distributed_check_and_reserve"]
        release_sha = script_shas["distributed_release_capacity"]
        keys = get_keys(req_id="gen-mismatch")
        state_key, pend_req_key, pend_tok_key, _req_map_key = keys

        # Reserve capacity
        await redis.evalsha(
            reserve_sha,
            4,
            *keys,
            1,
            100,
            20,
            500000,
            60,
            60,
            "gen-mismatch",
            1800,
        )

        # Manually simulate window rotation by updating generation
        await redis.hset(state_key, "gen_req", "2")
        await redis.hset(state_key, "gen_tok", "2")
        await redis.set(pend_req_key, "5")  # New pending for new generation
        await redis.set(pend_tok_key, "500")

        # Release - should skip decrement due to generation mismatch
        result = await redis.evalsha(release_sha, 4, *keys)

        assert result == 1, "Should still return success"

        # Pending should NOT be decremented (gen mismatch)
        pend_req = await redis.get(pend_req_key)
        pend_tok = await redis.get(pend_tok_key)
        assert pend_req == b"5", "Pending req should not be decremented"
        assert pend_tok == b"500", "Pending tok should not be decremented"


class TestReleaseStreaming:
    """Tests for distributed_release_streaming.lua."""

    @pytest.mark.asyncio
    async def test_streaming_release_with_refund(self, redis, script_shas):
        """Test streaming release calculates refund correctly."""
        release_sha = script_shas["distributed_release_streaming"]
        keys = get_keys(req_id="stream-test")
        state_key, pend_req_key, pend_tok_key, req_map_key = keys

        # Set up state with known values (simulating mid-usage scenario)
        now = int(time.time())
        await redis.hset(
            state_key,
            mapping={
                "v": "1",
                "rem_req": "15",
                "rem_tok": "10000",  # Start with 10000 remaining
                "lim_req": "20",
                "lim_tok": "500000",
                "rst_req": str(now + 60),
                "rst_tok": str(now + 60),
                "gen_req": "1",
                "gen_tok": "1",
            },
        )

        # Create request mapping: reserved 1000 tokens
        await redis.set(req_map_key, "1:1:1:1000")
        await redis.set(pend_req_key, "1")
        await redis.set(pend_tok_key, "1000")

        # Get initial remaining tokens
        state_before = await redis.hgetall(state_key)
        rem_tok_before = int(state_before[b"rem_tok"])
        assert rem_tok_before == 10000, "Pre-condition: rem_tok should be 10000"

        # Release with actual = 300 tokens (refund = 1000 - 300 = 700)
        result = await redis.evalsha(
            release_sha,
            4,
            *keys,
            1000,  # reserved_tokens
            300,  # actual_tokens
        )

        assert result == 1, "Should return 1 (success)"

        # Verify refund was applied
        state_after = await redis.hgetall(state_key)
        rem_tok_after = int(state_after[b"rem_tok"])

        # rem_tok should increase by refund (1000 - 300 = 700)
        assert rem_tok_after == rem_tok_before + 700, "Refund should be applied"
        assert rem_tok_after == 10700, "rem_tok should be 10000 + 700 = 10700"

    @pytest.mark.asyncio
    async def test_streaming_over_consumption_clamping(self, redis, script_shas):
        """Test over-consumption is clamped to 0."""
        release_sha = script_shas["distributed_release_streaming"]
        keys = get_keys(req_id="over-consume")
        state_key = keys[0]

        # Set up state with low remaining
        now = int(time.time())
        await redis.hset(
            state_key,
            mapping={
                "v": "1",
                "rem_req": "20",
                "rem_tok": "100",  # Low remaining
                "lim_req": "20",
                "lim_tok": "500000",
                "rst_req": str(now + 60),
                "rst_tok": str(now + 60),
                "gen_req": "1",
                "gen_tok": "1",
            },
        )

        # Create request mapping manually
        await redis.set(keys[3], "1:1:1:50")  # Reserved 50 tokens
        await redis.set(keys[1], "1")  # pending_req
        await redis.set(keys[2], "50")  # pending_tok

        # Release with actual > reserved (over-consumption)
        # actual=200, reserved=50 → refund=-150 → rem_tok would go to 100-150=-50
        # Should be clamped to 0
        result = await redis.evalsha(
            release_sha,
            4,
            *keys,
            50,  # reserved_tokens
            200,  # actual_tokens (over-consumption!)
        )

        assert result == 1, "Should succeed"

        state_after = await redis.hgetall(state_key)
        rem_tok = int(state_after[b"rem_tok"])
        assert rem_tok == 0, "Over-consumption should clamp to 0"

        # Verify over_consumption_tokens metric was recorded
        over_consumption = state_after.get(b"over_consumption_tokens")
        assert over_consumption is not None, "Over-consumption metric should be set"
        assert int(over_consumption) == 50, "Should track 50 over-consumed tokens"

    @pytest.mark.asyncio
    async def test_streaming_refund_clamping_to_limit(self, redis, script_shas):
        """Test refund is clamped to limit (pure streaming case)."""
        release_sha = script_shas["distributed_release_streaming"]
        keys = get_keys(req_id="pure-stream")
        state_key = keys[0]

        # Set up state near limit
        now = int(time.time())
        await redis.hset(
            state_key,
            mapping={
                "v": "1",
                "rem_req": "20",
                "rem_tok": "499900",  # Near limit
                "lim_req": "20",
                "lim_tok": "500000",
                "rst_req": str(now + 60),
                "rst_tok": str(now + 60),
                "gen_req": "1",
                "gen_tok": "1",
            },
        )

        # Create request mapping: reserved 1000 tokens
        await redis.set(keys[3], "1:1:1:1000")
        await redis.set(keys[1], "1")
        await redis.set(keys[2], "1000")

        # Release with actual=100, refund=900
        # rem_tok would be 499900 + 900 = 500800 > limit
        # Should be clamped to 500000
        result = await redis.evalsha(
            release_sha,
            4,
            *keys,
            1000,  # reserved_tokens
            100,  # actual_tokens
        )

        assert result == 1, "Should succeed"

        state_after = await redis.hgetall(state_key)
        rem_tok = int(state_after[b"rem_tok"])
        assert rem_tok == 500000, "Refund should be clamped to limit"


class TestUpdateRateLimits:
    """Tests for distributed_update_rate_limits.lua (2xx response handling)."""

    @pytest.mark.asyncio
    async def test_update_syncs_state_from_headers(self, redis, script_shas):
        """Test state is synced from API response headers."""
        reserve_sha = script_shas["distributed_check_and_reserve"]
        update_sha = script_shas["distributed_update_rate_limits"]
        keys = get_keys(req_id="update-test")
        state_key = keys[0]

        # Reserve capacity first
        await redis.evalsha(
            reserve_sha,
            4,
            *keys,
            1,
            100,
            20,
            500000,
            60,
            60,
            "update-test",
            1800,
        )

        # Get initial limit
        state_before = await redis.hgetall(state_key)
        lim_req_before = int(state_before[b"lim_req"])
        assert lim_req_before == 20, "Pre-condition: lim_req should be 20"

        # Simulate API response headers with higher limit
        # The script checks: head_rst_req >= (s.rst_req - stale_buffer)
        # Cold start sets rst_req = now + fb_win_req (60), so we need:
        # head_rst_req >= rst_req - stale_buffer = (now + 60) - 10 = now + 50
        # Using now + 60 ensures we pass the staleness check
        now = int(time.time())
        result = await redis.evalsha(
            update_sha,
            4,
            *keys,
            15,  # head_rem_req
            400000,  # head_rem_tok
            50,  # head_lim_req (higher than current 20)
            1000000,  # head_lim_tok (higher than current 500000)
            now + 60,  # head_rst_req (absolute) - must be >= rst_req - stale_buffer
            30,  # head_rst_tok_delta (relative seconds)
            10,  # stale_buffer
            120,  # max_token_delta
        )

        assert result == 1, "Should return 1 (success)"

        # Verify state was updated
        state = await redis.hgetall(state_key)
        # Header is authoritative: the header value replaces the stored value
        lim_req_after = int(state[b"lim_req"])
        assert lim_req_after == 50, (
            f"Request limit should be upgraded to 50, got {lim_req_after}"
        )

        # Request mapping should be deleted
        req_map = await redis.get(keys[3])
        assert req_map is None, "Mapping should be deleted after update"

    @pytest.mark.asyncio
    async def test_update_rejects_invalid_headers(self, redis, script_shas):
        """Test rejection of invalid header values."""
        reserve_sha = script_shas["distributed_check_and_reserve"]
        update_sha = script_shas["distributed_update_rate_limits"]
        keys = get_keys(req_id="invalid-headers")

        # Reserve first
        await redis.evalsha(
            reserve_sha,
            4,
            *keys,
            1,
            100,
            20,
            500000,
            60,
            60,
            "invalid-headers",
            1800,
        )

        now = int(time.time())

        # Test negative remaining
        result = await redis.evalsha(
            update_sha,
            4,
            *keys,
            -1,  # Invalid negative
            400000,
            50,
            1000000,
            now + 45,
            30,
            10,
            120,
        )
        assert result == 0, "Should reject negative remaining"

    @pytest.mark.asyncio
    async def test_update_accepts_limit_decrease(self, redis, script_shas):
        """Test that a server-reported limit decrease is honored (not silently ignored).

        When a server tier is downgraded mid-window the header limit will be lower
        than the stored limit.  Using math.max() would keep the old high value and
        cause over-admission.  The header must be treated as authoritative.
        """
        update_sha = script_shas["distributed_update_rate_limits"]
        keys = get_keys(req_id="decrease-test")
        state_key, pend_req_key, pend_tok_key, req_map_key = keys

        # Seed state with a HIGH limit (simulating a previous tier)
        now = int(time.time())
        await redis.hset(
            state_key,
            mapping={
                "v": "1",
                "rem_req": "1000",
                "rem_tok": "100000",
                "lim_req": "1000",  # HIGH stored limit
                "lim_tok": "100000",
                "rst_req": str(now + 60),
                "rst_tok": str(now + 60),
                "gen_req": "0",
                "gen_tok": "0",
            },
        )
        # Seed pending gauges at 0
        await redis.set(pend_req_key, "0")
        await redis.set(pend_tok_key, "0")
        # Seed a request mapping so the script proceeds past the mapping check
        # Format: gen_req:gen_tok:cost_req:cost_tok
        await redis.set(req_map_key, "0:0:1:10")

        # Run update with a LOWER limit reported by the server (tier downgrade).
        # head_rst_tok_delta=60 ensures calc_rst_tok = now+60 >= rst_tok(now+60)-10,
        # so the token-window branch is entered and lim_tok is updated.
        result = await redis.evalsha(
            update_sha,
            4,
            *keys,
            90,  # head_rem_req
            9000,  # head_rem_tok
            100,  # head_lim_req  <-- LOWER than stored 1000
            10000,  # head_lim_tok  <-- LOWER than stored 100000
            now + 60,  # head_rst_req (absolute unix ts, well above 1600000000)
            60,  # head_rst_tok_delta: calc_rst_tok=now+60 passes staleness check
            10,  # stale_buffer
            120,  # max_tok_delta
        )

        assert result == 1, "Script should return 1 (success)"

        state = await redis.hgetall(state_key)
        stored_lim_req = int(state[b"lim_req"])
        assert stored_lim_req == 100, (
            f"Limit decrease must be applied: expected 100, got {stored_lim_req}. "
            "math.max() keeps the old high value — header must be authoritative."
        )
        stored_lim_tok = int(state[b"lim_tok"])
        assert stored_lim_tok == 10000, (
            f"Token limit decrease must be applied: expected 10000, got {stored_lim_tok}."
        )

    @pytest.mark.asyncio
    async def test_update_fails_without_mapping(self, redis, script_shas):
        """Test update fails when request mapping doesn't exist."""
        update_sha = script_shas["distributed_update_rate_limits"]
        keys = get_keys(req_id="no-mapping")

        now = int(time.time())
        result = await redis.evalsha(
            update_sha,
            4,
            *keys,
            15,
            400000,
            50,
            1000000,
            now + 45,
            30,
            10,
            120,
        )

        assert result == 0, "Should return 0 when mapping not found"


class TestUpdateRateLimits429:
    """Tests for distributed_update_rate_limits_429.lua (429 response handling)."""

    @pytest.mark.asyncio
    async def test_429_releases_pending_and_updates_state(self, redis, script_shas):
        """Test 429 handling releases pending and syncs state."""
        reserve_sha = script_shas["distributed_check_and_reserve"]
        update_429_sha = script_shas["distributed_update_rate_limits_429"]
        keys = get_keys(req_id="429-test")
        pend_req_key = keys[1]

        # Reserve capacity first
        await redis.evalsha(
            reserve_sha,
            4,
            *keys,
            1,
            100,
            20,
            500000,
            60,
            60,
            "429-test",
            1800,
        )

        # Verify pending before 429
        pend_req_before = await redis.get(pend_req_key)
        assert pend_req_before == b"1"

        # Handle 429 response
        now = int(time.time())
        result = await redis.evalsha(
            update_429_sha,
            4,
            *keys,
            0,  # head_rem_req (429 means 0 remaining)
            0,  # head_rem_tok
            20,  # head_lim_req
            500000,  # head_lim_tok
            now + 60,  # head_rst_req
            60,  # head_rst_tok_delta
            10,  # stale_buffer
            120,  # max_token_delta
        )

        assert result == 1, "Should return 1 (success)"

        # Pending should be released (decremented) since 429 = request not consumed
        pend_req_after = await redis.get(pend_req_key)
        assert pend_req_after == b"0", "Pending should be released on 429"

    @pytest.mark.asyncio
    async def test_429_works_without_headers(self, redis, script_shas):
        """Test 429 handling still releases pending even with invalid headers."""
        reserve_sha = script_shas["distributed_check_and_reserve"]
        update_429_sha = script_shas["distributed_update_rate_limits_429"]
        keys = get_keys(req_id="429-no-headers")

        # Reserve first
        await redis.evalsha(
            reserve_sha,
            4,
            *keys,
            1,
            100,
            20,
            500000,
            60,
            60,
            "429-no-headers",
            1800,
        )

        # Handle 429 with invalid/missing headers
        result = await redis.evalsha(
            update_429_sha,
            4,
            *keys,
            -1,  # Invalid header
            0,
            0,
            0,
            0,
            0,
            10,
            120,
        )

        # Should still succeed (release pending) even with bad headers
        assert result == 1, "Should still release pending with bad headers"

    @pytest.mark.asyncio
    async def test_update_429_accepts_limit_decrease(self, redis, script_shas):
        """Test that a 429 response with a lower header limit stores the lower value.

        When a server tier is downgraded and signals this via a 429, the header
        limit will be lower than the stored limit.  The 429 script got the same
        authoritative-header change as the normal update script, so it must also
        apply the decrease rather than silently ignoring it.
        """
        update_429_sha = script_shas["distributed_update_rate_limits_429"]
        keys = get_keys(req_id="429-decrease-test")
        state_key, pend_req_key, pend_tok_key, req_map_key = keys

        # Seed state with a HIGH limit (simulating a previous tier)
        now = int(time.time())
        await redis.hset(
            state_key,
            mapping={
                "v": "1",
                "rem_req": "1000",
                "rem_tok": "100000",
                "lim_req": "1000",  # HIGH stored limit
                "lim_tok": "100000",
                "rst_req": str(now + 60),
                "rst_tok": str(now + 60),
                "gen_req": "0",
                "gen_tok": "0",
            },
        )
        # Seed pending gauges at 0
        await redis.set(pend_req_key, "0")
        await redis.set(pend_tok_key, "0")
        # Seed a request mapping so the script proceeds past the mapping check
        # Format: gen_req:gen_tok:cost_req:cost_tok
        await redis.set(req_map_key, "0:0:1:10")

        # Run 429 update with a LOWER limit reported by the server (tier downgrade).
        # head_rst_tok_delta=60 ensures calc_rst_tok = now+60 >= rst_tok(now+60)-10,
        # so the token-window branch is entered and lim_tok is updated.
        result = await redis.evalsha(
            update_429_sha,
            4,
            *keys,
            0,  # head_rem_req (429 means 0 remaining)
            0,  # head_rem_tok
            100,  # head_lim_req  <-- LOWER than stored 1000
            10000,  # head_lim_tok  <-- LOWER than stored 100000
            now + 60,  # head_rst_req (absolute unix ts)
            60,  # head_rst_tok_delta: calc_rst_tok=now+60 passes staleness check
            10,  # stale_buffer
            120,  # max_tok_delta
        )

        assert result == 1, "Script should return 1 (success)"

        state = await redis.hgetall(state_key)
        stored_lim_req = int(state[b"lim_req"])
        assert stored_lim_req == 100, (
            f"Limit decrease must be applied: expected 100, got {stored_lim_req}. "
            "math.max() keeps the old high value — header must be authoritative."
        )
        stored_lim_tok = int(state[b"lim_tok"])
        assert stored_lim_tok == 10000, (
            f"Token limit decrease must be applied: expected 10000, got {stored_lim_tok}."
        )


class TestRecoverOrphan:
    """Tests for distributed_recover_orphan.lua."""

    @pytest.mark.asyncio
    async def test_recover_decrements_pending(self, redis, script_shas):
        """Test orphan recovery decrements pending gauges."""
        recover_sha = script_shas["distributed_recover_orphan"]

        # Set up state with pending values
        state_key = "rl:test:orphan:state"
        pend_req_key = "rl:test:orphan:pend_req"
        pend_tok_key = "rl:test:orphan:pend_tok"

        now = int(time.time())
        await redis.hset(
            state_key,
            mapping={
                "v": "1",
                "gen_req": "1",
                "gen_tok": "1",
                "rst_req": str(now + 60),
                "rst_tok": str(now + 60),
            },
        )
        await redis.set(pend_req_key, "5")
        await redis.set(pend_tok_key, "500")

        # Recover orphan: cost_req=2, cost_tok=200, gen_req=1, gen_tok=1
        result = await redis.evalsha(
            recover_sha,
            3,
            pend_req_key,
            pend_tok_key,
            state_key,
            2,  # cost_req
            200,  # cost_tok
            1,  # expected_gen_req
            1,  # expected_gen_tok
        )

        assert result == 1, "Should return 1 (success)"

        pend_req = await redis.get(pend_req_key)
        pend_tok = await redis.get(pend_tok_key)
        assert pend_req == b"3", "Pending req should be 5 - 2 = 3"
        assert pend_tok == b"300", "Pending tok should be 500 - 200 = 300"

    @pytest.mark.asyncio
    async def test_recover_skips_on_generation_mismatch(self, redis, script_shas):
        """Test orphan recovery skips decrement when generation mismatches."""
        recover_sha = script_shas["distributed_recover_orphan"]

        state_key = "rl:test:orphan2:state"
        pend_req_key = "rl:test:orphan2:pend_req"
        pend_tok_key = "rl:test:orphan2:pend_tok"

        now = int(time.time())
        await redis.hset(
            state_key,
            mapping={
                "v": "1",
                "gen_req": "2",  # Current gen is 2
                "gen_tok": "2",
                "rst_req": str(now + 60),
                "rst_tok": str(now + 60),
            },
        )
        await redis.set(pend_req_key, "5")
        await redis.set(pend_tok_key, "500")

        # Try to recover with old generation (1)
        result = await redis.evalsha(
            recover_sha,
            3,
            pend_req_key,
            pend_tok_key,
            state_key,
            2,
            200,
            1,  # expected_gen_req = 1 (mismatches current 2)
            1,  # expected_gen_tok = 1 (mismatches current 2)
        )

        assert result == 1, "Should still return success"

        # Pending should NOT be decremented due to gen mismatch
        pend_req = await redis.get(pend_req_key)
        pend_tok = await redis.get(pend_tok_key)
        assert pend_req == b"5", "Pending req should not change"
        assert pend_tok == b"500", "Pending tok should not change"

    @pytest.mark.asyncio
    async def test_recover_clamps_negative_to_zero(self, redis, script_shas):
        """Test orphan recovery clamps pending to 0 if it would go negative."""
        recover_sha = script_shas["distributed_recover_orphan"]

        state_key = "rl:test:orphan3:state"
        pend_req_key = "rl:test:orphan3:pend_req"
        pend_tok_key = "rl:test:orphan3:pend_tok"

        now = int(time.time())
        await redis.hset(
            state_key,
            mapping={
                "v": "1",
                "gen_req": "1",
                "gen_tok": "1",
                "rst_req": str(now + 60),
                "rst_tok": str(now + 60),
            },
        )
        await redis.set(pend_req_key, "1")  # Only 1 pending
        await redis.set(pend_tok_key, "50")  # Only 50 pending

        # Recover with more than pending
        result = await redis.evalsha(
            recover_sha,
            3,
            pend_req_key,
            pend_tok_key,
            state_key,
            5,  # cost_req > current pending (1)
            200,  # cost_tok > current pending (50)
            1,
            1,
        )

        assert result == 1, "Should succeed"

        pend_req = await redis.get(pend_req_key)
        pend_tok = await redis.get(pend_tok_key)
        assert pend_req == b"0", "Pending req should be clamped to 0"
        assert pend_tok == b"0", "Pending tok should be clamped to 0"


class TestEndToEndScenarios:
    """End-to-end integration tests covering realistic scenarios."""

    @pytest.mark.asyncio
    async def test_full_request_lifecycle(self, redis, script_shas):
        """Test complete request lifecycle: reserve → execute → update."""
        reserve_sha = script_shas["distributed_check_and_reserve"]
        update_sha = script_shas["distributed_update_rate_limits"]
        keys = get_keys(req_id="lifecycle-test")

        # 1. Reserve capacity
        reserve_result = await redis.evalsha(
            reserve_sha,
            4,
            *keys,
            1,
            500,
            20,
            500000,
            60,
            60,
            "lifecycle-test",
            1800,
        )
        assert reserve_result[0] == 1, "Reserve should succeed"

        # 2. Simulate API call execution (external)
        # API returns: remaining_req=18, remaining_tok=499000

        # 3. Update state from response headers
        now = int(time.time())
        update_result = await redis.evalsha(
            update_sha,
            4,
            *keys,
            18,  # head_rem_req
            499000,  # head_rem_tok
            20,  # head_lim_req
            500000,  # head_lim_tok
            now + 55,  # head_rst_req
            55,  # head_rst_tok_delta
            10,
            120,
        )
        assert update_result == 1, "Update should succeed"

        # Verify final state
        state = await redis.hgetall(keys[0])
        # rem should be header value minus current pending (0 after update)
        assert int(state[b"rem_req"]) == 18
        # Mapping should be cleaned up
        assert await redis.get(keys[3]) is None

    @pytest.mark.asyncio
    async def test_multiple_concurrent_reservations(self, redis, script_shas):
        """Test multiple requests reserving from same pool."""
        reserve_sha = script_shas["distributed_check_and_reserve"]

        # Make 5 reservations
        for i in range(5):
            keys = get_keys(req_id=f"concurrent-{i}")
            result = await redis.evalsha(
                reserve_sha,
                4,
                *keys,
                1,
                100,
                20,
                500000,
                60,
                60,
                f"concurrent-{i}",
                1800,
            )
            assert result[0] == 1, f"Reservation {i} should succeed"

        # Check pending accumulation
        pend_req = await redis.get(get_keys()[1])
        pend_tok = await redis.get(get_keys()[2])
        assert int(pend_req) == 5, "Should have 5 pending requests"
        assert int(pend_tok) == 500, "Should have 500 pending tokens"

    @pytest.mark.asyncio
    async def test_streaming_workflow(self, redis, script_shas):
        """Test streaming request workflow with refund."""
        release_sha = script_shas["distributed_release_streaming"]
        keys = get_keys(req_id="stream-workflow")
        state_key, pend_req_key, pend_tok_key, req_map_key = keys

        # Set up state with known values (simulating mid-usage scenario)
        # This avoids the cold-start logic setting rem_tok to limit
        now = int(time.time())
        await redis.hset(
            state_key,
            mapping={
                "v": "1",
                "rem_req": "15",
                "rem_tok": "100000",  # Start with less than limit
                "lim_req": "20",
                "lim_tok": "500000",
                "rst_req": str(now + 60),
                "rst_tok": str(now + 60),
                "gen_req": "1",
                "gen_tok": "1",
            },
        )

        # Create request mapping: reserved 2000 tokens (streaming estimate)
        await redis.set(req_map_key, "1:1:1:2000")
        await redis.set(pend_req_key, "1")
        await redis.set(pend_tok_key, "2000")

        # Get remaining after "reserve"
        state_mid = await redis.hgetall(state_key)
        rem_tok_mid = int(state_mid[b"rem_tok"])
        assert rem_tok_mid == 100000, "Pre-condition: rem_tok should be 100000"

        # Stream completes with actual 800 tokens (less than estimated)
        # refund = 2000 - 800 = 1200
        release_result = await redis.evalsha(
            release_sha,
            4,
            *keys,
            2000,  # reserved
            800,  # actual (refund = 1200)
        )
        assert release_result == 1

        # Verify refund was applied
        state_final = await redis.hgetall(state_key)
        rem_tok_final = int(state_final[b"rem_tok"])

        # rem_tok should increase by 1200 (2000 - 800)
        assert rem_tok_final == rem_tok_mid + 1200, (
            f"Expected {rem_tok_mid + 1200}, got {rem_tok_final}"
        )
        assert rem_tok_final == 101200, "rem_tok should be 100000 + 1200 = 101200"


class TestResetWindowProvenance:
    """Reset-window provenance (``vrf_req`` / ``vrf_tok``) in the update scripts.

    Regression coverage for the venice-py upstream report (2026-08-23).

    ``distributed_check_and_reserve.lua`` fabricates a window (``now +
    fb_win_req``) on cold start and on every rotation. That guess is stored in
    the same field as an observed server reset, so the staleness comparison used
    to pit a real header against a fabrication -- and the fabrication, sitting
    further in the future, won. The flag records which of the two a stored
    window is, so staleness is only enforced between real observations.
    """

    RESERVE = "distributed_check_and_reserve"
    UPDATE = "distributed_update_rate_limits"
    UPDATE_429 = "distributed_update_rate_limits_429"

    @staticmethod
    async def _now(redis):
        """Redis server time in whole seconds - the clock the scripts use."""
        seconds, _micros = await redis.time()
        return int(seconds)

    async def _reserve(self, redis, script_shas, keys, req_id, win=60):
        return await redis.evalsha(
            script_shas[self.RESERVE],
            4,
            *keys,
            1,  # cost_req
            100,  # cost_tok
            20,  # fb_lim_req  (the conservative fallback)
            500000,  # fb_lim_tok
            win,  # fb_win_req
            win,  # fb_win_tok
            req_id,
            1800,
        )

    async def _update(
        self,
        redis,
        script_shas,
        keys,
        *,
        rst_req,
        tok_delta,
        script=None,
        lim_req=500,
        rem_req=499,
        lim_tok=1000000,
        rem_tok=999000,
    ):
        return await redis.evalsha(
            script_shas[script or self.UPDATE],
            4,
            *keys,
            rem_req,
            rem_tok,
            lim_req,
            lim_tok,
            rst_req,  # ARGV[5] absolute Unix seconds
            tok_delta,  # ARGV[6] relative seconds
            10,  # stale_buffer
            120,  # max_tok_delta
        )

    # The offsets that used to fail: every server reset more than stale_buffer
    # earlier than the fabricated now+60 window was silently discarded.
    @pytest.mark.parametrize("offset", [5, 15, 30, 45, 49, 50, 51, 55, 60, 75, 90])
    @pytest.mark.asyncio
    async def test_first_header_ingested_at_every_offset(
        self, redis, script_shas, offset
    ):
        """A real header must win over the fabricated cold-start window."""
        keys = get_keys(model=f"prov-{offset}", req_id=f"prov-req-{offset}")
        state_key = keys[0]

        await self._reserve(redis, script_shas, keys, f"prov-req-{offset}")
        now = await self._now(redis)

        result = await self._update(
            redis,
            script_shas,
            keys,
            rst_req=now + offset,
            tok_delta=min(offset, 120),
        )
        assert result == 1

        state = await redis.hgetall(state_key)
        assert int(state[b"lim_req"]) == 500, (
            f"server limit dropped at offset now+{offset}s "
            f"(fabricated window was now+60s)"
        )
        assert int(state[b"lim_tok"]) == 1000000
        assert int(state[b"vrf_req"]) == 1
        assert int(state[b"vrf_tok"]) == 1
        # The observed window is adopted outright, not max()'d against the guess
        assert int(state[b"rst_req"]) == now + offset

    @pytest.mark.asyncio
    async def test_stale_snapshot_still_rejected_once_verified(
        self, redis, script_shas
    ):
        """The guard must still reject out-of-order snapshots between observations.

        This is the behaviour the flag preserves; without this test the fix
        would be indistinguishable from deleting the staleness check.
        """
        keys = get_keys(model="prov-stale", req_id="prov-stale-1")
        state_key = keys[0]

        await self._reserve(redis, script_shas, keys, "prov-stale-1")
        now = await self._now(redis)

        # First observation establishes a real window at now+50
        await self._update(redis, script_shas, keys, rst_req=now + 50, tok_delta=50)
        state = await redis.hgetall(state_key)
        assert int(state[b"lim_req"]) == 500
        assert int(state[b"vrf_req"]) == 1

        # A late-arriving snapshot from an OLDER window must not overwrite it
        await self._reserve(redis, script_shas, keys, "prov-stale-2")
        await self._update(
            redis,
            script_shas,
            keys,
            rst_req=now + 10,  # older than (now+50) - stale_buffer
            tok_delta=10,
            lim_req=999,
            lim_tok=999,
        )

        state = await redis.hgetall(state_key)
        assert int(state[b"lim_req"]) == 500, "stale snapshot overwrote live state"
        assert int(state[b"rst_req"]) == now + 50, "stale snapshot rewound the window"

    @pytest.mark.asyncio
    async def test_window_rotation_clears_verification(self, redis, script_shas):
        """Rotation re-fabricates the window, so the flag must reset with it.

        Without this the bug returns once per window: the rotated guess would be
        treated as an observation and shadow the next real header.
        """
        keys = get_keys(model="prov-rot", req_id="prov-rot-1")
        state_key = keys[0]

        await self._reserve(redis, script_shas, keys, "prov-rot-1")
        now = await self._now(redis)
        await self._update(redis, script_shas, keys, rst_req=now + 50, tok_delta=50)
        assert int((await redis.hgetall(state_key))[b"vrf_req"]) == 1

        # Force the window to have elapsed, then reserve to trigger rotation
        await redis.hset(state_key, "rst_req", now - 1)
        await redis.hset(state_key, "rst_tok", now - 1)
        await self._reserve(redis, script_shas, keys, "prov-rot-2")

        state = await redis.hgetall(state_key)
        assert int(state[b"vrf_req"]) == 0, (
            "rotation left a fabricated window marked verified"
        )
        assert int(state[b"vrf_tok"]) == 0
        assert int(state[b"lim_req"]) == 20, (
            "rotation should restore the fallback limit"
        )

        # The next real header is adopted despite sitting well before now+60
        await self._update(redis, script_shas, keys, rst_req=now + 5, tok_delta=5)
        state = await redis.hgetall(state_key)
        assert int(state[b"lim_req"]) == 500
        assert int(state[b"vrf_req"]) == 1

    @pytest.mark.asyncio
    async def test_state_written_before_the_flag_existed(self, redis, script_shas):
        """Hashes from a pre-upgrade deployment have no vrf_* fields.

        A missing flag must read as unverified so existing state self-heals on
        the next response, without a schema version bump.
        """
        keys = get_keys(model="prov-legacy", req_id="prov-legacy-1")
        state_key = keys[0]
        now = await self._now(redis)

        # Exactly what the old check_and_reserve wrote: no vrf_req / vrf_tok
        await redis.hset(
            state_key,
            mapping={
                "v": "1",
                "rem_req": "20",
                "rem_tok": "500000",
                "lim_req": "20",
                "lim_tok": "500000",
                "rst_req": str(now + 60),
                "rst_tok": str(now + 60),
                "gen_req": "1",
                "gen_tok": "1",
            },
        )
        await self._reserve(redis, script_shas, keys, "prov-legacy-1")

        await self._update(redis, script_shas, keys, rst_req=now + 30, tok_delta=30)

        state = await redis.hgetall(state_key)
        assert int(state[b"lim_req"]) == 500, "legacy state did not self-heal"
        assert int(state[b"vrf_req"]) == 1

    @pytest.mark.asyncio
    async def test_429_script_shares_the_behaviour(self, redis, script_shas):
        """The 429 path silently skipped the update - no return code signalled it."""
        keys = get_keys(model="prov-429", req_id="prov-429-1")
        state_key = keys[0]

        await self._reserve(redis, script_shas, keys, "prov-429-1")
        now = await self._now(redis)

        result = await self._update(
            redis,
            script_shas,
            keys,
            rst_req=now + 30,
            tok_delta=30,
            script=self.UPDATE_429,
        )
        assert result == 1

        state = await redis.hgetall(state_key)
        assert int(state[b"lim_req"]) == 500
        assert int(state[b"vrf_req"]) == 1


class TestBackendToLuaResetUnits:
    """Full chain: absolute reset headers -> RedisBackend -> real Lua -> state.

    Answers a specific question the unit tests cannot: ``_parse_rate_limit_headers``
    emits ``tpm_reset`` as an **absolute** Unix timestamp, but
    ``distributed_update_rate_limits.lua`` validates ARGV[6] as a **relative**
    delta bounded by ``max_token_delta`` (120). An absolute epoch would blow past
    that and the script would silently reject the update.

    ``RedisBackend._token_reset_delta`` converts at that boundary. Everything
    else covering it either mocks ``evalsha`` (asserting on ARGV, never on real
    Lua) or passes a delta directly (bypassing the conversion), so without this
    class a regression there would still look green.
    """

    @pytest.fixture
    async def backend(self):
        """A real RedisBackend over fakeredis.

        Needs its own fixture: the module-level ``redis`` fixture uses
        ``decode_responses=False``, and RedisBackend requires True.
        """
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
            backend = RedisBackend(
                redis_url=redis_url or "redis://localhost:6379",
                namespace="units",
                account_id="acct",
            )
            backend._redis = client
            backend._connected = True
            await backend._load_scripts()
            yield backend

        if redis_url:
            await client.flushdb()
        await client.aclose()

    @staticmethod
    def _venice_headers(now, *, req_offset=30, tok_offset=45):
        """Venice sends a clean integer epoch in MILLISECONDS."""
        return {
            "x-ratelimit-limit-requests": "500",
            "x-ratelimit-remaining-requests": "499",
            "x-ratelimit-limit-tokens": "1000000",
            "x-ratelimit-remaining-tokens": "999000",
            "x-ratelimit-reset-requests": str(int((now + req_offset) * 1000)),
            "x-ratelimit-reset-tokens": str(int((now + tok_offset) * 1000)),
        }

    @pytest.mark.asyncio
    async def test_absolute_token_reset_reaches_the_state_hash(self, backend):
        """The token window must land, not be eaten by the ARGV[6] delta guard."""
        model = "units-tok"
        ok, req_id = await backend.check_and_reserve_capacity(
            key=model, requests=1, tokens=100
        )
        assert ok

        now = time.time()
        result = await backend.update_rate_limits(
            model=model,
            headers=self._venice_headers(now),
            request_id=req_id,
            status_code=200,
        )
        assert result == 1, "update rejected"

        state = await backend.get_rate_limits(model)
        assert state["tpm_limit"] == 1000000, "token limit never ingested"
        assert state["tpm_remaining"] == 999000

        # rst_tok is re-anchored to Redis time as now + delta, so it must be a
        # plausible absolute timestamp - not an epoch-ms value, not a bare delta.
        assert 1600000000 < state["tpm_reset"] < 4102444800
        assert abs(state["tpm_reset"] - (now + 45)) < 5

        # And the request side, for symmetry
        assert state["rpm_limit"] == 500
        assert abs(state["rpm_reset"] - (now + 30)) < 5

    @pytest.mark.asyncio
    async def test_long_token_window_is_not_clamped_to_an_early_reset(self, backend):
        """A token window beyond max_token_delta must not become an early reset.

        Why this direction matters: believing the window resets EARLIER than it
        does makes ``check_and_reserve`` rotate and refill ``rem_tok`` to the
        fallback limit before the server has actually refilled, so the limiter
        over-sends and earns 429s -- the exact failure it exists to prevent.
        Believing the window is longer than it is only over-throttles, which is
        recoverable. So an out-of-range delta must never be clamped into a
        window we then treat as observed.

        The observed *counts* are still ingested: they are directly reported and
        not in doubt, and dropping them would strand ``rem_tok`` at the fallback
        (far above the server's real remaining), which over-sends immediately
        rather than only at rotation.
        """
        model = "units-long"
        ok, req_id = await backend.check_and_reserve_capacity(
            key=model, requests=1, tokens=100
        )
        assert ok

        state_key = backend._get_state_key(model)
        before = await backend._redis.hgetall(state_key)
        rst_tok_before = float(before["rst_tok"])

        now = time.time()
        headers = self._venice_headers(now, tok_offset=3600)
        headers["x-ratelimit-remaining-tokens"] = "5000"
        result = await backend.update_rate_limits(
            model=model, headers=headers, request_id=req_id, status_code=200
        )
        assert result == 1, "the request-side update must still land"

        after = await backend._redis.hgetall(state_key)

        # Request side landed in full
        assert int(after["lim_req"]) == 500
        assert int(after["vrf_req"]) == 1

        # Token window was NOT adopted, and NOT clamped to an early reset
        assert float(after["rst_tok"]) == rst_tok_before, "a window was fabricated"
        assert int(after.get("vrf_tok", 0)) == 0, (
            "an unadopted window must stay unverified, or the next real header "
            "would be compared against a fabrication"
        )

        # ...but the observed counts were ingested
        assert int(after["lim_tok"]) == 1000000
        assert int(after["rem_tok"]) == 5000, "server's real remaining was discarded"

    @pytest.mark.asyncio
    async def test_in_range_token_window_is_still_adopted(self, backend):
        """The skip must be scoped to out-of-range deltas only."""
        model = "units-inrange"
        ok, req_id = await backend.check_and_reserve_capacity(
            key=model, requests=1, tokens=100
        )
        assert ok

        now = time.time()
        result = await backend.update_rate_limits(
            model=model,
            headers=self._venice_headers(now, tok_offset=45),
            request_id=req_id,
            status_code=200,
        )
        assert result == 1

        state = await backend.get_rate_limits(model)
        assert abs(state["tpm_reset"] - (now + 45)) < 5
        assert (
            int(
                (await backend._redis.hgetall(backend._get_state_key(model)))["vrf_tok"]
            )
            == 1
        )

    @pytest.mark.asyncio
    async def test_absurd_reset_value_cannot_poison_state(self, backend):
        """A finite-but-absurd reset must not freeze the model's state.

        1e308 clears the Lua floor, is stored in scientific notation (breaking
        get_rate_limits' int() read), and survives via math.max - so every later
        real header loses the staleness comparison. Rejected at the parser now,
        with the Lua ceiling as the second layer.
        """
        model = "units-poison"
        now = time.time()

        ok, req_id = await backend.check_and_reserve_capacity(
            key=model, requests=1, tokens=100
        )
        assert ok
        bad = self._venice_headers(now)
        bad["x-ratelimit-reset-requests"] = "1e308"
        await backend.update_rate_limits(
            model=model, headers=bad, request_id=req_id, status_code=200
        )

        # State must still be readable
        state = await backend.get_rate_limits(model)
        assert state != {}, "get_rate_limits broke on a poisoned reset value"
        assert state["rpm_reset"] < 4102444800

        # And a subsequent good header must still land
        ok, req_id2 = await backend.check_and_reserve_capacity(
            key=model, requests=1, tokens=100
        )
        assert ok
        good = self._venice_headers(now)
        good["x-ratelimit-limit-requests"] = "777"
        await backend.update_rate_limits(
            model=model, headers=good, request_id=req_id2, status_code=200
        )

        state = await backend.get_rate_limits(model)
        assert state["rpm_limit"] == 777, "state frozen by the earlier bad value"
