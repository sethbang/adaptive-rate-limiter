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


# Skip all tests if dependencies not available
pytestmark = [
    pytest.mark.skipif(fakeredis is None, reason="fakeredis not installed"),
    pytest.mark.skipif(lupa is None, reason="lupa not installed (required for Lua)"),
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
async def redis():
    """Create a fresh fakeredis instance for each test."""
    assert fakeredis is not None, "fakeredis not available"
    r = fakeredis.FakeRedis()
    yield r
    await r.aclose()


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
        # Limit takes max of current vs header, so should increase
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
