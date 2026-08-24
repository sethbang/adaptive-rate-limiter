-- Copyright 2026 Seth Bang
-- SPDX-License-Identifier: Apache-2.0

-- Distributed update_rate_limits_429.lua
-- Handle 429 (Rate Limited) responses from the API.
-- A 429 response includes valid rate limit headers, but the request consumed no server-side capacity.
-- This script releases the pending reservation AND updates state from the authoritative headers.
--
-- KEYS[1]: state_key - Hash storing limits, remaining, resets, generations
-- KEYS[2]: pend_req_key - String (Int) gauge of in-flight requests
-- KEYS[3]: pend_tok_key - String (Int) gauge of in-flight tokens
-- KEYS[4]: req_map_key - String storing reservation snapshot
--
-- ARGV[1]: head_rem_req - x-ratelimit-remaining-requests
-- ARGV[2]: head_rem_tok - x-ratelimit-remaining-tokens
-- ARGV[3]: head_lim_req - x-ratelimit-limit-requests
-- ARGV[4]: head_lim_tok - x-ratelimit-limit-tokens
-- ARGV[5]: head_rst_req - Absolute Unix timestamp, in SECONDS, when the request
--          window resets. Normalized by BaseBackend._parse_rate_limit_headers,
--          which accepts epoch milliseconds, epoch seconds or a relative delta
--          on the wire and always emits absolute seconds.
-- ARGV[6]: head_rst_tok_delta - RELATIVE seconds until the token window resets.
--          NOT the raw header value: derived by RedisBackend._token_reset_delta
--          from the same absolute-seconds representation, floored at 0 but
--          deliberately NOT capped at ARGV[8]. Kept relative so it can be
--          re-anchored to Redis server time below, which makes it immune to
--          clock skew between the calling process and Redis. A value exceeding
--          ARGV[8] is handled in the token block: the window is skipped, the
--          counts are still applied. Capping it here instead would turn a long
--          window into an early reset and cause over-sending.
-- ARGV[7]: stale_buffer - Buffer for stale detection (default 10)
-- ARGV[8]: max_tok_delta - Maximum accepted value for ARGV[6] (default 120)
--
-- Returns:
--   1 - Success (released pending and optionally updated state)
--   0 - Mapping not found

local state_key = KEYS[1]
local pend_req_key = KEYS[2]
local pend_tok_key = KEYS[3]
local req_map_key = KEYS[4]

local head_rem_req = tonumber(ARGV[1])
local head_rem_tok = tonumber(ARGV[2])
local head_lim_req = tonumber(ARGV[3])
local head_lim_tok = tonumber(ARGV[4])
local head_rst_req = tonumber(ARGV[5])
local head_rst_tok_delta = tonumber(ARGV[6])
local stale_buffer = tonumber(ARGV[7]) or 10
local max_tok_delta = tonumber(ARGV[8]) or 120

-- 1. Get Mapping
local map_val = redis.call('GET', req_map_key)
if not map_val then return 0 end

-- 2. Parse Mapping
local parts = {}
for part in string.gmatch(map_val, "([^:]+)") do table.insert(parts, part) end
local map_gen_req = tonumber(parts[1])
local map_gen_tok = tonumber(parts[2])
local cost_req = tonumber(parts[3])
local cost_tok = tonumber(parts[4])

-- 3. Get current time from Redis
local time = redis.call('TIME')
local now = tonumber(time[1])

-- 4. Load State
local state = redis.call('HGETALL', state_key)
local s = {}
for i = 1, #state, 2 do s[state[i]] = state[i+1] end

-- 5. Decrement Pending (with generation check)
-- For 429, the request was NOT consumed, so we release our pending reservation.
if map_gen_req == tonumber(s.gen_req or 0) then
    local new_p = redis.call('DECRBY', pend_req_key, cost_req)
    if new_p < 0 then redis.call('SET', pend_req_key, 0) end
    redis.call('EXPIRE', pend_req_key, 86400)
end

if map_gen_tok == tonumber(s.gen_tok or 0) then
    local new_p = redis.call('DECRBY', pend_tok_key, cost_tok)
    if new_p < 0 then redis.call('SET', pend_tok_key, 0) end
    redis.call('EXPIRE', pend_tok_key, 86400)
end

-- 6. Cleanup
redis.call('DEL', req_map_key)

-- If headers are missing or invalid, we've done the release; exit early
if not head_rem_req or not head_rem_tok or not head_lim_req or not head_lim_tok or not head_rst_req or not head_rst_tok_delta then
    return 1
end
-- Validate header format sanity
if head_rem_req < 0 or head_rem_tok < 0 then return 1 end
if head_lim_req < 1 or head_lim_tok < 1 then return 1 end
-- Floor: must be an absolute Unix timestamp (post-2020)
if head_rst_req < 1600000000 then return 1 end
-- Ceiling: beyond year 2100. A finite-but-absurd value (e.g. 1e308) clears
-- the floor, gets stored in scientific notation (breaking the int() read in
-- get_rate_limits), and can never be walked back because rst_req only ever
-- advances via math.max below.
if head_rst_req > 4102444800 then return 1 end
-- A negative delta is malformed input. The UPPER bound is deliberately not
-- checked here: exceeding it must not discard the whole update, only the
-- token window (see the token block below).
if head_rst_tok_delta < 0 then return 1 end

-- 7. Update State (with current pending AFTER the decrement)
local curr_pend_req = tonumber(redis.call('GET', pend_req_key) or 0)
local curr_pend_tok = tonumber(redis.call('GET', pend_tok_key) or 0)

-- Request Window
-- Staleness is only meaningful against a previously OBSERVED window. When the
-- stored window is unverified (fabricated by check_and_reserve from the
-- fallback duration) the first real header must win outright: the guess sits
-- further in the future than the server's real reset roughly 5 times out of 6,
-- and would otherwise reject every genuine update.
local vrf_req = tonumber(s.vrf_req or 0)
if vrf_req == 0 or head_rst_req >= (tonumber(s.rst_req or 0) - stale_buffer) then
    s.rem_req = math.max(0, head_rem_req - curr_pend_req)
    s.lim_req = head_lim_req  -- Header is authoritative: accept increases AND decreases (tier changes)
    if vrf_req == 0 then
        -- Adopt the observed window; never math.max against a fabricated value
        s.rst_req = head_rst_req
    else
        s.rst_req = math.max(tonumber(s.rst_req or 0), head_rst_req)
    end
    s.vrf_req = 1
end

-- Token Window
local calc_rst_tok = now + head_rst_tok_delta
local vrf_tok = tonumber(s.vrf_tok or 0)

-- A delta beyond max_tok_delta means the observed token window is not what this
-- deployment is configured to expect. Do NOT clamp it into range: the script
-- would then treat a shortened window as observed, rotate early in
-- check_and_reserve, and refill rem_tok to the fallback limit before the server
-- actually refilled - i.e. over-send. Believing the window is longer only
-- over-throttles, which is recoverable.
--
-- The observed COUNTS are still applied either way: they are directly reported
-- and not in doubt, and dropping them would strand rem_tok at the fallback,
-- which over-sends immediately rather than only at rotation.
if head_rst_tok_delta > max_tok_delta then
    -- No trustworthy window, so no staleness comparison is possible. Take the
    -- lower of local and reported remaining, matching the conservative update
    -- the in-memory backend uses. rst_tok and vrf_tok are left untouched so a
    -- later in-range header is still adopted outright rather than being
    -- compared against a fabricated window.
    s.rem_tok = math.min(tonumber(s.rem_tok or head_rem_tok), math.max(0, head_rem_tok - curr_pend_tok))
    s.lim_tok = head_lim_tok
elseif vrf_tok == 0 or calc_rst_tok >= (tonumber(s.rst_tok or 0) - stale_buffer) then
    -- Not stale (with buffer) - update
    s.rem_tok = math.max(0, head_rem_tok - curr_pend_tok)
    s.lim_tok = head_lim_tok  -- Header is authoritative: accept increases AND decreases (tier changes)
    if vrf_tok == 0 then
        -- Adopt the observed window; never math.max against a fabricated value
        s.rst_tok = calc_rst_tok
    else
        s.rst_tok = math.max(tonumber(s.rst_tok or 0), calc_rst_tok)
    end
    s.vrf_tok = 1
end

local save_args = {}
for k, v in pairs(s) do table.insert(save_args, k); table.insert(save_args, v) end
redis.call('HSET', state_key, unpack(save_args))
redis.call('EXPIRE', state_key, 86400)

return 1
