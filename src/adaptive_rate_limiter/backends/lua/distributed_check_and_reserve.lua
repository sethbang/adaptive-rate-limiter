-- Copyright 2026 Seth Bang
-- SPDX-License-Identifier: Apache-2.0

-- ==== BITWISE EMULATION (Redis 6.x & 7.x Compatible) ====
-- Place at TOP of script, BEFORE all other code
-- Replaces dependency on `bit` library removed in Redis 7.0
-- @BITWISE_EMULATION_START

local function bxor(a, b)
    local result = 0
    local pow = 1
    -- Only iterate while bits remain; handles a=0 or b=0 efficiently
    while a > 0 or b > 0 do
        local a_bit = a % 2
        local b_bit = b % 2
        if a_bit ~= b_bit then
            result = result + pow
        end
        a = math.floor(a / 2)
        b = math.floor(b / 2)
        pow = pow * 2
    end
    return result
end

local function calculate_jitter(req_id, micro)
    local hash = 2166136261  -- FNV offset basis (32-bit)
    if req_id then
        for i = 1, #req_id do
            hash = bxor(hash, string.byte(req_id, i))
            -- Modulo keeps hash in 32-bit unsigned range, preventing overflow
            hash = (hash * 16777619) % 4294967296
        end
    end
    -- Final XOR with microseconds, then 0-3000ms range
    return (bxor(hash, micro) % 3000) / 1000.0
end

-- @BITWISE_EMULATION_END

-- Distributed check_and_reserve.lua
-- Atomically check capacity, reserve resources, and handle window resets.
--
-- KEYS[1]: state_key - Hash storing limits, remaining, resets, generations
-- KEYS[2]: pend_req_key - String (Int) gauge of in-flight requests
-- KEYS[3]: pend_tok_key - String (Int) gauge of in-flight tokens
-- KEYS[4]: req_map_key - String storing reservation snapshot
--
-- ARGV[1]: cost_req - Number of requests to reserve
-- ARGV[2]: cost_tok - Number of tokens to reserve
-- ARGV[3]: fb_lim_req - Fallback request limit (from API or conservative default)
-- ARGV[4]: fb_lim_tok - Fallback token limit (from API or conservative default)
-- ARGV[5]: fb_win_req - Fallback request window duration in seconds (60)
-- ARGV[6]: fb_win_tok - Fallback token window duration in seconds (60)
-- ARGV[7]: req_id - Unique request ID (UUIDv4)
-- ARGV[8]: req_map_ttl - TTL for request mapping in seconds (1800)
--
-- Returns:
--   {1, 0, remaining_req, remaining_tok, gen_req, gen_tok} - Allowed
--   {0, wait_time, avail_req, avail_tok, 0, 0} - Rate Limited
--   {-1, 0, 0, 0, 0, 0} - Invalid Input
--   {-2, 0, 0, 0, 0, 0} - Collision (req_id already exists)
--   {-3, 0, lim_req, lim_tok, 0, 0} - Cost Exceeds Limit

local state_key = KEYS[1]
local pend_req_key = KEYS[2]
local pend_tok_key = KEYS[3]
local req_map_key = KEYS[4]

local cost_req = tonumber(ARGV[1]) or 0
local cost_tok = tonumber(ARGV[2]) or 0
local fb_lim_req = tonumber(ARGV[3]) or 20
local fb_lim_tok = tonumber(ARGV[4]) or 500000
local fb_win_req = tonumber(ARGV[5]) or 60
local fb_win_tok = tonumber(ARGV[6]) or 60
local req_id = ARGV[7]
local req_map_ttl = tonumber(ARGV[8]) or 1800

-- Validation (fail fast, before any writes)
if cost_req < 0 or cost_tok < 0 then return {-1, 0, 0, 0, 0, 0} end
if not req_id or req_id == "" then return {-1, 0, 0, 0, 0, 0} end
if fb_lim_req < 1 or fb_lim_tok < 1 then return {-1, 0, 0, 0, 0, 0} end
if redis.call('EXISTS', req_map_key) == 1 then return {-2, 0, 0, 0, 0, 0} end

local time = redis.call('TIME')
local now = tonumber(time[1])
local micro = tonumber(time[2])

-- Load State
local state = redis.call('HGETALL', state_key)
local s = {}
for i = 1, #state, 2 do s[state[i]] = state[i+1] end

-- Schema version check (for future migration)
if s.v and s.v ~= "1" then
    return {-1, 0, 0, 0, 0, 0}
end

-- Init Defaults (cold start)
if not s.v then
    s.v = "1"
    s.rem_req = fb_lim_req
    s.rem_tok = fb_lim_tok
    s.lim_req = fb_lim_req
    s.lim_tok = fb_lim_tok
    s.rst_req = now + fb_win_req
    s.rst_tok = now + fb_win_tok
    s.gen_req = 1
    s.gen_tok = 1
end

local state_changed = false

-- Check Request Window Rotation
if now >= tonumber(s.rst_req) then
    -- Apply new fallback limit on rotation (allows tier changes)
    s.lim_req = fb_lim_req
    s.rem_req = fb_lim_req
    s.rst_req = now + fb_win_req
    s.gen_req = tonumber(s.gen_req) + 1
    redis.call('SET', pend_req_key, 0)
    redis.call('EXPIRE', pend_req_key, 86400)
    state_changed = true
end

-- Check Token Window Rotation
if now >= tonumber(s.rst_tok) then
    -- Apply new fallback limit on rotation (allows tier changes)
    s.lim_tok = fb_lim_tok
    s.rem_tok = fb_lim_tok
    s.rst_tok = now + fb_win_tok
    s.gen_tok = tonumber(s.gen_tok) + 1
    redis.call('SET', pend_tok_key, 0)
    redis.call('EXPIRE', pend_tok_key, 86400)
    state_changed = true
end

-- Check if cost exceeds limits (returns limits so client can adapt)
if cost_req > tonumber(s.lim_req) or cost_tok > tonumber(s.lim_tok) then
    if state_changed then
        local save_args = {}
        for k, v in pairs(s) do table.insert(save_args, k); table.insert(save_args, v) end
        redis.call('HSET', state_key, unpack(save_args))
        redis.call('EXPIRE', state_key, 86400)
    end
    return {-3, 0, tonumber(s.lim_req), tonumber(s.lim_tok), 0, 0}
end

-- Calculate Available Capacity
local pend_req = tonumber(redis.call('GET', pend_req_key) or 0)
local pend_tok = tonumber(redis.call('GET', pend_tok_key) or 0)

local avail_req = tonumber(s.rem_req) - pend_req
local avail_tok = tonumber(s.rem_tok) - pend_tok

if avail_req >= cost_req and avail_tok >= cost_tok then
    -- Reserve capacity
    redis.call('INCRBY', pend_req_key, cost_req)
    redis.call('INCRBY', pend_tok_key, cost_tok)
    redis.call('EXPIRE', pend_req_key, 86400)
    redis.call('EXPIRE', pend_tok_key, 86400)

    -- Store mapping for later release/update
    local map_val = s.gen_req .. ":" .. s.gen_tok .. ":" .. cost_req .. ":" .. cost_tok
    redis.call('SETEX', req_map_key, req_map_ttl, map_val)

    -- Save state
    local save_args = {}
    for k, v in pairs(s) do table.insert(save_args, k); table.insert(save_args, v) end
    redis.call('HSET', state_key, unpack(save_args))
    redis.call('EXPIRE', state_key, 86400)

    return {1, 0, avail_req - cost_req, avail_tok - cost_tok, tonumber(s.gen_req), tonumber(s.gen_tok)}
else
    -- Rate limited - save state if changed
    if state_changed then
        local save_args = {}
        for k, v in pairs(s) do table.insert(save_args, k); table.insert(save_args, v) end
        redis.call('HSET', state_key, unpack(save_args))
        redis.call('EXPIRE', state_key, 86400)
    end

    -- Calculate wait time until capacity available
    local wait_req = (avail_req < cost_req) and (tonumber(s.rst_req) - now) or 0
    local wait_tok = (avail_tok < cost_tok) and (tonumber(s.rst_tok) - now) or 0

    -- @JITTER_CALCULATION
    -- Use emulated jitter calculation (Redis 6.x & 7.x compatible)
    local jitter = calculate_jitter(req_id, micro)
    -- @END_JITTER_CALCULATION

    local wait_final = math.max(wait_req, wait_tok)
    if wait_final > 0 then wait_final = wait_final + jitter end

    return {0, wait_final, avail_req, avail_tok, 0, 0}
end
