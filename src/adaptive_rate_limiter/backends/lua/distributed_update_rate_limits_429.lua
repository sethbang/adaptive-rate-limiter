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
-- ARGV[5]: head_rst_req - x-ratelimit-reset-requests (Unix timestamp)
-- ARGV[6]: head_rst_tok_delta - x-ratelimit-reset-tokens (seconds until reset)
-- ARGV[7]: stale_buffer - Buffer for stale detection (default 10)
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
if head_rst_req < 1600000000 then return 1 end
if head_rst_tok_delta < 0 or head_rst_tok_delta > max_tok_delta then return 1 end  -- Delta must be reasonable (0-2min for per-minute windows)

-- 7. Update State (with current pending AFTER the decrement)
local curr_pend_req = tonumber(redis.call('GET', pend_req_key) or 0)
local curr_pend_tok = tonumber(redis.call('GET', pend_tok_key) or 0)

-- Request Window
if head_rst_req >= (tonumber(s.rst_req or 0) - stale_buffer) then
    s.rem_req = math.max(0, head_rem_req - curr_pend_req)
    s.lim_req = math.max(tonumber(s.lim_req or 0), head_lim_req)
    s.rst_req = math.max(tonumber(s.rst_req or 0), head_rst_req)
end

-- Token Window
local calc_rst_tok = now + head_rst_tok_delta
if calc_rst_tok >= (tonumber(s.rst_tok or 0) - stale_buffer) then
    s.rem_tok = math.max(0, head_rem_tok - curr_pend_tok)
    s.lim_tok = math.max(tonumber(s.lim_tok or 0), head_lim_tok)
    s.rst_tok = math.max(tonumber(s.rst_tok or 0), calc_rst_tok)
end

local save_args = {}
for k, v in pairs(s) do table.insert(save_args, k); table.insert(save_args, v) end
redis.call('HSET', state_key, unpack(save_args))
redis.call('EXPIRE', state_key, 86400)

return 1
