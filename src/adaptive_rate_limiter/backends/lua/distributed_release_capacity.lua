-- Copyright 2026 Seth Bang
-- SPDX-License-Identifier: Apache-2.0

-- Distributed release_capacity.lua
-- Handle client failures—release reserved capacity without server headers.
-- Used for: Timeouts, client errors (pre-send), 4xx errors (non-429)
--
-- KEYS[1]: state_key - Hash storing limits, remaining, resets, generations
-- KEYS[2]: pend_req_key - String (Int) gauge of in-flight requests
-- KEYS[3]: pend_tok_key - String (Int) gauge of in-flight tokens
-- KEYS[4]: req_map_key - String storing reservation snapshot
--
-- Returns:
--   1 - Success (released or already cleaned up)
--   0 - Mapping not found (but this is still idempotent success)

local state_key = KEYS[1]
local pend_req_key = KEYS[2]
local pend_tok_key = KEYS[3]
local req_map_key = KEYS[4]

-- 1. Get Mapping
local map_val = redis.call('GET', req_map_key)
if not map_val then return 1 end  -- Already cleaned up, idempotent success

-- 2. Parse Mapping
local parts = {}
for part in string.gmatch(map_val, "([^:]+)") do table.insert(parts, part) end
local map_gen_req = tonumber(parts[1])
local map_gen_tok = tonumber(parts[2])
local cost_req = tonumber(parts[3])
local cost_tok = tonumber(parts[4])

-- 3. Load State (for generation check)
local state = redis.call('HGETALL', state_key)
local s = {}
for i = 1, #state, 2 do s[state[i]] = state[i+1] end

-- 4. Decrement Pending (with generation check)
-- Only decrement if the mapping's generation matches the current window generation.
-- This prevents corrupting the pending gauge after window rotation.
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

-- 5. Cleanup
redis.call('DEL', req_map_key)
return 1
