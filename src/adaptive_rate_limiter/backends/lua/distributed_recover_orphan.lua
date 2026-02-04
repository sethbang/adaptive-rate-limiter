-- Copyright 2026 Seth Bang
-- SPDX-License-Identifier: Apache-2.0

-- Distributed recover_orphan.lua
-- Recover capacity from orphaned pending reservations when a request mapping has expired
-- but the pending gauge was never decremented. This is called by a background reconciliation process.
--
-- KEYS[1]: pend_req_key - String (Int) gauge of in-flight requests
-- KEYS[2]: pend_tok_key - String (Int) gauge of in-flight tokens
-- KEYS[3]: state_key - Hash storing limits, remaining, resets, generations
--
-- ARGV[1]: cost_req - The request cost to recover
-- ARGV[2]: cost_tok - The token cost to recover
-- ARGV[3]: expected_gen_req - The generation when the reservation was made
-- ARGV[4]: expected_gen_tok - The generation when the reservation was made
--
-- Returns:
--   1 - Success

local pend_req_key = KEYS[1]
local pend_tok_key = KEYS[2]
local state_key = KEYS[3]

local cost_req = tonumber(ARGV[1]) or 0
local cost_tok = tonumber(ARGV[2]) or 0
local expected_gen_req = tonumber(ARGV[3]) or 0
local expected_gen_tok = tonumber(ARGV[4]) or 0

-- 1. Load State (for generation check)
local state = redis.call('HGETALL', state_key)
local s = {}
for i = 1, #state, 2 do s[state[i]] = state[i+1] end

-- 2. Decrement pending for requests (only if generation matches)
-- If generations differ, the window has rotated and pending was already reset—skip decrement.
if cost_req > 0 and expected_gen_req == tonumber(s.gen_req or 0) then
    local new_p = redis.call('DECRBY', pend_req_key, cost_req)
    if new_p < 0 then redis.call('SET', pend_req_key, 0) end
    redis.call('EXPIRE', pend_req_key, 86400)
end

-- 3. Decrement pending for tokens (only if generation matches)
if cost_tok > 0 and expected_gen_tok == tonumber(s.gen_tok or 0) then
    local new_p = redis.call('DECRBY', pend_tok_key, cost_tok)
    if new_p < 0 then redis.call('SET', pend_tok_key, 0) end
    redis.call('EXPIRE', pend_tok_key, 86400)
end

return 1
