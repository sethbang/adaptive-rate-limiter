-- Copyright 2026 Seth Bang
-- SPDX-License-Identifier: Apache-2.0

-- Distributed release_streaming.lua
-- Handle streaming completion with REFUND-BASED accounting.
--
-- This script is used when a streaming request completes successfully.
-- Unlike non-streaming requests which sync with server headers, streaming
-- uses refund-based accounting because headers arrive at stream START,
-- not completion.
--
-- REFUND-BASED ACCOUNTING:
--   At stream completion, we know the actual tokens consumed vs reserved.
--   refund = reserved_tokens - actual_tokens
--   rem_tok = current_rem + refund
--
-- CLAMPING (Critical):
--   For pure streaming workloads (no header syncs), refund can push rem_tok > limit.
--   For over-consumption (actual > reserved), negative refund can push rem_tok < 0.
--   Both cases must be clamped to prevent corruption.
--
-- GENERATION CHECKING:
--   Streaming requests can span window rotations (60-second windows).
--   If the window has rotated since reservation:
--   - Generation will mismatch
--   - Skip pending decrement AND skip refund
--   - Window rotation already reset pending to 0 and rem_tok to limit
--
-- KEYS[1]: state_key - Hash storing limits, remaining, resets, generations
-- KEYS[2]: pend_req_key - String (Int) gauge of in-flight requests
-- KEYS[3]: pend_tok_key - String (Int) gauge of in-flight tokens
-- KEYS[4]: req_map_key - String storing reservation snapshot
--
-- ARGV[1]: reserved_tokens - Tokens that were reserved at stream start
-- ARGV[2]: actual_tokens - Actual tokens consumed (from stream completion)
--
-- Returns:
--   1 - Success (released, or already cleaned up - idempotent)

local state_key = KEYS[1]
local pend_req_key = KEYS[2]
local pend_tok_key = KEYS[3]
local req_map_key = KEYS[4]

local reserved_tokens = tonumber(ARGV[1]) or 0
local actual_tokens = tonumber(ARGV[2]) or 0

-- Validate inputs (treat negative as 0)
if reserved_tokens < 0 then reserved_tokens = 0 end
if actual_tokens < 0 then actual_tokens = 0 end

-- 1. Get Request Mapping
-- If mapping doesn't exist, this request was already cleaned up (idempotent success)
local map_val = redis.call('GET', req_map_key)
if not map_val then return 1 end

-- 2. Parse Mapping
-- Format: "gen_req:gen_tok:cost_req:cost_tok"
local parts = {}
for part in string.gmatch(map_val, "([^:]+)") do table.insert(parts, part) end
local map_gen_req = tonumber(parts[1])
local map_gen_tok = tonumber(parts[2])
local cost_req = tonumber(parts[3])
local cost_tok = tonumber(parts[4])

-- 3. Load Current State
local state = redis.call('HGETALL', state_key)
local s = {}
for i = 1, #state, 2 do s[state[i]] = state[i+1] end

local current_gen_req = tonumber(s.gen_req or 0)
local current_gen_tok = tonumber(s.gen_tok or 0)
local current_rem_tok = tonumber(s.rem_tok or 0)
local current_limit = tonumber(s.lim_tok or 0)

-- 4. Decrement Pending (CONDITIONAL on generation match)
-- Only decrement if the mapping's generation matches the current window generation.
-- If generation mismatches, the window has rotated and pending was already reset to 0.

-- Request pending
if map_gen_req == current_gen_req then
    local new_p = redis.call('DECRBY', pend_req_key, cost_req)
    if new_p < 0 then redis.call('SET', pend_req_key, 0) end
    redis.call('EXPIRE', pend_req_key, 86400)
end

-- Token pending
if map_gen_tok == current_gen_tok then
    local new_p = redis.call('DECRBY', pend_tok_key, cost_tok)
    if new_p < 0 then redis.call('SET', pend_tok_key, 0) end
    redis.call('EXPIRE', pend_tok_key, 86400)
end

-- 5. Apply Refund (CONDITIONAL on generation match)
-- Only apply refund if we're still in the same token window.
-- If generation mismatches, skip - the window rotated and rem_tok was reset.
if map_gen_tok == current_gen_tok then
    -- Calculate refund: positive if actual < reserved, negative if actual > reserved
    local refund = reserved_tokens - actual_tokens
    local new_rem = current_rem_tok + refund

    -- CRITICAL: Clamp to [0, limit]
    -- This prevents state corruption in edge cases:
    --   - Pure streaming (no header syncs): refund could push rem_tok > limit
    --   - Over-consumption: negative refund could push rem_tok < 0

    if new_rem < 0 then
        -- Over-consumption: actual > reserved
        -- Track for observability before clamping to 0
        local over_consumption = -new_rem
        redis.call('HINCRBY', state_key, 'over_consumption_tokens', over_consumption)
        new_rem = 0
    elseif current_limit > 0 and new_rem > current_limit then
        -- Would exceed limit (pure streaming case)
        -- Clamp to limit to prevent artificial capacity inflation
        new_rem = current_limit
    end

    -- Update rem_tok
    redis.call('HSET', state_key, 'rem_tok', new_rem)
    redis.call('EXPIRE', state_key, 86400)
end

-- 6. Cleanup: Delete request mapping
redis.call('DEL', req_map_key)

return 1
