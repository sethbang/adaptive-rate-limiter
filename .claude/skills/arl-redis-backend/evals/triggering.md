# Triggering coverage — arl-redis-backend

## Should trigger

- "Move my rate limiter to a distributed Redis backend."
- "How do I configure RedisBackend with cluster mode?"
- "I changed a Lua reservation script — what else must change?"
- "Importing my module raises ImportError about the redis extra."

## Should NOT trigger

- "Set up a scheduler and submit a request." (→ arl-integrate)
- "Write a provider that parses rate-limit headers." (→ arl-custom-provider)
- "Wrap an SSE stream for refund-based accounting." (→ arl-streaming)
- "Fix a flaky timing-dependent unit test." (unrelated)
