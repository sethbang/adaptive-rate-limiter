# Triggering coverage — arl-custom-provider

## Should trigger

- "Write a provider that parses rate-limit headers for adaptive-rate-limiter."
- "How do I implement ProviderInterface?"
- "My discover_limits method — what should it return?"
- "Map model IDs to rate-limit buckets in my provider."

## Should NOT trigger

- "How do I set up a Scheduler and submit requests?" (→ arl-integrate)
- "Wrap my SSE stream for refund-based token accounting." (→ arl-streaming)
- "Configure RedisBackend for a distributed deployment." (→ arl-redis-backend)
- "Bump the package version in pyproject.toml." (unrelated)
