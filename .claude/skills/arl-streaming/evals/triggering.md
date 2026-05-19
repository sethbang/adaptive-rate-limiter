# Triggering coverage — arl-streaming

## Should trigger

- "How do I wrap an SSE stream for token accounting in adaptive-rate-limiter?"
- "My streaming reservation isn't being refunded — why?"
- "What is RateLimitedAsyncIterator for?"
- "Capacity leaks when I break out of the stream early."

## Should NOT trigger

- "Set up a scheduler and submit a non-streaming request." (→ arl-integrate)
- "Implement ProviderInterface for my API." (→ arl-custom-provider)
- "Configure RedisBackend with cluster mode." (→ arl-redis-backend)
- "Run the test suite with coverage." (unrelated)
