# Triggering coverage — arl-integrate

## Should trigger

- "How do I set up a Scheduler with adaptive-rate-limiter?"
- "Wire adaptive-rate-limiter into my FastAPI service."
- "What does create_scheduler do and what mode should I pass?"
- "How do I implement ClientProtocol for this rate limiter?"

## Should NOT trigger

- "Write a custom provider that parses rate-limit headers." (→ arl-custom-provider)
- "How do I wrap an SSE stream for token accounting?" (→ arl-streaming)
- "Switch my backend to Redis for distributed rate limiting." (→ arl-redis-backend)
- "Add a new pytest fixture for the cleanup-loop tests." (unrelated)
