---
name: arl-streaming
description: Use adaptive-rate-limiter's refund-based token accounting for streaming responses — wrap a raw async/SSE stream with RateLimitedAsyncIterator and a StreamingReservationContext so unused token capacity is refunded on completion. Use this skill whenever the user is streaming responses through adaptive-rate-limiter, mentions StreamingReservationContext, RateLimitedAsyncIterator, token refunds, streaming reservations, or capacity leaking on early break — even if they do not name the skill.
---

# Streaming with refund-based token accounting

Streaming responses cannot be token-counted up front, so the library reserves
an optimistic token estimate, then refunds the unused portion when the stream
finishes. Your job: wrap the raw stream so that accounting happens.

## The two pieces

```python
from adaptive_rate_limiter import (
    StreamingReservationContext, RateLimitedAsyncIterator,
)
```

- `StreamingReservationContext` — holds one stream's reservation
  (`reserved_tokens`) and the `backend` used to release capacity.
- `RateLimitedAsyncIterator` — wraps the raw SSE/async iterator, extracts the
  real token count from the final chunk's usage, and releases the reservation
  with a refund when iteration ends.

## Wrap the raw stream

```python
ctx = StreamingReservationContext(
    reservation_id="res-1",
    bucket_id="gpt-5",
    request_id="req-1",
    reserved_tokens=4000,   # optimistic up-front estimate
    backend=backend,        # the scheduler's backend instance
)
stream = RateLimitedAsyncIterator(raw_iterator, ctx)

async for chunk in stream:
    handle(chunk)
# Stream exhausted: actual tokens extracted, unused capacity refunded.
```

The wrapper is transparent — it yields exactly the chunks the inner iterator
yields. The accounting is a side effect.

## Early break needs the context manager

A plain `async for ... break` leaves the inner stream open and the reservation
unreleased. To break early, enter the iterator as an async context manager —
`__aexit__` calls `aclose()`, which runs the refund:

```python
async with stream:
    async for chunk in stream:
        if done:
            break   # __aexit__ -> aclose() -> reservation released
```

## Keep reservation and refund consistent

The reserve path and the refund path must agree on token units and bucket id.
If you change how `reserved_tokens` is estimated, change extraction and refund
to match — a mismatch silently leaks or over-refunds capacity. The orphan
layer (`ReservationTracker`) reclaims reservations from streams that are
abandoned or hang, but that is a safety net, not a substitute for closing
streams properly.
