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

## Early break needs explicit cleanup

`RateLimitedAsyncIterator` releases its reservation automatically when the
`async for` loop runs to completion, and on the error path if the stream
raises. `asyncio.CancelledError` and `GeneratorExit` are also handled
automatically — the iterator catches them, releases the reservation with a
conservative zero-refund fallback, and re-raises, so cancelling the task or
wrapping the call in `asyncio.wait_for()` / `asyncio.timeout()` is safe
**without** a `finally`/`aclose()`.

But a plain `async for ... break` stops early without triggering any of those
paths — the reservation is left unreleased and capacity leaks. The
`aclose()` pattern below is only needed for an explicit `break`.

The iterator is **not** an async context manager. To clean up on early break,
call `aclose()` yourself, in a `try`/`finally` so it runs on every exit path:

```python
stream = RateLimitedAsyncIterator(raw_iterator, ctx)
try:
    async for chunk in stream:
        if done:
            break
finally:
    await stream.aclose()   # releases the reservation (refund-based)
```

`aclose()` is the public cleanup method; it runs the refund-based release and
is safe to call even after the stream has finished normally.

## Keep reservation and refund consistent

The reserve path and the refund path must agree on token units and bucket id.
If you change how `reserved_tokens` is estimated, change extraction and refund
to match — a mismatch silently leaks or over-refunds capacity. The orphan
layer (`ReservationTracker`) reclaims reservations from streams that are
abandoned or hang, but that is a safety net, not a substitute for closing
streams properly.
