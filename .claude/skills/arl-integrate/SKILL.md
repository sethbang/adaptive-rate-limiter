---
name: arl-integrate
description: Integrate the adaptive-rate-limiter library into an application — implement ClientProtocol, build a scheduler with create_scheduler(), choose a scheduling mode (basic/intelligent/account), and submit requests. Use this skill whenever the user is wiring adaptive-rate-limiter into their app, asks how to set up a Scheduler, mentions create_scheduler, RateLimiterConfig, submit_request, RequestMetadata, or SchedulerMode, or wants to rate-limit API calls with this library — even if they do not name the skill.
---

# Integrating adaptive-rate-limiter

Wire the library into an app that calls a rate-limited API. The shape is:
implement a client, create a scheduler, submit requests through it.

## Import only from the public surface

Public names come from `adaptive_rate_limiter` and
`adaptive_rate_limiter.scheduler`. Any `_`-prefixed module, class, or function
is private and may change between any versions — never import or rely on those.

## Step 1 — Implement ClientProtocol

`ClientProtocol` is a `runtime_checkable` `typing.Protocol`. Satisfy it by
structural typing (matching the members) — you do not need to subclass it.
Three members are required:

```python
from adaptive_rate_limiter import ClientProtocol

class MyClient:  # structurally satisfies ClientProtocol
    @property
    def base_url(self) -> str:
        return "https://api.example.com"

    @property
    def timeout(self) -> float:
        return 30.0

    def get_headers(self) -> dict[str, str]:
        return {"Authorization": "Bearer ..."}
```

The core library does not make HTTP calls itself — your `request_func` does.
`ClientProtocol` only supplies identification and timeout for scheduling.

## Step 2 — Create the scheduler with the factory

Always use `create_scheduler(...)`. Do not instantiate `Scheduler` directly —
the factory wires dependencies and applies mode defaults.

```python
from adaptive_rate_limiter import create_scheduler
from adaptive_rate_limiter.scheduler import RateLimiterConfig

scheduler = create_scheduler(client=MyClient(), mode="intelligent")
```

`mode` is one of `"basic"`, `"intelligent"`, `"account"` (case-insensitive).
If omitted it falls back to `config.mode`, defaulting to `"intelligent"`.

## Choosing a mode

- `basic` — pass-through with retry. No provider or classifier needed. Use for
  the simplest case, or when an upstream system already manages limits.
- `intelligent` — the production path: capacity reservation, queue management,
  streaming refunds. For full function, pass a `provider` (see the
  arl-custom-provider skill) and a `classifier`.
- `account` — multi-tenant, per-account tracking. Use when one process serves
  many billing accounts, each with its own quota.

Adding a brand-new mode is a library change, not an integration task — it
means a new mode-strategy class wired in `BaseScheduler._setup_mode_strategy`,
not editing `Scheduler`.

## Step 3 — Submit requests

`Scheduler` is an async context manager; enter it before submitting.

```python
from adaptive_rate_limiter import RequestMetadata, TEXT

async with scheduler:
    metadata = RequestMetadata(
        request_id="req-1",
        model_id="gpt-5",
        resource_type=TEXT,
        estimated_tokens=500,
    )
    result = await scheduler.submit_request(metadata, request_func)
```

`request_func` is a zero-argument async callable that performs the actual API
call. `submit_request`'s return type is **mode-dependent** — `basic` returns
the call result directly; `intelligent`/`account` may return a `ScheduleResult`.
Check the return type for the mode you chose rather than assuming.

## Configuration

`RateLimiterConfig` is a dataclass with `__post_init__` validation. Build it
explicitly when you need non-default behavior and pass it as `config=` to
`create_scheduler`.
