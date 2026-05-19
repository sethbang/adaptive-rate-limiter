---
name: arl-redis-backend
description: Use the Redis distributed backend of adaptive-rate-limiter — install the redis extra, construct RedisBackend, wire it through StateManager, and preserve the lazy-import boundary and atomic Lua reservation scripts. Use this skill whenever the user wants distributed or multi-process rate limiting, mentions RedisBackend, Redis cluster mode, the lua reservation scripts, an ImportError about the redis extra, or moving off MemoryBackend — even if they do not name the skill.
---

# Using the Redis distributed backend

`RedisBackend` shares rate-limit state across processes and hosts. It is
optional — gated behind the `redis` extra — so the import path matters.

## Install the extra

```bash
pip install adaptive-rate-limiter[redis]   # or: uv sync --extra redis
```

## Import it lazily — never eagerly

`RedisBackend` is exposed through `__getattr__` lazy imports in both
`adaptive_rate_limiter/__init__.py` and
`adaptive_rate_limiter/backends/__init__.py`, so the package imports cleanly
for users who never installed `redis`.

Preserve that. Import it at the point of use:

```python
from adaptive_rate_limiter import RedisBackend   # resolved lazily — OK
```

Do **not** add an eager top-level `from .backends import RedisBackend` to any
module — that drags `redis` into every import and breaks installs without the
extra. The same applies to `FallbackRateLimiter`, `InFlightRequest`, and
`ModelLimits`.

## Construct it

```python
backend = RedisBackend(
    redis_url="redis://localhost:6379/0",
    namespace="rate_limiter",   # key prefix
    account_id="default",       # tenant scope
)
```

Pass either `redis_url` or a pre-built `redis_client`. For production
multi-key atomicity, run Redis in cluster mode and pass `cluster_mode=True`
with `cluster_url=` (env var fallback: `REDIS_CLUSTER_URL`).

## Atomic operations live in Lua

All check-and-reserve, release, and orphan-recovery operations are atomic Lua
scripts in `src/adaptive_rate_limiter/backends/lua/*.lua`. The scripts ship
inside the wheel (see the hatch sdist include in `pyproject.toml`).

When you change reservation semantics, the Lua script and its Python caller
must move together — they share a contract (key layout, argument order, return
shape). Changing one side alone corrupts state. After such a change, run the
`requires_redis_pool` and `cluster` marked tests against a real Redis;
`fakeredis` does not exercise true multi-key atomicity. See
`docker-compose.redis-cluster.yml` for a local cluster.

## Wiring it in

A backend reaches the scheduler through the state manager, not directly:

```python
from adaptive_rate_limiter import RedisBackend, create_scheduler
from adaptive_rate_limiter.scheduler import StateManager

backend = RedisBackend(redis_url="redis://localhost:6379/0")
state_manager = StateManager(backend=backend)
scheduler = create_scheduler(
    client=MyClient(), mode="intelligent", state_manager=state_manager,
)
```
