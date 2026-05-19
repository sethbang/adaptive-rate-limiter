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

## Import it inside the function that uses it

The package uses `__getattr__` lazy imports so that bare `import
adaptive_rate_limiter` succeeds even without the `redis` extra. That protects
the *package* import only — it does not make the `RedisBackend` name free to
import.

`from adaptive_rate_limiter import RedisBackend` triggers that `__getattr__`,
which imports the `redis` dependency **at the point the import statement
runs**. Put it at module top level and `redis` becomes a load-time
requirement — without the extra it raises
`ImportError: 'RedisBackend' requires the 'redis' extra`.

So import `RedisBackend` inside the function that builds the backend:

```python
def build_backend():
    from adaptive_rate_limiter import RedisBackend  # redis loaded only when called
    return RedisBackend(redis_url="redis://localhost:6379/0")
```

Never import it at module top level, and never import from the submodule path
`from adaptive_rate_limiter.backends.redis import ...` — both pull `redis` in
at import time and break installs without the extra. The same applies to
`FallbackRateLimiter`, `InFlightRequest`, and `ModelLimits`.

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

## Lifecycle: start() and stop()

When a `RedisBackend` is used via `StateManager` or `create_scheduler` (the normal path), the scheduler calls `start()` and `stop()` automatically — `start()` connects to Redis and begins background orphan recovery; `stop()` halts it.

If you construct and use a `RedisBackend` directly outside a scheduler, manage the lifecycle yourself:

```python
backend = RedisBackend(redis_url="redis://localhost:6379/0")
await backend.start()   # connects + starts orphan recovery
try:
    ...
finally:
    await backend.stop()
```

Or use `async with backend:` — it calls `start()` on entry and `stop()` on exit, including on exceptions, and is the recommended pattern.

## Wiring it in

A backend reaches the scheduler through the state manager, not directly. Keep
the `RedisBackend` import inside the builder function (`create_scheduler` and
`StateManager` carry no `redis` dependency, so those stay at module top level):

```python
from adaptive_rate_limiter import create_scheduler
from adaptive_rate_limiter.scheduler import StateManager

def build_scheduler(client):
    from adaptive_rate_limiter import RedisBackend  # redis loaded only here
    backend = RedisBackend(redis_url="redis://localhost:6379/0")
    state_manager = StateManager(backend=backend)
    return create_scheduler(
        client=client, mode="intelligent", state_manager=state_manager,
    )
```
