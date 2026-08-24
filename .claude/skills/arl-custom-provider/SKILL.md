---
name: arl-custom-provider
description: Implement a custom provider for the adaptive-rate-limiter library by subclassing ProviderInterface — discover_limits, parse_rate_limit_response, get_bucket_for_model, and the name property. Use this skill whenever the user wants to teach adaptive-rate-limiter about a specific API's rate limits, mentions ProviderInterface, DiscoveredBucket, RateLimitInfo, parsing rate-limit headers, or bucket-for-model mapping, or asks to write or debug a provider — even if they do not name the skill.
---

# Implementing a custom provider

A provider teaches the library how a specific API expresses its rate limits.
The library ships only the abstract `ProviderInterface` — every concrete
provider is user-supplied, then passed to `create_scheduler(provider=...)`.

## Import only from the public surface

```python
from adaptive_rate_limiter import (
    ProviderInterface, DiscoveredBucket, RateLimitInfo, ProviderError,
)
```

Anything `_`-prefixed is private. `RateLimitBucket` in the `types` package is
internal queue state — use `DiscoveredBucket` for discovery results.

## Subclass ProviderInterface

`ProviderInterface` is an `abc.ABC`. Unlike `ClientProtocol`, you **do**
subclass it, and you must implement all four abstract members or instantiation
fails:

| Member | Kind | Returns |
|---|---|---|
| `name` | sync property | `str` — unique provider id, e.g. `"openai"` |
| `discover_limits` | async method | `dict[str, DiscoveredBucket]` |
| `parse_rate_limit_response` | sync method | `RateLimitInfo` |
| `get_bucket_for_model` | async method | `str` (bucket id) |

```python
def _int(v: str | None) -> int | None:
    # Coerce through float(): some APIs render integral header values
    # float-formatted ("499.0"), which a bare int() rejects. Return None
    # rather than raising -- an unparseable header means "unknown".
    try:
        return int(float(v))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None

class MyProvider(ProviderInterface):
    @property
    def name(self) -> str:
        return "myapi"

    async def discover_limits(
        self, force_refresh: bool = False, timeout: float = 30.0,
    ) -> dict[str, DiscoveredBucket]:
        # Query the API for limits. Return {} if discovery is unsupported.
        return {
            "gpt-5": DiscoveredBucket(
                bucket_id="gpt-5", rpm_limit=500, tpm_limit=200_000,
            ),
        }

    def parse_rate_limit_response(
        self, headers: dict[str, str],
        body: dict | None = None, status_code: int | None = None,
    ) -> RateLimitInfo:
        return RateLimitInfo(
            rpm_remaining=_int(headers.get("x-ratelimit-remaining-requests")),
            tpm_remaining=_int(headers.get("x-ratelimit-remaining-tokens")),
            retry_after=_int(headers.get("retry-after")),
            is_rate_limited=status_code == 429,
        )

    async def get_bucket_for_model(
        self, model_id: str, resource_type: str | None = None,
    ) -> str:
        return model_id  # safe fallback: guarantees per-model isolation
```

## Behavior contract

- `parse_rate_limit_response` is **not called by the library** on the response
  path. The scheduler calls only `discover_limits` and `get_bucket_for_model`;
  response headers go to `Backend.update_rate_limits(headers=...)` and are
  parsed by `BaseBackend._parse_rate_limit_headers`, which normalizes reset
  values (epoch ms / epoch seconds / relative delta) to absolute Unix seconds.
  Implement this method for your own use — inspection, logging, tests, or a
  custom backend that calls it — but do not expect it to change what the
  limiter ingests.
  Match header names case-insensitively. Leave unknown fields as `None` — never
  guess. Set `is_rate_limited=True` exactly when `status_code == 429`.
- `discover_limits` should cache results unless `force_refresh=True`, and
  return `{}` (not raise) when the API has no discovery endpoint.
- `get_bucket_for_model` maps model ids to bucket ids; returning `model_id`
  itself is a sound fallback for unknown models (it guarantees isolation).
- Raise `ProviderError` (from `adaptive_rate_limiter`) when `discover_limits`,
  `parse_rate_limit_response`, or `get_bucket_for_model` cannot complete — for
  example, when the upstream discovery API returns a 4xx/5xx response. Do
  **not** raise `BackendConnectionError` from provider code; that exception is
  reserved for storage-backend failures (Redis, in-memory).

## Wiring it in

```python
scheduler = create_scheduler(
    client=MyClient(), mode="intelligent", provider=MyProvider(),
)
```
