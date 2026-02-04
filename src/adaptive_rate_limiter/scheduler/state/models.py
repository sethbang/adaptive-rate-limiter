# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
State models for Adaptive Rate Limiter.

Contains all data models, enums, and dataclasses for state management.
"""

import contextlib
import logging
import random  # Used in PendingUpdate.get_backoff_delay()
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    # Forward reference for type hints only - no runtime import
    pass

logger = logging.getLogger(__name__)  # adaptive_rate_limiter.scheduler.state.models


class StateType(Enum):
    """Types of state entries."""

    RATE_LIMIT = "rate_limit"
    RESERVATION = "reservation"
    MODEL_CONFIG = "model_config"
    BUCKET_INFO = "bucket_info"
    ACCOUNT_LIMIT = "account_limit"


@dataclass
class StateMetrics:
    """State management metrics."""

    cache_hits: int = 0
    cache_misses: int = 0
    cache_evictions: int = 0
    backend_writes: int = 0
    backend_reads: int = 0
    version_conflicts: int = 0
    bulk_operations: int = 0
    flush_retries: int = 0
    flush_drops: int = 0

    @property
    def hit_ratio(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class StateEntry(BaseModel):
    """
    Unified state entry model using Pydantic for validation.

    Consolidates different state formats from scheduler and account systems.
    """

    key: str
    data: dict[str, Any]
    state_type: StateType = StateType.RATE_LIMIT
    version: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    metadata: dict[str, Any] | None = None
    namespace: str = "default"

    @model_validator(mode="after")
    def _validate_expiration(self) -> "StateEntry":
        """Validate that expires_at is after created_at."""
        if self.expires_at and self.expires_at <= self.created_at:
            raise ValueError("expires_at must be after created_at")
        return self

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return (
            self.expires_at is not None
            and datetime.now(timezone.utc) >= self.expires_at
        )

    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()

    def update_data(self, new_data: dict[str, Any]) -> None:
        """Update data and increment version."""
        self.data.update(new_data)
        self.version += 1
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for backend storage."""
        return {
            "key": self.key,
            "data": self.data,
            "state_type": self.state_type.value,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
            "namespace": self.namespace,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateEntry":
        """Create StateEntry from dict."""
        return cls(
            key=data["key"],
            data=data["data"],
            state_type=StateType(data.get("state_type", "rate_limit")),
            version=data.get("version", 1),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if data.get("updated_at")
            else datetime.now(timezone.utc),
            expires_at=datetime.fromisoformat(data["expires_at"])
            if data.get("expires_at")
            else None,
            metadata=data.get("metadata"),
            namespace=data.get("namespace", "default"),
        )


@dataclass
class PendingUpdate:
    """Wrapper for pending state updates with retry tracking.

    Tracks retry attempts to prevent infinite retry loops when backend
    writes fail persistently.
    """

    entry: "StateEntry"
    retry_count: int = 0
    last_attempt_time: float = 0.0

    def should_retry(self, max_retries: int) -> bool:
        """Check if this update should be retried."""
        return self.retry_count < max_retries

    def get_backoff_delay(
        self, base_delay: float = 1.0, max_delay: float = 60.0
    ) -> float:
        """Calculate exponential backoff delay for this update.

        Uses exponential backoff with jitter: base_delay * 2^retry_count + jitter
        """
        delay = min(base_delay * (2**self.retry_count), max_delay)
        # Add up to 25% jitter to prevent thundering herd
        jitter: float = delay * 0.25 * random.random()  # noqa: S311  # nosec B311
        return float(delay + jitter)


class RateLimitState(BaseModel):
    """
    Unified rate limit state model.

    Tracks rate limit information including remaining requests/tokens,
    reset times, and verification status.
    """

    model_id: str
    remaining_requests: int | None = None
    remaining_requests_daily: int | None = None
    remaining_tokens: int | None = None
    reset_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    reset_at_daily: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_request_time: datetime | None = None

    # Limit values
    request_limit: int | None = None
    token_limit: int | None = None
    bucket_id: str | None = None

    # Cold start verification
    is_verified: bool = False

    @property
    def is_exhausted(self) -> bool:
        """Check if any rate limit is exhausted."""
        return (
            self.remaining_requests is not None and self.remaining_requests <= 0
        ) or (self.remaining_tokens is not None and self.remaining_tokens <= 0)

    @property
    def time_until_reset(self) -> float:
        """Time in seconds until next reset."""
        now = datetime.now(timezone.utc)
        times = []

        if self.reset_at > now:
            times.append((self.reset_at - now).total_seconds())
        if self.reset_at_daily > now:
            times.append((self.reset_at_daily - now).total_seconds())

        return min(times) if times else 0.0

    @property
    def usage_percentage(self) -> float:
        """Calculate usage percentage for requests."""
        if self.request_limit is None or self.remaining_requests is None:
            return 0.0
        used = self.request_limit - self.remaining_requests
        return (used / self.request_limit) * 100.0 if self.request_limit > 0 else 0.0

    def update_from_headers(self, headers: dict[str, str]) -> None:
        """
        Update state from API response headers with safety checks.

        Implements conservative update logic: if local state is lower than server state
        (due to in-flight requests), we keep the local state to prevent 429s.
        """
        # Parse headers
        new_requests = None
        if "x-ratelimit-remaining-requests" in headers:
            with contextlib.suppress(ValueError, TypeError):
                new_requests = int(headers["x-ratelimit-remaining-requests"])

        new_tokens = None
        if "x-ratelimit-remaining-tokens" in headers:
            with contextlib.suppress(ValueError, TypeError):
                new_tokens = int(headers["x-ratelimit-remaining-tokens"])

        # Parse limits
        if "x-ratelimit-limit-requests" in headers:
            with contextlib.suppress(ValueError, TypeError):
                self.request_limit = int(headers["x-ratelimit-limit-requests"])

        if "x-ratelimit-limit-tokens" in headers:
            with contextlib.suppress(ValueError, TypeError):
                self.token_limit = int(headers["x-ratelimit-limit-tokens"])

        # Update reset time
        if "x-ratelimit-reset-requests" in headers:
            with contextlib.suppress(ValueError, TypeError):
                self.reset_at = datetime.fromtimestamp(
                    float(headers["x-ratelimit-reset-requests"]), tz=timezone.utc
                )

        # Apply conservative update logic
        used_local_value = False

        if new_requests is not None:
            local_exhausted = (
                self.remaining_requests is None or self.remaining_requests <= 0
            )
            server_at_limit = (
                self.request_limit is not None
                and new_requests >= self.request_limit * 0.9
            )

            if not self.is_verified or local_exhausted or server_at_limit:
                self.remaining_requests = new_requests
            elif (
                self.remaining_requests is not None
                and self.remaining_requests < new_requests
            ):
                used_local_value = True
            else:
                self.remaining_requests = new_requests

        if new_tokens is not None:
            local_exhausted = (
                self.remaining_tokens is None or self.remaining_tokens <= 0
            )
            server_at_limit = (
                self.token_limit is not None and new_tokens >= self.token_limit * 0.9
            )

            if not self.is_verified or local_exhausted or server_at_limit:
                self.remaining_tokens = new_tokens
            elif (
                self.remaining_tokens is not None and self.remaining_tokens < new_tokens
            ):
                used_local_value = True
            else:
                self.remaining_tokens = new_tokens

        if not used_local_value or self.last_updated is None:
            self.last_updated = datetime.now(timezone.utc)

        self.is_verified = True

    @classmethod
    def create_fallback_state(
        cls, model_id: str, bucket_id: str | None = None
    ) -> "RateLimitState":
        """Create conservative fallback state."""
        current_time = datetime.now(timezone.utc)
        return cls(
            model_id=model_id,
            remaining_requests=30,  # Conservative default
            remaining_requests_daily=500,
            remaining_tokens=5000,
            reset_at=current_time + timedelta(minutes=1),
            reset_at_daily=current_time + timedelta(days=1),
            last_updated=current_time,
            bucket_id=bucket_id,
        )


__all__ = [
    "PendingUpdate",
    "RateLimitState",
    "StateEntry",
    "StateMetrics",
    "StateType",
]
