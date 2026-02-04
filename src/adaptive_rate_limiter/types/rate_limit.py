# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
Rate limit types and configurations.

This module defines the rate limit type enums, bucket configurations,
and result types for rate limit checks.
"""

from dataclasses import dataclass
from enum import Enum


class RateLimitType(Enum):
    """
    Types of rate limits enforced by an API.

    The API may implement multiple types of rate limiting to ensure fair
    usage and system stability. Each limit type has different characteristics
    and reset behaviors that affect queuing and scheduling decisions.

    Limit Types:
        * **RPM**: Requests Per Minute - Short-term burst protection
        * **RPD**: Requests Per Day - Long-term usage quotas
        * **TPM**: Tokens Per Minute - Content-based limiting for text operations

    Reset Behaviors:
        * RPM limits reset every minute (sliding or fixed window)
        * RPD limits reset daily (typically at UTC midnight)
        * TPM limits reset every minute and are model-specific
    """

    RPM = "RPM"  # Requests Per Minute
    RPD = "RPD"  # Requests Per Day
    TPM = "TPM"  # Tokens Per Minute


@dataclass
class RateLimitBucket:
    """
    Configuration for rate limits on a specific model/resource.

    Contains the rate limit parameters for a particular model and resource
    type combination. Used by the queue system to determine request capacity.

    Attributes:
        model_id: The model identifier this bucket applies to
        resource_type: Classification of the resource (string)
        rpm_limit: Requests per minute limit
        rpd_limit: Optional requests per day limit
        tpm_limit: Optional tokens per minute limit (for text resources)
    """

    model_id: str
    resource_type: str  # Uses string instead of Enum
    rpm_limit: int
    rpd_limit: int | None = None
    tpm_limit: int | None = None  # Tokens per minute for text resources


@dataclass
class LimitCheckResult:
    """
    Result of checking rate limits for a request.

    Contains information about whether a request can proceed and, if not,
    how long it should wait and why.

    Attributes:
        can_proceed: Whether the request can proceed immediately
        wait_time: Suggested wait time in seconds if request cannot proceed
        reason: Human-readable reason if request cannot proceed
        limiting_factor: Which rate limit type is preventing the request
        remaining_requests: Number of requests remaining in current window
        remaining_tokens: Number of tokens remaining in current window
    """

    can_proceed: bool
    wait_time: float = 0.0
    reason: str | None = None
    limiting_factor: RateLimitType | None = None
    remaining_requests: int | None = None
    remaining_tokens: int | None = None


__all__ = [
    "LimitCheckResult",
    "RateLimitBucket",
    "RateLimitType",
]
