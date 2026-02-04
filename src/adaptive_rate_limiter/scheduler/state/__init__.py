# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
State Management for Adaptive Rate Limiter

Re-exports all public symbols for backward compatibility.
Use `from adaptive_rate_limiter.scheduler.state import StateManager` etc.
"""

import logging

from .cache import Cache
from .manager import StateManager
from .models import (
    PendingUpdate,
    RateLimitState,
    StateEntry,
    StateMetrics,
    StateType,
)

# Ensure package-level logger exists for backward-compatible log filtering
logging.getLogger(__name__)  # Creates: adaptive_rate_limiter.scheduler.state

__all__ = [
    # Cache
    "Cache",
    "PendingUpdate",
    "RateLimitState",
    "StateEntry",
    # Manager
    "StateManager",
    "StateMetrics",
    # Models
    "StateType",
]
