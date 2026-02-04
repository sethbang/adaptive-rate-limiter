# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
Scheduler for managing rate limit checks and reservations.

This module provides:
- RateLimiterConfig: Configuration for scheduler behavior
- StateConfig: Configuration for state management
- StateManager: Unified state management with caching
- RateLimitState: Rate limit state model
- PendingUpdate: Wrapper for pending state updates with retry tracking
"""

from .base import BaseScheduler, SchedulingStrategy
from .config import (
    CachePolicy,
    RateLimiterConfig,
    SchedulerMode,
    StateConfig,
)
from .scheduler import Scheduler, create_scheduler
from .state import (
    Cache,
    PendingUpdate,
    RateLimitState,
    StateEntry,
    StateManager,
    StateMetrics,
    StateType,
)

__all__ = [
    # Base
    "BaseScheduler",
    "Cache",
    "CachePolicy",
    "PendingUpdate",
    "RateLimitState",
    # Config
    "RateLimiterConfig",
    # Scheduler
    "Scheduler",
    "SchedulerMode",
    "SchedulingStrategy",
    "StateConfig",
    "StateEntry",
    "StateManager",
    "StateMetrics",
    # State
    "StateType",
    "create_scheduler",
]
