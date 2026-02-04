# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
Scheduling mode strategies for the adaptive rate limiter.

This package contains the mode strategy implementations that define how
requests are scheduled and processed. Each mode provides different trade-offs
between simplicity, performance, and rate limit compliance.

Available Modes:
    - BASIC: Simple direct execution with retry logic
    - INTELLIGENT: Advanced queuing with bucket-based scheduling (requires dependencies)
    - ACCOUNT: Account-level request management

The base class `BaseSchedulingModeStrategy` defines the interface that all
mode implementations must follow.
"""

from typing import TYPE_CHECKING, Any, Optional

from .account import AccountModeStrategy
from .base import BaseSchedulingModeStrategy
from .basic import BasicModeStrategy
from .intelligent import IntelligentModeStrategy

if TYPE_CHECKING:
    from ...protocols.classifier import ClassifierProtocol
    from ...protocols.client import ClientProtocol
    from ...providers.base import ProviderInterface
    from ...scheduler.config import RateLimiterConfig
    from ...scheduler.state import StateManager
    from ...strategies.scheduling import BaseSchedulingStrategy


def create_mode_strategy(
    mode: str,
    scheduler: Any,
    config: "RateLimiterConfig",
    client: "ClientProtocol",
    provider: Optional["ProviderInterface"] = None,
    classifier: Optional["ClassifierProtocol"] = None,
    state_manager: Optional["StateManager"] = None,
    scheduling_strategy: Optional["BaseSchedulingStrategy"] = None,
) -> BaseSchedulingModeStrategy:
    """
    Factory function to create the appropriate mode strategy.

    Args:
        mode: Mode name ("basic", "intelligent", "account")
        scheduler: Main scheduler instance
        config: Scheduler configuration
        client: Client instance
        provider: Provider interface (required for intelligent mode)
        classifier: Request classifier (required for intelligent mode)
        state_manager: State manager (required for intelligent mode)
        scheduling_strategy: Optional scheduling strategy

    Returns:
        Appropriate mode strategy instance

    Raises:
        ValueError: If mode is unknown or required dependencies are missing
    """
    mode = mode.lower()

    if mode == "basic":
        return BasicModeStrategy(scheduler, config, client)

    elif mode == "intelligent":
        if not provider:
            raise ValueError("ProviderInterface is required for INTELLIGENT mode")
        if not classifier:
            raise ValueError("ClassifierProtocol is required for INTELLIGENT mode")
        if not state_manager:
            raise ValueError("StateManager is required for INTELLIGENT mode")

        return IntelligentModeStrategy(
            scheduler,
            config,
            client,
            provider,
            classifier,
            state_manager,
            scheduling_strategy,
        )

    elif mode == "account":
        return AccountModeStrategy(scheduler, config, client)

    else:
        raise ValueError(f"Unknown scheduler mode: {mode}")


__all__ = [
    "AccountModeStrategy",
    "BaseSchedulingModeStrategy",
    "BasicModeStrategy",
    "IntelligentModeStrategy",
    "create_mode_strategy",
]
