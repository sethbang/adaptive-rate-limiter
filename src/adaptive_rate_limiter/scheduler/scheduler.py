# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
Refactored Scheduler implementation for Adaptive Rate Limiter.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import (
    Any,
)

from ..exceptions import RateLimiterError
from ..protocols.classifier import ClassifierProtocol
from ..protocols.client import ClientProtocol
from ..providers.base import ProviderInterface
from ..strategies.scheduling import BaseSchedulingStrategy
from ..types.request import RequestMetadata
from .base import BaseScheduler
from .config import RateLimiterConfig, SchedulerMode
from .state import StateManager

logger = logging.getLogger(__name__)


class Scheduler(BaseScheduler):
    """
    The main request scheduler for the Adaptive Rate Limiter.

    This class acts as a facade that delegates all mode-specific operations to
    a dedicated strategy object (`BaseSchedulingModeStrategy`). This design
    simplifies the `Scheduler` class by removing complex conditional logic
    and promoting a clean separation of concerns.

    The `Scheduler` is responsible for:
    - Creating the appropriate mode strategy based on the configuration.
    - Delegating incoming requests to the current strategy.
    - Managing the lifecycle (start and stop) of the strategy.
    - Aggregating and exposing metrics from the strategy.
    """

    def __init__(
        self,
        client: ClientProtocol,
        config: RateLimiterConfig,
        provider: ProviderInterface | None = None,
        classifier: ClassifierProtocol | None = None,
        state_manager: StateManager | None = None,
        scheduling_strategy: BaseSchedulingStrategy | None = None,
        test_rate_multiplier: float = 1.0,
        metrics_enabled: bool = False,
        default_rate_limits: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the unified Scheduler with Strategy Pattern.

        Args:
            client: Async client for API calls
            config: Unified scheduler configuration
            provider: Service for discovering rate limit buckets (optional for basic mode)
            classifier: Request classifier for routing (optional for basic mode)
            state_manager: Manager for rate limit state (optional for basic mode)
            scheduling_strategy: Optional custom scheduling strategy
            test_rate_multiplier: Rate multiplier for testing scenarios
            metrics_enabled: Whether to enable metrics collection
            default_rate_limits: Default rate limits for models
        """
        # Initialize base scheduler
        super().__init__(
            client=client,
            config=config,
            provider=provider,
            classifier=classifier,
            state_manager=state_manager,
            scheduling_strategy=scheduling_strategy,
            test_rate_multiplier=test_rate_multiplier,
            metrics_enabled=metrics_enabled,
            default_rate_limits=default_rate_limits,
        )

        # BaseScheduler handles mode strategy creation in _setup_mode_strategy

    async def submit_request(
        self, metadata: RequestMetadata, request_func: Callable[[], Awaitable[Any]]
    ) -> Any:
        """
        Submit request for execution.

        This method delegates to the mode strategy.

        Args:
            metadata: Request metadata
            request_func: Async function to execute the request

        Returns:
            Result from the mode strategy
        """
        logger.debug(
            f"Scheduler.submit_request called! Mode: {self.config.mode}, Model: {metadata.model_id}"
        )

        if not self.is_running():
            raise RateLimiterError("Scheduler is not running")

        # Simple delegation to strategy
        return await self.mode_strategy.submit_request(metadata, request_func)

    async def _scheduler_loop(self) -> None:
        """
        Main scheduler loop.
        Delegates to mode strategy if it supports it.
        """
        # Only run loop for modes that need it
        if self.config.mode == SchedulerMode.BASIC:
            return

        if hasattr(self.mode_strategy, "run_scheduling_loop"):
            await self.mode_strategy.run_scheduling_loop()

    def get_metrics(self) -> dict[str, Any]:
        """
        Get current scheduler metrics.

        Combines base scheduler metrics with mode-specific metrics from the strategy.
        """
        base_metrics = super().get_metrics()
        mode_metrics = self.mode_strategy.get_metrics()

        # Merge metrics with mode strategy taking precedence for overlapping keys
        combined_metrics = {**base_metrics, **mode_metrics}

        # Ensure scheduler_mode is string
        mode_val = (
            self.config.mode.value
            if hasattr(self.config.mode, "value")
            else str(self.config.mode)
        )
        combined_metrics["scheduler_mode"] = mode_val

        return combined_metrics


# Factory function for easy creation with dependency injection
def create_scheduler(
    client: ClientProtocol,
    mode: str | None = None,
    config: RateLimiterConfig | None = None,
    provider: ProviderInterface | None = None,
    classifier: ClassifierProtocol | None = None,
    state_manager: StateManager | None = None,
    scheduling_strategy: BaseSchedulingStrategy | None = None,
    **kwargs: Any,
) -> Scheduler:
    """
    Factory function to create a Scheduler with proper dependency injection.

    Args:
        client: Async client
        mode: Scheduler mode ("basic", "intelligent", "account"). If None,
            uses config.mode (or defaults to "intelligent" if config is also None)
        config: Optional scheduler config (will create default if not provided)
        provider: Optional provider interface
        classifier: Optional request classifier
        state_manager: Optional state manager
        scheduling_strategy: Optional scheduling strategy
        **kwargs: Additional arguments passed to Scheduler constructor

    Returns:
        Configured Scheduler instance

    Raises:
        ValueError: If mode is unknown or required dependencies are missing
    """
    if config is None:
        config = RateLimiterConfig()

    # Only override mode in config if explicitly provided (not None)
    if mode is not None:
        try:
            scheduler_mode = SchedulerMode(mode.lower())
            config.mode = scheduler_mode
        except ValueError as e:
            raise ValueError(f"Unknown scheduler mode: {mode}") from e

    return Scheduler(
        client=client,
        config=config,
        provider=provider,
        classifier=classifier,
        state_manager=state_manager,
        scheduling_strategy=scheduling_strategy,
        **kwargs,
    )
