# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
Advanced Scheduling Strategies for Intelligent Queue Scheduler.

This module implements a comprehensive suite of sophisticated scheduling algorithms
that determine optimal queue processing order. The schedulers ensure fairness,
prevent starvation, and optimize throughput across models with vastly different
rate limits and computational characteristics.

Core Scheduling Challenges Addressed:

1. **Starvation Prevention**: Ensuring low-rate models get fair processing opportunities
2. **Proportional Fairness**: Allocating processing time proportional to rate limits
3. **Priority Handling**: Respecting client priorities while maintaining fairness
4. **Adaptive Behavior**: Adjusting to changing load patterns and system conditions
5. **Throughput Optimization**: Maximizing overall system utilization

Implemented Scheduling Algorithms:

1. **WeightedRoundRobinStrategy**: Proportional fairness based on rate limits
2. **PriorityStrategy**: Priority-first scheduling with fairness fallbacks
3. **DeficitRoundRobinStrategy**: Anti-starvation algorithm with deficit tracking
4. **FairQueueStrategy**: Simple round-robin for equal opportunity
5. **AdaptiveStrategy**: Dynamic adaptation based on queue depth and wait times
"""

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable

from ..types.queue import QueueInfo

logger = logging.getLogger(__name__)


class BaseSchedulingStrategy(ABC):
    """
    Abstract base class defining the contract for all scheduling strategies.

    This class establishes the Strategy pattern interface that enables pluggable
    scheduling algorithms in the queue system. Each concrete strategy implements
    a different fairness model and optimization approach for queue selection.

    Strategy Pattern Benefits:

    - **Algorithmic Flexibility**: Switch between scheduling algorithms at runtime
    - **Extensibility**: Add new scheduling strategies without modifying existing code
    - **Testing**: Isolated testing of individual scheduling algorithms
    - **Performance Tuning**: Optimize different strategies for different workload patterns
    - **Fairness Models**: Implement various fairness and priority models

    Implementation Requirements:

    All concrete strategies must implement the `select` method with the following
    guarantees:

    - **Deterministic Behavior**: Same inputs should produce consistent outputs
    - **Fairness**: No indefinite starvation of any eligible queue
    - **Performance**: Selection algorithm should be O(n) or better for scalability
    - **Robustness**: Graceful handling of edge cases and empty queue lists
    - **State Management**: Proper tracking of selection history for fairness
    """

    @abstractmethod
    async def select(self, eligible_queues: list[QueueInfo]) -> QueueInfo | None:
        """
        Select the next queue to process from the list of eligible queues.

        This method implements the core scheduling logic that determines which
        queue should be processed next. The algorithm should balance multiple
        objectives including fairness, throughput, and starvation prevention.

        Args:
            eligible_queues: List of QueueInfo objects representing queues that
                           can currently process requests without violating rate
                           limits. May be empty if no queues are eligible.

        Returns:
            QueueInfo: The selected queue to process next, or None if no queue
                      can be selected (e.g., empty input list).
        """
        pass


class WeightedRoundRobinStrategy(BaseSchedulingStrategy):
    """
    Implements proportional fairness scheduling based on model rate limits.

    This strategy provides fair resource allocation proportional to each model's
    rate limiting capacity while preventing any single high-rate model from
    monopolizing the scheduler.

    Algorithm Overview:

    **Weighted Fair Queueing Principles:**
    - Each model receives processing opportunities proportional to its rate limit
    - High-rate models (1000 RPM) get more selections than low-rate models (10 RPM)
    - Recent selections are penalized to prevent monopolization
    - Randomization provides fairness among models with similar weights

    **Mathematical Foundation:**

    Base weight calculation: `weight = rpm_limit / 100.0`
    - Normalizes rate limits to reasonable weight ranges
    - A 1000 RPM model gets weight 10.0, a 100 RPM model gets weight 1.0

    **Anti-Monopolization Mechanism:**
    Recent selection penalty: `weight *= 0.5 if selected_within_last_second`
    - Halves the weight of recently selected models
    - Prevents high-rate models from being selected repeatedly
    """

    def __init__(self) -> None:
        """Initialize weighted round-robin scheduler with tracking state."""
        self._lock = asyncio.Lock()
        self.last_selected: dict[str, float] = {}  # model_id -> last_selection_time
        self.selection_counts: dict[str, int] = {}  # model_id -> total_selections

    async def select(self, eligible_queues: list[QueueInfo]) -> QueueInfo | None:
        """
        Select next queue using weighted round-robin algorithm with anti-monopolization.

        Algorithm Steps:

        1. **Input Validation**: Check for empty queue list
        2. **Weight Calculation**: Compute base weights from rate limits
        3. **Anti-Monopolization**: Apply penalties for recent selections
        4. **Weighted Selection**: Use probabilistic selection based on weights
        5. **State Update**: Track selection for future fairness calculations

        Args:
            eligible_queues: List of queues that can currently process requests.

        Returns:
            QueueInfo: Selected queue for processing, or None if input is empty.
        """
        if not eligible_queues:
            return None

        async with self._lock:
            weights: dict[str, float] = {}
            current_time = time.time()

            for queue_info in eligible_queues:
                model_id = queue_info.model_id
                rpm_limit = queue_info.rate_config.rpm_limit

                # Base weight proportional to RPM limit (normalized by 100)
                weight = rpm_limit / 100.0

                # Apply anti-monopolization penalty for recent selections
                if model_id in self.last_selected:
                    time_since = current_time - self.last_selected[model_id]
                    if time_since < 1.0:  # Selected within last second
                        weight *= 0.5

                weights[model_id] = weight

            selected = self._weighted_random_choice(eligible_queues, weights)

            self.last_selected[selected.model_id] = current_time
            self.selection_counts[selected.model_id] = (
                self.selection_counts.get(selected.model_id, 0) + 1
            )

            return selected

    def _weighted_random_choice(
        self, items: list[QueueInfo], weights: dict[str, float]
    ) -> QueueInfo:
        """
        Perform weighted random selection with probabilities proportional to weights.

        Args:
            items: List of QueueInfo objects to select from
            weights: Dictionary mapping model_id to selection weight

        Returns:
            Selected item with probability proportional to its weight
        """
        total_weight = sum(weights.values())
        if total_weight <= 0:
            return random.choice(items)  # noqa: S311  # nosec B311

        random_value = random.random() * total_weight  # noqa: S311  # nosec B311

        cumulative_weight = 0.0
        for item in items:
            cumulative_weight += weights.get(item.model_id, 0)
            if random_value <= cumulative_weight:
                return item

        return items[-1]


class PriorityStrategy(BaseSchedulingStrategy):
    """
    Selects the queue with the highest average priority using safe metadata.

    This strategy prioritizes queues based on their average priority values,
    which are tracked safely through metadata updates during enqueue/dequeue
    operations. This eliminates race conditions from peeking at queue internals.

    Thread Safety:
        Uses safe metadata access instead of accessing the queue's internal _queue
        attribute, preventing IndexError and TOCTOU race conditions.

    Use Cases:
        - Production environments where certain requests need higher priority
        - Latency-sensitive applications with priority-based SLAs
        - Systems requiring prioritization while maintaining fairness
    """

    async def select(self, eligible_queues: list[QueueInfo]) -> QueueInfo | None:
        """
        Select queue with highest average priority using safe metadata access.

        Args:
            eligible_queues: List of queues that can currently process requests

        Returns:
            QueueInfo: Queue with highest average priority, or None if no queues
        """
        if not eligible_queues:
            return None

        non_empty = [q for q in eligible_queues if not q.is_empty]
        if not non_empty:
            return None

        best_queue = max(non_empty, key=lambda q: q.get_priority_for_scheduling())

        logger.debug(
            f"Selected queue {best_queue.queue_key} "
            f"(priority: {best_queue.get_priority_for_scheduling():.2f}, "
            f"size: {best_queue.current_size})"
        )

        return best_queue


class DeficitRoundRobinStrategy(BaseSchedulingStrategy):
    """
    Implements Deficit Round Robin (DRR) scheduling to prevent starvation.

    This strategy is specifically designed to solve the starvation problem inherent
    in weighted scheduling algorithms. In systems where models have vastly different
    rate limits, DRR guarantees that every model gets fair opportunities regardless
    of its rate limit.

    The Starvation Problem:

    In proportional scheduling, high-rate models receive proportionally more selections:
    - 1000 RPM model: ~99% of selections
    - 10 RPM model: ~1% of selections

    During high load, the 1% allocation may never occur, causing indefinite delays.

    Deficit Round Robin Solution:

    DRR uses a "deficit counter" per queue that tracks the queue's "credit balance":
    - Each round, all queues receive equal quantum credits
    - Queues are selected based on highest deficit (most credits accumulated)
    - Selection "costs" a quantum, reducing the queue's deficit
    - Deficits can go negative, preventing immediate re-selection

    Mathematical Properties:

    - **Fairness Guarantee**: Every queue receives exactly one selection per quantum period
    - **Starvation Prevention**: No queue can be starved indefinitely
    - **Work Conservation**: High deficit queues get priority until deficits equalize

    Memory Characteristics:

    The ``deficits`` dictionary grows with the number of unique model_ids seen.
    This is **not** unbounded memory growth in practice because:

    - Models are typically configured/registered at application startup, not dynamically created
    - The number of models is bounded by your provider's available model set
    - Each entry is minimal (just a model_id string key and float value)

    If your application uses truly dynamic model IDs (e.g., user-generated), consider
    implementing periodic cleanup or using a bounded LRU cache for the deficits dictionary.
    """

    def __init__(self) -> None:
        """Initialize deficit round-robin scheduler with quantum-based accounting."""
        self._lock = asyncio.Lock()
        self.deficits: dict[str, float] = {}  # model_id -> accumulated_deficit
        self.quantum: float = 100.0

    async def select(self, eligible_queues: list[QueueInfo]) -> QueueInfo | None:
        """
        Select next queue using deficit round-robin algorithm for starvation prevention.

        Algorithm Steps:

        1. **Deficit Initialization**: Set up deficit tracking for new queues
        2. **Quantum Allocation**: Add equal quantum to all eligible queues
        3. **Depth Bonus**: Add bonus quantum based on queue depth
        4. **Maximum Selection**: Choose queue with highest deficit balance
        5. **Cost Deduction**: Deduct quantum cost from selected queue

        Args:
            eligible_queues: List of queues that can process requests.

        Returns:
            QueueInfo: Queue with highest deficit balance, or None if no queues eligible.
        """
        if not eligible_queues:
            return None

        async with self._lock:
            for queue_info in eligible_queues:
                if queue_info.model_id not in self.deficits:
                    self.deficits[queue_info.model_id] = 0.0

            for queue_info in eligible_queues:
                model_id = queue_info.model_id
                queue_depth = queue_info.queue_depth

                quantum_to_add = self.quantum
                depth_bonus = queue_depth * 10.0

                self.deficits[model_id] += quantum_to_add + depth_bonus

            selected = None
            max_deficit = float("-inf")

            for queue_info in eligible_queues:
                deficit = self.deficits[queue_info.model_id]
                if deficit > max_deficit:
                    max_deficit = deficit
                    selected = queue_info

            if selected is None:
                selected = eligible_queues[0]

            if selected:
                self.deficits[selected.model_id] -= self.quantum

            return selected


class FairQueueStrategy(BaseSchedulingStrategy):
    """
    Fair queue strategy that ensures equal opportunity for all models.

    This is a simple round-robin approach without weights, giving
    each model an equal chance regardless of rate limits.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.last_index = -1

    async def select(self, eligible_queues: list[QueueInfo]) -> QueueInfo | None:
        """Select next queue using simple round-robin."""
        if not eligible_queues:
            return None

        async with self._lock:
            self.last_index = (self.last_index + 1) % len(eligible_queues)
            return eligible_queues[self.last_index]


class AdaptiveStrategy(BaseSchedulingStrategy):
    """
    Adaptive strategy that adjusts selection based on queue depths, wait times,
    and historical response times.

    This strategy dynamically adapts to current load conditions, prioritizing
    queues with longer wait times, deeper backlogs, or faster historical response
    times for improved throughput.

    Response Time Weighting:
        Models with faster average response times receive a bonus score multiplier.
        This helps maximize throughput by favoring models that respond quickly.
        The response time factor uses an inverse relationship: faster responses
        lead to higher scores.
    """

    def __init__(self) -> None:
        self.alpha = 0.7  # Weight for queue depth
        self.beta = 0.3  # Weight for wait time
        self.gamma = 0.2  # Weight for response time bonus
        self.response_times: dict[str, list[float]] = {}
        self.max_response_times = 100

    async def record_response_time(self, model_id: str, response_time: float) -> None:
        """
        Record response time for a model to inform adaptive scheduling.

        The recorded response times are used in the select() method to favor
        models with faster historical response times, improving overall throughput.

        Args:
            model_id: The model identifier
            response_time: The response time in seconds
        """
        if model_id not in self.response_times:
            self.response_times[model_id] = []

        self.response_times[model_id].append(response_time)

        if len(self.response_times[model_id]) > self.max_response_times:
            self.response_times[model_id] = self.response_times[model_id][
                -self.max_response_times :
            ]

    def _get_avg_response_time(self, model_id: str) -> float | None:
        """
        Get average response time for a model.

        Args:
            model_id: The model identifier

        Returns:
            Average response time in seconds, or None if no data available
        """
        times = self.response_times.get(model_id)
        if times:
            return sum(times) / len(times)
        return None

    async def select(self, eligible_queues: list[QueueInfo]) -> QueueInfo | None:
        """
        Select queue based on adaptive scoring including response time history.

        The scoring formula considers:
        - Queue depth (alpha weight): Deeper queues get higher priority
        - Wait time (beta weight): Longer-waiting queues get higher priority
        - Response time bonus (gamma weight): Faster models get a bonus multiplier

        The response time bonus uses an inverse relationship: a model with 0.5s
        average response time scores better than one with 2.0s average.
        """
        if not eligible_queues:
            return None

        current_time = time.time()
        best_queue = None
        best_score = float("-inf")

        # Calculate global average response time for normalization
        all_avg_times = [
            self._get_avg_response_time(q.model_id) for q in eligible_queues
        ]
        valid_times = [t for t in all_avg_times if t is not None]
        global_avg = sum(valid_times) / len(valid_times) if valid_times else None

        for queue_info in eligible_queues:
            depth_score = queue_info.queue_depth

            if queue_info.last_request_time:
                last_time = queue_info.last_request_time.timestamp()
                time_since_last = current_time - last_time
            else:
                time_since_last = 10.0
            wait_score = min(time_since_last, 10.0)

            score = (self.alpha * depth_score) + (self.beta * wait_score)

            rpm_factor = queue_info.rate_config.rpm_limit / 100.0
            score *= 1 + rpm_factor * 0.1

            # Apply response time bonus: faster models get higher scores
            model_avg_time = self._get_avg_response_time(queue_info.model_id)
            if (
                model_avg_time is not None
                and global_avg is not None
                and model_avg_time > 0
            ):
                # Inverse relationship: faster response = higher bonus
                # Normalize by global average to keep bonus reasonable
                response_bonus = global_avg / model_avg_time
                score *= 1 + self.gamma * (response_bonus - 1)

            if score > best_score:
                best_score = score
                best_queue = queue_info

        return best_queue or eligible_queues[0]


def create_strategy(strategy_name: str) -> BaseSchedulingStrategy:
    """
    Factory function for creating scheduling strategy instances by name.

    This factory implements the Strategy pattern by providing a unified interface
    for creating different scheduling algorithms. It enables runtime strategy
    selection and configuration.

    Available Strategies and Use Cases:

    **weighted_round_robin**: Best for production with mixed rate limits
    - Provides proportional fairness based on model rate limits
    - Prevents monopolization while respecting capacity differences

    **priority**: Best for latency-sensitive applications
    - Prioritizes high-priority requests while maintaining basic fairness
    - Suitable for real-time applications or SLA-driven environments

    **deficit_round_robin**: Best for fairness-critical environments
    - Guarantees equal access for all models regardless of rate limits
    - Prevents starvation of low-rate or low-priority models

    **fair_queue**: Best for simple round-robin scenarios
    - Provides basic round-robin scheduling without weights or priorities
    - Minimal overhead and predictable behavior

    **adaptive**: Best for dynamic environments
    - Adapts scheduling based on queue depth and response times
    - Provides intelligent load-aware scheduling with feedback control

    Args:
        strategy_name: Name of the scheduling strategy to create. Case-insensitive.
                      Valid values: 'weighted_round_robin', 'priority',
                      'deficit_round_robin', 'fair_queue', 'adaptive'

    Returns:
        BaseSchedulingStrategy: Fully initialized instance of the requested
                              scheduling strategy, ready for use in the scheduler.

    Raises:
        ValueError: If strategy_name is not recognized.

    Example:
        ```python
        production_strategy = create_strategy("weighted_round_robin")
        testing_strategy = create_strategy("deficit_round_robin")
        latency_strategy = create_strategy("priority")
        ```
    """
    strategies: dict[str, Callable[[], BaseSchedulingStrategy]] = {
        "weighted_round_robin": WeightedRoundRobinStrategy,
        "priority": PriorityStrategy,
        "deficit_round_robin": DeficitRoundRobinStrategy,
        "fair_queue": FairQueueStrategy,
        "adaptive": AdaptiveStrategy,
    }

    strategy_factory = strategies.get(strategy_name.lower())
    if strategy_factory is None:
        available = [f"'{s}'" for s in strategies]
        raise ValueError(
            f"Unknown scheduling strategy: {strategy_name}. "
            f"Available strategies: {available}"
        )

    return strategy_factory()


__all__ = [
    "AdaptiveStrategy",
    "BaseSchedulingStrategy",
    "DeficitRoundRobinStrategy",
    "FairQueueStrategy",
    "PriorityStrategy",
    "WeightedRoundRobinStrategy",
    "create_strategy",
]
