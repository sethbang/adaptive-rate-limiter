"""
Unit tests for scheduling strategies.

Tests the scheduling strategies used by the intelligent queue scheduler for
determining optimal queue processing order. Covers all strategy implementations:
- BaseSchedulingStrategy (abstract base class)
- WeightedRoundRobinStrategy
- DeficitRoundRobinStrategy
- PriorityStrategy
- FairQueueStrategy
- AdaptiveStrategy
- create_strategy factory function
"""

import asyncio
from asyncio import PriorityQueue
from datetime import datetime, timezone

import pytest

from adaptive_rate_limiter.strategies.scheduling import (
    AdaptiveStrategy,
    BaseSchedulingStrategy,
    DeficitRoundRobinStrategy,
    FairQueueStrategy,
    PriorityStrategy,
    WeightedRoundRobinStrategy,
    create_strategy,
)
from adaptive_rate_limiter.types.queue import QueueInfo
from adaptive_rate_limiter.types.rate_limit import RateLimitBucket

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def rate_config_high() -> RateLimitBucket:
    """Rate config with high RPM limit (1000)."""
    return RateLimitBucket(
        model_id="model-high",
        resource_type="chat",
        rpm_limit=1000,
        tpm_limit=100000,
    )


@pytest.fixture
def rate_config_medium() -> RateLimitBucket:
    """Rate config with medium RPM limit (100)."""
    return RateLimitBucket(
        model_id="model-medium",
        resource_type="chat",
        rpm_limit=100,
        tpm_limit=10000,
    )


@pytest.fixture
def rate_config_low() -> RateLimitBucket:
    """Rate config with low RPM limit (10)."""
    return RateLimitBucket(
        model_id="model-low",
        resource_type="chat",
        rpm_limit=10,
        tpm_limit=1000,
    )


def create_queue_info(
    queue_key: str,
    model_id: str,
    rate_config: RateLimitBucket,
    queue_depth: int = 0,
    is_empty: bool = False,
    avg_priority: float = 0.0,
    last_request_time: datetime | None = None,
) -> QueueInfo:
    """Helper to create QueueInfo with mock priority queue."""
    queue = PriorityQueue()

    # Add items to make non-empty if needed
    if not is_empty:
        queue.put_nowait((0, "item"))

    info = QueueInfo(
        queue_key=queue_key,
        model_id=model_id,
        resource_type=rate_config.resource_type,
        queue=queue,
        rate_config=rate_config,
        queue_depth=queue_depth,
        last_request_time=last_request_time,
    )

    # Set priority tracking if specified
    if avg_priority > 0:
        info.priority_sum = avg_priority
        info.total_enqueued = 1

    return info


@pytest.fixture
def high_rate_queue(rate_config_high) -> QueueInfo:
    """Queue with high rate limit (1000 RPM)."""
    return create_queue_info(
        queue_key="high-queue",
        model_id="model-high",
        rate_config=rate_config_high,
        queue_depth=5,
    )


@pytest.fixture
def medium_rate_queue(rate_config_medium) -> QueueInfo:
    """Queue with medium rate limit (100 RPM)."""
    return create_queue_info(
        queue_key="medium-queue",
        model_id="model-medium",
        rate_config=rate_config_medium,
        queue_depth=3,
    )


@pytest.fixture
def low_rate_queue(rate_config_low) -> QueueInfo:
    """Queue with low rate limit (10 RPM)."""
    return create_queue_info(
        queue_key="low-queue",
        model_id="model-low",
        rate_config=rate_config_low,
        queue_depth=1,
    )


# ============================================================================
# BaseSchedulingStrategy Tests
# ============================================================================


class TestBaseSchedulingStrategy:
    """Tests for BaseSchedulingStrategy abstract base class."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseSchedulingStrategy cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseSchedulingStrategy()  # type: ignore

    def test_requires_select_method(self):
        """Test that subclasses must implement select method."""

        class IncompleteStrategy(BaseSchedulingStrategy):
            pass

        with pytest.raises(TypeError):
            IncompleteStrategy()  # type: ignore

    def test_concrete_subclass_works(self):
        """Test that a complete subclass can be instantiated."""

        class ConcreteStrategy(BaseSchedulingStrategy):
            async def select(self, eligible_queues):
                return eligible_queues[0] if eligible_queues else None

        strategy = ConcreteStrategy()
        assert strategy is not None

    @pytest.mark.asyncio
    async def test_concrete_subclass_select_works(self):
        """Test that a concrete subclass select method works."""

        class ConcreteStrategy(BaseSchedulingStrategy):
            async def select(self, eligible_queues):
                return eligible_queues[0] if eligible_queues else None

        strategy = ConcreteStrategy()
        rate_config = RateLimitBucket(
            model_id="test", resource_type="chat", rpm_limit=100
        )
        queue = create_queue_info("test", "test", rate_config)

        result = await strategy.select([queue])
        assert result == queue


# ============================================================================
# WeightedRoundRobinStrategy Tests
# ============================================================================


class TestWeightedRoundRobinStrategy:
    """Tests for WeightedRoundRobinStrategy."""

    @pytest.fixture
    def strategy(self) -> WeightedRoundRobinStrategy:
        """Create a WeightedRoundRobinStrategy instance."""
        return WeightedRoundRobinStrategy()

    def test_init_creates_empty_tracking(self, strategy):
        """Test initialization creates empty tracking structures."""
        assert strategy.last_selected == {}
        assert strategy.selection_counts == {}

    @pytest.mark.asyncio
    async def test_select_empty_list_returns_none(self, strategy):
        """Test selecting from empty list returns None."""
        result = await strategy.select([])
        assert result is None

    @pytest.mark.asyncio
    async def test_select_single_queue_returns_queue(self, strategy, high_rate_queue):
        """Test selecting from single queue returns that queue."""
        result = await strategy.select([high_rate_queue])
        assert result == high_rate_queue

    @pytest.mark.asyncio
    async def test_select_updates_last_selected(self, strategy, high_rate_queue):
        """Test selection updates last_selected tracking."""
        await strategy.select([high_rate_queue])

        assert "model-high" in strategy.last_selected
        assert strategy.last_selected["model-high"] > 0

    @pytest.mark.asyncio
    async def test_select_increments_selection_count(self, strategy, high_rate_queue):
        """Test selection increments selection count."""
        await strategy.select([high_rate_queue])
        await strategy.select([high_rate_queue])

        assert strategy.selection_counts["model-high"] == 2

    @pytest.mark.asyncio
    async def test_weighted_selection_favors_high_rate(
        self, strategy, high_rate_queue, low_rate_queue
    ):
        """Test weighted selection favors higher rate queues over many iterations."""
        queues = [high_rate_queue, low_rate_queue]

        # Run many selections to check distribution
        high_count = 0
        low_count = 0

        for _ in range(100):
            result = await strategy.select(queues)
            if result.model_id == "model-high":
                high_count += 1
            else:
                low_count += 1

        # High rate (1000) should be selected more often than low rate (10)
        # Due to anti-monopolization, this ratio won't be 100:1 but should still favor high
        assert high_count > low_count

    @pytest.mark.asyncio
    async def test_anti_monopolization_penalty(self, strategy, high_rate_queue):
        """Test that recently selected queues get penalty."""
        # First selection
        await strategy.select([high_rate_queue])
        first_time = strategy.last_selected["model-high"]

        # Immediate second selection - should still work but with penalty
        await strategy.select([high_rate_queue])
        second_time = strategy.last_selected["model-high"]

        assert second_time >= first_time

    @pytest.mark.asyncio
    async def test_weighted_random_choice_zero_weights(self, strategy):
        """Test weighted random choice handles zero total weight."""
        rate_config = RateLimitBucket(
            model_id="zero", resource_type="chat", rpm_limit=0
        )
        queue = create_queue_info("zero-queue", "zero", rate_config)

        # With zero RPM, weight is 0, but should still return a queue
        result = await strategy.select([queue])
        assert result == queue

    @pytest.mark.asyncio
    async def test_weighted_random_choice_fallback(self, strategy):
        """Test weighted random choice returns last item as fallback."""
        # Create queues with weights that don't sum correctly for edge case
        rate_config = RateLimitBucket(
            model_id="test", resource_type="chat", rpm_limit=100
        )
        queues = [
            create_queue_info(f"queue-{i}", f"model-{i}", rate_config) for i in range(3)
        ]

        result = await strategy.select(queues)
        assert result in queues


# ============================================================================
# PriorityStrategy Tests
# ============================================================================


class TestPriorityStrategy:
    """Tests for PriorityStrategy."""

    @pytest.fixture
    def strategy(self) -> PriorityStrategy:
        """Create a PriorityStrategy instance."""
        return PriorityStrategy()

    @pytest.mark.asyncio
    async def test_select_empty_list_returns_none(self, strategy):
        """Test selecting from empty list returns None."""
        result = await strategy.select([])
        assert result is None

    @pytest.mark.asyncio
    async def test_select_all_empty_queues_returns_none(
        self, strategy, rate_config_high
    ):
        """Test selecting when all queues are empty returns None."""
        queue = create_queue_info(
            "empty-queue", "model", rate_config_high, is_empty=True
        )
        # Ensure the queue is actually empty
        while not queue.queue.empty():
            queue.queue.get_nowait()

        result = await strategy.select([queue])
        assert result is None

    @pytest.mark.asyncio
    async def test_select_highest_priority(self, strategy, rate_config_high):
        """Test selects queue with highest priority."""
        low_priority = create_queue_info(
            "low-priority", "model-low", rate_config_high, avg_priority=1.0
        )
        high_priority = create_queue_info(
            "high-priority", "model-high", rate_config_high, avg_priority=10.0
        )
        medium_priority = create_queue_info(
            "medium-priority", "model-med", rate_config_high, avg_priority=5.0
        )

        result = await strategy.select([low_priority, high_priority, medium_priority])
        assert result == high_priority

    @pytest.mark.asyncio
    async def test_select_single_non_empty_queue(self, strategy, rate_config_high):
        """Test selects the only non-empty queue."""
        non_empty = create_queue_info(
            "non-empty", "model", rate_config_high, avg_priority=5.0
        )

        result = await strategy.select([non_empty])
        assert result == non_empty

    @pytest.mark.asyncio
    async def test_select_skips_empty_queues(self, strategy, rate_config_high):
        """Test skips empty queues even with higher priority."""
        empty_queue = create_queue_info(
            "empty", "model-empty", rate_config_high, avg_priority=100.0, is_empty=True
        )
        # Make it actually empty
        while not empty_queue.queue.empty():
            empty_queue.queue.get_nowait()

        non_empty = create_queue_info(
            "non-empty", "model-ok", rate_config_high, avg_priority=1.0
        )

        result = await strategy.select([empty_queue, non_empty])
        assert result == non_empty


# ============================================================================
# DeficitRoundRobinStrategy Tests
# ============================================================================


class TestDeficitRoundRobinStrategy:
    """Tests for DeficitRoundRobinStrategy."""

    @pytest.fixture
    def strategy(self) -> DeficitRoundRobinStrategy:
        """Create a DeficitRoundRobinStrategy instance."""
        return DeficitRoundRobinStrategy()

    def test_init_creates_empty_deficits(self, strategy):
        """Test initialization creates empty deficit tracking."""
        assert strategy.deficits == {}
        assert strategy.quantum == 100.0

    @pytest.mark.asyncio
    async def test_select_empty_list_returns_none(self, strategy):
        """Test selecting from empty list returns None."""
        result = await strategy.select([])
        assert result is None

    @pytest.mark.asyncio
    async def test_select_initializes_deficit(self, strategy, high_rate_queue):
        """Test first selection initializes deficit tracking."""
        await strategy.select([high_rate_queue])

        assert "model-high" in strategy.deficits

    @pytest.mark.asyncio
    async def test_select_adds_quantum(self, strategy, high_rate_queue):
        """Test selection adds quantum to deficit."""
        await strategy.select([high_rate_queue])

        # Deficit should be positive after adding quantum
        # Then deducted quantum for selection
        # queue_depth=5, so depth_bonus = 50
        # quantum (100) + depth_bonus (50) - quantum (100) = 50
        expected_deficit = strategy.quantum + (5 * 10.0) - strategy.quantum
        assert strategy.deficits["model-high"] == pytest.approx(expected_deficit)

    @pytest.mark.asyncio
    async def test_select_considers_queue_depth(
        self, strategy, rate_config_high, rate_config_low
    ):
        """Test selection considers queue depth bonus."""
        deep_queue = create_queue_info(
            "deep", "model-deep", rate_config_low, queue_depth=10
        )
        shallow_queue = create_queue_info(
            "shallow", "model-shallow", rate_config_high, queue_depth=1
        )

        # First selection - deep queue should get higher deficit due to depth
        result = await strategy.select([deep_queue, shallow_queue])

        # Deep queue gets depth_bonus of 100 (10 * 10), shallow gets 10 (1 * 10)
        assert result == deep_queue

    @pytest.mark.asyncio
    async def test_deficit_prevents_monopolization(self, strategy, rate_config_high):
        """Test deficit mechanism prevents monopolization."""
        queue1 = create_queue_info("q1", "model-1", rate_config_high, queue_depth=5)
        queue2 = create_queue_info("q2", "model-2", rate_config_high, queue_depth=5)

        # Run multiple selections
        selections = {"model-1": 0, "model-2": 0}
        for _ in range(10):
            result = await strategy.select([queue1, queue2])
            selections[result.model_id] += 1

        # Both should get roughly equal selections due to DRR fairness
        assert selections["model-1"] >= 4
        assert selections["model-2"] >= 4

    @pytest.mark.asyncio
    async def test_select_deducts_quantum(self, strategy, high_rate_queue):
        """Test selection deducts quantum from selected queue."""
        await strategy.select([high_rate_queue])

        # Should have added quantum + depth_bonus, then deducted quantum
        deficit = strategy.deficits["model-high"]

        # Now select again
        await strategy.select([high_rate_queue])
        new_deficit = strategy.deficits["model-high"]

        # Second selection adds quantum + depth_bonus again
        # deficit + quantum + depth_bonus - quantum (for selection)
        expected = deficit + strategy.quantum + (5 * 10.0) - strategy.quantum
        assert new_deficit == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_fallback_to_first_queue(self, strategy, rate_config_high):
        """Test fallback to first queue when selection is ambiguous."""
        queues = [
            create_queue_info(f"q-{i}", f"model-{i}", rate_config_high, queue_depth=0)
            for i in range(3)
        ]

        result = await strategy.select(queues)
        assert result is not None


# ============================================================================
# FairQueueStrategy Tests
# ============================================================================


class TestFairQueueStrategy:
    """Tests for FairQueueStrategy."""

    @pytest.fixture
    def strategy(self) -> FairQueueStrategy:
        """Create a FairQueueStrategy instance."""
        return FairQueueStrategy()

    def test_init_starts_at_negative_one(self, strategy):
        """Test initialization starts last_index at -1."""
        assert strategy.last_index == -1

    @pytest.mark.asyncio
    async def test_select_empty_list_returns_none(self, strategy):
        """Test selecting from empty list returns None."""
        result = await strategy.select([])
        assert result is None

    @pytest.mark.asyncio
    async def test_select_increments_index(self, strategy, rate_config_high):
        """Test selection increments the index."""
        queues = [
            create_queue_info(f"q-{i}", f"model-{i}", rate_config_high)
            for i in range(3)
        ]

        # First selection
        await strategy.select(queues)
        assert strategy.last_index == 0

        # Second selection
        await strategy.select(queues)
        assert strategy.last_index == 1

    @pytest.mark.asyncio
    async def test_select_wraps_around(self, strategy, rate_config_high):
        """Test selection wraps around when reaching end."""
        queues = [
            create_queue_info(f"q-{i}", f"model-{i}", rate_config_high)
            for i in range(3)
        ]

        # Select through all queues
        for _ in range(3):
            await strategy.select(queues)

        assert strategy.last_index == 2

        # Next selection should wrap to 0
        await strategy.select(queues)
        assert strategy.last_index == 0

    @pytest.mark.asyncio
    async def test_round_robin_fairness(self, strategy, rate_config_high):
        """Test round robin provides equal selections."""
        queues = [
            create_queue_info(f"q-{i}", f"model-{i}", rate_config_high)
            for i in range(3)
        ]

        selections = {"model-0": 0, "model-1": 0, "model-2": 0}
        for _ in range(9):
            result = await strategy.select(queues)
            selections[result.model_id] += 1

        # Each should be selected exactly 3 times
        assert selections["model-0"] == 3
        assert selections["model-1"] == 3
        assert selections["model-2"] == 3

    @pytest.mark.asyncio
    async def test_single_queue_always_selected(self, strategy, high_rate_queue):
        """Test single queue is always selected."""
        for _ in range(5):
            result = await strategy.select([high_rate_queue])
            assert result == high_rate_queue


# ============================================================================
# AdaptiveStrategy Tests
# ============================================================================


class TestAdaptiveStrategy:
    """Tests for AdaptiveStrategy."""

    @pytest.fixture
    def strategy(self) -> AdaptiveStrategy:
        """Create an AdaptiveStrategy instance."""
        return AdaptiveStrategy()

    def test_init_sets_weights(self, strategy):
        """Test initialization sets alpha and beta weights."""
        assert strategy.alpha == 0.7  # Weight for queue depth
        assert strategy.beta == 0.3  # Weight for wait time

    def test_init_creates_empty_response_times(self, strategy):
        """Test initialization creates empty response times tracking."""
        assert strategy.response_times == {}
        assert strategy.max_response_times == 100

    @pytest.mark.asyncio
    async def test_select_empty_list_returns_none(self, strategy):
        """Test selecting from empty list returns None."""
        result = await strategy.select([])
        assert result is None

    @pytest.mark.asyncio
    async def test_select_prefers_deep_queues(self, strategy, rate_config_high):
        """Test selection prefers queues with more items."""
        deep_queue = create_queue_info(
            "deep", "model-deep", rate_config_high, queue_depth=10
        )
        shallow_queue = create_queue_info(
            "shallow", "model-shallow", rate_config_high, queue_depth=1
        )

        result = await strategy.select([shallow_queue, deep_queue])
        assert result == deep_queue

    @pytest.mark.asyncio
    async def test_select_considers_wait_time(self, strategy, rate_config_high):
        """Test selection considers time since last request."""
        old_queue = create_queue_info(
            "old",
            "model-old",
            rate_config_high,
            queue_depth=1,
            last_request_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        new_queue = create_queue_info(
            "new",
            "model-new",
            rate_config_high,
            queue_depth=1,
            last_request_time=datetime.now(timezone.utc),
        )

        # Old queue should have higher wait time score
        result = await strategy.select([new_queue, old_queue])
        assert result == old_queue

    @pytest.mark.asyncio
    async def test_select_considers_rpm_factor(
        self, strategy, rate_config_high, rate_config_low
    ):
        """Test selection applies RPM factor bonus."""
        high_rpm = create_queue_info(
            "high-rpm", "model-high", rate_config_high, queue_depth=1
        )
        low_rpm = create_queue_info(
            "low-rpm", "model-low", rate_config_low, queue_depth=1
        )

        # Both have same depth and no last_request_time
        # High RPM (1000) should get slightly more score due to rpm_factor
        high_rpm.last_request_time = None
        low_rpm.last_request_time = None

        result = await strategy.select([low_rpm, high_rpm])
        # High RPM gets bonus: score *= 1 + (1000/100) * 0.1 = 1 + 1.0 = 2.0
        # Low RPM gets: score *= 1 + (10/100) * 0.1 = 1 + 0.01 = 1.01
        assert result == high_rpm

    @pytest.mark.asyncio
    async def test_select_no_last_request_uses_default(
        self, strategy, rate_config_high
    ):
        """Test selection uses default wait time when no last request."""
        queue = create_queue_info("no-last", "model", rate_config_high, queue_depth=1)
        queue.last_request_time = None

        result = await strategy.select([queue])
        assert result == queue

    @pytest.mark.asyncio
    async def test_record_response_time(self, strategy):
        """Test recording response time."""
        await strategy.record_response_time("model-1", 0.5)
        await strategy.record_response_time("model-1", 0.8)

        assert len(strategy.response_times["model-1"]) == 2
        assert strategy.response_times["model-1"] == [0.5, 0.8]

    @pytest.mark.asyncio
    async def test_record_response_time_new_model(self, strategy):
        """Test recording response time for new model."""
        await strategy.record_response_time("new-model", 1.0)

        assert "new-model" in strategy.response_times
        assert strategy.response_times["new-model"] == [1.0]

    @pytest.mark.asyncio
    async def test_record_response_time_truncates_to_max(self, strategy):
        """Test recording truncates to max_response_times."""
        strategy.max_response_times = 5

        for i in range(10):
            await strategy.record_response_time("model", float(i))

        # Should only keep last 5
        assert len(strategy.response_times["model"]) == 5
        assert strategy.response_times["model"] == [5.0, 6.0, 7.0, 8.0, 9.0]

    @pytest.mark.asyncio
    async def test_select_wait_time_capped(self, strategy, rate_config_high):
        """Test wait time is capped at 10 seconds."""
        very_old_queue = create_queue_info(
            "very-old",
            "model",
            rate_config_high,
            queue_depth=1,
            last_request_time=datetime(2000, 1, 1, tzinfo=timezone.utc),
        )

        # Even with very old time, wait score is capped
        result = await strategy.select([very_old_queue])
        assert result == very_old_queue

    @pytest.mark.asyncio
    async def test_select_returns_fallback(self, strategy, rate_config_high):
        """Test select returns first queue as fallback."""
        queue = create_queue_info("only", "model", rate_config_high, queue_depth=0)

        result = await strategy.select([queue])
        assert result == queue


# ============================================================================
# create_strategy Factory Tests
# ============================================================================


class TestCreateStrategy:
    """Tests for create_strategy factory function."""

    def test_create_weighted_round_robin(self):
        """Test creating weighted round robin strategy."""
        strategy = create_strategy("weighted_round_robin")
        assert isinstance(strategy, WeightedRoundRobinStrategy)

    def test_create_priority(self):
        """Test creating priority strategy."""
        strategy = create_strategy("priority")
        assert isinstance(strategy, PriorityStrategy)

    def test_create_deficit_round_robin(self):
        """Test creating deficit round robin strategy."""
        strategy = create_strategy("deficit_round_robin")
        assert isinstance(strategy, DeficitRoundRobinStrategy)

    def test_create_fair_queue(self):
        """Test creating fair queue strategy."""
        strategy = create_strategy("fair_queue")
        assert isinstance(strategy, FairQueueStrategy)

    def test_create_adaptive(self):
        """Test creating adaptive strategy."""
        strategy = create_strategy("adaptive")
        assert isinstance(strategy, AdaptiveStrategy)

    def test_create_case_insensitive(self):
        """Test strategy name is case insensitive."""
        strategy = create_strategy("WEIGHTED_ROUND_ROBIN")
        assert isinstance(strategy, WeightedRoundRobinStrategy)

        strategy = create_strategy("Adaptive")
        assert isinstance(strategy, AdaptiveStrategy)

    def test_create_unknown_raises_value_error(self):
        """Test unknown strategy name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            create_strategy("unknown_strategy")

        assert "unknown_strategy" in str(exc_info.value).lower()
        assert "available strategies" in str(exc_info.value).lower()

    def test_error_message_lists_available(self):
        """Test error message lists available strategies."""
        try:
            create_strategy("invalid")
        except ValueError as e:
            error_msg = str(e).lower()
            assert "weighted_round_robin" in error_msg
            assert "priority" in error_msg
            assert "deficit_round_robin" in error_msg
            assert "fair_queue" in error_msg
            assert "adaptive" in error_msg


# ============================================================================
# Concurrent Access Tests
# ============================================================================


class TestConcurrentAccess:
    """Tests for concurrent access safety."""

    @pytest.mark.asyncio
    async def test_weighted_round_robin_concurrent(self, rate_config_high):
        """Test WeightedRoundRobinStrategy handles concurrent access."""
        strategy = WeightedRoundRobinStrategy()
        queues = [
            create_queue_info(f"q-{i}", f"model-{i}", rate_config_high)
            for i in range(3)
        ]

        async def select_many():
            for _ in range(10):
                await strategy.select(queues)

        # Run concurrent selections
        await asyncio.gather(*[select_many() for _ in range(5)])

        # Should complete without errors
        total_selections = sum(strategy.selection_counts.values())
        assert total_selections == 50

    @pytest.mark.asyncio
    async def test_deficit_round_robin_concurrent(self, rate_config_high):
        """Test DeficitRoundRobinStrategy handles concurrent access."""
        strategy = DeficitRoundRobinStrategy()
        queues = [
            create_queue_info(f"q-{i}", f"model-{i}", rate_config_high)
            for i in range(3)
        ]

        async def select_many():
            for _ in range(10):
                await strategy.select(queues)

        # Run concurrent selections
        await asyncio.gather(*[select_many() for _ in range(5)])

        # Should complete without errors and all models tracked
        assert len(strategy.deficits) == 3

    @pytest.mark.asyncio
    async def test_fair_queue_concurrent(self, rate_config_high):
        """Test FairQueueStrategy handles concurrent access."""
        strategy = FairQueueStrategy()
        queues = [
            create_queue_info(f"q-{i}", f"model-{i}", rate_config_high)
            for i in range(3)
        ]

        async def select_many():
            results = []
            for _ in range(9):
                result = await strategy.select(queues)
                results.append(result)
            return results

        # Run concurrent selections
        all_results = await asyncio.gather(*[select_many() for _ in range(3)])

        # Should complete without errors
        assert len(all_results) == 3
        for results in all_results:
            assert len(results) == 9


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_weighted_rr_single_zero_rpm(self):
        """Test WeightedRoundRobin with single zero RPM queue."""
        strategy = WeightedRoundRobinStrategy()
        rate_config = RateLimitBucket(
            model_id="zero", resource_type="chat", rpm_limit=0
        )
        queue = create_queue_info("zero-queue", "zero", rate_config)

        result = await strategy.select([queue])
        assert result == queue

    @pytest.mark.asyncio
    async def test_deficit_rr_negative_deficit_handling(self):
        """Test DeficitRoundRobin handles negative deficit."""
        strategy = DeficitRoundRobinStrategy()
        rate_config = RateLimitBucket(
            model_id="test", resource_type="chat", rpm_limit=100
        )
        queue = create_queue_info("q", "test", rate_config, queue_depth=0)

        # Multiple selections should work even with negative deficits
        for _ in range(5):
            result = await strategy.select([queue])
            assert result == queue

    @pytest.mark.asyncio
    async def test_priority_all_zero_priority(self):
        """Test PriorityStrategy with all zero priority queues."""
        strategy = PriorityStrategy()
        rate_config = RateLimitBucket(
            model_id="test", resource_type="chat", rpm_limit=100
        )

        queues = [
            create_queue_info(f"q-{i}", f"model-{i}", rate_config, avg_priority=0.0)
            for i in range(3)
        ]

        result = await strategy.select(queues)
        assert result is not None

    @pytest.mark.asyncio
    async def test_adaptive_all_same_conditions(self):
        """Test AdaptiveStrategy with identical queue conditions."""
        strategy = AdaptiveStrategy()
        rate_config = RateLimitBucket(
            model_id="test", resource_type="chat", rpm_limit=100
        )
        now = datetime.now(timezone.utc)

        queues = [
            create_queue_info(
                f"q-{i}",
                f"model-{i}",
                rate_config,
                queue_depth=5,
                last_request_time=now,
            )
            for i in range(3)
        ]

        result = await strategy.select(queues)
        assert result is not None

    @pytest.mark.asyncio
    async def test_fair_queue_changing_list_size(self):
        """Test FairQueueStrategy with changing queue list size."""
        strategy = FairQueueStrategy()
        rate_config = RateLimitBucket(
            model_id="test", resource_type="chat", rpm_limit=100
        )

        # Start with 3 queues
        queues = [
            create_queue_info(f"q-{i}", f"model-{i}", rate_config) for i in range(3)
        ]
        await strategy.select(queues)
        await strategy.select(queues)

        # Now with only 2 queues
        queues = queues[:2]
        result = await strategy.select(queues)

        # Should wrap around correctly
        assert result in queues


# ============================================================================
# Integration-like Tests
# ============================================================================


class TestStrategyIntegration:
    """Tests that verify strategies work correctly in realistic scenarios."""

    @pytest.mark.asyncio
    async def test_weighted_rr_realistic_distribution(self):
        """Test WeightedRR produces realistic distribution over many selections."""
        strategy = WeightedRoundRobinStrategy()

        high_config = RateLimitBucket(
            model_id="high", resource_type="chat", rpm_limit=1000
        )
        low_config = RateLimitBucket(
            model_id="low", resource_type="chat", rpm_limit=100
        )

        high_queue = create_queue_info("high", "high", high_config)
        low_queue = create_queue_info("low", "low", low_config)

        selections = {"high": 0, "low": 0}
        for _ in range(1000):
            result = await strategy.select([high_queue, low_queue])
            assert result is not None
            selections[result.model_id] += 1

        # High should be selected more often
        # Ratio won't be exactly 10:1 due to anti-monopolization
        # But high should still significantly exceed low
        assert selections["high"] > selections["low"] * 2

    @pytest.mark.asyncio
    async def test_drr_starvation_prevention(self):
        """Test DRR prevents starvation of low-depth queues when depths are similar."""
        strategy = DeficitRoundRobinStrategy()

        # Use same RPM since DRR doesn't consider rate limits, only depth
        config = RateLimitBucket(model_id="test", resource_type="chat", rpm_limit=100)

        # Give similar queue depths to test fair rotation
        queue1 = create_queue_info("q1", "model-1", config, queue_depth=5)
        queue2 = create_queue_info("q2", "model-2", config, queue_depth=5)

        selections = {"model-1": 0, "model-2": 0}
        for _ in range(100):
            result = await strategy.select([queue1, queue2])
            if result:
                selections[result.model_id] += 1

        # Both queues should get roughly equal selections due to DRR fairness
        # Each should get at least 40% of selections
        assert selections["model-1"] >= 40
        assert selections["model-2"] >= 40

    @pytest.mark.asyncio
    async def test_adaptive_responds_to_deep_queues(self):
        """Test Adaptive strategy prioritizes deep queues correctly."""
        strategy = AdaptiveStrategy()

        config = RateLimitBucket(model_id="test", resource_type="chat", rpm_limit=100)

        # One queue is much deeper than others
        deep_queue = create_queue_info("deep", "deep", config, queue_depth=50)
        shallow_queues = [
            create_queue_info(f"shallow-{i}", f"shallow-{i}", config, queue_depth=1)
            for i in range(3)
        ]

        all_queues = [deep_queue, *shallow_queues]

        # Deep queue should be selected first (has highest depth score)
        result = await strategy.select(all_queues)
        assert result == deep_queue
