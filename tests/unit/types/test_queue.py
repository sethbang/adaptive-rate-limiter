"""
Unit tests for QueueInfo thread-safety methods.

These tests verify the thread-safety of QueueInfo's update_on_enqueue and
update_on_dequeue methods under concurrent access.
"""

import asyncio

import pytest

from adaptive_rate_limiter.types.queue import QueueInfo
from adaptive_rate_limiter.types.rate_limit import RateLimitBucket


class TestQueueInfoBasicOperations:
    """Tests for basic QueueInfo enqueue/dequeue operations."""

    @pytest.fixture
    def queue_info(self):
        """Create a QueueInfo instance for testing."""
        mock_queue = asyncio.PriorityQueue()
        rate_config = RateLimitBucket(
            model_id="test-model",
            resource_type="chat",
            rpm_limit=100,
        )
        return QueueInfo(
            queue_key="test-key",
            model_id="test-model",
            resource_type="chat",
            queue=mock_queue,
            rate_config=rate_config,
        )

    @pytest.mark.asyncio
    async def test_update_on_enqueue_basic(self, queue_info):
        """Test basic enqueue tracking updates counters correctly."""
        initial_enqueued = queue_info.total_enqueued
        initial_sum = queue_info.priority_sum

        await queue_info.update_on_enqueue(priority=5.0)

        assert queue_info.total_enqueued == initial_enqueued + 1
        assert queue_info.priority_sum == initial_sum + 5.0
        assert queue_info.current_priority == 5.0
        assert queue_info.last_enqueue_time is not None

    @pytest.mark.asyncio
    async def test_update_on_dequeue_basic(self, queue_info):
        """Test basic dequeue tracking updates counters correctly."""
        # First enqueue something
        await queue_info.update_on_enqueue(priority=5.0)
        initial_dequeued = queue_info.total_dequeued

        await queue_info.update_on_dequeue()

        assert queue_info.total_dequeued == initial_dequeued + 1
        assert queue_info.last_dequeue_time is not None

    @pytest.mark.asyncio
    async def test_avg_priority_calculation(self, queue_info):
        """Verify avg_priority is calculated correctly."""
        # Enqueue items with known priorities
        await queue_info.update_on_enqueue(priority=10.0)
        await queue_info.update_on_enqueue(priority=20.0)
        await queue_info.update_on_enqueue(priority=30.0)

        # Average should be (10 + 20 + 30) / 3 = 20.0
        assert queue_info.avg_priority == 20.0

    @pytest.mark.asyncio
    async def test_avg_priority_empty_queue(self, queue_info):
        """Verify avg_priority returns 0.0 for empty queue."""
        assert queue_info.total_enqueued == 0
        assert queue_info.avg_priority == 0.0

    @pytest.mark.asyncio
    async def test_min_max_priority_tracking(self, queue_info):
        """Verify min_priority and max_priority are tracked correctly."""
        await queue_info.update_on_enqueue(priority=5.0)
        await queue_info.update_on_enqueue(priority=15.0)
        await queue_info.update_on_enqueue(priority=10.0)

        assert queue_info.min_priority_seen == 5.0
        assert queue_info.max_priority_seen == 15.0

    @pytest.mark.asyncio
    async def test_min_max_priority_with_negative_values(self, queue_info):
        """Verify min/max priority tracks negative values correctly."""
        await queue_info.update_on_enqueue(priority=-10.0)
        await queue_info.update_on_enqueue(priority=0.0)
        await queue_info.update_on_enqueue(priority=10.0)

        assert queue_info.min_priority_seen == -10.0
        assert queue_info.max_priority_seen == 10.0

    @pytest.mark.asyncio
    async def test_items_pending_calculation(self, queue_info):
        """Verify items_pending tracks difference correctly."""
        await queue_info.update_on_enqueue(priority=1.0)
        await queue_info.update_on_enqueue(priority=2.0)
        await queue_info.update_on_enqueue(priority=3.0)

        assert queue_info.items_pending == 3

        await queue_info.update_on_dequeue()
        assert queue_info.items_pending == 2

        await queue_info.update_on_dequeue()
        await queue_info.update_on_dequeue()
        assert queue_info.items_pending == 0


class TestQueueInfoConcurrentOperations:
    """Tests for QueueInfo thread-safety under concurrent access."""

    @pytest.fixture
    def queue_info(self):
        """Create a QueueInfo instance for testing."""
        mock_queue = asyncio.PriorityQueue()
        rate_config = RateLimitBucket(
            model_id="test-model",
            resource_type="chat",
            rpm_limit=100,
        )
        return QueueInfo(
            queue_key="test-key",
            model_id="test-model",
            resource_type="chat",
            queue=mock_queue,
            rate_config=rate_config,
        )

    @pytest.mark.asyncio
    async def test_update_on_enqueue_concurrent(self, queue_info):
        """Run 100 concurrent update_on_enqueue calls and verify counters are correct."""
        num_operations = 100
        priority_value = 1.0

        async def enqueue_task():
            await queue_info.update_on_enqueue(priority=priority_value)

        # Run 100 concurrent enqueue operations
        tasks = [enqueue_task() for _ in range(num_operations)]
        await asyncio.gather(*tasks)

        # Verify counters are accurate after concurrent operations
        assert queue_info.total_enqueued == num_operations
        assert queue_info.priority_sum == num_operations * priority_value
        assert queue_info.avg_priority == priority_value

    @pytest.mark.asyncio
    async def test_update_on_dequeue_concurrent(self, queue_info):
        """Run 100 concurrent update_on_dequeue calls and verify counters are correct."""
        num_operations = 100

        # First, enqueue 100 items
        for i in range(num_operations):
            await queue_info.update_on_enqueue(priority=float(i))

        async def dequeue_task():
            await queue_info.update_on_dequeue()

        # Run 100 concurrent dequeue operations
        tasks = [dequeue_task() for _ in range(num_operations)]
        await asyncio.gather(*tasks)

        # Verify counters are accurate after concurrent operations
        assert queue_info.total_dequeued == num_operations
        assert queue_info.items_pending == 0

    @pytest.mark.asyncio
    async def test_mixed_concurrent_operations(self, queue_info):
        """Run concurrent enqueue and dequeue operations."""
        num_enqueue = 50
        num_dequeue = 30

        async def enqueue_task(priority: float):
            await queue_info.update_on_enqueue(priority=priority)

        async def dequeue_task():
            await queue_info.update_on_dequeue()

        # Create mixed tasks
        enqueue_tasks = [enqueue_task(float(i)) for i in range(num_enqueue)]
        dequeue_tasks = [dequeue_task() for _ in range(num_dequeue)]

        # Run all tasks concurrently
        all_tasks = enqueue_tasks + dequeue_tasks
        await asyncio.gather(*all_tasks)

        # Verify final state
        assert queue_info.total_enqueued == num_enqueue
        assert queue_info.total_dequeued == num_dequeue
        assert queue_info.items_pending == num_enqueue - num_dequeue

        # Verify priority tracking
        expected_sum = sum(float(i) for i in range(num_enqueue))
        assert queue_info.priority_sum == expected_sum
        assert queue_info.min_priority_seen == 0.0
        assert queue_info.max_priority_seen == float(num_enqueue - 1)

    @pytest.mark.asyncio
    async def test_concurrent_enqueue_with_varying_priorities(self, queue_info):
        """Test concurrent enqueues with different priority values."""
        priorities = [1.0, 5.0, 10.0, 15.0, 20.0] * 20  # 100 items

        async def enqueue_task(priority: float):
            await queue_info.update_on_enqueue(priority=priority)

        tasks = [enqueue_task(p) for p in priorities]
        await asyncio.gather(*tasks)

        assert queue_info.total_enqueued == 100
        assert queue_info.priority_sum == sum(priorities)
        assert queue_info.min_priority_seen == 1.0
        assert queue_info.max_priority_seen == 20.0
        assert queue_info.avg_priority == sum(priorities) / 100


class TestQueueInfoProperties:
    """Tests for QueueInfo property accessors."""

    @pytest.fixture
    def queue_info(self):
        """Create a QueueInfo instance for testing."""
        mock_queue = asyncio.PriorityQueue()
        rate_config = RateLimitBucket(
            model_id="test-model",
            resource_type="chat",
            rpm_limit=100,
        )
        return QueueInfo(
            queue_key="test-key",
            model_id="test-model",
            resource_type="chat",
            queue=mock_queue,
            rate_config=rate_config,
        )

    def test_current_size_empty(self, queue_info):
        """Test current_size returns 0 for empty queue."""
        assert queue_info.current_size == 0

    def test_is_empty_true(self, queue_info):
        """Test is_empty returns True for empty queue."""
        assert queue_info.is_empty is True

    @pytest.mark.asyncio
    async def test_current_size_after_put(self, queue_info):
        """Test current_size after adding items to queue."""
        await queue_info.queue.put((1, "item1"))
        await queue_info.queue.put((2, "item2"))
        assert queue_info.current_size == 2
        assert queue_info.is_empty is False

    def test_get_priority_for_scheduling(self, queue_info):
        """Test get_priority_for_scheduling returns avg_priority."""
        # With no items, should return 0.0
        assert queue_info.get_priority_for_scheduling() == 0.0

    @pytest.mark.asyncio
    async def test_get_priority_for_scheduling_with_items(self, queue_info):
        """Test get_priority_for_scheduling after enqueues."""
        await queue_info.update_on_enqueue(priority=10.0)
        await queue_info.update_on_enqueue(priority=20.0)

        # Should return average: (10 + 20) / 2 = 15.0
        assert queue_info.get_priority_for_scheduling() == 15.0
