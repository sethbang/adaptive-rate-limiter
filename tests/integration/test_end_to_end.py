"""
End-to-end integration tests for the adaptive rate limiter.

These tests verify complete request flows from submission through execution
and cleanup, including both success and error paths.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from adaptive_rate_limiter.backends.memory import MemoryBackend
from adaptive_rate_limiter.providers.base import (
    DiscoveredBucket,
    ProviderInterface,
    RateLimitInfo,
)
from adaptive_rate_limiter.scheduler.config import RateLimiterConfig, SchedulerMode
from adaptive_rate_limiter.scheduler.state import StateManager
from adaptive_rate_limiter.strategies.modes.intelligent import IntelligentModeStrategy
from adaptive_rate_limiter.types.request import RequestMetadata


class MockProvider(ProviderInterface):
    """Minimal provider for testing."""

    @property
    def name(self) -> str:
        return "mock"

    async def discover_limits(
        self,
        force_refresh: bool = False,
        timeout: float = 30.0,
    ) -> dict[str, DiscoveredBucket]:
        return {
            "test-model": DiscoveredBucket(
                bucket_id="test-model",
                rpm_limit=100,
                tpm_limit=10000,
            )
        }

    def parse_rate_limit_response(
        self,
        headers: dict[str, str],
        body: dict[str, Any] | None = None,
        status_code: int | None = None,
    ) -> RateLimitInfo:
        return RateLimitInfo(
            rpm_remaining=90,
            rpm_limit=100,
            is_rate_limited=False,
        )

    async def get_bucket_for_model(
        self,
        model_id: str,
        resource_type: str | None = None,
    ) -> str:
        return model_id


class MockClassifier:
    """Minimal classifier for testing."""

    async def classify(self, request: dict[str, Any]) -> RequestMetadata:
        return RequestMetadata(
            request_id="test-123",
            model_id="test-model",
            resource_type="text",
            estimated_tokens=100,
        )


class MockClient:
    """Minimal client for testing."""

    @property
    def base_url(self) -> str:
        return "https://api.test.com"

    @property
    def timeout(self) -> float:
        return 30.0

    def get_headers(self) -> dict[str, str]:
        return {"Authorization": "Bearer test"}


class TestEndToEndFlows:
    """Test complete request lifecycles."""

    @pytest.fixture
    async def backend(self):
        backend = MemoryBackend()
        await backend.start()
        yield backend
        await backend.stop()

    @pytest.fixture
    def config(self):
        return RateLimiterConfig(
            mode=SchedulerMode.INTELLIGENT,
            max_queue_size=100,
            max_concurrent_executions=10,
        )

    @pytest.fixture
    async def state_manager(self, backend):
        sm = StateManager(backend=backend)
        await sm.start()
        yield sm
        await sm.stop()

    @pytest.fixture
    def mock_scheduler(self):
        """Create a mock scheduler with minimal interface."""
        scheduler = MagicMock()
        scheduler.metrics_enabled = False
        scheduler.metrics = {}
        scheduler._circuit_breaker_always_closed = True
        scheduler.circuit_breaker = None
        scheduler.extract_response_headers = MagicMock(return_value={})
        return scheduler

    @pytest.mark.asyncio
    async def test_successful_request_flow(
        self, backend, config, state_manager, mock_scheduler
    ):
        """Test a complete successful request flow."""
        strategy = IntelligentModeStrategy(
            scheduler=mock_scheduler,
            config=config,
            client=MockClient(),
            provider=MockProvider(),
            classifier=MockClassifier(),
            state_manager=state_manager,
        )

        await strategy.start()

        try:
            # Create test request
            metadata = RequestMetadata(
                request_id="e2e-test-1",
                model_id="test-model",
                resource_type="text",
                estimated_tokens=50,
            )

            async def mock_request():
                return {"result": "success", "tokens": 45}

            # Submit request
            result = await strategy.submit_request(metadata, mock_request)

            assert result is not None
            assert result.request is not None
            assert result.should_retry is True

        finally:
            await strategy.stop()

    @pytest.mark.asyncio
    async def test_reservation_cleanup_on_completion(
        self, backend, config, state_manager, mock_scheduler
    ):
        """Test that reservations are properly cleaned up after request completion."""
        strategy = IntelligentModeStrategy(
            scheduler=mock_scheduler,
            config=config,
            client=MockClient(),
            provider=MockProvider(),
            classifier=MockClassifier(),
            state_manager=state_manager,
        )

        await strategy.start()

        try:
            # Store a reservation
            await strategy._reservation_tracker.store(
                request_id="cleanup-test",
                bucket_id="test-model",
                reservation_id="res-123",
                estimated_tokens=100,
            )

            # Verify it exists
            ctx = await strategy._reservation_tracker.get(
                request_id="cleanup-test",
                bucket_id="test-model",
            )
            assert ctx is not None

            # Clear it
            cleared = await strategy._reservation_tracker.get_and_clear(
                request_id="cleanup-test",
                bucket_id="test-model",
            )
            assert cleared is not None
            assert cleared.reservation_id == "res-123"

            # Verify it's gone
            ctx_after = await strategy._reservation_tracker.get(
                request_id="cleanup-test",
                bucket_id="test-model",
            )
            assert ctx_after is None

        finally:
            await strategy.stop()

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(
        self, backend, config, state_manager, mock_scheduler
    ):
        """Test handling multiple concurrent requests."""
        strategy = IntelligentModeStrategy(
            scheduler=mock_scheduler,
            config=config,
            client=MockClient(),
            provider=MockProvider(),
            classifier=MockClassifier(),
            state_manager=state_manager,
        )

        await strategy.start()

        try:
            # Submit multiple requests
            results = []
            for i in range(5):
                metadata = RequestMetadata(
                    request_id=f"concurrent-{i}",
                    model_id="test-model",
                    resource_type="text",
                    estimated_tokens=20,
                )

                async def make_request(idx=i):
                    return {"result": f"success-{idx}"}

                result = await strategy.submit_request(metadata, make_request)
                results.append(result)

            # All should be submitted
            assert len(results) == 5
            for result in results:
                assert result is not None
                assert result.request is not None

        finally:
            await strategy.stop()

    @pytest.mark.asyncio
    async def test_queue_management(
        self, backend, config, state_manager, mock_scheduler
    ):
        """Test queue creation and tracking."""
        strategy = IntelligentModeStrategy(
            scheduler=mock_scheduler,
            config=config,
            client=MockClient(),
            provider=MockProvider(),
            classifier=MockClassifier(),
            state_manager=state_manager,
        )

        await strategy.start()

        try:
            # Submit a request to create a queue
            metadata = RequestMetadata(
                request_id="queue-test-1",
                model_id="test-model",
                resource_type="text",
                estimated_tokens=50,
            )

            async def mock_request():
                return {"result": "success"}

            await strategy.submit_request(metadata, mock_request)

            # Verify queue was created
            queue_key = "test-model:text"
            assert queue_key in strategy.fast_queues
            assert strategy._queue_has_items.get(queue_key, False) is True

        finally:
            await strategy.stop()

    @pytest.mark.asyncio
    async def test_metrics_tracking(
        self, backend, config, state_manager, mock_scheduler
    ):
        """Test that metrics are properly tracked."""
        # Enable metrics
        mock_scheduler.metrics_enabled = True
        mock_scheduler.metrics = {
            "requests_scheduled": 0,
            "requests_completed": 0,
            "requests_failed": 0,
            "queue_overflows": 0,
            "scheduler_loops": 0,
        }

        strategy = IntelligentModeStrategy(
            scheduler=mock_scheduler,
            config=config,
            client=MockClient(),
            provider=MockProvider(),
            classifier=MockClassifier(),
            state_manager=state_manager,
        )

        await strategy.start()

        try:
            # Submit a request
            metadata = RequestMetadata(
                request_id="metrics-test-1",
                model_id="test-model",
                resource_type="text",
                estimated_tokens=50,
            )

            async def mock_request():
                return {"result": "success"}

            await strategy.submit_request(metadata, mock_request)

            # Check that requests_scheduled was incremented
            assert mock_scheduler.metrics["requests_scheduled"] == 1

            # Get strategy metrics
            metrics = strategy.get_metrics()
            assert metrics["mode"] == "intelligent"
            assert "reservation_metrics" in metrics
            assert "streaming_metrics" in metrics

        finally:
            await strategy.stop()
