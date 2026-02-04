"""
Shared fixtures for benchmark tests.
"""

import pytest
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

from adaptive_rate_limiter.backends.memory import MemoryBackend
from adaptive_rate_limiter.scheduler.config import RateLimiterConfig, SchedulerMode
from adaptive_rate_limiter.scheduler.state import StateManager
from adaptive_rate_limiter.providers.base import (
    ProviderInterface,
    RateLimitInfo,
    DiscoveredBucket,
)


class BenchmarkProvider(ProviderInterface):
    """Minimal provider for benchmarking overhead only."""

    @property
    def name(self) -> str:
        return "benchmark"

    async def discover_limits(
        self,
        force_refresh: bool = False,
        timeout: float = 30.0,
    ) -> Dict[str, DiscoveredBucket]:
        return {
            "benchmark-model": DiscoveredBucket(
                bucket_id="benchmark-model",
                rpm_limit=10000,  # High limit for benchmarks
                tpm_limit=1000000,
            )
        }

    def parse_rate_limit_response(
        self,
        headers: Dict[str, str],
        body: Optional[Dict[str, Any]] = None,
        status_code: Optional[int] = None,
    ) -> RateLimitInfo:
        return RateLimitInfo(
            rpm_remaining=9999,
            rpm_limit=10000,
            is_rate_limited=False,
        )

    async def get_bucket_for_model(
        self,
        model_id: str,
        resource_type: Optional[str] = None,
    ) -> str:
        return model_id


class BenchmarkClassifier:
    """Minimal classifier for benchmarking."""

    async def classify(self, request: Dict[str, Any]) -> Any:
        from adaptive_rate_limiter.types.request import RequestMetadata

        return RequestMetadata(
            request_id="bench-1",
            model_id="benchmark-model",
            resource_type="text",
            estimated_tokens=100,
        )


class BenchmarkClient:
    """Minimal client for benchmarking."""

    @property
    def base_url(self) -> str:
        return "https://benchmark.test"

    @property
    def timeout(self) -> float:
        return 30.0

    def get_headers(self) -> Dict[str, str]:
        return {"Authorization": "Bearer benchmark"}


@pytest.fixture
async def memory_backend():
    """Create and start a memory backend for benchmarks."""
    backend = MemoryBackend()
    await backend.start()
    yield backend
    await backend.stop()


@pytest.fixture
def benchmark_config():
    """Configuration optimized for benchmarking."""
    return RateLimiterConfig(
        mode=SchedulerMode.INTELLIGENT,
        max_queue_size=10000,
        max_concurrent_executions=1000,
    )


@pytest.fixture
async def state_manager(memory_backend):
    """State manager for benchmarks."""
    sm = StateManager(backend=memory_backend)
    await sm.start()
    yield sm
    await sm.stop()


@pytest.fixture
def mock_scheduler():
    """Create a mock scheduler with minimal interface for benchmarks."""
    scheduler = MagicMock()
    scheduler.metrics_enabled = False
    scheduler.metrics = {}
    scheduler._circuit_breaker_always_closed = True
    scheduler.circuit_breaker = None
    scheduler.extract_response_headers = MagicMock(return_value={})
    return scheduler


@pytest.fixture
def benchmark_provider():
    """Benchmark provider instance."""
    return BenchmarkProvider()


@pytest.fixture
def benchmark_classifier():
    """Benchmark classifier instance."""
    return BenchmarkClassifier()


@pytest.fixture
def benchmark_client():
    """Benchmark client instance."""
    return BenchmarkClient()
