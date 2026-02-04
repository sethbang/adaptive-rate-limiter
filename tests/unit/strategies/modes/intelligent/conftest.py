"""
Shared fixtures for IntelligentModeStrategy tests.

This module contains all common fixtures used across the intelligent mode test suite.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from adaptive_rate_limiter.strategies.modes.intelligent import IntelligentModeStrategy
from adaptive_rate_limiter.types.rate_limit import RateLimitBucket
from adaptive_rate_limiter.types.request import RequestMetadata

# ============================================================================
# Mock Fixtures
# ============================================================================


@pytest.fixture
def mock_provider():
    """Create a mock provider."""
    provider = Mock()
    provider.get_bucket_for_model = AsyncMock(return_value="bucket-1")
    provider.discover_limits = AsyncMock(
        return_value={
            "bucket-1": RateLimitBucket(
                model_id="test-model",
                resource_type="chat",
                rpm_limit=100,
                tpm_limit=10000,
            )
        }
    )
    return provider


@pytest.fixture
def mock_classifier():
    """Create a mock classifier."""
    classifier = Mock()
    return classifier


@pytest.fixture
def mock_backend():
    """Create a mock backend."""
    backend = Mock()
    backend.check_and_reserve_capacity = AsyncMock(return_value=(True, "res-123"))
    backend.release_reservation = AsyncMock(return_value=True)
    backend.release_streaming_reservation = AsyncMock(return_value=True)
    return backend


@pytest.fixture
def mock_state_manager(mock_backend):
    """Create a mock state manager."""
    state_manager = Mock()
    state_manager.backend = mock_backend
    state_manager.get_state = AsyncMock(return_value=None)
    state_manager.update_state_from_headers = AsyncMock(return_value=1)
    state_manager.stop = AsyncMock()
    return state_manager


@pytest.fixture
def mock_scheduler():
    """Create a mock scheduler."""
    scheduler = Mock()
    scheduler.circuit_breaker = None
    scheduler._circuit_breaker_always_closed = True
    scheduler.metrics_enabled = False
    scheduler.metrics = {}
    scheduler.extract_response_headers = Mock(return_value={})
    return scheduler


@pytest.fixture
def mock_scheduler_with_metrics():
    """Create a mock scheduler with metrics enabled."""
    scheduler = Mock()
    scheduler.circuit_breaker = None
    scheduler._circuit_breaker_always_closed = True
    scheduler.metrics_enabled = True
    scheduler.metrics = {
        "scheduler_loops": 0,
        "requests_scheduled": 0,
        "requests_completed": 0,
        "requests_failed": 0,
        "queue_overflows": 0,
    }
    scheduler.extract_response_headers = Mock(return_value={})
    return scheduler


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = Mock()
    config.batch_size = 50
    config.scheduler_interval = 0.001
    config.rate_limit_buffer_ratio = 0.9
    config.max_queue_size = 1000
    config.overflow_policy = "reject"
    config.max_concurrent_executions = 100
    config.request_timeout = 30.0
    return config


@pytest.fixture
def mock_client():
    """Create a mock client."""
    return Mock()


# ============================================================================
# Strategy Fixture
# ============================================================================


@pytest.fixture
def strategy(
    mock_scheduler,
    mock_config,
    mock_client,
    mock_provider,
    mock_classifier,
    mock_state_manager,
):
    """Create an IntelligentModeStrategy instance."""
    return IntelligentModeStrategy(
        scheduler=mock_scheduler,
        config=mock_config,
        client=mock_client,
        provider=mock_provider,
        classifier=mock_classifier,
        state_manager=mock_state_manager,
    )


@pytest.fixture
def strategy_with_metrics(
    mock_scheduler_with_metrics,
    mock_config,
    mock_client,
    mock_provider,
    mock_classifier,
    mock_state_manager,
):
    """Create an IntelligentModeStrategy instance with metrics enabled."""
    return IntelligentModeStrategy(
        scheduler=mock_scheduler_with_metrics,
        config=mock_config,
        client=mock_client,
        provider=mock_provider,
        classifier=mock_classifier,
        state_manager=mock_state_manager,
    )


# ============================================================================
# Request Metadata Fixture
# ============================================================================


@pytest.fixture
def metadata():
    """Create request metadata."""
    return RequestMetadata(
        request_id="req-123",
        model_id="test-model",
        resource_type="chat",
        estimated_tokens=100,
        priority=0,
    )
