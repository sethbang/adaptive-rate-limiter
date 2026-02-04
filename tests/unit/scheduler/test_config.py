import pytest

from adaptive_rate_limiter.scheduler.config import (
    CachePolicy,
    RateLimiterConfig,
    SchedulerMode,
    StateConfig,
)


class TestSchedulerMode:
    def test_enum_values(self):
        """Test SchedulerMode enum values."""
        assert SchedulerMode.BASIC.value == "basic"
        assert SchedulerMode.INTELLIGENT.value == "intelligent"
        assert SchedulerMode.ACCOUNT.value == "account"


class TestCachePolicy:
    def test_enum_values(self):
        """Test CachePolicy enum values."""
        assert CachePolicy.WRITE_THROUGH.value == "write_through"
        assert CachePolicy.WRITE_BACK.value == "write_back"
        assert CachePolicy.WRITE_AROUND.value == "write_around"


class TestRateLimiterConfig:
    def test_default_values(self):
        """Test default configuration values."""
        config = RateLimiterConfig()

        # Core Scheduling
        assert config.mode == SchedulerMode.INTELLIGENT
        assert config.max_concurrent_executions == 100
        assert config.max_queue_size == 1000
        assert config.overflow_policy == "reject"
        assert config.scheduler_interval == 0.01

        # Request Processing
        assert config.request_timeout == 30.0
        assert config.enable_priority_scheduling is True

        # Rate Limiting
        assert config.enable_rate_limiting is True
        assert config.rate_limit_buffer_ratio == 0.9

        # Failure Handling
        assert config.failure_window == 30.0
        assert config.max_failures == 20
        assert config.backoff_base == 2.0
        assert config.max_backoff == 60.0

        # Graceful Degradation
        assert config.enable_graceful_degradation is True
        assert config.health_check_interval == 30.0
        assert config.max_consecutive_failures == 3
        assert config.conservative_multiplier == 0.6

        # Metrics
        assert config.metrics_enabled is True
        assert config.prometheus_host == "0.0.0.0"  # noqa: S104
        assert config.prometheus_port == 9090
        assert config.metrics_export_interval == 60.0
        assert config.enable_performance_tracking is True

        # State Management
        assert config.enable_state_persistence is True

        # Testing
        assert config.test_mode is False
        assert config.test_rate_multiplier == 1.0

    def test_validation_rate_limit_buffer_ratio(self):
        """Test validation of rate_limit_buffer_ratio."""
        # Valid values
        RateLimiterConfig(rate_limit_buffer_ratio=0.1)
        RateLimiterConfig(rate_limit_buffer_ratio=1.0)

        # Invalid values
        with pytest.raises(ValueError, match="rate_limit_buffer_ratio"):
            RateLimiterConfig(rate_limit_buffer_ratio=0.0)
        with pytest.raises(ValueError, match="rate_limit_buffer_ratio"):
            RateLimiterConfig(rate_limit_buffer_ratio=1.1)
        with pytest.raises(ValueError, match="rate_limit_buffer_ratio"):
            RateLimiterConfig(rate_limit_buffer_ratio=-0.5)

    def test_validation_conservative_multiplier(self):
        """Test validation of conservative_multiplier."""
        # Valid values
        RateLimiterConfig(conservative_multiplier=0.1)
        RateLimiterConfig(conservative_multiplier=1.0)

        # Invalid values
        with pytest.raises(ValueError, match="conservative_multiplier"):
            RateLimiterConfig(conservative_multiplier=0.0)
        with pytest.raises(ValueError, match="conservative_multiplier"):
            RateLimiterConfig(conservative_multiplier=1.1)
        with pytest.raises(ValueError, match="conservative_multiplier"):
            RateLimiterConfig(conservative_multiplier=-0.5)

    def test_validation_max_concurrent_executions(self):
        """Test validation of max_concurrent_executions."""
        # Valid values
        RateLimiterConfig(max_concurrent_executions=1)
        RateLimiterConfig(max_concurrent_executions=100)

        # Invalid values
        with pytest.raises(ValueError, match="max_concurrent_executions"):
            RateLimiterConfig(max_concurrent_executions=0)
        with pytest.raises(ValueError, match="max_concurrent_executions"):
            RateLimiterConfig(max_concurrent_executions=-1)

    def test_validation_max_queue_size(self):
        """Test validation of max_queue_size."""
        # Valid values
        RateLimiterConfig(max_queue_size=1)
        RateLimiterConfig(max_queue_size=1000)

        # Invalid values
        with pytest.raises(ValueError, match="max_queue_size"):
            RateLimiterConfig(max_queue_size=0)
        with pytest.raises(ValueError, match="max_queue_size"):
            RateLimiterConfig(max_queue_size=-1)

    def test_validation_overflow_policy(self):
        """Test validation of overflow_policy."""
        # Valid values
        RateLimiterConfig(overflow_policy="reject")
        RateLimiterConfig(overflow_policy="drop_oldest")

        # Invalid values
        with pytest.raises(ValueError, match="overflow_policy"):
            RateLimiterConfig(overflow_policy="invalid")
        with pytest.raises(ValueError, match="overflow_policy"):
            RateLimiterConfig(overflow_policy="")


class TestStateConfig:
    def test_default_values(self):
        """Test default configuration values."""
        config = StateConfig()

        # Cache configuration
        assert config.cache_ttl == 1.0
        assert config.max_cache_size == 1000
        # Default is WRITE_THROUGH for production safety (prevents data loss on crashes)
        assert config.cache_policy == CachePolicy.WRITE_THROUGH

        # Production safety
        assert config.warn_write_back_production is True
        assert config.is_production is False

        # Batch processing
        assert config.batch_size == 50
        assert config.batch_timeout == 0.1

        # Cleanup
        assert config.cleanup_interval == 30.0
        assert config.enable_background_cleanup is True
        assert config.cleanup_task_cancel_timeout == 2.0
        assert config.cleanup_task_wait_timeout == 1.0

        # State persistence
        assert config.state_ttl == 3600

        # Reservation
        assert config.reservation_cleanup_interval == 3600.0
        assert config.reservation_ttl == 300.0

        # Account state
        assert config.account_state_ttl == 86400.0
        assert config.account_state_max_size == 10000

        # Versioning
        assert config.enable_versioning is True
        assert config.max_versions == 10

        # Concurrency
        assert config.lock_free_reads is True
        assert config.max_concurrent_operations == 100

        # Namespace
        assert config.namespace == "default"

    def test_validation_cache_ttl(self):
        """Test validation of cache_ttl."""
        # Valid values
        StateConfig(cache_ttl=0.1)
        StateConfig(cache_ttl=100.0)

        # Invalid values
        with pytest.raises(ValueError, match="cache_ttl"):
            StateConfig(cache_ttl=0.0)
        with pytest.raises(ValueError, match="cache_ttl"):
            StateConfig(cache_ttl=-1.0)

    def test_validation_max_cache_size(self):
        """Test validation of max_cache_size."""
        # Valid values
        StateConfig(max_cache_size=1)
        StateConfig(max_cache_size=None)

        # Invalid values
        with pytest.raises(ValueError, match="max_cache_size"):
            StateConfig(max_cache_size=0)
        with pytest.raises(ValueError, match="max_cache_size"):
            StateConfig(max_cache_size=-1)

    def test_validation_batch_size(self):
        """Test validation of batch_size."""
        # Valid values
        StateConfig(batch_size=1)
        StateConfig(batch_size=100)

        # Invalid values
        with pytest.raises(ValueError, match="batch_size"):
            StateConfig(batch_size=0)
        with pytest.raises(ValueError, match="batch_size"):
            StateConfig(batch_size=-1)

    def test_validation_state_ttl(self):
        """Test validation of state_ttl."""
        # Valid values
        StateConfig(state_ttl=60)
        StateConfig(state_ttl=3600)

        # Invalid values
        with pytest.raises(ValueError, match="state_ttl"):
            StateConfig(state_ttl=59)
        with pytest.raises(ValueError, match="state_ttl"):
            StateConfig(state_ttl=0)
