import time
from datetime import datetime, timezone

import pytest

from adaptive_rate_limiter.backends.base import (
    BaseBackend,
    HealthCheckResult,
    validate_safety_margin,
)


class ConcreteBackend(BaseBackend):
    """Concrete implementation of BaseBackend for testing."""

    async def get_state(self, key):
        return {}

    async def set_state(self, key, state):
        pass  # type: ignore

    async def get_all_states(self):
        return {}

    async def clear(self):
        pass

    async def check_and_reserve_capacity(
        self,
        key,
        requests,
        tokens,
        bucket_limits=None,
        safety_margin=1.0,
        request_id=None,
    ):
        return True, "res-id"

    async def release_reservation(self, key, reservation_id):
        return True

    async def release_streaming_reservation(
        self, key, reservation_id, reserved_tokens, actual_tokens
    ):
        return True

    async def check_capacity(self, model, request_type="default"):
        return True, 0.0

    async def record_request(self, model, request_type="default", tokens_used=None):
        pass

    async def record_failure(self, error_type, error_message=""):
        pass

    async def get_failure_count(self, window_seconds=30):
        return 0

    async def is_circuit_broken(self):
        return False

    async def update_rate_limits(
        self, model, headers, bucket_id=None, request_id=None, status_code=None
    ):
        return 0

    async def get_rate_limits(self, model):
        return {}

    async def reserve_capacity(self, model, request_id, tokens_estimated=0):
        return True

    async def release_reservation_by_id(self, request_id):
        pass

    async def cache_bucket_info(self, bucket_data, ttl_seconds=3600):
        pass

    async def get_cached_bucket_info(self):
        return {}

    async def cache_model_info(self, model, model_data, ttl_seconds=3600):
        pass

    async def get_cached_model_info(self, model):
        return {}

    async def health_check(self):
        return HealthCheckResult(healthy=True, backend_type="test", namespace="test")

    async def get_all_stats(self):
        return {}

    async def cleanup(self):
        pass

    async def clear_failures(self):
        pass

    async def force_circuit_break(self, duration):
        pass


class TestBaseBackendHelpers:
    def test_validate_safety_margin_valid(self):
        validate_safety_margin(0.0)
        validate_safety_margin(0.5)
        validate_safety_margin(1.0)
        validate_safety_margin(1)

    def test_validate_safety_margin_invalid_type(self):
        with pytest.raises(ValueError, match="must be a number"):
            validate_safety_margin("0.5")  # type: ignore

    def test_validate_safety_margin_invalid_range(self):
        with pytest.raises(ValueError, match=r"must be between 0\.0 and 1\.0"):
            validate_safety_margin(-0.1)
        with pytest.raises(ValueError, match=r"must be between 0\.0 and 1\.0"):
            validate_safety_margin(1.1)


class TestBaseBackendConcreteMethods:
    @pytest.fixture
    def backend(self):
        return ConcreteBackend(namespace="test")  # type: ignore

    def test_init(self, backend):
        assert backend.namespace == "test"

    def test_parse_rate_limit_headers_empty(self, backend):
        headers = {}
        result = backend._parse_rate_limit_headers(headers)
        assert "timestamp" in result
        assert len(result) == 1

    def test_parse_rate_limit_headers_full(self, backend):
        headers = {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "50",
            "x-ratelimit-reset-requests": "10",
            "x-ratelimit-limit-tokens": "1000",
            "x-ratelimit-remaining-tokens": "500",
            "x-ratelimit-reset-tokens": "20",
            "retry-after": "5",
        }
        result = backend._parse_rate_limit_headers(headers)

        assert result["rpm_limit"] == 100
        assert result["rpm_remaining"] == 50
        assert result["rpm_reset"] == 10
        assert result["tpm_limit"] == 1000
        assert result["tpm_remaining"] == 500
        assert result["tpm_reset"] == 20
        assert result["retry_after"] == 5
        assert "timestamp" in result

    def test_parse_reset_time_seconds_timestamp(self, backend):
        # Future timestamp in seconds
        future_ts = time.time() + 60
        result = backend._parse_reset_time(str(future_ts))
        assert abs(result - future_ts) < 1.0

    def test_parse_reset_time_milliseconds_timestamp(self, backend):
        # Future timestamp in milliseconds
        future_ts = time.time() + 60
        future_ms = future_ts * 1000
        result = backend._parse_reset_time(str(future_ms))
        assert abs(result - future_ts) < 1.0

    def test_parse_reset_time_delta(self, backend):
        # Delta seconds
        delta = 60
        expected = time.time() + delta
        result = backend._parse_reset_time(str(delta))
        assert abs(result - expected) < 1.0

    def test_parse_reset_time_iso(self, backend):
        # ISO format
        dt = datetime.now(timezone.utc)
        iso_str = dt.isoformat()
        result = backend._parse_reset_time(iso_str)
        assert abs(result - dt.timestamp()) < 1.0

    def test_parse_reset_time_invalid(self, backend):
        # Invalid format should fallback to 60s
        expected = time.time() + 60
        result = backend._parse_reset_time("invalid")
        assert abs(result - expected) < 1.0

    def test_calculate_wait_time_no_limits(self, backend):
        assert backend._calculate_wait_time({}) == 0.0

    def test_calculate_wait_time_rpm_exhausted(self, backend):
        reset_time = time.time() + 10
        rate_limits = {"rpm_remaining": 0, "rpm_reset": str(reset_time)}
        wait_time = backend._calculate_wait_time(rate_limits)
        assert abs(wait_time - 10) < 1.0

    def test_calculate_wait_time_tpm_exhausted(self, backend):
        reset_time = time.time() + 10
        rate_limits = {"tpm_remaining": 0, "tpm_reset": str(reset_time)}
        wait_time = backend._calculate_wait_time(rate_limits)
        assert abs(wait_time - 10) < 1.0

    def test_calculate_wait_time_retry_after(self, backend):
        rate_limits = {"retry_after": 5}
        wait_time = backend._calculate_wait_time(rate_limits)
        assert wait_time == 5

    def test_calculate_wait_time_max_wait(self, backend):
        # Should take the maximum wait time
        reset_time = time.time() + 10
        rate_limits = {
            "rpm_remaining": 0,
            "rpm_reset": str(reset_time),
            "retry_after": 5,
        }
        wait_time = backend._calculate_wait_time(rate_limits)
        assert abs(wait_time - 10) < 1.0

    def test_calculate_wait_time_has_capacity(self, backend):
        rate_limits = {"rpm_remaining": 10, "tpm_remaining": 100}
        assert backend._calculate_wait_time(rate_limits) == 0.0

    def test_calculate_wait_time_rpm_exhausted_no_reset(self, backend):
        """Covers line 571: rpm_remaining <= 0 but rpm_reset missing."""
        rate_limits = {"rpm_remaining": 0}  # No rpm_reset
        assert backend._calculate_wait_time(rate_limits) == 0.0

    def test_calculate_wait_time_tpm_exhausted_no_reset(self, backend):
        """Covers line 579: tpm_remaining <= 0 but tpm_reset missing."""
        rate_limits = {"tpm_remaining": 0}  # No tpm_reset
        assert backend._calculate_wait_time(rate_limits) == 0.0

    def test_parse_rate_limit_headers_malformed_rpm_limit(self, backend):
        """Covers lines 532-533: malformed x-ratelimit-limit-requests header."""
        headers = {"x-ratelimit-limit-requests": "invalid"}
        result = backend._parse_rate_limit_headers(headers)
        assert "rpm_limit" not in result
        assert "timestamp" in result

    def test_parse_rate_limit_headers_malformed_rpm_remaining(self, backend):
        """Covers lines 540-541: malformed x-ratelimit-remaining-requests header."""
        headers = {"x-ratelimit-remaining-requests": "not_a_number"}
        result = backend._parse_rate_limit_headers(headers)
        assert "rpm_remaining" not in result
        assert "timestamp" in result

    def test_parse_rate_limit_headers_malformed_rpm_reset(self, backend):
        """Covers lines 548-549: malformed x-ratelimit-reset-requests header."""
        headers = {"x-ratelimit-reset-requests": "abc"}
        result = backend._parse_rate_limit_headers(headers)
        assert "rpm_reset" not in result
        assert "timestamp" in result

    def test_parse_rate_limit_headers_malformed_tpm_limit(self, backend):
        """Covers lines 558-559: malformed x-ratelimit-limit-tokens header."""
        headers = {"x-ratelimit-limit-tokens": "xyz"}
        result = backend._parse_rate_limit_headers(headers)
        assert "tpm_limit" not in result
        assert "timestamp" in result

    def test_parse_rate_limit_headers_malformed_tpm_remaining(self, backend):
        """Covers lines 566-567: malformed x-ratelimit-remaining-tokens header."""
        headers = {"x-ratelimit-remaining-tokens": "!@#"}
        result = backend._parse_rate_limit_headers(headers)
        assert "tpm_remaining" not in result
        assert "timestamp" in result

    def test_parse_rate_limit_headers_malformed_tpm_reset(self, backend):
        """Covers lines 574-575: malformed x-ratelimit-reset-tokens header."""
        headers = {"x-ratelimit-reset-tokens": "bad_value"}
        result = backend._parse_rate_limit_headers(headers)
        assert "tpm_reset" not in result
        assert "timestamp" in result

    def test_parse_rate_limit_headers_malformed_retry_after(self, backend):
        """Covers lines 584-585: malformed retry-after header."""
        headers = {"retry-after": "not_valid"}
        result = backend._parse_rate_limit_headers(headers)
        assert "retry_after" not in result
        assert "timestamp" in result

    def test_parse_rate_limit_headers_mixed_valid_invalid(self, backend):
        """Test mix of valid and malformed headers."""
        headers = {
            "x-ratelimit-limit-requests": "100",  # valid
            "x-ratelimit-remaining-requests": "invalid",  # invalid
            "x-ratelimit-limit-tokens": "1000",  # valid
            "x-ratelimit-remaining-tokens": "bad",  # invalid
            "retry-after": "5.5",  # valid (float)
        }
        result = backend._parse_rate_limit_headers(headers)
        assert result["rpm_limit"] == 100
        assert "rpm_remaining" not in result
        assert result["tpm_limit"] == 1000
        assert "tpm_remaining" not in result
        assert result["retry_after"] == 5  # converted to int
        assert "timestamp" in result

    def test_parse_rate_limit_headers_float_retry_after(self, backend):
        """Test retry-after with float value."""
        headers = {"retry-after": "10.7"}
        result = backend._parse_rate_limit_headers(headers)
        assert result["retry_after"] == 10  # int(float("10.7")) = 10


class TestHealthCheckResult:
    """Tests for HealthCheckResult dataclass."""

    def test_health_check_result_basic(self):
        """Test HealthCheckResult with minimal fields."""
        result = HealthCheckResult(
            healthy=True,
            backend_type="memory",
            namespace="test_ns",
        )
        assert result.healthy is True
        assert result.backend_type == "memory"
        assert result.namespace == "test_ns"
        assert result.error is None
        assert result.metadata is None

    def test_health_check_result_with_error(self):
        """Test HealthCheckResult with error."""
        result = HealthCheckResult(
            healthy=False,
            backend_type="redis",
            namespace="prod",
            error="Connection refused",
        )
        assert result.healthy is False
        assert result.backend_type == "redis"
        assert result.namespace == "prod"
        assert result.error == "Connection refused"
        assert result.metadata is None

    def test_health_check_result_with_metadata(self):
        """Test HealthCheckResult with metadata."""
        metadata = {"version": "7.0", "connected_clients": 42}
        result = HealthCheckResult(
            healthy=True,
            backend_type="redis",
            namespace="default",
            metadata=metadata,
        )
        assert result.healthy is True
        assert result.metadata == metadata
        assert result.metadata is not None
        assert result.metadata["version"] == "7.0"

    def test_health_check_result_full(self):
        """Test HealthCheckResult with all fields."""
        result = HealthCheckResult(
            healthy=False,
            backend_type="redis_cluster",
            namespace="cluster-ns",
            error="Cluster node down",
            metadata={"nodes_ok": 5, "nodes_failed": 1},
        )
        assert result.healthy is False
        assert result.backend_type == "redis_cluster"
        assert result.namespace == "cluster-ns"
        assert result.error == "Cluster node down"
        assert result.metadata is not None
        assert result.metadata["nodes_ok"] == 5
