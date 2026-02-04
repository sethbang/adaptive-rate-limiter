"""
Unit tests for rate limit types.

These tests verify the structure, defaults, and behavior of:
- RateLimitType enum
- RateLimitBucket dataclass
- LimitCheckResult dataclass
"""

import pytest

from adaptive_rate_limiter.types import LimitCheckResult, RateLimitBucket, RateLimitType


class TestRateLimitType:
    """Tests for RateLimitType enum."""

    def test_rpm_value(self):
        """Test RPM enum value."""
        assert RateLimitType.RPM.value == "RPM"

    def test_rpd_value(self):
        """Test RPD enum value."""
        assert RateLimitType.RPD.value == "RPD"

    def test_tpm_value(self):
        """Test TPM enum value."""
        assert RateLimitType.TPM.value == "TPM"

    def test_all_enum_values(self):
        """Test that all expected enum members exist."""
        values = [e.value for e in RateLimitType]
        assert "RPM" in values
        assert "RPD" in values
        assert "TPM" in values
        assert len(values) == 3

    def test_enum_member_count(self):
        """Test that RateLimitType has exactly 3 members."""
        assert len(RateLimitType) == 3

    def test_enum_from_value(self):
        """Test creating enum from string value."""
        assert RateLimitType("RPM") == RateLimitType.RPM
        assert RateLimitType("RPD") == RateLimitType.RPD
        assert RateLimitType("TPM") == RateLimitType.TPM

    def test_enum_invalid_value(self):
        """Test that invalid values raise ValueError."""
        with pytest.raises(ValueError):
            RateLimitType("INVALID")

    def test_enum_comparison(self):
        """Test enum comparison."""
        assert RateLimitType.RPM == RateLimitType.RPM
        assert RateLimitType.RPM != RateLimitType.RPD
        assert RateLimitType.TPM != RateLimitType.RPD


class TestRateLimitBucket:
    """Tests for RateLimitBucket dataclass."""

    def test_required_fields_only(self):
        """Test creating RateLimitBucket with only required fields."""
        bucket = RateLimitBucket(
            model_id="gpt-5",
            resource_type="chat",
            rpm_limit=100,
        )

        assert bucket.model_id == "gpt-5"
        assert bucket.resource_type == "chat"
        assert bucket.rpm_limit == 100

    def test_default_rpd_limit(self):
        """Test that rpd_limit defaults to None."""
        bucket = RateLimitBucket(
            model_id="gpt-5",
            resource_type="chat",
            rpm_limit=100,
        )
        assert bucket.rpd_limit is None

    def test_default_tpm_limit(self):
        """Test that tpm_limit defaults to None."""
        bucket = RateLimitBucket(
            model_id="gpt-5",
            resource_type="chat",
            rpm_limit=100,
        )
        assert bucket.tpm_limit is None

    def test_all_fields_specified(self):
        """Test creating RateLimitBucket with all fields specified."""
        bucket = RateLimitBucket(
            model_id="gpt-5.1",
            resource_type="completions",
            rpm_limit=200,
            rpd_limit=10000,
            tpm_limit=40000,
        )

        assert bucket.model_id == "gpt-5.1"
        assert bucket.resource_type == "completions"
        assert bucket.rpm_limit == 200
        assert bucket.rpd_limit == 10000
        assert bucket.tpm_limit == 40000

    def test_zero_rpm_limit(self):
        """Test that zero rpm_limit is allowed."""
        bucket = RateLimitBucket(
            model_id="gpt-5",
            resource_type="chat",
            rpm_limit=0,
        )
        assert bucket.rpm_limit == 0

    def test_large_limits(self):
        """Test that large limit values are allowed."""
        bucket = RateLimitBucket(
            model_id="gpt-5",
            resource_type="chat",
            rpm_limit=1_000_000,
            rpd_limit=100_000_000,
            tpm_limit=10_000_000,
        )
        assert bucket.rpm_limit == 1_000_000
        assert bucket.rpd_limit == 100_000_000
        assert bucket.tpm_limit == 10_000_000

    def test_empty_string_model_id(self):
        """Test that empty string is allowed for model_id."""
        bucket = RateLimitBucket(
            model_id="",
            resource_type="chat",
            rpm_limit=100,
        )
        assert bucket.model_id == ""

    def test_empty_string_resource_type(self):
        """Test that empty string is allowed for resource_type."""
        bucket = RateLimitBucket(
            model_id="gpt-5",
            resource_type="",
            rpm_limit=100,
        )
        assert bucket.resource_type == ""

    def test_equality_same_values(self):
        """Test that two RateLimitBucket with same values are equal."""
        bucket1 = RateLimitBucket(
            model_id="gpt-5",
            resource_type="chat",
            rpm_limit=100,
            rpd_limit=1000,
        )
        bucket2 = RateLimitBucket(
            model_id="gpt-5",
            resource_type="chat",
            rpm_limit=100,
            rpd_limit=1000,
        )

        assert bucket1 == bucket2

    def test_inequality_different_values(self):
        """Test that two RateLimitBucket with different values are not equal."""
        bucket1 = RateLimitBucket(
            model_id="gpt-5",
            resource_type="chat",
            rpm_limit=100,
        )
        bucket2 = RateLimitBucket(
            model_id="gpt-5",
            resource_type="chat",
            rpm_limit=200,  # Different rpm_limit
        )

        assert bucket1 != bucket2


class TestLimitCheckResult:
    """Tests for LimitCheckResult dataclass."""

    def test_can_proceed_true(self):
        """Test creating a passing LimitCheckResult."""
        result = LimitCheckResult(can_proceed=True)

        assert result.can_proceed is True
        assert result.wait_time == 0.0
        assert result.reason is None
        assert result.limiting_factor is None
        assert result.remaining_requests is None
        assert result.remaining_tokens is None

    def test_can_proceed_false_with_defaults(self):
        """Test creating a failing LimitCheckResult with defaults."""
        result = LimitCheckResult(can_proceed=False)

        assert result.can_proceed is False
        assert result.wait_time == 0.0

    def test_default_wait_time(self):
        """Test that wait_time defaults to 0.0."""
        result = LimitCheckResult(can_proceed=True)
        assert result.wait_time == 0.0

    def test_default_reason(self):
        """Test that reason defaults to None."""
        result = LimitCheckResult(can_proceed=True)
        assert result.reason is None

    def test_default_limiting_factor(self):
        """Test that limiting_factor defaults to None."""
        result = LimitCheckResult(can_proceed=True)
        assert result.limiting_factor is None

    def test_default_remaining_requests(self):
        """Test that remaining_requests defaults to None."""
        result = LimitCheckResult(can_proceed=True)
        assert result.remaining_requests is None

    def test_default_remaining_tokens(self):
        """Test that remaining_tokens defaults to None."""
        result = LimitCheckResult(can_proceed=True)
        assert result.remaining_tokens is None

    def test_all_fields_specified_rate_limited(self):
        """Test creating a rate-limited LimitCheckResult with all fields."""
        result = LimitCheckResult(
            can_proceed=False,
            wait_time=5.5,
            reason="Rate limit exceeded for RPM",
            limiting_factor=RateLimitType.RPM,
            remaining_requests=0,
            remaining_tokens=5000,
        )

        assert result.can_proceed is False
        assert result.wait_time == 5.5
        assert result.reason == "Rate limit exceeded for RPM"
        assert result.limiting_factor == RateLimitType.RPM
        assert result.remaining_requests == 0
        assert result.remaining_tokens == 5000

    def test_limiting_factor_tpm(self):
        """Test LimitCheckResult with TPM as limiting factor."""
        result = LimitCheckResult(
            can_proceed=False,
            limiting_factor=RateLimitType.TPM,
            reason="Token limit exceeded",
        )

        assert result.limiting_factor == RateLimitType.TPM

    def test_limiting_factor_rpd(self):
        """Test LimitCheckResult with RPD as limiting factor."""
        result = LimitCheckResult(
            can_proceed=False,
            limiting_factor=RateLimitType.RPD,
            reason="Daily limit exceeded",
        )

        assert result.limiting_factor == RateLimitType.RPD

    def test_positive_remaining_values(self):
        """Test LimitCheckResult with positive remaining values."""
        result = LimitCheckResult(
            can_proceed=True,
            remaining_requests=50,
            remaining_tokens=10000,
        )

        assert result.remaining_requests == 50
        assert result.remaining_tokens == 10000

    def test_zero_remaining_values(self):
        """Test LimitCheckResult with zero remaining values."""
        result = LimitCheckResult(
            can_proceed=False,
            remaining_requests=0,
            remaining_tokens=0,
        )

        assert result.remaining_requests == 0
        assert result.remaining_tokens == 0

    def test_large_wait_time(self):
        """Test LimitCheckResult with large wait time."""
        result = LimitCheckResult(
            can_proceed=False,
            wait_time=86400.0,  # 24 hours
            reason="Daily limit exceeded, wait until reset",
        )

        assert result.wait_time == 86400.0

    def test_equality_same_values(self):
        """Test that two LimitCheckResult with same values are equal."""
        result1 = LimitCheckResult(
            can_proceed=True,
            wait_time=0.0,
            remaining_requests=100,
        )
        result2 = LimitCheckResult(
            can_proceed=True,
            wait_time=0.0,
            remaining_requests=100,
        )

        assert result1 == result2

    def test_inequality_different_values(self):
        """Test that two LimitCheckResult with different values are not equal."""
        result1 = LimitCheckResult(can_proceed=True)
        result2 = LimitCheckResult(can_proceed=False)

        assert result1 != result2


class TestRateLimitTypesIntegration:
    """Integration tests between rate limit types."""

    def test_limit_check_with_bucket_context(self):
        """Test that LimitCheckResult limiting_factor matches rate limit types in bucket."""
        bucket = RateLimitBucket(
            model_id="gpt-5",
            resource_type="chat",
            rpm_limit=100,
            rpd_limit=1000,
            tpm_limit=40000,
        )

        # Simulate RPM limit hit
        result = LimitCheckResult(
            can_proceed=False,
            wait_time=1.0,
            reason=f"RPM limit ({bucket.rpm_limit}) exceeded",
            limiting_factor=RateLimitType.RPM,
            remaining_requests=0,
        )

        assert result.limiting_factor == RateLimitType.RPM
        assert result.reason is not None
        assert str(bucket.rpm_limit) in result.reason

    def test_all_rate_limit_types_as_limiting_factors(self):
        """Test that all RateLimitType values can be used as limiting_factor."""
        for rate_type in RateLimitType:
            result = LimitCheckResult(
                can_proceed=False,
                limiting_factor=rate_type,
            )
            assert result.limiting_factor == rate_type
