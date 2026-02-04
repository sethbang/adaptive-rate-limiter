"""
Unit tests for RequestMetadata dataclass.

These tests verify the structure, defaults, and behavior of the RequestMetadata
type used for request classification, routing, and queue management.
"""

from datetime import datetime, timedelta, timezone

from adaptive_rate_limiter.types import RequestMetadata


class TestRequestMetadataDefaults:
    """Tests for RequestMetadata default values."""

    def test_required_fields_only(self):
        """Test creating RequestMetadata with only required fields."""
        meta = RequestMetadata(
            request_id="req-123",
            model_id="gpt-5",
            resource_type="chat",
        )

        assert meta.request_id == "req-123"
        assert meta.model_id == "gpt-5"
        assert meta.resource_type == "chat"

    def test_default_estimated_tokens(self):
        """Test that estimated_tokens defaults to None."""
        meta = RequestMetadata(
            request_id="req-123",
            model_id="gpt-5",
            resource_type="chat",
        )
        assert meta.estimated_tokens is None

    def test_default_priority(self):
        """Test that priority defaults to 0."""
        meta = RequestMetadata(
            request_id="req-123",
            model_id="gpt-5",
            resource_type="chat",
        )
        assert meta.priority == 0

    def test_default_submitted_at(self):
        """Test that submitted_at defaults to current UTC time."""
        before = datetime.now(timezone.utc)
        meta = RequestMetadata(
            request_id="req-123",
            model_id="gpt-5",
            resource_type="chat",
        )
        after = datetime.now(timezone.utc)

        assert before <= meta.submitted_at <= after
        assert meta.submitted_at.tzinfo == timezone.utc

    def test_default_timeout(self):
        """Test that timeout defaults to 60.0 seconds."""
        meta = RequestMetadata(
            request_id="req-123",
            model_id="gpt-5",
            resource_type="chat",
        )
        assert meta.timeout == 60.0

    def test_default_client_id(self):
        """Test that client_id defaults to None."""
        meta = RequestMetadata(
            request_id="req-123",
            model_id="gpt-5",
            resource_type="chat",
        )
        assert meta.client_id is None

    def test_default_endpoint(self):
        """Test that endpoint defaults to None."""
        meta = RequestMetadata(
            request_id="req-123",
            model_id="gpt-5",
            resource_type="chat",
        )
        assert meta.endpoint is None

    def test_default_requires_model(self):
        """Test that requires_model defaults to True."""
        meta = RequestMetadata(
            request_id="req-123",
            model_id="gpt-5",
            resource_type="chat",
        )
        assert meta.requires_model is True


class TestRequestMetadataCustomValues:
    """Tests for RequestMetadata with custom values."""

    def test_all_fields_specified(self):
        """Test creating RequestMetadata with all fields specified."""
        custom_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        meta = RequestMetadata(
            request_id="req-456",
            model_id="gpt-5.1",
            resource_type="completions",
            estimated_tokens=500,
            priority=5,
            submitted_at=custom_time,
            timeout=120.0,
            client_id="client-abc",
            endpoint="/v1/completions",
            requires_model=False,
        )

        assert meta.request_id == "req-456"
        assert meta.model_id == "gpt-5.1"
        assert meta.resource_type == "completions"
        assert meta.estimated_tokens == 500
        assert meta.priority == 5
        assert meta.submitted_at == custom_time
        assert meta.timeout == 120.0
        assert meta.client_id == "client-abc"
        assert meta.endpoint == "/v1/completions"
        assert meta.requires_model is False

    def test_negative_priority(self):
        """Test that negative priority values are allowed (for background ops)."""
        meta = RequestMetadata(
            request_id="req-123",
            model_id="gpt-5",
            resource_type="chat",
            priority=-5,
        )
        assert meta.priority == -5

    def test_high_priority(self):
        """Test that high priority values are allowed."""
        meta = RequestMetadata(
            request_id="req-123",
            model_id="gpt-5",
            resource_type="chat",
            priority=9,
        )
        assert meta.priority == 9

    def test_none_timeout(self):
        """Test that timeout can be set to None."""
        meta = RequestMetadata(
            request_id="req-123",
            model_id="gpt-5",
            resource_type="chat",
            timeout=None,
        )
        assert meta.timeout is None

    def test_zero_estimated_tokens(self):
        """Test that estimated_tokens can be zero."""
        meta = RequestMetadata(
            request_id="req-123",
            model_id="gpt-5",
            resource_type="chat",
            estimated_tokens=0,
        )
        assert meta.estimated_tokens == 0


class TestRequestMetadataEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_string_request_id(self):
        """Test that empty string is allowed for request_id."""
        meta = RequestMetadata(
            request_id="",
            model_id="gpt-5",
            resource_type="chat",
        )
        assert meta.request_id == ""

    def test_empty_string_model_id(self):
        """Test that empty string is allowed for model_id."""
        meta = RequestMetadata(
            request_id="req-123",
            model_id="",
            resource_type="chat",
        )
        assert meta.model_id == ""

    def test_empty_string_resource_type(self):
        """Test that empty string is allowed for resource_type."""
        meta = RequestMetadata(
            request_id="req-123",
            model_id="gpt-5",
            resource_type="",
        )
        assert meta.resource_type == ""

    def test_large_estimated_tokens(self):
        """Test that large token values are allowed."""
        meta = RequestMetadata(
            request_id="req-123",
            model_id="gpt-5",
            resource_type="chat",
            estimated_tokens=1_000_000,
        )
        assert meta.estimated_tokens == 1_000_000

    def test_very_long_timeout(self):
        """Test that very long timeout values are allowed."""
        meta = RequestMetadata(
            request_id="req-123",
            model_id="gpt-5",
            resource_type="chat",
            timeout=3600.0,  # 1 hour
        )
        assert meta.timeout == 3600.0

    def test_submitted_at_with_different_timezone(self):
        """Test that submitted_at works with different timezones."""
        # Create a timezone at UTC+5

        custom_tz = timezone(timedelta(hours=5))
        custom_time = datetime(2024, 6, 15, 10, 30, 0, tzinfo=custom_tz)

        meta = RequestMetadata(
            request_id="req-123",
            model_id="gpt-5",
            resource_type="chat",
            submitted_at=custom_time,
        )

        assert meta.submitted_at == custom_time
        assert meta.submitted_at.tzinfo == custom_tz


class TestRequestMetadataMutability:
    """Tests for RequestMetadata mutability behavior."""

    def test_is_mutable(self):
        """Test that RequestMetadata is mutable (not frozen)."""
        meta = RequestMetadata(
            request_id="req-123",
            model_id="gpt-5",
            resource_type="chat",
        )

        # Should be able to modify fields (dataclass is not frozen)
        meta.priority = 10
        assert meta.priority == 10

        meta.model_id = "gpt-5.1"
        assert meta.model_id == "gpt-5.1"


class TestRequestMetadataEquality:
    """Tests for RequestMetadata equality comparison."""

    def test_equality_same_values(self):
        """Test that two RequestMetadata with same values are equal."""
        fixed_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        meta1 = RequestMetadata(
            request_id="req-123",
            model_id="gpt-5",
            resource_type="chat",
            submitted_at=fixed_time,
        )
        meta2 = RequestMetadata(
            request_id="req-123",
            model_id="gpt-5",
            resource_type="chat",
            submitted_at=fixed_time,
        )

        assert meta1 == meta2

    def test_inequality_different_values(self):
        """Test that two RequestMetadata with different values are not equal."""
        fixed_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        meta1 = RequestMetadata(
            request_id="req-123",
            model_id="gpt-5",
            resource_type="chat",
            submitted_at=fixed_time,
        )
        meta2 = RequestMetadata(
            request_id="req-456",  # Different request_id
            model_id="gpt-5",
            resource_type="chat",
            submitted_at=fixed_time,
        )

        assert meta1 != meta2
