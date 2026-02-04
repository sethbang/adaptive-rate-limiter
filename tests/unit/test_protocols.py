from collections.abc import AsyncIterator
from dataclasses import is_dataclass
from typing import Any

import pytest

from adaptive_rate_limiter.protocols import (
    ClassifierProtocol,
    ClientProtocol,
    RequestMetadata,
)
from adaptive_rate_limiter.protocols.streaming import StreamingResponseProtocol
from adaptive_rate_limiter.providers import (
    DiscoveredBucket,
    ProviderInterface,
    RateLimitInfo,
)


class TestProtocols:
    def test_client_protocol_runtime_checkable(self):
        """Verify ClientProtocol is runtime checkable."""

        class ValidClient:
            @property
            def base_url(self) -> str:
                return "https://api.example.com"

            @property
            def timeout(self) -> float:
                return 30.0

            def get_headers(self) -> dict[str, str]:
                return {}

        assert isinstance(ValidClient(), ClientProtocol)

    def test_classifier_protocol_runtime_checkable(self):
        """Verify ClassifierProtocol is runtime checkable."""

        class ValidClassifier:
            async def classify(self, request: dict[str, Any]) -> RequestMetadata:
                return RequestMetadata(
                    request_id="req-1", model_id="model-1", resource_type="chat"
                )

        assert isinstance(ValidClassifier(), ClassifierProtocol)

    def test_streaming_protocol_runtime_checkable(self):
        """Verify StreamingResponseProtocol is runtime checkable."""

        class MockStreamingResponse:
            def __init__(self):
                self._iterator: AsyncIterator[Any] = self._default_iterator()

            async def _default_iterator(self) -> AsyncIterator[Any]:
                yield "test"

            def get_iterator(self) -> AsyncIterator[Any]:
                return self._iterator

            def set_iterator(self, iterator: AsyncIterator[Any]) -> None:
                self._iterator = iterator

        mock = MockStreamingResponse()
        assert isinstance(mock, StreamingResponseProtocol)


class TestDataclasses:
    def test_request_metadata_defaults(self):
        """Verify RequestMetadata defaults."""
        meta = RequestMetadata(
            request_id="req-1", model_id="model-1", resource_type="chat"
        )
        assert is_dataclass(meta)
        assert isinstance(meta.request_id, str)
        assert len(meta.request_id) > 0
        assert meta.resource_type == "chat"
        assert meta.estimated_tokens is None
        assert meta.priority == 0
        assert meta.requires_model is True

    def test_discovered_bucket_defaults(self):
        """Verify DiscoveredBucket defaults."""
        bucket = DiscoveredBucket(bucket_id="test-bucket")
        assert is_dataclass(bucket)
        assert bucket.bucket_id == "test-bucket"
        assert bucket.rpm_limit is None
        assert bucket.tpm_limit is None

    def test_rate_limit_info_defaults(self):
        """Verify RateLimitInfo defaults."""
        info = RateLimitInfo()
        assert is_dataclass(info)
        assert info.is_rate_limited is False
        assert isinstance(info.timestamp, float)
        assert info.rpm_remaining is None

    def test_rate_limit_info_negative_remaining(self):
        """Test RateLimitInfo with negative rpm_remaining/tpm_remaining values.

        Some APIs may report negative values when limits are exceeded.
        The dataclass should accept these values without validation errors.
        """
        info = RateLimitInfo(
            rpm_remaining=-5,
            tpm_remaining=-1000,
            rpm_limit=100,
            tpm_limit=10000,
        )
        assert info.rpm_remaining == -5
        assert info.tpm_remaining == -1000
        # Limits should still be positive
        assert info.rpm_limit == 100
        assert info.tpm_limit == 10000

    def test_rate_limit_info_zero_limits(self):
        """Test RateLimitInfo with zero rpm_limit/tpm_limit.

        Zero limits may indicate a disabled or suspended account.
        """
        info = RateLimitInfo(
            rpm_limit=0,
            tpm_limit=0,
            rpm_remaining=0,
            tpm_remaining=0,
        )
        assert info.rpm_limit == 0
        assert info.tpm_limit == 0
        assert info.rpm_remaining == 0
        assert info.tpm_remaining == 0
        assert info.is_rate_limited is False  # Default, not set

    def test_rate_limit_info_future_reset_time(self):
        """Test RateLimitInfo with reset time in the future.

        Reset times are typically in the future, indicating when limits will reset.
        """
        import time

        future_time = time.time() + 3600  # 1 hour in the future

        info = RateLimitInfo(
            rpm_reset=future_time,
            tpm_reset=future_time,
            rpm_remaining=50,
            rpm_limit=100,
        )
        assert info.rpm_reset == future_time
        assert info.tpm_reset == future_time
        # Verify reset is actually in the future
        assert info.rpm_reset is not None and info.rpm_reset > time.time()

    def test_rate_limit_info_past_reset_time(self):
        """Test RateLimitInfo with reset time in the distant past.

        Past reset times may occur with stale cached data or API inconsistencies.
        The dataclass should store the value without validation.
        """
        import time

        past_time = time.time() - 86400  # 24 hours ago

        info = RateLimitInfo(
            rpm_reset=past_time,
            tpm_reset=past_time,
            rpm_remaining=100,
            rpm_limit=100,
        )
        assert info.rpm_reset == past_time
        assert info.tpm_reset == past_time
        # Verify reset is actually in the past
        assert info.rpm_reset is not None and info.rpm_reset < time.time()

    def test_rate_limit_info_rate_limited_with_retry_after(self):
        """Test RateLimitInfo when rate limited with retry_after set."""
        info = RateLimitInfo(
            is_rate_limited=True,
            retry_after=60,
            rpm_remaining=0,
            tpm_remaining=0,
        )
        assert info.is_rate_limited is True
        assert info.retry_after == 60
        assert info.rpm_remaining == 0
        assert info.tpm_remaining == 0

    def test_rate_limit_info_all_none_values(self):
        """Test RateLimitInfo with explicit None values."""
        info = RateLimitInfo(
            rpm_remaining=None,
            rpm_limit=None,
            rpm_reset=None,
            tpm_remaining=None,
            tpm_limit=None,
            tpm_reset=None,
            retry_after=None,
        )
        assert info.rpm_remaining is None
        assert info.rpm_limit is None
        assert info.rpm_reset is None
        assert info.tpm_remaining is None
        assert info.tpm_limit is None
        assert info.tpm_reset is None
        assert info.retry_after is None


class TestProviderInterface:
    def test_provider_interface_is_abstract(self):
        """Verify ProviderInterface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ProviderInterface()  # type: ignore

    def test_provider_implementation(self):
        """Verify a concrete implementation works."""

        class TestProvider(ProviderInterface):
            @property
            def name(self) -> str:
                return "test"

            async def discover_limits(self, force_refresh=False, timeout=30.0):
                return {}

            def parse_rate_limit_response(self, headers, body=None, status_code=None):
                return RateLimitInfo()

            async def get_bucket_for_model(self, model_id, resource_type=None):
                return "default"

        provider = TestProvider()
        assert provider.name == "test"
        assert isinstance(provider, ProviderInterface)
