"""
Multi-provider integration tests for adaptive-rate-limiter.

These tests verify that the library works correctly with different
provider implementations and that providers can be switched at runtime.
"""

from typing import Any

import pytest

from adaptive_rate_limiter.backends.memory import MemoryBackend
from adaptive_rate_limiter.providers.base import (
    DiscoveredBucket,
    ProviderInterface,
    RateLimitInfo,
)


class MockOpenAIProvider(ProviderInterface):
    """Mock provider simulating OpenAI rate limit behavior."""

    @property
    def name(self) -> str:
        return "openai"

    async def discover_limits(
        self, force_refresh: bool = False, timeout: float = 30.0
    ) -> dict[str, DiscoveredBucket]:
        return {
            "gpt-5": DiscoveredBucket(
                bucket_id="gpt-5",
                rpm_limit=100,
                tpm_limit=40000,
            ),
            "gpt-5.1": DiscoveredBucket(
                bucket_id="gpt-5.1",
                rpm_limit=500,
                tpm_limit=90000,
            ),
        }

    def parse_rate_limit_response(
        self,
        headers: dict[str, str],
        body: dict[str, Any] | None = None,
        status_code: int | None = None,
    ) -> RateLimitInfo:
        return RateLimitInfo(
            rpm_remaining=int(headers.get("x-ratelimit-remaining-requests", 0)),
            rpm_limit=int(headers.get("x-ratelimit-limit-requests", 100)),
            tpm_remaining=int(headers.get("x-ratelimit-remaining-tokens", 0)),
            tpm_limit=int(headers.get("x-ratelimit-limit-tokens", 40000)),
            is_rate_limited=(status_code == 429),
        )

    async def get_bucket_for_model(
        self, model_id: str, resource_type: str | None = None
    ) -> str:
        return model_id


class MockAnthropicProvider(ProviderInterface):
    """Mock provider simulating Anthropic rate limit behavior."""

    @property
    def name(self) -> str:
        return "anthropic"

    async def discover_limits(
        self, force_refresh: bool = False, timeout: float = 30.0
    ) -> dict[str, DiscoveredBucket]:
        return {
            "claude-haiku-4.5": DiscoveredBucket(
                bucket_id="claude-haiku-4.5",
                rpm_limit=50,
                tpm_limit=100000,
            ),
        }

    def parse_rate_limit_response(
        self,
        headers: dict[str, str],
        body: dict[str, Any] | None = None,
        status_code: int | None = None,
    ) -> RateLimitInfo:
        # Anthropic uses different header names
        return RateLimitInfo(
            rpm_remaining=int(headers.get("anthropic-ratelimit-requests-remaining", 0)),
            rpm_limit=int(headers.get("anthropic-ratelimit-requests-limit", 50)),
            tpm_remaining=int(headers.get("anthropic-ratelimit-tokens-remaining", 0)),
            tpm_limit=int(headers.get("anthropic-ratelimit-tokens-limit", 100000)),
            is_rate_limited=(status_code == 429),
        )

    async def get_bucket_for_model(
        self, model_id: str, resource_type: str | None = None
    ) -> str:
        return model_id


class TestMultiProviderIntegration:
    """Test multi-provider scenarios."""

    @pytest.fixture
    def openai_provider(self):
        return MockOpenAIProvider()

    @pytest.fixture
    def anthropic_provider(self):
        return MockAnthropicProvider()

    @pytest.fixture
    def memory_backend(self):
        return MemoryBackend()

    @pytest.mark.asyncio
    async def test_openai_provider_integration(self, openai_provider, memory_backend):
        """Test scheduler works with OpenAI-style provider."""
        # Discover limits
        limits = await openai_provider.discover_limits()

        assert "gpt-5" in limits
        assert limits["gpt-5"].rpm_limit == 100

        # Parse response headers
        headers = {
            "x-ratelimit-remaining-requests": "95",
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-tokens": "39000",
            "x-ratelimit-limit-tokens": "40000",
        }
        info = openai_provider.parse_rate_limit_response(headers, status_code=200)

        assert info.rpm_remaining == 95
        assert info.rpm_limit == 100
        assert info.is_rate_limited is False

    @pytest.mark.asyncio
    async def test_anthropic_provider_integration(
        self, anthropic_provider, memory_backend
    ):
        """Test scheduler works with Anthropic-style provider."""
        limits = await anthropic_provider.discover_limits()

        assert "claude-haiku-4.5" in limits
        assert limits["claude-haiku-4.5"].tpm_limit == 100000

        # Parse Anthropic-style headers
        headers = {
            "anthropic-ratelimit-requests-remaining": "45",
            "anthropic-ratelimit-requests-limit": "50",
            "anthropic-ratelimit-tokens-remaining": "95000",
            "anthropic-ratelimit-tokens-limit": "100000",
        }
        info = anthropic_provider.parse_rate_limit_response(headers, status_code=200)

        assert info.rpm_remaining == 45
        assert info.is_rate_limited is False

    @pytest.mark.asyncio
    async def test_rate_limit_detection(self, openai_provider):
        """Test 429 rate limit detection."""
        headers = {
            "x-ratelimit-remaining-requests": "0",
            "x-ratelimit-limit-requests": "100",
            "retry-after": "30",
        }
        info = openai_provider.parse_rate_limit_response(headers, status_code=429)

        assert info.is_rate_limited is True
        assert info.rpm_remaining == 0

    @pytest.mark.asyncio
    async def test_bucket_isolation(self, openai_provider, memory_backend):
        """Test that different models use different buckets."""
        bucket1 = await openai_provider.get_bucket_for_model("gpt-5", "text")
        bucket2 = await openai_provider.get_bucket_for_model("gpt-5.1", "text")

        assert bucket1 == "gpt-5"
        assert bucket2 == "gpt-5.1"
        assert bucket1 != bucket2
