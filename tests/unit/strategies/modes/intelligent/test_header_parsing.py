from unittest.mock import Mock

import pytest

from adaptive_rate_limiter.types.request import RequestMetadata


class TestHeaderParsing:
    @pytest.mark.asyncio
    async def test_parse_duration_string(self, strategy):
        """Test parsing of duration strings."""
        assert strategy._parse_duration_string("2s") == 2.0
        assert strategy._parse_duration_string("500ms") == 0.5
        assert strategy._parse_duration_string("1m") == 60.0
        assert strategy._parse_duration_string("1m30s") == 90.0
        assert strategy._parse_duration_string("1.5s") == 1.5
        assert strategy._parse_duration_string("1h") == 3600.0
        assert strategy._parse_duration_string("1d") == 86400.0
        assert strategy._parse_duration_string("invalid") is None
        assert strategy._parse_duration_string("") is None
        assert strategy._parse_duration_string(None) is None

        # Test mixed units with spaces (regex finds all matches)
        assert strategy._parse_duration_string("1m 30s") == 90.0

    @pytest.mark.asyncio
    async def test_normalization_logic(
        self, strategy, mock_scheduler, mock_state_manager
    ):
        """Test that normalization logic works as expected.

        Drives the real ``_update_rate_limit_state``. This test previously
        re-implemented that method's normalization loop inline, which meant it
        asserted nothing about production and would have stayed green through
        any change to it.
        """
        mock_scheduler.extract_response_headers.return_value = {
            "x-ratelimit-remaining-requests": "99",
            "x-ratelimit-remaining-tokens": "9900",
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-limit-tokens": "10000",
            "x-ratelimit-reset-requests": "2s",
            "x-ratelimit-reset-tokens": "500ms",
        }

        metadata = RequestMetadata(
            request_id="req-normalization",
            model_id="test-model",
            resource_type="chat",
        )

        await strategy._update_rate_limit_state(
            metadata, result=Mock(), status_code=200
        )

        call = mock_state_manager.update_state_from_headers.call_args
        headers = call.args[2] if len(call.args) > 2 else call.kwargs["headers"]

        assert strategy._assess_header_availability(headers) == "full"
        assert headers["x-ratelimit-reset-requests"] == "2.0"
        assert headers["x-ratelimit-reset-tokens"] == "0.5"
