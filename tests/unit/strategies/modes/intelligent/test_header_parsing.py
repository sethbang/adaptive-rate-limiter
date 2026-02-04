import pytest


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
    async def test_normalization_logic(self, strategy):
        """Test that normalization logic works as expected."""
        headers = {
            "x-ratelimit-remaining-requests": "99",
            "x-ratelimit-remaining-tokens": "9900",
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-limit-tokens": "10000",
            "x-ratelimit-reset-requests": "2s",
            "x-ratelimit-reset-tokens": "500ms",
        }

        # Simulate the logic in _update_rate_limit_state
        for key in ["x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"]:
            if key in headers:
                val = headers[key]
                parsed = strategy._parse_duration_string(val)
                if parsed is not None:
                    headers[key] = str(parsed)

        # Now assess should return full
        status = strategy._assess_header_availability(headers)
        assert status == "full"
        assert headers["x-ratelimit-reset-requests"] == "2.0"
        assert headers["x-ratelimit-reset-tokens"] == "0.5"
