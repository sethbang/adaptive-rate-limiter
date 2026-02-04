"""
Unit tests for RateLimitedAsyncIterator.

Tests cover:
- Initialization
- __anext__ method (chunk yielding, token extraction, cleanup)
- _extract_tokens() method
- _release_capacity() method
- _release_capacity_fallback() method
- aclose() method
- __aiter__ method
- context property
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from adaptive_rate_limiter.streaming.context import StreamingReservationContext
from adaptive_rate_limiter.streaming.iterator import RateLimitedAsyncIterator


class TestRateLimitedAsyncIteratorInit:
    """Tests for RateLimitedAsyncIterator initialization."""

    @pytest.fixture
    def mock_backend(self) -> Mock:
        """Create a mock backend."""
        backend = Mock()
        backend.release_streaming_reservation = AsyncMock(return_value=True)
        return backend

    @pytest.fixture
    def context(self, mock_backend: Mock) -> StreamingReservationContext:
        """Create a test context."""
        return StreamingReservationContext(
            reservation_id="res-1",
            bucket_id="bucket-1",
            request_id="req-1",
            reserved_tokens=1000,
            backend=mock_backend,
        )

    @pytest.mark.asyncio
    async def test_init_sets_inner_iterator(
        self, context: StreamingReservationContext
    ) -> None:
        """Verify _inner is set to the provided iterator."""

        async def mock_iter() -> AsyncIterator[dict[str, str]]:
            yield {"content": "test"}

        inner = mock_iter()
        iterator = RateLimitedAsyncIterator(inner, context)

        assert iterator._inner is inner

    @pytest.mark.asyncio
    async def test_init_sets_context(
        self, context: StreamingReservationContext
    ) -> None:
        """Verify _ctx is set to the provided context."""

        async def mock_iter() -> AsyncIterator[dict[str, str]]:
            yield {"content": "test"}

        iterator = RateLimitedAsyncIterator(mock_iter(), context)

        assert iterator._ctx is context

    @pytest.mark.asyncio
    async def test_init_sets_released_flag_false(
        self, context: StreamingReservationContext
    ) -> None:
        """Verify _released is initialized to False."""

        async def mock_iter() -> AsyncIterator[dict[str, str]]:
            yield {"content": "test"}

        iterator = RateLimitedAsyncIterator(mock_iter(), context)

        assert iterator._released is False

    @pytest.mark.asyncio
    async def test_init_sets_closed_flag_false(
        self, context: StreamingReservationContext
    ) -> None:
        """Verify _closed is initialized to False."""

        async def mock_iter() -> AsyncIterator[dict[str, str]]:
            yield {"content": "test"}

        iterator = RateLimitedAsyncIterator(mock_iter(), context)

        assert iterator._closed is False

    @pytest.mark.asyncio
    async def test_init_sets_warned_long_running_false(
        self, context: StreamingReservationContext
    ) -> None:
        """Verify _warned_long_running is initialized to False."""

        async def mock_iter() -> AsyncIterator[dict[str, str]]:
            yield {"content": "test"}

        iterator = RateLimitedAsyncIterator(mock_iter(), context)

        assert iterator._warned_long_running is False


class TestRateLimitedAsyncIteratorAnext:
    """Tests for RateLimitedAsyncIterator.__anext__ method."""

    @pytest.fixture
    def mock_backend(self) -> Mock:
        """Create a mock backend."""
        backend = Mock()
        backend.release_streaming_reservation = AsyncMock(return_value=True)
        return backend

    @pytest.fixture
    def context(self, mock_backend: Mock) -> StreamingReservationContext:
        """Create a test context."""
        return StreamingReservationContext(
            reservation_id="res-1",
            bucket_id="bucket-1",
            request_id="req-1",
            reserved_tokens=1000,
            backend=mock_backend,
        )

    @pytest.mark.asyncio
    async def test_yields_chunks_from_inner_iterator(
        self, context: StreamingReservationContext
    ) -> None:
        """Verify chunks from inner iterator are yielded."""

        async def mock_iter() -> AsyncIterator[dict[str, str]]:
            yield {"content": "Hello"}
            yield {"content": "World"}

        iterator = RateLimitedAsyncIterator(mock_iter(), context)
        chunks = [chunk async for chunk in iterator]

        assert len(chunks) == 2
        assert chunks[0]["content"] == "Hello"
        assert chunks[1]["content"] == "World"

    @pytest.mark.asyncio
    async def test_updates_activity_on_each_chunk(
        self, context: StreamingReservationContext
    ) -> None:
        """Verify activity is updated on each chunk."""
        assert context.chunk_count == 0

        async def mock_iter() -> AsyncIterator[dict[str, str]]:
            yield {"content": "Hello"}
            yield {"content": "World"}
            yield {"content": "!"}

        iterator = RateLimitedAsyncIterator(mock_iter(), context)
        [chunk async for chunk in iterator]

        assert context.chunk_count == 3

    @pytest.mark.asyncio
    async def test_extracts_tokens_from_chunk_with_usage_object(
        self, context: StreamingReservationContext
    ) -> None:
        """Verify tokens are extracted from chunks with usage.total_tokens attribute."""

        class MockUsage:
            total_tokens = 500

        class MockChunk:
            content = "test"
            usage = MockUsage()

        async def mock_iter() -> AsyncIterator[MockChunk]:
            yield MockChunk()

        iterator = RateLimitedAsyncIterator(mock_iter(), context)
        [chunk async for chunk in iterator]

        assert context.final_tokens == 500

    @pytest.mark.asyncio
    async def test_extracts_tokens_from_chunk_with_usage_dict(
        self, context: StreamingReservationContext
    ) -> None:
        """Verify tokens are extracted from dict chunks with usage.total_tokens."""

        async def mock_iter() -> AsyncIterator[dict[str, Any]]:
            yield {"content": "test", "usage": {"total_tokens": 750}}

        iterator = RateLimitedAsyncIterator(mock_iter(), context)
        [chunk async for chunk in iterator]

        assert context.final_tokens == 750

    @pytest.mark.asyncio
    async def test_releases_capacity_on_stop_async_iteration(
        self, context: StreamingReservationContext, mock_backend: Mock
    ) -> None:
        """Verify _release_capacity() is called on StopAsyncIteration."""

        async def mock_iter() -> AsyncIterator[dict[str, str]]:
            yield {"content": "Hello"}

        iterator = RateLimitedAsyncIterator(mock_iter(), context)
        [chunk async for chunk in iterator]

        mock_backend.release_streaming_reservation.assert_called_once()

    @pytest.mark.asyncio
    async def test_releases_capacity_fallback_on_exception(
        self, context: StreamingReservationContext, mock_backend: Mock
    ) -> None:
        """Verify _release_capacity_fallback() is called on exception."""

        async def mock_iter() -> AsyncIterator[dict[str, str]]:
            yield {"content": "Hello"}
            raise RuntimeError("Stream error")

        iterator = RateLimitedAsyncIterator(mock_iter(), context)

        with pytest.raises(RuntimeError, match="Stream error"):
            [chunk async for chunk in iterator]

        mock_backend.release_streaming_reservation.assert_called_once()
        # Fallback uses reserved_tokens as actual
        call_args = mock_backend.release_streaming_reservation.call_args
        assert call_args.kwargs["actual_tokens"] == 1000

    @pytest.mark.asyncio
    async def test_handles_empty_iterator(
        self, context: StreamingReservationContext, mock_backend: Mock
    ) -> None:
        """Verify empty iterator completes without error."""

        async def mock_iter() -> AsyncIterator[dict[str, str]]:
            return
            yield  # Make it a generator

        iterator = RateLimitedAsyncIterator(mock_iter(), context)
        chunks = [chunk async for chunk in iterator]

        assert len(chunks) == 0
        mock_backend.release_streaming_reservation.assert_called_once()


class TestRateLimitedAsyncIteratorExtractTokens:
    """Tests for RateLimitedAsyncIterator._extract_tokens() method."""

    @pytest.fixture
    def mock_backend(self) -> Mock:
        """Create a mock backend."""
        backend = Mock()
        backend.release_streaming_reservation = AsyncMock(return_value=True)
        return backend

    @pytest.fixture
    def context(self, mock_backend: Mock) -> StreamingReservationContext:
        """Create a test context."""
        return StreamingReservationContext(
            reservation_id="res-1",
            bucket_id="bucket-1",
            request_id="req-1",
            reserved_tokens=1000,
            backend=mock_backend,
        )

    @pytest.fixture
    def iterator(
        self, context: StreamingReservationContext
    ) -> RateLimitedAsyncIterator[Any]:
        """Create a test iterator."""

        async def mock_iter() -> AsyncIterator[dict[str, str]]:
            yield {"content": "test"}

        return RateLimitedAsyncIterator(mock_iter(), context)

    def test_extracts_from_object_with_usage_total_tokens(
        self, iterator: RateLimitedAsyncIterator[Any]
    ) -> None:
        """Verify extraction from object with usage.total_tokens attribute."""

        class MockUsage:
            total_tokens = 500

        class MockChunk:
            usage = MockUsage()

        result = iterator._extract_tokens(MockChunk())
        assert result == 500

    def test_extracts_from_object_with_usage_total_tokens_int(
        self, iterator: RateLimitedAsyncIterator[Any]
    ) -> None:
        """Verify extraction handles int values."""

        class MockUsage:
            total_tokens = 1234

        class MockChunk:
            usage = MockUsage()

        result = iterator._extract_tokens(MockChunk())
        assert result == 1234

    def test_extracts_from_dict_with_usage_total_tokens(
        self, iterator: RateLimitedAsyncIterator[Any]
    ) -> None:
        """Verify extraction from dict with usage.total_tokens key."""
        chunk = {"usage": {"total_tokens": 750}}
        result = iterator._extract_tokens(chunk)
        assert result == 750

    def test_returns_none_for_chunk_without_usage(
        self, iterator: RateLimitedAsyncIterator[Any]
    ) -> None:
        """Verify returns None for chunks without usage."""
        chunk = {"content": "test"}
        result = iterator._extract_tokens(chunk)
        assert result is None

    def test_returns_none_for_chunk_with_none_usage(
        self, iterator: RateLimitedAsyncIterator[Any]
    ) -> None:
        """Verify returns None for chunks with None usage."""

        class MockChunk:
            usage = None

        result = iterator._extract_tokens(MockChunk())
        assert result is None

    def test_returns_none_for_dict_with_none_usage(
        self, iterator: RateLimitedAsyncIterator[Any]
    ) -> None:
        """Verify returns None for dict with None usage."""
        chunk = {"usage": None}
        result = iterator._extract_tokens(chunk)
        assert result is None

    def test_returns_none_for_usage_without_total_tokens(
        self, iterator: RateLimitedAsyncIterator[Any]
    ) -> None:
        """Verify returns None when usage lacks total_tokens."""

        class MockUsage:
            prompt_tokens = 100

        class MockChunk:
            usage = MockUsage()

        result = iterator._extract_tokens(MockChunk())
        assert result is None

    def test_returns_none_for_dict_usage_without_total_tokens(
        self, iterator: RateLimitedAsyncIterator[Any]
    ) -> None:
        """Verify returns None when dict usage lacks total_tokens."""
        chunk = {"usage": {"prompt_tokens": 100}}
        result = iterator._extract_tokens(chunk)
        assert result is None

    def test_returns_none_for_invalid_total_tokens_value(
        self, iterator: RateLimitedAsyncIterator[Any]
    ) -> None:
        """Verify returns None for non-numeric total_tokens."""

        class MockUsage:
            total_tokens = "invalid"

        class MockChunk:
            usage = MockUsage()

        result = iterator._extract_tokens(MockChunk())
        assert result is None

    def test_returns_none_for_invalid_dict_total_tokens_value(
        self, iterator: RateLimitedAsyncIterator[Any]
    ) -> None:
        """Verify returns None for non-numeric dict total_tokens."""
        chunk = {"usage": {"total_tokens": "invalid"}}
        result = iterator._extract_tokens(chunk)
        assert result is None

    def test_converts_string_numeric_to_int(
        self, iterator: RateLimitedAsyncIterator[Any]
    ) -> None:
        """Verify string numbers are converted to int."""
        chunk = {"usage": {"total_tokens": "500"}}
        result = iterator._extract_tokens(chunk)
        assert result == 500

    def test_returns_none_for_usage_with_none_total_tokens(
        self, iterator: RateLimitedAsyncIterator[Any]
    ) -> None:
        """Verify returns None when total_tokens is None."""

        class MockUsage:
            total_tokens = None

        class MockChunk:
            usage = MockUsage()

        result = iterator._extract_tokens(MockChunk())
        assert result is None


class TestRateLimitedAsyncIteratorReleaseCapacity:
    """Tests for RateLimitedAsyncIterator._release_capacity() method."""

    @pytest.fixture
    def mock_backend(self) -> Mock:
        """Create a mock backend."""
        backend = Mock()
        backend.release_streaming_reservation = AsyncMock(return_value=True)
        return backend

    @pytest.fixture
    def context(self, mock_backend: Mock) -> StreamingReservationContext:
        """Create a test context."""
        return StreamingReservationContext(
            reservation_id="res-1",
            bucket_id="bucket-1",
            request_id="req-1",
            reserved_tokens=1000,
            backend=mock_backend,
        )

    @pytest.mark.asyncio
    async def test_calls_backend_release_with_correct_args(
        self, context: StreamingReservationContext, mock_backend: Mock
    ) -> None:
        """Verify backend.release_streaming_reservation is called with correct args."""

        async def mock_iter() -> AsyncIterator[dict[str, str]]:
            yield {"content": "test"}

        iterator = RateLimitedAsyncIterator(mock_iter(), context)
        [chunk async for chunk in iterator]

        mock_backend.release_streaming_reservation.assert_called_once_with(
            "bucket-1",
            "res-1",
            reserved_tokens=1000,
            actual_tokens=1000,  # No final_tokens extracted
        )

    @pytest.mark.asyncio
    async def test_uses_final_tokens_when_set(
        self, context: StreamingReservationContext, mock_backend: Mock
    ) -> None:
        """Verify actual_tokens uses final_tokens when extracted."""

        async def mock_iter() -> AsyncIterator[dict[str, Any]]:
            yield {"content": "test", "usage": {"total_tokens": 500}}

        iterator = RateLimitedAsyncIterator(mock_iter(), context)
        [chunk async for chunk in iterator]

        mock_backend.release_streaming_reservation.assert_called_once_with(
            "bucket-1",
            "res-1",
            reserved_tokens=1000,
            actual_tokens=500,
        )

    @pytest.mark.asyncio
    async def test_is_idempotent(
        self, context: StreamingReservationContext, mock_backend: Mock
    ) -> None:
        """Verify second release call does nothing."""

        async def mock_iter() -> AsyncIterator[dict[str, str]]:
            yield {"content": "test"}

        iterator = RateLimitedAsyncIterator(mock_iter(), context)

        # First release via iteration
        [chunk async for chunk in iterator]
        assert mock_backend.release_streaming_reservation.call_count == 1

        # Second release attempt
        await iterator._release_capacity()
        assert mock_backend.release_streaming_reservation.call_count == 1

    @pytest.mark.asyncio
    async def test_calls_metrics_callback_on_success(self, mock_backend: Mock) -> None:
        """Verify metrics_callback is called on successful release."""
        metrics_callback = Mock()
        context = StreamingReservationContext(
            reservation_id="res-1",
            bucket_id="bucket-1",
            request_id="req-1",
            reserved_tokens=1000,
            backend=mock_backend,
            metrics_callback=metrics_callback,
        )

        async def mock_iter() -> AsyncIterator[dict[str, Any]]:
            yield {"content": "test", "usage": {"total_tokens": 500}}

        iterator = RateLimitedAsyncIterator(mock_iter(), context)
        [chunk async for chunk in iterator]

        metrics_callback.assert_called_once()
        call_args = metrics_callback.call_args[0]
        assert call_args[0] == 1000  # reserved
        assert call_args[1] == 500  # actual
        assert call_args[2] is True  # extraction_succeeded
        assert call_args[3] == "bucket-1"  # bucket_id

    @pytest.mark.asyncio
    async def test_handles_backend_exception(
        self, context: StreamingReservationContext, mock_backend: Mock
    ) -> None:
        """Verify backend exception is handled gracefully."""
        mock_backend.release_streaming_reservation.side_effect = RuntimeError(
            "Backend error"
        )

        async def mock_iter() -> AsyncIterator[dict[str, str]]:
            yield {"content": "test"}

        iterator = RateLimitedAsyncIterator(mock_iter(), context)

        # Should not raise
        [chunk async for chunk in iterator]

    @pytest.mark.asyncio
    async def test_handles_metrics_callback_exception(self, mock_backend: Mock) -> None:
        """Verify metrics_callback exception is handled gracefully."""
        metrics_callback = Mock(side_effect=RuntimeError("Callback error"))
        context = StreamingReservationContext(
            reservation_id="res-1",
            bucket_id="bucket-1",
            request_id="req-1",
            reserved_tokens=1000,
            backend=mock_backend,
            metrics_callback=metrics_callback,
        )

        async def mock_iter() -> AsyncIterator[dict[str, str]]:
            yield {"content": "test"}

        iterator = RateLimitedAsyncIterator(mock_iter(), context)

        # Should not raise
        [chunk async for chunk in iterator]

    @pytest.mark.asyncio
    async def test_sets_released_flag(
        self, context: StreamingReservationContext
    ) -> None:
        """Verify _released flag is set after release."""

        async def mock_iter() -> AsyncIterator[dict[str, str]]:
            yield {"content": "test"}

        iterator = RateLimitedAsyncIterator(mock_iter(), context)
        assert iterator._released is False

        [chunk async for chunk in iterator]

        assert iterator._released is True


class TestRateLimitedAsyncIteratorReleaseCapacityFallback:
    """Tests for RateLimitedAsyncIterator._release_capacity_fallback() method."""

    @pytest.fixture
    def mock_backend(self) -> Mock:
        """Create a mock backend."""
        backend = Mock()
        backend.release_streaming_reservation = AsyncMock(return_value=True)
        return backend

    @pytest.fixture
    def context(self, mock_backend: Mock) -> StreamingReservationContext:
        """Create a test context."""
        return StreamingReservationContext(
            reservation_id="res-1",
            bucket_id="bucket-1",
            request_id="req-1",
            reserved_tokens=1000,
            backend=mock_backend,
        )

    @pytest.mark.asyncio
    async def test_uses_reserved_tokens_as_actual(
        self, context: StreamingReservationContext, mock_backend: Mock
    ) -> None:
        """Verify actual_tokens equals reserved_tokens (zero refund)."""

        async def mock_iter() -> AsyncIterator[dict[str, str]]:
            yield {"content": "test"}
            raise RuntimeError("Stream error")

        iterator = RateLimitedAsyncIterator(mock_iter(), context)

        with pytest.raises(RuntimeError):
            [chunk async for chunk in iterator]

        mock_backend.release_streaming_reservation.assert_called_once_with(
            "bucket-1",
            "res-1",
            reserved_tokens=1000,
            actual_tokens=1000,  # Zero refund
        )

    @pytest.mark.asyncio
    async def test_calls_error_metrics_callback(self, mock_backend: Mock) -> None:
        """Verify error_metrics_callback is called."""
        error_callback = Mock()
        context = StreamingReservationContext(
            reservation_id="res-1",
            bucket_id="bucket-1",
            request_id="req-1",
            reserved_tokens=1000,
            backend=mock_backend,
            error_metrics_callback=error_callback,
        )

        async def mock_iter() -> AsyncIterator[dict[str, str]]:
            yield {"content": "test"}
            raise RuntimeError("Stream error")

        iterator = RateLimitedAsyncIterator(mock_iter(), context)

        with pytest.raises(RuntimeError):
            [chunk async for chunk in iterator]

        error_callback.assert_called_once_with("bucket-1")

    @pytest.mark.asyncio
    async def test_is_idempotent(
        self, context: StreamingReservationContext, mock_backend: Mock
    ) -> None:
        """Verify second fallback release call does nothing."""

        async def mock_iter() -> AsyncIterator[dict[str, str]]:
            yield {"content": "test"}

        iterator = RateLimitedAsyncIterator(mock_iter(), context)

        # First fallback release
        await iterator._release_capacity_fallback()
        assert mock_backend.release_streaming_reservation.call_count == 1

        # Second attempt
        await iterator._release_capacity_fallback()
        assert mock_backend.release_streaming_reservation.call_count == 1

    @pytest.mark.asyncio
    async def test_handles_backend_exception(
        self, context: StreamingReservationContext, mock_backend: Mock
    ) -> None:
        """Verify backend exception is handled gracefully."""
        mock_backend.release_streaming_reservation.side_effect = RuntimeError(
            "Backend error"
        )

        async def mock_iter() -> AsyncIterator[dict[str, str]]:
            yield {"content": "test"}

        iterator = RateLimitedAsyncIterator(mock_iter(), context)

        # Should not raise
        await iterator._release_capacity_fallback()

    @pytest.mark.asyncio
    async def test_handles_error_callback_exception(self, mock_backend: Mock) -> None:
        """Verify error_metrics_callback exception is handled gracefully."""
        error_callback = Mock(side_effect=RuntimeError("Callback error"))
        context = StreamingReservationContext(
            reservation_id="res-1",
            bucket_id="bucket-1",
            request_id="req-1",
            reserved_tokens=1000,
            backend=mock_backend,
            error_metrics_callback=error_callback,
        )

        async def mock_iter() -> AsyncIterator[dict[str, str]]:
            yield {"content": "test"}

        iterator = RateLimitedAsyncIterator(mock_iter(), context)

        # Should not raise
        await iterator._release_capacity_fallback()


class TestRateLimitedAsyncIteratorAclose:
    """Tests for RateLimitedAsyncIterator.aclose() method."""

    @pytest.fixture
    def mock_backend(self) -> Mock:
        """Create a mock backend."""
        backend = Mock()
        backend.release_streaming_reservation = AsyncMock(return_value=True)
        return backend

    @pytest.fixture
    def context(self, mock_backend: Mock) -> StreamingReservationContext:
        """Create a test context."""
        return StreamingReservationContext(
            reservation_id="res-1",
            bucket_id="bucket-1",
            request_id="req-1",
            reserved_tokens=1000,
            backend=mock_backend,
        )

    @pytest.mark.asyncio
    async def test_releases_capacity_if_not_released(
        self, context: StreamingReservationContext, mock_backend: Mock
    ) -> None:
        """Verify aclose() releases capacity if not already released."""

        async def mock_iter() -> AsyncIterator[dict[str, str]]:
            yield {"content": "test"}

        iterator = RateLimitedAsyncIterator(mock_iter(), context)

        await iterator.aclose()

        mock_backend.release_streaming_reservation.assert_called_once()

    @pytest.mark.asyncio
    async def test_does_not_release_if_already_released(
        self, context: StreamingReservationContext, mock_backend: Mock
    ) -> None:
        """Verify aclose() doesn't release if already released."""

        async def mock_iter() -> AsyncIterator[dict[str, str]]:
            yield {"content": "test"}

        iterator = RateLimitedAsyncIterator(mock_iter(), context)

        # Release via iteration
        [chunk async for chunk in iterator]
        assert mock_backend.release_streaming_reservation.call_count == 1

        # aclose should not release again
        await iterator.aclose()
        assert mock_backend.release_streaming_reservation.call_count == 1

    @pytest.mark.asyncio
    async def test_closes_inner_iterator_with_aclose(
        self, context: StreamingReservationContext
    ) -> None:
        """Verify aclose() closes inner iterator if it has aclose()."""
        inner_aclose = AsyncMock()

        class MockAsyncIterator:
            def __init__(self) -> None:
                self._yielded = False
                self.aclose = inner_aclose

            def __aiter__(self) -> MockAsyncIterator:
                return self

            async def __anext__(self) -> dict[str, str]:
                if self._yielded:
                    raise StopAsyncIteration
                self._yielded = True
                return {"content": "test"}

        inner = MockAsyncIterator()

        iterator = RateLimitedAsyncIterator(inner, context)
        await iterator.aclose()

        inner_aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_inner_without_aclose(
        self, context: StreamingReservationContext
    ) -> None:
        """Verify aclose() works when inner iterator has no aclose()."""

        class SimpleIterator:
            def __aiter__(self) -> SimpleIterator:
                return self

            async def __anext__(self) -> dict[str, str]:
                raise StopAsyncIteration

        inner = SimpleIterator()
        iterator = RateLimitedAsyncIterator(inner, context)

        # Should not raise
        await iterator.aclose()

    @pytest.mark.asyncio
    async def test_is_idempotent(
        self, context: StreamingReservationContext, mock_backend: Mock
    ) -> None:
        """Verify multiple aclose() calls are safe."""

        async def mock_iter() -> AsyncIterator[dict[str, str]]:
            yield {"content": "test"}

        iterator = RateLimitedAsyncIterator(mock_iter(), context)

        await iterator.aclose()
        await iterator.aclose()
        await iterator.aclose()

        # Only one release
        mock_backend.release_streaming_reservation.assert_called_once()

    @pytest.mark.asyncio
    async def test_sets_closed_flag(self, context: StreamingReservationContext) -> None:
        """Verify _closed flag is set after aclose()."""

        async def mock_iter() -> AsyncIterator[dict[str, str]]:
            yield {"content": "test"}

        iterator = RateLimitedAsyncIterator(mock_iter(), context)
        assert iterator._closed is False

        await iterator.aclose()

        assert iterator._closed is True

    @pytest.mark.asyncio
    async def test_handles_inner_aclose_exception(
        self, context: StreamingReservationContext
    ) -> None:
        """Verify inner aclose() exception is handled gracefully."""
        inner_aclose = AsyncMock(side_effect=RuntimeError("Close error"))

        class MockAsyncIterator:
            def __init__(self) -> None:
                self._yielded = False
                self.aclose = inner_aclose

            def __aiter__(self) -> MockAsyncIterator:
                return self

            async def __anext__(self) -> dict[str, str]:
                if self._yielded:
                    raise StopAsyncIteration
                self._yielded = True
                return {"content": "test"}

        inner = MockAsyncIterator()

        iterator = RateLimitedAsyncIterator(inner, context)

        # Should not raise
        await iterator.aclose()


class TestRateLimitedAsyncIteratorAiter:
    """Tests for RateLimitedAsyncIterator.__aiter__ method."""

    @pytest.fixture
    def mock_backend(self) -> Mock:
        """Create a mock backend."""
        backend = Mock()
        backend.release_streaming_reservation = AsyncMock(return_value=True)
        return backend

    @pytest.fixture
    def context(self, mock_backend: Mock) -> StreamingReservationContext:
        """Create a test context."""
        return StreamingReservationContext(
            reservation_id="res-1",
            bucket_id="bucket-1",
            request_id="req-1",
            reserved_tokens=1000,
            backend=mock_backend,
        )

    @pytest.mark.asyncio
    async def test_returns_self(self, context: StreamingReservationContext) -> None:
        """Verify __aiter__ returns self."""

        async def mock_iter() -> AsyncIterator[dict[str, str]]:
            yield {"content": "test"}

        iterator = RateLimitedAsyncIterator(mock_iter(), context)

        assert iterator.__aiter__() is iterator


class TestRateLimitedAsyncIteratorContextProperty:
    """Tests for RateLimitedAsyncIterator.context property."""

    @pytest.fixture
    def mock_backend(self) -> Mock:
        """Create a mock backend."""
        backend = Mock()
        backend.release_streaming_reservation = AsyncMock(return_value=True)
        return backend

    @pytest.fixture
    def context(self, mock_backend: Mock) -> StreamingReservationContext:
        """Create a test context."""
        return StreamingReservationContext(
            reservation_id="res-1",
            bucket_id="bucket-1",
            request_id="req-1",
            reserved_tokens=1000,
            backend=mock_backend,
        )

    @pytest.mark.asyncio
    async def test_returns_context(self, context: StreamingReservationContext) -> None:
        """Verify context property returns the context."""

        async def mock_iter() -> AsyncIterator[dict[str, str]]:
            yield {"content": "test"}

        iterator = RateLimitedAsyncIterator(mock_iter(), context)

        assert iterator.context is context


class TestRateLimitedAsyncIteratorLongRunningWarning:
    """Tests for long-running stream warning functionality."""

    @pytest.fixture
    def mock_backend(self) -> Mock:
        """Create a mock backend."""
        backend = Mock()
        backend.release_streaming_reservation = AsyncMock(return_value=True)
        return backend

    @pytest.mark.asyncio
    async def test_warns_for_long_running_stream(self, mock_backend: Mock) -> None:
        """Verify warning is logged for long-running streams."""
        import time

        # Create context with old created_at (25 minutes ago)
        context = StreamingReservationContext(
            reservation_id="res-1",
            bucket_id="bucket-1",
            request_id="req-1",
            reserved_tokens=1000,
            backend=mock_backend,
            created_at=time.time() - 1500,  # 25 minutes ago
        )

        async def mock_iter() -> AsyncIterator[dict[str, str]]:
            yield {"content": "Hello"}
            yield {"content": "World"}

        iterator = RateLimitedAsyncIterator(mock_iter(), context)

        with patch("adaptive_rate_limiter.streaming.iterator.logger") as mock_logger:
            [chunk async for chunk in iterator]

            # Should warn about long-running stream
            assert any(
                "running for" in str(call)
                for call in mock_logger.warning.call_args_list
            )

    @pytest.mark.asyncio
    async def test_warns_only_once(self, mock_backend: Mock) -> None:
        """Verify long-running warning is only logged once."""
        import time

        context = StreamingReservationContext(
            reservation_id="res-1",
            bucket_id="bucket-1",
            request_id="req-1",
            reserved_tokens=1000,
            backend=mock_backend,
            created_at=time.time() - 1500,  # 25 minutes ago
        )

        async def mock_iter() -> AsyncIterator[dict[str, str]]:
            yield {"content": "1"}
            yield {"content": "2"}
            yield {"content": "3"}
            yield {"content": "4"}
            yield {"content": "5"}

        iterator = RateLimitedAsyncIterator(mock_iter(), context)

        with patch("adaptive_rate_limiter.streaming.iterator.logger") as mock_logger:
            [chunk async for chunk in iterator]

            # Should warn only once
            long_running_warnings = [
                call
                for call in mock_logger.warning.call_args_list
                if "running for" in str(call)
            ]
            assert len(long_running_warnings) == 1

    @pytest.mark.asyncio
    async def test_no_warning_for_short_stream(self, mock_backend: Mock) -> None:
        """Verify no warning for streams under threshold."""
        context = StreamingReservationContext(
            reservation_id="res-1",
            bucket_id="bucket-1",
            request_id="req-1",
            reserved_tokens=1000,
            backend=mock_backend,
            # Default created_at is "now", so duration will be very short
        )

        async def mock_iter() -> AsyncIterator[dict[str, str]]:
            yield {"content": "Hello"}

        iterator = RateLimitedAsyncIterator(mock_iter(), context)

        with patch("adaptive_rate_limiter.streaming.iterator.logger") as mock_logger:
            [chunk async for chunk in iterator]

            # Should not warn about long-running
            long_running_warnings = [
                call
                for call in mock_logger.warning.call_args_list
                if "running for" in str(call)
            ]
            assert len(long_running_warnings) == 0
