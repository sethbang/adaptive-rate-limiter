# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for StreamingHandler."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

import adaptive_rate_limiter.strategies.modes.streaming_handler as sh
from adaptive_rate_limiter.strategies.modes.streaming_handler import StreamingHandler


@pytest.mark.asyncio
async def test_on_completion_forwards_bucket_id() -> None:
    """The completion callback must forward bucket_id to record_completion."""
    metrics = MagicMock()
    backend = MagicMock()
    backend.release_reservation = AsyncMock()
    reservation_tracker = MagicMock()
    reservation_tracker.get_and_clear = AsyncMock()

    handler = StreamingHandler(
        reservation_tracker=reservation_tracker,
        backend=backend,
        streaming_metrics=metrics,
        register_callback=MagicMock(),
    )
    reservation = MagicMock()
    reservation.reservation_id = "r1"
    reservation.bucket_id = "b1"
    reservation.estimated_tokens = 1000
    metadata = MagicMock()
    metadata.request_id = "req1"

    real_ctx = sh.StreamingReservationContext
    captured: dict[str, object] = {}

    def _spy_ctx(*args: object, **kwargs: object) -> object:
        captured["metrics_callback"] = kwargs.get("metrics_callback")
        return real_ctx(*args, **kwargs)  # type: ignore[arg-type]

    sh.StreamingReservationContext = _spy_ctx  # type: ignore[assignment]
    try:
        await handler.wrap_streaming_response(object(), reservation, metadata)
    finally:
        sh.StreamingReservationContext = real_ctx  # type: ignore[assignment]

    on_completion = captured["metrics_callback"]
    assert callable(on_completion), "metrics_callback was not captured"
    on_completion(1000, 600, True, "b1", 2.5)

    metrics.record_completion.assert_called_once()
    _, kwargs = metrics.record_completion.call_args
    assert kwargs.get("bucket_id") == "b1", (
        f"Expected bucket_id='b1' in keyword args, got call_args={metrics.record_completion.call_args}"
    )


@pytest.mark.asyncio
async def test_on_completion_forwards_none_bucket_id() -> None:
    """bucket_id=None should also be forwarded (not silently dropped)."""
    metrics = MagicMock()
    backend = MagicMock()
    backend.release_reservation = AsyncMock()
    reservation_tracker = MagicMock()
    reservation_tracker.get_and_clear = AsyncMock()

    handler = StreamingHandler(
        reservation_tracker=reservation_tracker,
        backend=backend,
        streaming_metrics=metrics,
        register_callback=MagicMock(),
    )
    reservation = MagicMock()
    reservation.reservation_id = "r2"
    reservation.bucket_id = None
    reservation.estimated_tokens = 500
    metadata = MagicMock()
    metadata.request_id = "req2"

    real_ctx = sh.StreamingReservationContext
    captured: dict[str, object] = {}

    def _spy_ctx(*args: object, **kwargs: object) -> object:
        captured["metrics_callback"] = kwargs.get("metrics_callback")
        return real_ctx(*args, **kwargs)  # type: ignore[arg-type]

    sh.StreamingReservationContext = _spy_ctx  # type: ignore[assignment]
    try:
        await handler.wrap_streaming_response(object(), reservation, metadata)
    finally:
        sh.StreamingReservationContext = real_ctx  # type: ignore[assignment]

    on_completion = captured["metrics_callback"]
    assert callable(on_completion), "metrics_callback was not captured"
    on_completion(500, 500, False, None, None)

    metrics.record_completion.assert_called_once()
    _, kwargs = metrics.record_completion.call_args
    assert "bucket_id" in kwargs, (
        f"bucket_id key missing from kwargs: {metrics.record_completion.call_args}"
    )
    assert kwargs["bucket_id"] is None
