# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""Protocol for API client integration."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class ClientProtocol(Protocol):
    """
    Minimal protocol for API client integration.

    This protocol is intentionally minimal. The core library does NOT need
    HTTP capabilities - those stay in the SDK-side Provider implementations.

    The core library only needs:
    1. Client identification for metrics/logging
    2. Timeout values for scheduling decisions
    """

    @property
    def base_url(self) -> str:
        """Base URL of the API (for logging/debugging)."""
        ...

    @property
    def timeout(self) -> float:
        """Request timeout in seconds."""
        ...

    def get_headers(self) -> dict[str, str]:
        """Get default headers (for debugging context)."""
        ...
