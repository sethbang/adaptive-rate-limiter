# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""Protocol for streaming response handling."""

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StreamingResponseProtocol(Protocol):
    """
    Protocol for streaming responses that can be wrapped.

    SDK implementations provide an adapter that exposes the
    underlying iterator in a provider-agnostic way.
    """

    def get_iterator(self) -> AsyncIterator[Any]:
        """Get the underlying async iterator for wrapping."""
        ...

    def set_iterator(self, iterator: AsyncIterator[Any]) -> None:
        """Replace the underlying iterator with a wrapped version."""
        ...
