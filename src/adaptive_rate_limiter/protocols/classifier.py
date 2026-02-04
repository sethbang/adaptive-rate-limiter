# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""Protocol for request classification."""

from typing import Any, Protocol, runtime_checkable

from ..types.request import RequestMetadata


@runtime_checkable
class ClassifierProtocol(Protocol):
    """
    Protocol for request classification.

    Providers implement this to classify requests based on their
    specific API structure and resource types.
    """

    async def classify(self, request: dict[str, Any]) -> RequestMetadata:
        """
        Classify a request and return metadata for routing.

        Args:
            request: Raw request dictionary

        Returns:
            RequestMetadata with resource_type (string), estimated_tokens, etc.
        """
        ...
