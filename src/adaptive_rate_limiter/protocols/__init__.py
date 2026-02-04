# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
Protocol definitions for rate limiter components.

This module provides Protocol classes that define the interfaces for
pluggable components in the adaptive rate limiter system.

Available protocols:
- ClientProtocol: Interface for HTTP clients that make rate-limited requests
- ClassifierProtocol: Interface for classifying requests by priority tier
- StreamingResponseProtocol: Interface for streaming response objects

Supporting types:
- RequestMetadata: Dataclass for request metadata used in rate limiting decisions
"""

from ..types.request import RequestMetadata
from .classifier import ClassifierProtocol
from .client import ClientProtocol
from .streaming import StreamingResponseProtocol

__all__ = [
    "ClassifierProtocol",
    "ClientProtocol",
    "RequestMetadata",
    "StreamingResponseProtocol",
]
