# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
# adaptive_rate_limiter/types/resource.py
"""
Resource type constants for request classification.

This module defines standard resource types used to classify API requests
for routing and rate limit bucket selection.

Constants:
    TEXT: Text generation requests (chat, completion)
    IMAGE: Image generation/editing requests
    AUDIO: Audio transcription/generation requests
    EMBEDDING: Embedding generation requests
    GENERIC: Fallback for unclassified requests
    RESOURCE_TYPES: Frozenset of all standard types

Example:
    >>> from adaptive_rate_limiter import TEXT, IMAGE
    >>> request_type = TEXT  # For chat/completion requests
"""

# Core library uses plain strings for resource types
ResourceType = str  # Type alias for clarity

# Standard resource types (as constants)
TEXT = "text"
IMAGE = "image"
AUDIO = "audio"
EMBEDDING = "embedding"
GENERIC = "generic"

# Collection of all standard resource types
RESOURCE_TYPES = frozenset({TEXT, IMAGE, AUDIO, EMBEDDING, GENERIC})
