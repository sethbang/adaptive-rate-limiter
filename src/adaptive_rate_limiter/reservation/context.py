# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""ReservationContext dataclass for tracking reservation state."""

import time
from dataclasses import dataclass, field


@dataclass
class ReservationContext:
    """
    Tracks reservation state for a specific request+bucket combination.

    This is used to associate a reservation with the request that created it,
    enabling proper cleanup and accounting when the request completes or fails.

    Attributes:
        reservation_id: Unique identifier for this reservation
        bucket_id: The rate limit bucket this reservation is against
        estimated_tokens: The number of tokens reserved
        created_at: Unix timestamp when the reservation was created
    """

    reservation_id: str
    bucket_id: str
    estimated_tokens: int
    created_at: float = field(default_factory=time.time)
