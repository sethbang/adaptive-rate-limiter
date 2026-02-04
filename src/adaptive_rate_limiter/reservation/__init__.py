# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""Reservation system for capacity tracking and management.

This module provides tools for tracking and managing request capacity
reservations. It ensures accurate accounting of in-flight requests and
supports orphan recovery for distributed backends.

Exports:
    ReservationTracker: Core tracker for managing capacity reservations
    ReservationContext: Context object representing a single reservation
"""

from .context import ReservationContext
from .tracker import ReservationTracker

__all__ = ["ReservationContext", "ReservationTracker"]
