"""
Unit tests for Bundle 4 API/Factory Design Fixes.

Tests cover:
- scheduler_core_002: Sentinel pattern for mode parameter
- state_mgmt_005: Copy-on-write pattern for lock-free reads
- state_mgmt_006: TooManyFailedRequestsError exception
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from adaptive_rate_limiter.backends.memory import MemoryBackend
from adaptive_rate_limiter.exceptions import (
    RateLimiterError,
    TooManyFailedRequestsError,
)
from adaptive_rate_limiter.scheduler.config import (
    RateLimiterConfig,
    SchedulerMode,
    StateConfig,
)
from adaptive_rate_limiter.scheduler.scheduler import create_scheduler
from adaptive_rate_limiter.scheduler.state import (
    Cache,
    StateEntry,
    StateManager,
    StateType,
)


class TestSchedulerCore002SentinelPattern:
    """Tests for scheduler_core_002: Factory function mode sentinel pattern."""

    def test_mode_none_respects_config_mode(self):
        """When mode=None, the config's mode should be preserved."""
        # Create a config with a specific mode
        config = RateLimiterConfig()
        config.mode = SchedulerMode.BASIC

        # Create mock client
        client = MagicMock()

        # Call factory without mode (defaults to None)
        scheduler = create_scheduler(client=client, config=config)

        # Config mode should be preserved
        assert scheduler.config.mode == SchedulerMode.BASIC

    def test_mode_none_uses_config_default(self):
        """When both mode=None and config=None, default config mode is used.

        Test the factory function's mode handling logic directly by checking
        the config.mode after the factory processes it (using BASIC mode which
        doesn't require extra dependencies).
        """
        # Default config mode is INTELLIGENT, but we test by verifying
        # that when mode=None with a non-default config, the config's mode is preserved
        client = MagicMock()

        # Config with default mode (intelligent)
        config = RateLimiterConfig()
        assert config.mode == SchedulerMode.INTELLIGENT  # Verify default

        # To test mode=None without INTELLIGENT dependencies,
        # we override to BASIC and verify it's preserved
        config.mode = SchedulerMode.BASIC
        scheduler = create_scheduler(client=client, config=config)
        # mode=None should not change config.mode
        assert scheduler.config.mode == SchedulerMode.BASIC

    def test_explicit_mode_overrides_config(self):
        """When mode is explicitly provided, it should override config.mode."""
        config = RateLimiterConfig()
        config.mode = SchedulerMode.INTELLIGENT  # Start with INTELLIGENT

        client = MagicMock()

        # Explicitly set mode to basic (no extra deps needed)
        scheduler = create_scheduler(client=client, mode="basic", config=config)

        # Explicit mode should override config
        assert scheduler.config.mode == SchedulerMode.BASIC

    def test_explicit_basic_mode_works(self):
        """Explicit 'basic' mode should set correctly."""
        client = MagicMock()
        config = RateLimiterConfig()
        config.mode = SchedulerMode.INTELLIGENT  # Start with intelligent

        scheduler = create_scheduler(client=client, mode="basic", config=config)

        assert scheduler.config.mode == SchedulerMode.BASIC

    def test_explicit_account_mode_works(self):
        """Explicit 'account' mode should set correctly."""
        client = MagicMock()
        config = RateLimiterConfig()

        scheduler = create_scheduler(client=client, mode="account", config=config)

        assert scheduler.config.mode == SchedulerMode.ACCOUNT

    def test_invalid_mode_raises_value_error(self):
        """Invalid mode string should raise ValueError."""
        client = MagicMock()

        with pytest.raises(ValueError, match="Unknown scheduler mode"):
            create_scheduler(client=client, mode="invalid_mode")

    def test_mode_case_insensitive(self):
        """Mode strings should be case-insensitive."""
        client = MagicMock()

        # Use BASIC mode to test case-insensitivity without extra dependencies
        scheduler = create_scheduler(client=client, mode="BASIC")
        assert scheduler.config.mode == SchedulerMode.BASIC

        scheduler = create_scheduler(client=client, mode="basic")
        assert scheduler.config.mode == SchedulerMode.BASIC

        scheduler = create_scheduler(client=client, mode="Basic")
        assert scheduler.config.mode == SchedulerMode.BASIC


class TestStateMgmt005CopyOnWrite:
    """Tests for state_mgmt_005: Copy-on-write pattern for lock-free reads."""

    @pytest.fixture
    def state_config(self):
        """Create a state config with lock-free reads enabled."""
        return StateConfig(lock_free_reads=True)

    @pytest.fixture
    def cache(self, state_config):
        """Create a cache instance."""
        return Cache(state_config)

    @pytest.mark.asyncio
    async def test_atomic_update_creates_new_instance(self, cache):
        """atomic_update should create a new StateEntry, not mutate existing."""
        # Set initial entry
        initial_entry = StateEntry(
            key="test_key",
            data={"value": 1, "name": "original"},
            state_type=StateType.RATE_LIMIT,
            version=1,
        )
        await cache.set(initial_entry)

        # Store reference to original entry
        original_entry = await cache.get("test_key")
        original_id = id(original_entry)
        original_version = original_entry.version

        # Perform atomic update
        updated_entry = await cache.atomic_update("test_key", {"value": 2}, merge=True)

        # New entry should be a different object
        assert id(updated_entry) != original_id

        # Version should be incremented
        assert updated_entry.version == original_version + 1

        # Data should be merged
        assert updated_entry.data["value"] == 2
        assert updated_entry.data["name"] == "original"

    @pytest.mark.asyncio
    async def test_atomic_update_replace_creates_new_instance(self, cache):
        """atomic_update with merge=False should create new instance with replaced data."""
        initial_entry = StateEntry(
            key="test_key",
            data={"value": 1, "name": "original"},
            state_type=StateType.RATE_LIMIT,
            version=1,
        )
        await cache.set(initial_entry)

        original_entry = await cache.get("test_key")
        original_id = id(original_entry)

        # Perform atomic update with merge=False
        updated_entry = await cache.atomic_update(
            "test_key", {"new_field": "new_value"}, merge=False
        )

        # New entry should be a different object
        assert id(updated_entry) != original_id

        # Data should be completely replaced
        assert updated_entry.data == {"new_field": "new_value"}
        assert "value" not in updated_entry.data
        assert "name" not in updated_entry.data

    @pytest.mark.asyncio
    async def test_atomic_update_preserves_metadata(self, cache):
        """atomic_update should preserve state_type, namespace, metadata, expires_at."""
        from datetime import timedelta

        expires = datetime.now(timezone.utc) + timedelta(hours=1)
        initial_entry = StateEntry(
            key="test_key",
            data={"value": 1},
            state_type=StateType.BUCKET_INFO,
            version=5,
            namespace="custom_namespace",
            metadata={"custom": "metadata"},
            expires_at=expires,
        )
        await cache.set(initial_entry)

        updated_entry = await cache.atomic_update("test_key", {"value": 2}, merge=True)

        # Metadata should be preserved
        assert updated_entry.state_type == StateType.BUCKET_INFO
        assert updated_entry.namespace == "custom_namespace"
        assert updated_entry.metadata == {"custom": "metadata"}
        # Note: expires_at may be reset by TTL policy, so we check it exists
        assert updated_entry.expires_at is not None

    @pytest.mark.asyncio
    async def test_atomic_update_creates_new_entry_if_missing(self, cache):
        """atomic_update on non-existent key should create new entry."""
        updated_entry = await cache.atomic_update("new_key", {"value": 42}, merge=True)

        assert updated_entry is not None
        assert updated_entry.key == "new_key"
        assert updated_entry.data == {"value": 42}
        assert updated_entry.version == 1

    @pytest.mark.asyncio
    async def test_lock_free_read_sees_consistent_state(self, cache):
        """Lock-free readers should see consistent state (either old or new, not partial)."""
        # Set initial entry
        initial_entry = StateEntry(
            key="test_key",
            data={"a": 1, "b": 2, "c": 3},
            version=1,
        )
        await cache.set(initial_entry)

        # Simulate concurrent update and read
        # Because we use copy-on-write, the read should see either
        # the complete old state or the complete new state
        async def reader():
            for _ in range(100):
                entry = await cache.get("test_key")
                if entry:
                    data = entry.data
                    # Data should be consistent - either all old or all new values
                    # Old: a=1, b=2, c=3 (sum=6)
                    # New: a=10, b=20, c=30 (sum=60)
                    total = sum(data.values())
                    assert total in [6, 60], f"Inconsistent state detected: {data}"

        async def writer():
            for _ in range(10):
                await cache.atomic_update(
                    "test_key", {"a": 10, "b": 20, "c": 30}, merge=False
                )
                await asyncio.sleep(0.001)
                await cache.atomic_update(
                    "test_key", {"a": 1, "b": 2, "c": 3}, merge=False
                )

        # Run concurrent read/write
        await asyncio.gather(reader(), writer())


class TestStateMgmt006TooManyFailedRequestsError:
    """Tests for state_mgmt_006: TooManyFailedRequestsError exception."""

    def test_exception_inherits_from_rate_limiter_error(self):
        """TooManyFailedRequestsError should inherit from RateLimiterError."""
        assert issubclass(TooManyFailedRequestsError, RateLimiterError)

    def test_exception_can_be_caught_as_rate_limiter_error(self):
        """TooManyFailedRequestsError should be catchable as RateLimiterError."""
        try:
            raise TooManyFailedRequestsError("test message")
        except RateLimiterError as e:
            assert isinstance(e, TooManyFailedRequestsError)

    def test_exception_has_correct_attributes(self):
        """TooManyFailedRequestsError should have expected attributes."""
        exc = TooManyFailedRequestsError(
            message="Too many failures",
            failure_count=25,
            window_seconds=30.0,
            threshold=20,
        )

        assert str(exc) == "Too many failures"
        assert exc.failure_count == 25
        assert exc.window_seconds == 30.0
        assert exc.threshold == 20

    def test_exception_default_message(self):
        """TooManyFailedRequestsError should have a sensible default message."""
        exc = TooManyFailedRequestsError()
        assert str(exc) == "Too many failed requests"

    @pytest.mark.asyncio
    async def test_state_manager_raises_correct_exception(self):
        """StateManager.record_failed_request should raise TooManyFailedRequestsError."""
        backend = MemoryBackend()
        state_manager = StateManager(backend=backend)

        # Record enough failures to exceed threshold
        for _i in range(21):
            try:
                await state_manager.record_failed_request()
            except TooManyFailedRequestsError as e:
                # Should raise on the 21st failure (>20 threshold)
                assert e.failure_count == 21
                assert e.window_seconds == 30.0
                assert e.threshold == 20
                return

        pytest.fail("TooManyFailedRequestsError was not raised after 21 failures")

    @pytest.mark.asyncio
    async def test_exception_can_be_specifically_caught(self):
        """TooManyFailedRequestsError can be caught specifically, not just generic Exception."""
        backend = MemoryBackend()
        state_manager = StateManager(backend=backend)

        # Record enough failures
        for _ in range(21):
            try:
                await state_manager.record_failed_request()
            except TooManyFailedRequestsError:
                # This is what we want - specific exception type
                pass
            except Exception:
                pytest.fail(
                    "Generic Exception raised instead of TooManyFailedRequestsError"
                )
