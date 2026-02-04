"""
Tests for StateManager signal handler and atexit handler functionality.

This module covers:
- Signal handler installation for WRITE_BACK graceful shutdown
- Signal handler restoration on close
- Signal handler invocation and flush behavior
- Atexit handler registration
- Synchronous flush of pending updates
"""

import atexit
import logging
import signal
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adaptive_rate_limiter.backends.base import BaseBackend
from adaptive_rate_limiter.scheduler.config import CachePolicy, StateConfig
from adaptive_rate_limiter.scheduler.state import (
    PendingUpdate,
    StateEntry,
    StateManager,
)


@pytest.fixture
def mock_backend():
    """Create a mock backend."""
    backend = AsyncMock(spec=BaseBackend)
    backend.namespace = "default"
    return backend


@pytest.fixture
def write_back_manager(mock_backend):
    """Create a StateManager with WRITE_BACK policy."""
    config = StateConfig(cache_policy=CachePolicy.WRITE_BACK)
    return StateManager(backend=mock_backend, config=config)


@pytest.fixture
def write_through_manager(mock_backend):
    """Create a StateManager with WRITE_THROUGH policy."""
    config = StateConfig(cache_policy=CachePolicy.WRITE_THROUGH)
    return StateManager(backend=mock_backend, config=config)


class TestSignalHandlerInstallation:
    """Tests for _install_signal_handlers."""

    def test_signal_handlers_installed_on_write_back_policy(self, write_back_manager):
        """Test signal handlers are installed when using WRITE_BACK policy."""
        # Verify initial state
        assert not write_back_manager._signal_handlers_installed

        # Install handlers
        write_back_manager._install_signal_handlers()

        # Verify handlers installed
        assert write_back_manager._signal_handlers_installed

        # Verify our handler is set
        current_sigterm = signal.getsignal(signal.SIGTERM)
        current_sigint = signal.getsignal(signal.SIGINT)

        assert current_sigterm == write_back_manager._signal_handler
        assert current_sigint == write_back_manager._signal_handler

        # Clean up
        write_back_manager._restore_signal_handlers()

    def test_signal_handlers_not_installed_twice(self, write_back_manager):
        """Test signal handlers are only installed once."""
        write_back_manager._install_signal_handlers()

        # Store the original handlers we saved
        first_original_sigterm = write_back_manager._original_sigterm
        first_original_sigint = write_back_manager._original_sigint

        # Try to install again
        write_back_manager._install_signal_handlers()

        # Should not have changed the saved originals
        assert write_back_manager._original_sigterm is first_original_sigterm
        assert write_back_manager._original_sigint is first_original_sigint

        # Clean up
        write_back_manager._restore_signal_handlers()

    @pytest.mark.asyncio
    async def test_signal_handlers_installed_on_start_with_write_back(
        self, write_back_manager
    ):
        """Test signal handlers are installed when start() is called with WRITE_BACK."""
        with (
            patch.object(write_back_manager, "_batch_loop", new_callable=AsyncMock),
            patch.object(write_back_manager, "_cleanup_loop", new_callable=AsyncMock),
        ):
            await write_back_manager.start()

            assert write_back_manager._signal_handlers_installed
            assert write_back_manager._atexit_registered

            await write_back_manager.stop()

    @pytest.mark.asyncio
    async def test_signal_handlers_not_installed_on_write_through(
        self, write_through_manager
    ):
        """Test signal handlers are NOT installed for WRITE_THROUGH policy."""
        with (
            patch.object(write_through_manager, "_batch_loop", new_callable=AsyncMock),
            patch.object(
                write_through_manager, "_cleanup_loop", new_callable=AsyncMock
            ),
        ):
            await write_through_manager.start()

            # Should not install handlers for WRITE_THROUGH
            assert not write_through_manager._signal_handlers_installed
            assert not write_through_manager._atexit_registered

            await write_through_manager.stop()

    def test_signal_handlers_skip_non_main_thread(self, write_back_manager):
        """Test signal handlers are skipped on non-main thread."""
        result_holder = {"installed": None}

        def run_in_thread():
            write_back_manager._install_signal_handlers()
            result_holder["installed"] = write_back_manager._signal_handlers_installed

        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()

        # Should not have installed handlers in non-main thread
        assert result_holder["installed"] is False


class TestSignalHandlerRestoration:
    """Tests for _restore_signal_handlers."""

    def test_restore_signal_handlers(self, write_back_manager):
        """Test original signal handlers are restored on close."""
        # Store original handlers before our installation
        original_sigterm = signal.getsignal(signal.SIGTERM)
        original_sigint = signal.getsignal(signal.SIGINT)

        # Install our handlers
        write_back_manager._install_signal_handlers()

        # Verify handlers changed
        assert signal.getsignal(signal.SIGTERM) == write_back_manager._signal_handler
        assert signal.getsignal(signal.SIGINT) == write_back_manager._signal_handler

        # Restore
        write_back_manager._restore_signal_handlers()

        # Verify restoration
        assert not write_back_manager._signal_handlers_installed
        assert signal.getsignal(signal.SIGTERM) == original_sigterm
        assert signal.getsignal(signal.SIGINT) == original_sigint

    def test_restore_signal_handlers_noop_if_not_installed(self, write_back_manager):
        """Test restore is no-op if handlers weren't installed."""
        assert not write_back_manager._signal_handlers_installed

        # Should not raise, just return early
        write_back_manager._restore_signal_handlers()

        assert not write_back_manager._signal_handlers_installed

    @pytest.mark.asyncio
    async def test_signal_handlers_restored_on_stop(self, write_back_manager):
        """Test signal handlers are restored when stop() is called."""
        original_sigterm = signal.getsignal(signal.SIGTERM)
        original_sigint = signal.getsignal(signal.SIGINT)

        with (
            patch.object(write_back_manager, "_batch_loop", new_callable=AsyncMock),
            patch.object(write_back_manager, "_cleanup_loop", new_callable=AsyncMock),
        ):
            await write_back_manager.start()

            assert write_back_manager._signal_handlers_installed

            await write_back_manager.stop()

            # Handlers should be restored
            assert not write_back_manager._signal_handlers_installed
            assert signal.getsignal(signal.SIGTERM) == original_sigterm
            assert signal.getsignal(signal.SIGINT) == original_sigint


class TestSignalHandler:
    """Tests for _signal_handler behavior."""

    def test_signal_handler_calls_flush_pending_updates_sync(self, write_back_manager):
        """Test signal handler calls _flush_pending_updates_sync."""
        write_back_manager._install_signal_handlers()

        with patch.object(
            write_back_manager, "_flush_pending_updates_sync"
        ) as mock_flush:
            # Set up a handler that captures the call but prevents actual signal raise
            with patch.object(signal, "raise_signal"):
                write_back_manager._signal_handler(signal.SIGTERM, None)

            mock_flush.assert_called_once()

        write_back_manager._restore_signal_handlers()

    def test_signal_handler_handles_flush_error(self, write_back_manager, caplog):
        """Test signal handler handles flush errors gracefully."""
        write_back_manager._install_signal_handlers()

        with (
            patch.object(
                write_back_manager,
                "_flush_pending_updates_sync",
                side_effect=Exception("Flush failed"),
            ),
            patch.object(signal, "raise_signal"),
            caplog.at_level(logging.ERROR),
        ):
            write_back_manager._signal_handler(signal.SIGTERM, None)

            # Should log the error
            assert "Failed to flush pending updates" in caplog.text

        write_back_manager._restore_signal_handlers()

    def test_signal_handler_calls_original_handler_if_callable(
        self, write_back_manager
    ):
        """Test signal handler calls original handler if it was callable."""
        original_handler = MagicMock()

        # Manually set the original handler without actually changing signals
        write_back_manager._original_sigterm = original_handler
        write_back_manager._original_sigint = original_handler
        write_back_manager._signal_handlers_installed = True

        with patch.object(write_back_manager, "_flush_pending_updates_sync"):
            write_back_manager._signal_handler(signal.SIGTERM, None)

        # Original handler should be called
        original_handler.assert_called_once_with(signal.SIGTERM, None)

        # Clean up
        write_back_manager._signal_handlers_installed = False

    def test_signal_handler_re_raises_signal_for_default(self, write_back_manager):
        """Test signal handler re-raises signal when original was SIG_DFL."""
        # Manually set the original handler to SIG_DFL
        write_back_manager._original_sigterm = signal.SIG_DFL
        write_back_manager._original_sigint = signal.SIG_DFL
        write_back_manager._signal_handlers_installed = True

        with (
            patch.object(write_back_manager, "_flush_pending_updates_sync"),
            patch.object(signal, "signal") as mock_signal,
            patch.object(signal, "raise_signal") as mock_raise,
        ):
            write_back_manager._signal_handler(signal.SIGTERM, None)

            # Should set to SIG_DFL and re-raise
            mock_signal.assert_called_once_with(signal.SIGTERM, signal.SIG_DFL)
            mock_raise.assert_called_once_with(signal.SIGTERM)

        # Clean up
        write_back_manager._signal_handlers_installed = False


class TestFlushPendingUpdatesSync:
    """Tests for _flush_pending_updates_sync."""

    def test_flush_pending_updates_sync_flushes_data(self, write_back_manager):
        """Test _flush_pending_updates_sync actually flushes pending updates."""
        # Add pending updates with PendingUpdate wrapper
        entry1 = StateEntry(key="test1", data={"value": 1})
        entry2 = StateEntry(key="test2", data={"value": 2})
        write_back_manager._pending_updates = [
            PendingUpdate(entry=entry1, retry_count=0),
            PendingUpdate(entry=entry2, retry_count=0),
        ]

        # Create a mock loop that's not running
        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_loop.is_running.return_value = False
            mock_loop.run_until_complete = MagicMock()
            mock_get_loop.return_value = mock_loop

            write_back_manager._flush_pending_updates_sync()

            # Should have called run_until_complete for each entry
            assert mock_loop.run_until_complete.call_count == 2

        # Pending updates should be cleared
        assert len(write_back_manager._pending_updates) == 0

    def test_flush_pending_updates_sync_with_pending_update(self, write_back_manager):
        """Test _flush_pending_updates_sync handles PendingUpdate objects."""
        # Add pending updates as PendingUpdate wrapper
        entry = StateEntry(key="test", data={"value": 1})
        write_back_manager._pending_updates = [
            PendingUpdate(entry=entry, retry_count=0)
        ]

        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_loop.is_running.return_value = False
            mock_loop.run_until_complete = MagicMock()
            mock_get_loop.return_value = mock_loop

            write_back_manager._flush_pending_updates_sync()

            mock_loop.run_until_complete.assert_called_once()

        assert len(write_back_manager._pending_updates) == 0

    def test_flush_pending_updates_sync_empty_list(self, write_back_manager):
        """Test _flush_pending_updates_sync with empty list returns early."""
        write_back_manager._pending_updates = []

        # Should not raise, just return early
        write_back_manager._flush_pending_updates_sync()

        # Still empty
        assert len(write_back_manager._pending_updates) == 0

    def test_flush_pending_updates_sync_handles_running_loop(
        self, write_back_manager, caplog
    ):
        """Test _flush_pending_updates_sync uses run_coroutine_threadsafe when loop is running."""
        entry = StateEntry(key="test", data={"value": 1})
        write_back_manager._pending_updates = [
            PendingUpdate(entry=entry, retry_count=0)
        ]

        with patch(
            "adaptive_rate_limiter.scheduler.state.manager.asyncio.get_running_loop"
        ) as mock_get_running_loop:
            mock_loop = MagicMock()
            mock_get_running_loop.return_value = mock_loop

            # Mock the concurrent.futures.Future that run_coroutine_threadsafe returns
            mock_future = MagicMock()
            mock_future.result.return_value = None

            with patch(
                "adaptive_rate_limiter.scheduler.state.manager.asyncio.run_coroutine_threadsafe"
            ) as mock_run_threadsafe:
                mock_run_threadsafe.return_value = mock_future

                write_back_manager._flush_pending_updates_sync()

                # Should have called run_coroutine_threadsafe to block until completion
                mock_run_threadsafe.assert_called_once()
                # Should have called result() to block until the future completes
                mock_future.result.assert_called_once_with(timeout=5.0)

    def test_flush_pending_updates_sync_handles_errors(
        self, write_back_manager, caplog
    ):
        """Test _flush_pending_updates_sync handles errors during flush."""
        entry = StateEntry(key="test", data={"value": 1})
        write_back_manager._pending_updates = [
            PendingUpdate(entry=entry, retry_count=0)
        ]

        with (
            patch("asyncio.get_event_loop") as mock_get_loop,
            caplog.at_level(logging.ERROR),
        ):
            mock_loop = MagicMock()
            mock_loop.is_running.return_value = False
            mock_loop.run_until_complete.side_effect = Exception("Backend error")
            mock_get_loop.return_value = mock_loop

            write_back_manager._flush_pending_updates_sync()

            assert "Failed to flush state" in caplog.text


class TestAtexitHandler:
    """Tests for _register_atexit_handler."""

    def test_atexit_handler_registered(self, write_back_manager):
        """Test atexit handler is registered for WRITE_BACK policy."""
        with patch.object(atexit, "register") as mock_register:
            write_back_manager._register_atexit_handler()

            assert write_back_manager._atexit_registered
            mock_register.assert_called_once()

    def test_atexit_handler_not_registered_twice(self, write_back_manager):
        """Test atexit handler is only registered once."""
        with patch.object(atexit, "register") as mock_register:
            write_back_manager._register_atexit_handler()
            write_back_manager._register_atexit_handler()

            # Should only be called once
            assert mock_register.call_count == 1

    @pytest.mark.asyncio
    async def test_atexit_handler_registered_on_start(self, write_back_manager):
        """Test atexit handler is registered when start() is called."""
        with (
            patch.object(write_back_manager, "_batch_loop", new_callable=AsyncMock),
            patch.object(write_back_manager, "_cleanup_loop", new_callable=AsyncMock),
            patch.object(atexit, "register") as mock_register,
        ):
            await write_back_manager.start()

            assert write_back_manager._atexit_registered
            mock_register.assert_called_once()

            await write_back_manager.stop()

    def test_atexit_handler_function_flushes_pending(self, write_back_manager):
        """Test the atexit handler function actually flushes pending updates."""
        atexit_func = None

        with patch.object(atexit, "register") as mock_register:
            write_back_manager._register_atexit_handler()
            atexit_func = mock_register.call_args[0][0]

        # Add pending updates
        entry = StateEntry(key="test", data={"value": 1})
        write_back_manager._pending_updates = [
            PendingUpdate(entry=entry, retry_count=0)
        ]

        with patch.object(
            write_back_manager, "_flush_pending_updates_sync"
        ) as mock_flush:
            atexit_func()
            mock_flush.assert_called_once()

    def test_atexit_handler_function_does_nothing_if_empty(self, write_back_manager):
        """Test the atexit handler does nothing if no pending updates."""
        atexit_func = None

        with patch.object(atexit, "register") as mock_register:
            write_back_manager._register_atexit_handler()
            atexit_func = mock_register.call_args[0][0]

        # No pending updates
        write_back_manager._pending_updates = []

        with patch.object(
            write_back_manager, "_flush_pending_updates_sync"
        ) as mock_flush:
            atexit_func()
            mock_flush.assert_not_called()


class TestIntegration:
    """Integration tests for signal and atexit handlers."""

    @pytest.mark.asyncio
    async def test_full_lifecycle_with_write_back(self, write_back_manager):
        """Test full lifecycle: start -> signal handling setup -> stop."""
        original_sigterm = signal.getsignal(signal.SIGTERM)
        original_sigint = signal.getsignal(signal.SIGINT)

        with (
            patch.object(write_back_manager, "_batch_loop", new_callable=AsyncMock),
            patch.object(write_back_manager, "_cleanup_loop", new_callable=AsyncMock),
        ):
            # Start
            await write_back_manager.start()

            assert write_back_manager._running
            assert write_back_manager._signal_handlers_installed
            assert write_back_manager._atexit_registered

            # Verify our handlers are installed
            assert (
                signal.getsignal(signal.SIGTERM) == write_back_manager._signal_handler
            )
            assert signal.getsignal(signal.SIGINT) == write_back_manager._signal_handler

            # Stop
            await write_back_manager.stop()

            assert not write_back_manager._running
            assert not write_back_manager._signal_handlers_installed

            # Verify original handlers restored
            assert signal.getsignal(signal.SIGTERM) == original_sigterm
            assert signal.getsignal(signal.SIGINT) == original_sigint

    @pytest.mark.asyncio
    async def test_pending_updates_flushed_on_signal(self, write_back_manager):
        """Test that pending updates are flushed when signal is received."""
        with (
            patch.object(write_back_manager, "_batch_loop", new_callable=AsyncMock),
            patch.object(write_back_manager, "_cleanup_loop", new_callable=AsyncMock),
        ):
            await write_back_manager.start()

            # Add pending updates
            entry = StateEntry(key="test", data={"value": 1})
            write_back_manager._pending_updates = [
                PendingUpdate(entry=entry, retry_count=0)
            ]

            # Simulate signal handling
            with (
                patch.object(signal, "raise_signal"),
                patch(
                    "adaptive_rate_limiter.scheduler.state.manager.asyncio.get_running_loop"
                ) as mock_get_running_loop,
            ):
                mock_loop = MagicMock()
                mock_get_running_loop.return_value = mock_loop

                # Mock the concurrent.futures.Future
                mock_future = MagicMock()
                mock_future.result.return_value = None

                with patch(
                    "adaptive_rate_limiter.scheduler.state.manager.asyncio.run_coroutine_threadsafe"
                ) as mock_run_threadsafe:
                    mock_run_threadsafe.return_value = mock_future

                    write_back_manager._signal_handler(signal.SIGTERM, None)

                    # Should have called run_coroutine_threadsafe (blocks for completion)
                    mock_run_threadsafe.assert_called()
                    # Should have waited for result with timeout
                    mock_future.result.assert_called()

            await write_back_manager.stop()
