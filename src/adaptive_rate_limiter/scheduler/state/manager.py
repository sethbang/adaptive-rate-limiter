# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
Unified State Manager for the Adaptive Rate Limiter.

Provides thread-safe state management with caching and persistence.
"""

import asyncio
import atexit
import contextlib
import logging
import signal
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Optional, cast

from ...backends.base import BaseBackend
from ...exceptions import TooManyFailedRequestsError
from ...observability.protocols import MetricsCollectorProtocol
from ..config import CachePolicy, StateConfig
from .cache import Cache
from .models import PendingUpdate, RateLimitState, StateEntry, StateType

if TYPE_CHECKING:
    from ...providers.base import ProviderInterface

logger = logging.getLogger(__name__)  # adaptive_rate_limiter.scheduler.state.manager


class StateManager:
    """
    Unified State Manager for the Adaptive Rate Limiter.

    Features:
    - Thread-safe operations with configurable locking
    - Unified caching with multiple policies
    - State versioning and recovery
    - Atomic operations and bulk operations
    - Integration with backends
    - Comprehensive metrics and monitoring

    Thread Safety:
        All public methods are thread-safe.
    """

    def __init__(
        self,
        backend: BaseBackend,
        config: StateConfig | None = None,
        provider: Optional["ProviderInterface"] = None,
        metrics_collector: MetricsCollectorProtocol | None = None,
    ) -> None:
        """
        Initialize the StateManager.

        Args:
            backend: Backend for persistence
            config: Configuration for state management
            provider: Optional provider for bucket discovery
            metrics_collector: Optional unified metrics collector for Prometheus export
        """
        self.backend = backend
        self.config = config or StateConfig()
        self.provider = provider
        self._metrics_collector = metrics_collector

        # Production safety check for WRITE_BACK policy
        if (
            self.config.cache_policy == CachePolicy.WRITE_BACK
            and self.config.is_production
        ):
            if not self.config.acknowledge_write_back_risk:
                raise ValueError(
                    "WRITE_BACK cache policy in production requires explicit acknowledgment. "
                    "Set acknowledge_write_back_risk=True in StateConfig to proceed, understanding that "
                    "data loss may occur on unexpected shutdown."
                )
            if self.config.warn_write_back_production:
                logger.warning(
                    "WRITE_BACK cache policy in production - data loss possible on crash. "
                    "Risk has been explicitly acknowledged via acknowledge_write_back_risk=True."
                )

        # Unified cache - pass metrics collector for Prometheus integration
        self.cache = Cache(self.config, metrics_collector=self._metrics_collector)

        # Failed request tracking
        self.failed_request_counter = self._create_failed_counter()

        # Batch processing - using PendingUpdate wrapper for retry tracking
        self._pending_updates: list[PendingUpdate] = []
        self._batch_lock = asyncio.Lock()
        self._last_batch_time = time.time()

        # Reservation management
        self._reservations: dict[str, dict[str, Any]] = {}
        self._reservation_timestamps: dict[str, float] = {}
        self._reservation_lock = asyncio.Lock()

        # Account-level state tracking
        self._account_states: dict[str, dict[str, Any]] = {}
        self._account_state_timestamps: dict[str, float] = {}
        self._account_lock = asyncio.Lock()

        # Background tasks
        self._batch_task: asyncio.Task[None] | None = None
        self._cleanup_task: asyncio.Task[None] | None = None
        self._running = False

        # Signal handling for graceful shutdown
        self._original_sigterm: Any = None
        self._original_sigint: Any = None
        self._signal_handlers_installed = False
        self._atexit_registered = False

        logger.info(f"StateManager initialized with backend {type(backend).__name__}")

    def _create_failed_counter(self) -> Any:
        """Create optimized failed request counter."""

        class OptimizedFailedRequestCounter:
            def __init__(self, window_seconds: float = 30.0, max_failures: int = 20):
                self.window_seconds = window_seconds
                self.max_failures = max_failures
                self.failure_times: deque[float] = deque()
                self._count = 0
                self._last_cleanup = time.time()
                self._lock = threading.RLock()

            def increment(self) -> int:
                current_time = time.time()
                with self._lock:
                    if current_time - self._last_cleanup > 1.0:
                        self._cleanup(current_time)
                        self._last_cleanup = current_time
                    self.failure_times.append(current_time)
                    self._count = len(self.failure_times)
                    return self._count

            def get_count(self) -> int:
                if time.time() - self._last_cleanup < 0.1:
                    return self._count
                with self._lock:
                    current_time = time.time()
                    self._cleanup(current_time)
                    self._count = len(self.failure_times)
                    return self._count

            def is_limit_exceeded(self) -> bool:
                return self.get_count() >= self.max_failures

            def _cleanup(self, current_time: float) -> None:
                cutoff_time = current_time - self.window_seconds
                while self.failure_times and self.failure_times[0] <= cutoff_time:
                    self.failure_times.popleft()

        return OptimizedFailedRequestCounter()

    def _install_signal_handlers(self) -> None:
        """Install signal handlers for graceful shutdown with WRITE_BACK policy.

        This ensures pending updates are flushed on SIGTERM/SIGINT signals,
        reducing the risk of data loss when using WRITE_BACK cache policy.
        """
        if self._signal_handlers_installed:
            return

        try:
            # Only install signal handlers on the main thread
            import threading

            if threading.current_thread() is not threading.main_thread():
                logger.debug("Skipping signal handlers on non-main thread")
                return

            self._original_sigterm = signal.getsignal(signal.SIGTERM)
            self._original_sigint = signal.getsignal(signal.SIGINT)

            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)

            self._signal_handlers_installed = True
            logger.info("Signal handlers installed for graceful WRITE_BACK shutdown")

        except (ValueError, OSError) as e:
            # Signal handlers can only be set in main thread
            logger.debug(f"Could not install signal handlers: {e}")

    def _restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        if not self._signal_handlers_installed:
            return

        try:
            import threading

            if threading.current_thread() is not threading.main_thread():
                return

            if self._original_sigterm is not None:
                signal.signal(signal.SIGTERM, self._original_sigterm)
            if self._original_sigint is not None:
                signal.signal(signal.SIGINT, self._original_sigint)

            self._signal_handlers_installed = False
            logger.debug("Original signal handlers restored")

        except (ValueError, OSError) as e:
            logger.debug(f"Could not restore signal handlers: {e}")

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle SIGTERM/SIGINT signals by flushing pending updates.

        This is a synchronous handler that flushes pending updates before
        allowing the process to terminate, reducing data loss with WRITE_BACK.
        """
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, flushing pending WRITE_BACK updates...")

        try:
            # Synchronously flush pending updates
            self._flush_pending_updates_sync()
            logger.info(f"Pending updates flushed successfully after {sig_name}")
        except Exception as e:
            logger.error(f"Failed to flush pending updates on {sig_name}: {e}")

        # Call original handler or re-raise signal
        original = (
            self._original_sigterm
            if signum == signal.SIGTERM
            else self._original_sigint
        )

        if original in (signal.SIG_DFL, signal.SIG_IGN, None):
            # Re-raise the signal with default handler
            signal.signal(signum, signal.SIG_DFL)
            signal.raise_signal(signum)
        elif callable(original):
            original(signum, frame)

    def _flush_pending_updates_sync(self) -> None:
        """Synchronously flush pending updates to backend.

        Used by signal handlers and atexit handlers which cannot use async.
        Note: This method does not implement retry logic since it's a best-effort
        flush during shutdown. Retry state is not preserved across shutdowns.

        For running event loops, uses asyncio.run_coroutine_threadsafe() with a timeout
        to block until writes complete, ensuring data is flushed before signal handler exits.
        """
        if not self._pending_updates:
            return

        # Get a copy of pending updates under lock
        with threading.Lock():
            updates = self._pending_updates[:]
            self._pending_updates.clear()

        logger.info(f"Sync flushing {len(updates)} pending state updates")

        # Determine if we have a running event loop
        try:
            loop = asyncio.get_running_loop()
            loop_is_running = True
        except RuntimeError:
            # No running event loop
            loop_is_running = False
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # No event loop at all, create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

        # Use synchronous backend operations if available
        for pending in updates:
            entry = pending.entry
            try:
                if loop_is_running:
                    # Use run_coroutine_threadsafe and wait with timeout to ensure
                    # the write completes before signal handler exits
                    future = asyncio.run_coroutine_threadsafe(
                        self.backend.set_state(entry.key, entry.data), loop
                    )
                    try:
                        future.result(
                            timeout=5.0
                        )  # Block with 5 second timeout per write
                    except TimeoutError:
                        logger.warning(
                            f"Flush timed out for state key '{entry.key}' during signal handling"
                        )
                    except Exception as e:
                        logger.error(f"Failed to flush state {entry.key}: {e}")
                else:
                    # Run synchronously in a non-running loop
                    loop.run_until_complete(
                        self.backend.set_state(entry.key, entry.data)
                    )
            except Exception as e:
                logger.error(f"Failed to flush state {entry.key}: {e}")

    def _register_atexit_handler(self) -> None:
        """Register atexit handler as a fallback safety net.

        This provides additional protection for WRITE_BACK mode by ensuring
        pending updates are flushed when the Python interpreter exits normally.
        """
        if self._atexit_registered:
            return

        def _atexit_flush() -> None:
            if self._pending_updates:
                logger.info("Atexit: Flushing pending WRITE_BACK updates...")
                self._flush_pending_updates_sync()

        atexit.register(_atexit_flush)
        self._atexit_registered = True
        logger.debug("Atexit handler registered for WRITE_BACK safety")

    async def start(self) -> None:
        """Start the StateManager and background tasks."""
        if self._running:
            return

        self._running = True
        await self.cache.start()

        self._batch_task = asyncio.create_task(self._batch_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        # Install signal handlers for graceful shutdown with WRITE_BACK policy
        if self.config.cache_policy == CachePolicy.WRITE_BACK:
            self._install_signal_handlers()
            self._register_atexit_handler()

        logger.info("StateManager started with all background tasks")

    async def stop(self) -> None:
        """Stop StateManager and flush all pending data."""
        if not self._running:
            return

        self._running = False

        await self._flush_pending_updates()

        if self._batch_task:
            self._batch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._batch_task

        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task

        await self.cache.stop()

        # Restore original signal handlers
        self._restore_signal_handlers()

        logger.info("StateManager stopped and all data flushed")

    # === Core State Management API ===

    async def get_state(
        self,
        key: str,
        state_type: StateType = StateType.RATE_LIMIT,
        force_refresh: bool = False,
    ) -> Any | None:
        """Get state for a key."""
        if not force_refresh:
            entry = await self.cache.get(key)
            if entry:
                if state_type == StateType.RATE_LIMIT:
                    return RateLimitState(**entry.data)
                return entry.data

        try:
            state_dict = await self.backend.get_state(key)
            if state_dict:
                entry = StateEntry(
                    key=key,
                    data=state_dict,
                    state_type=state_type,
                    namespace=self.config.namespace,
                )
                await self.cache.set(entry)

                if state_type == StateType.RATE_LIMIT:
                    return RateLimitState(**state_dict)
                return state_dict
        except (AttributeError, ValueError, OSError, TypeError) as e:
            logger.warning(f"Backend failure for key {key}: {e}")

        if state_type == StateType.RATE_LIMIT:
            return await self._initialize_rate_limit_state(key)

        return None

    async def set_state(
        self,
        key: str,
        data: dict[str, Any] | RateLimitState,
        state_type: StateType = StateType.RATE_LIMIT,
    ) -> None:
        """Set state for a key."""
        if isinstance(data, RateLimitState):
            data_dict = data.model_dump(mode="json")
        else:
            data_dict = data

        entry = StateEntry(
            key=key,
            data=data_dict,
            state_type=state_type,
            namespace=self.config.namespace,
        )

        await self.cache.set(entry)

        if self.config.cache_policy == CachePolicy.WRITE_THROUGH:
            await self.backend.set_state(key, data_dict)
        elif self.config.cache_policy == CachePolicy.WRITE_BACK:
            await self._add_to_batch(entry)
        elif self.config.cache_policy == CachePolicy.WRITE_AROUND:
            await self.backend.set_state(key, data_dict)
            await self.cache.delete(key)

    async def update_state(
        self, key: str, updates: dict[str, Any], merge: bool = True
    ) -> Any | None:
        """Update existing state with new data using atomic operations."""
        entry = await self.cache.atomic_update(key, updates, merge)

        if entry and self.config.cache_policy != CachePolicy.WRITE_AROUND:
            if self.config.cache_policy == CachePolicy.WRITE_THROUGH:
                await self.backend.set_state(key, entry.data)
            else:
                await self._add_to_batch(entry)

        return entry.data if entry else None

    # === Scheduler Compatibility API ===

    async def update_state_from_headers(
        self,
        model_id: str,
        resource_type: str,
        headers: dict[str, str],
        request_id: str | None = None,
        status_code: int | None = None,
    ) -> int:
        """Update state from API response headers."""
        if not self.provider:
            logger.error("Provider required for header-based state updates")
            return 0

        bucket_id = await self.provider.get_bucket_for_model(model_id, resource_type)
        if not bucket_id:
            logger.warning(f"Could not resolve bucket for model {model_id}")
            return 0

        result = await self.backend.update_rate_limits(
            model_id,
            headers,
            bucket_id=bucket_id,
            request_id=request_id,
            status_code=status_code,
        )

        if result == 1:
            state = await self.get_state(bucket_id, StateType.RATE_LIMIT)

            if not state:
                state = RateLimitState.create_fallback_state(
                    model_id=bucket_id, bucket_id=bucket_id
                )

            state.update_from_headers(headers)
            await self.set_state(bucket_id, state)

        return result

    async def check_and_reserve_capacity(
        self,
        model_id: str,
        resource_type: str,
        request_count: int = 1,
        token_count: int | None = None,
    ) -> tuple[bool, str | None]:
        """Check and reserve capacity atomically."""
        if not self.provider:
            raise ValueError("provider is required for check_and_reserve_capacity")

        try:
            bucket_id = await self.provider.get_bucket_for_model(
                model_id, resource_type
            )
            if not bucket_id:
                logger.warning(f"No bucket found for model {model_id}")
                return False, None

            # Discover bucket limits
            buckets = await self.provider.discover_limits()
            bucket = buckets.get(bucket_id)
            if not bucket:
                raise ValueError(f"Bucket {bucket_id} not found in provider")

            bucket_limits = {
                "rpm_limit": bucket.rpm_limit or 60,
                "tpm_limit": bucket.tpm_limit or 10000,
            }

            if bucket_limits["rpm_limit"] <= 0:
                raise ValueError(f"Invalid rpm_limit for bucket {bucket_id}")
            if bucket_limits["tpm_limit"] <= 0:
                raise ValueError(f"Invalid tpm_limit for bucket {bucket_id}")

            success, reservation_id = await self.backend.check_and_reserve_capacity(
                bucket_id, request_count, token_count or 0, bucket_limits=bucket_limits
            )

            # Sync cache with backend
            backend_state = await self.backend.get_state(bucket_id)
            if backend_state:
                try:
                    entry = StateEntry(
                        key=bucket_id,
                        data=backend_state,
                        state_type=StateType.RATE_LIMIT,
                        namespace=self.config.namespace,
                    )
                    await self.cache.set(entry)
                except Exception as sync_error:
                    logger.debug(f"Cache sync failed: {sync_error}")

            return success, reservation_id
        except (AttributeError, TypeError, OSError) as e:
            logger.warning(f"Provider failure for model {model_id}: {e}")
            return False, None
        except ValueError:
            raise

    async def release_reservation(
        self, reservation_id: str, model_id: str, resource_type: str
    ) -> None:
        """Release a capacity reservation."""
        if not self.provider:
            logger.warning(
                f"Cannot release reservation {reservation_id}: provider not available"
            )
            return

        bucket_id = await self.provider.get_bucket_for_model(model_id, resource_type)
        if not bucket_id:
            logger.warning(
                f"Cannot release reservation {reservation_id}: bucket not found"
            )
            return

        await self.backend.release_reservation(bucket_id, reservation_id)

    async def record_failed_request(self, model_id: str | None = None) -> int:
        """Record failed request.

        Args:
            model_id: Optional model identifier for logging context.

        Returns:
            The current failure count within the tracking window.

        Raises:
            TooManyFailedRequestsError: If more than 20 requests have failed
                within the 30-second tracking window.
        """
        count = self.failed_request_counter.increment()
        if count > 20:
            logger.error("Failed request limit exceeded: 20 failures in 30s")
            raise TooManyFailedRequestsError(
                message="Too many failed requests: 20 failures in 30 seconds",
                failure_count=count,
                window_seconds=30.0,
                threshold=20,
            )
        return int(count)

    def get_failed_count_sync(self) -> int:
        """Get failed count synchronously."""
        return int(self.failed_request_counter.get_count())

    def is_failed_limit_exceeded(self) -> bool:
        """Check if failed limit exceeded."""
        return bool(self.failed_request_counter.is_limit_exceeded())

    async def get_next_reset_time(self, bucket_id: str) -> datetime | None:
        """Get next reset time for bucket."""
        state = await self.get_state(bucket_id, StateType.RATE_LIMIT)
        if not state:
            return None

        current_time = datetime.now(timezone.utc)
        earliest_reset = None

        if (
            state.remaining_requests is not None
            and state.remaining_requests <= 0
            and state.reset_at > current_time
        ):
            earliest_reset = state.reset_at

        if (
            state.remaining_requests_daily is not None
            and state.remaining_requests_daily <= 0
            and state.reset_at_daily > current_time
        ) and (earliest_reset is None or state.reset_at_daily < earliest_reset):
            earliest_reset = state.reset_at_daily

        return earliest_reset

    # === Caching API ===

    async def cache_bucket_info(
        self, bucket_data: dict[str, Any], ttl_seconds: int = 3600
    ) -> None:
        """Cache bucket discovery information."""
        entry = StateEntry(
            key="bucket_info",
            data=bucket_data,
            state_type=StateType.BUCKET_INFO,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds),
            namespace=self.config.namespace,
        )
        await self.cache.set(entry)
        await self.backend.cache_bucket_info(bucket_data, ttl_seconds)

    async def get_cached_bucket_info(self) -> dict[str, Any] | None:
        """Get cached bucket information."""
        entry = await self.cache.get("bucket_info")
        if entry and entry.state_type == StateType.BUCKET_INFO:
            return entry.data

        return await self.backend.get_cached_bucket_info()

    async def cache_model_info(
        self, model: str, model_data: dict[str, Any], ttl_seconds: int = 300
    ) -> None:
        """Cache model information."""
        entry = StateEntry(
            key=f"model:{model}",
            data=model_data,
            state_type=StateType.MODEL_CONFIG,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds),
            namespace=self.config.namespace,
        )
        await self.cache.set(entry)
        await self.backend.cache_model_info(model, model_data, ttl_seconds)

    async def get_cached_model_info(self, model: str) -> dict[str, Any] | None:
        """Get cached model information."""
        entry = await self.cache.get(f"model:{model}")
        if entry and entry.state_type == StateType.MODEL_CONFIG:
            return entry.data

        return await self.backend.get_cached_model_info(model)

    # === Reservation Management ===

    async def create_reservation(
        self,
        request_id: str,
        reservation_data: dict[str, Any],
        ttl_seconds: float = 300.0,
    ) -> str:
        """Create a time-limited reservation for request tracking."""
        async with self._reservation_lock:
            self._reservations[request_id] = {
                **reservation_data,
                "created_at": time.time(),
                "expires_at": time.time() + ttl_seconds,
            }
            self._reservation_timestamps[request_id] = time.time()

        entry = StateEntry(
            key=f"reservation:{request_id}",
            data=reservation_data,
            state_type=StateType.RESERVATION,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds),
            namespace=self.config.namespace,
        )
        await self.cache.set(entry)

        return request_id

    async def get_reservation(self, request_id: str) -> dict[str, Any] | None:
        """Get reservation data."""
        entry = await self.cache.get(f"reservation:{request_id}")
        if entry and entry.state_type == StateType.RESERVATION:
            return entry.data

        async with self._reservation_lock:
            reservation = self._reservations.get(request_id)
            if reservation and time.time() < reservation["expires_at"]:
                return reservation

        return None

    async def release_reservation_data(self, request_id: str) -> bool:
        """Release reservation data."""
        async with self._reservation_lock:
            if request_id in self._reservations:
                del self._reservations[request_id]

        await self.cache.delete(f"reservation:{request_id}")
        return True

    # === Bulk Operations ===

    async def bulk_get_states(
        self, keys: list[str], state_type: StateType = StateType.RATE_LIMIT
    ) -> dict[str, Any]:
        """Get multiple states efficiently."""
        results = {}
        cache_results = await self.cache.bulk_get(keys)

        backend_keys = []
        for key, entry in cache_results.items():
            if entry:
                if state_type == StateType.RATE_LIMIT:
                    results[key] = RateLimitState(**entry.data)
                else:
                    results[key] = cast(Any, entry.data)
            else:
                backend_keys.append(key)

        if backend_keys:
            backend_states = await self.backend.get_all_states()
            for key in backend_keys:
                if key in backend_states:
                    data = backend_states[key]
                    entry = StateEntry(
                        key=key,
                        data=data,
                        state_type=state_type,
                        namespace=self.config.namespace,
                    )
                    await self.cache.set(entry)

                    if state_type == StateType.RATE_LIMIT:
                        results[key] = RateLimitState(**data)
                    else:
                        results[key] = cast(Any, data)

        return results

    async def bulk_set_states(
        self,
        states: dict[str, dict[str, Any] | RateLimitState],
        state_type: StateType = StateType.RATE_LIMIT,
    ) -> None:
        """Set multiple states efficiently."""
        entries = []

        for key, data in states.items():
            data_dict = data.model_dump() if isinstance(data, RateLimitState) else data

            entry = StateEntry(
                key=key,
                data=data_dict,
                state_type=state_type,
                namespace=self.config.namespace,
            )
            entries.append(entry)

        await self.cache.bulk_set(entries)

        if self.config.cache_policy == CachePolicy.WRITE_THROUGH:
            for entry in entries:
                await self.backend.set_state(entry.key, entry.data)
        elif self.config.cache_policy == CachePolicy.WRITE_BACK:
            for entry in entries:
                await self._add_to_batch(entry)

    # === Internal Methods ===

    async def _initialize_rate_limit_state(self, key: str) -> RateLimitState | None:
        """Initialize a new rate limit state."""
        if self.provider:
            try:
                buckets = await self.provider.discover_limits()
                bucket = buckets.get(key)
                if bucket:
                    current_time = datetime.now(timezone.utc)
                    state = RateLimitState(
                        model_id=key,
                        remaining_requests=1,  # Conservative initialization
                        remaining_requests_daily=1000,
                        remaining_tokens=1,
                        reset_at=current_time + timedelta(minutes=1),
                        reset_at_daily=self._tomorrow_midnight(),
                        last_updated=current_time,
                        request_limit=bucket.rpm_limit,
                        token_limit=bucket.tpm_limit or 10000,
                        bucket_id=key,
                        is_verified=False,
                    )
                    await self.set_state(key, state)
                    return state
            except Exception as e:
                logger.debug(f"Could not initialize from provider: {e}")

        logger.warning(f"Initializing fallback state for {key}")
        state = RateLimitState.create_fallback_state(model_id=key)
        await self.set_state(key, state)
        return state

    def _tomorrow_midnight(self) -> datetime:
        """Get tomorrow's midnight in UTC."""
        now = datetime.now(timezone.utc)
        tomorrow = now + timedelta(days=1)
        return tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)

    async def _add_to_batch(self, entry: StateEntry) -> None:
        """Add entry to pending batch.

        Wraps the entry in a PendingUpdate with retry tracking.
        """
        should_flush = False
        pending = PendingUpdate(entry=entry, retry_count=0, last_attempt_time=0.0)

        async with self._batch_lock:
            self._pending_updates.append(pending)

            if (
                len(self._pending_updates) >= self.config.batch_size
                or time.time() - self._last_batch_time > self.config.batch_timeout
            ):
                should_flush = True

        if should_flush:
            await self._flush_pending_updates()

    async def _batch_loop(self) -> None:
        """Background task for batch processing."""
        while self._running:
            try:
                await asyncio.sleep(self.config.batch_timeout)
                if self._pending_updates:
                    await self._flush_pending_updates()
            except asyncio.CancelledError:
                break
            except (AttributeError, ValueError, OSError) as e:
                logger.exception(f"Error in batch processing: {e}")

    async def _cleanup_loop(self) -> None:
        """Background task for periodic cleanup."""
        while self._running:
            try:
                await asyncio.sleep(self.config.reservation_cleanup_interval)
                await self._cleanup_expired_reservations()
                await self._cleanup_account_states()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}", exc_info=True)

    async def _cleanup_expired_reservations(self) -> None:
        """Clean up expired reservations."""
        async with self._reservation_lock:
            current_time = time.time()
            expired_ids = []

            for request_id, reservation in self._reservations.items():
                expires_at = reservation.get("expires_at")
                if expires_at and current_time > expires_at:
                    expired_ids.append(request_id)

            for request_id in expired_ids:
                del self._reservations[request_id]
                if request_id in self._reservation_timestamps:
                    del self._reservation_timestamps[request_id]

            if expired_ids:
                logger.info(f"Cleaned up {len(expired_ids)} expired reservations")

    async def _cleanup_account_states(self) -> None:
        """Clean up account states using LRU eviction."""
        async with self._account_lock:
            current_time = time.time()

            ttl_expired = []
            for key, timestamp in self._account_state_timestamps.items():
                if current_time - timestamp > self.config.account_state_ttl:
                    ttl_expired.append(key)

            for key in ttl_expired:
                if key in self._account_states:
                    del self._account_states[key]
                del self._account_state_timestamps[key]

            if ttl_expired:
                logger.info(f"Cleaned up {len(ttl_expired)} TTL-expired account states")

            if self.config.account_state_max_size:
                overflow = (
                    len(self._account_states) - self.config.account_state_max_size
                )
                if overflow > 0:
                    sorted_entries = sorted(
                        self._account_state_timestamps.items(), key=lambda x: x[1]
                    )

                    evicted_keys = [key for key, _ in sorted_entries[:overflow]]

                    for key in evicted_keys:
                        if key in self._account_states:
                            del self._account_states[key]
                        del self._account_state_timestamps[key]

                    logger.info(f"Evicted {len(evicted_keys)} account states (LRU)")

    async def _flush_pending_updates(self) -> None:
        """Flush pending updates to backend with retry tracking.

        Implements retry logic with exponential backoff and max retry limit.
        Updates that exceed the max retry limit are dropped and logged as errors.
        """
        async with self._batch_lock:
            if not self._pending_updates:
                return

            updates = self._pending_updates[:]
            self._pending_updates.clear()
            self._last_batch_time = time.time()

        current_time = time.time()
        failed_updates: list[PendingUpdate] = []
        dropped_count = 0
        success_count = 0

        for pending in updates:
            # Check if we need to wait for backoff delay
            if pending.retry_count > 0:
                backoff_delay = pending.get_backoff_delay(
                    base_delay=self.config.flush_backoff_base,
                    max_delay=self.config.flush_backoff_max,
                )
                time_since_last = current_time - pending.last_attempt_time
                if time_since_last < backoff_delay:
                    # Not ready to retry yet, re-queue
                    failed_updates.append(pending)
                    continue

            try:
                await self.backend.set_state(pending.entry.key, pending.entry.data)
                success_count += 1

            except (AttributeError, ValueError, OSError, TypeError) as e:
                pending.retry_count += 1
                pending.last_attempt_time = time.time()

                if pending.should_retry(self.config.flush_max_retries):
                    # Re-queue with incremented retry count
                    failed_updates.append(pending)
                    self.cache.record_flush_retry()
                    logger.warning(
                        f"Backend write failed for key '{pending.entry.key}' "
                        f"(attempt {pending.retry_count}/{self.config.flush_max_retries}): {e}"
                    )
                else:
                    # Max retries exceeded - drop the update and log error
                    dropped_count += 1
                    self.cache.record_flush_drop()
                    logger.error(
                        f"DROPPING state update for key '{pending.entry.key}' after "
                        f"{pending.retry_count} failed attempts. Last error: {e}. "
                        f"This may indicate a persistent backend issue."
                    )

        if success_count > 0:
            self.cache.record_flush_success(success_count)
            logger.debug(f"Flushed {success_count} state updates to backend")

        if dropped_count > 0:
            logger.error(
                f"Dropped {dropped_count} state update(s) due to persistent backend failures. "
                f"Check backend connectivity and health."
            )

        # Re-queue failed updates (still under retry limit)
        if failed_updates:
            async with self._batch_lock:
                # Prepend to give them priority on next flush
                self._pending_updates = failed_updates + self._pending_updates
            logger.warning(
                f"Re-queued {len(failed_updates)} state update(s) for retry "
                f"(max retries: {self.config.flush_max_retries})"
            )

    # === Monitoring and Metrics ===

    def get_metrics(self) -> dict[str, Any]:
        """Get comprehensive state manager metrics."""
        cache_stats = self.cache.get_stats()

        return {
            "state_manager": {
                "namespace": self.config.namespace,
                "cache_policy": self.config.cache_policy.value,
                "running": self._running,
                "pending_updates": len(self._pending_updates),
                "failed_request_count": self.get_failed_count_sync(),
            },
            "cache": cache_stats,
            "backend": {
                "type": type(self.backend).__name__,
                "namespace": self.backend.namespace,
            },
            "reservations": len(self._reservations),
        }

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on state manager."""
        try:
            test_key = f"health_check_{uuid.uuid4().hex[:8]}"
            test_entry = StateEntry(
                key=test_key, data={"test": True}, namespace=self.config.namespace
            )

            await self.cache.set(test_entry)
            retrieved = await self.cache.get(test_key)
            await self.cache.delete(test_key)

            cache_healthy = retrieved is not None
            backend_healthy = await self.backend.health_check()

            return {
                "healthy": cache_healthy and backend_healthy.healthy,
                "cache_healthy": cache_healthy,
                "backend_healthy": backend_healthy,
                "metrics": self.get_metrics(),
            }

        except (AttributeError, ValueError, OSError, TypeError) as e:
            logger.exception(f"Health check failed: {e}")
            return {"healthy": False, "error": str(e), "metrics": self.get_metrics()}


__all__ = ["StateManager"]
