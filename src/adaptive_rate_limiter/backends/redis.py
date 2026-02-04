# Copyright 2026 Seth Bang
# SPDX-License-Identifier: Apache-2.0
"""
RedisBackend for Adaptive Rate Limiter

This module provides the RedisBackend that implements distributed rate limiting
with atomic Lua scripts for race-condition-free operations.

Key Features:
- Per-model rate limiting with pending gauge tracking
- Generation-based window management for drift correction
- Atomic Lua scripts for race-condition-free operations
- Support for Redis Cluster with hash tagging
- Orphan recovery for client crash resilience
- Fallback to MemoryBackend when Redis is unavailable
"""

import asyncio
import base64
import contextlib
import json
import logging
import os
import random
import threading
import time
import uuid
import weakref
from collections.abc import Awaitable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, cast

from redis.asyncio import ConnectionPool, Redis
from redis.asyncio.cluster import ClusterNode, RedisCluster
from redis.exceptions import (
    ConnectionError,
    NoScriptError,
    RedisError,
    ResponseError,
    TimeoutError,
)

from .base import BaseBackend, HealthCheckResult, validate_safety_margin
from .memory import MemoryBackend  # Import at module level for fallback

logger = logging.getLogger(__name__)


class _NoOpMetric:
    """No-op metric for when prometheus_client is not available."""

    def labels(self, **kwargs: Any) -> "_NoOpMetric":
        return self

    def set(self, value: Any) -> None:
        pass

    def observe(self, value: Any) -> None:
        pass


# Declare metrics with Any type to allow both Prometheus and no-op implementations
FALLBACK_ACTIVE_METRIC: Any
FALLBACK_DURATION_METRIC: Any
FALLBACK_BACKOFF_METRIC: Any

try:
    from prometheus_client import Gauge, Histogram

    FALLBACK_ACTIVE_METRIC = Gauge(
        "rate_limiter_fallback_mode_active",
        "Whether the backend is operating in fallback mode (1=active, 0=inactive)",
        ["backend"],
    )
    FALLBACK_DURATION_METRIC = Histogram(
        "rate_limiter_fallback_mode_duration_seconds",
        "Duration of time spent in fallback mode",
        ["reason"],
        buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600],
    )
    FALLBACK_BACKOFF_METRIC = Histogram(
        "rate_limiter_fallback_backoff_seconds",
        "Backoff duration applied during fallback mode",
        ["reason"],
        buckets=[0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10],
    )
except ImportError:
    # Fallback to no-op metrics if prometheus_client is not installed
    FALLBACK_ACTIVE_METRIC = _NoOpMetric()
    FALLBACK_DURATION_METRIC = _NoOpMetric()
    FALLBACK_BACKOFF_METRIC = _NoOpMetric()


class FallbackRateLimiter:
    """
    Per-request rate limiter for fallback mode.

    This provides protection against burst synchronization when multiple instances
    enter fallback simultaneously. Unlike designs that only stagger the first request,
    this enforces a minimum delay between ALL requests.

    Key Properties:
    - Lock-free jitter: Sleep happens OUTSIDE any locks
    - Per-request delay: Every request is rate-limited, not just the first
    - Exponential backoff on 429s: Backs off aggressively to prevent cascade
    - AIMD (Additive Increase, Multiplicative Decrease) for recovery
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the fallback rate limiter.

        Args:
            config: Configuration dict with keys:
                - min_delay_ms: Minimum delay between requests (default 50ms)
                - max_delay_ms: Maximum delay under heavy backoff (default 5000ms)
                - delay_decrease_ms: Amount to decrease delay on success (default 10ms)
        """
        self._config = config
        self._last_request_time: float = 0.0
        self._429_count: int = 0
        self._current_delay_ms: float = config.get("min_delay_ms", 50)
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        """
        Acquire permission to make a request.

        Returns the delay (in seconds) that was applied.

        Uses a two-phase approach:
        1. Calculate delay while holding lock (fast)
        2. Sleep OUTSIDE the lock (allows concurrent calculation for next request)
        """
        # Phase 1: Calculate delay under lock (fast, no I/O)
        async with self._lock:
            now = time.time()
            elapsed = now - self._last_request_time
            required_delay_sec = self._current_delay_ms / 1000.0

            # Calculate how long we need to wait
            wait_time = max(0.0, required_delay_sec - elapsed)

            # Add jitter (0-50% of required delay) to prevent thundering herd
            jitter = random.uniform(0, self._current_delay_ms * 0.5) / 1000.0  # nosec B311 # noqa: S311
            total_wait = wait_time + jitter

            # Update last request time to NOW + wait (reserves our slot)
            self._last_request_time = now + total_wait

        # Phase 2: Sleep OUTSIDE the lock
        if total_wait > 0:
            await asyncio.sleep(total_wait)

        return total_wait

    async def record_429(self) -> float:
        """
        Record a 429 response and increase backoff.

        Uses multiplicative increase (double delay on each 429).
        Returns the new delay in milliseconds.
        """
        async with self._lock:
            self._429_count += 1

            # Multiplicative increase: double the delay (AIMD backoff)
            self._current_delay_ms = min(
                self._current_delay_ms * 2, self._config.get("max_delay_ms", 5000)
            )

            return self._current_delay_ms

    async def record_success(self) -> None:
        """
        Record a successful request and slowly decrease delay.

        Uses additive decrease (reduce by fixed amount on success).
        This creates AIMD (Additive Increase Multiplicative Decrease) dynamics
        which is proven to find equilibrium rate.
        """
        async with self._lock:
            # Additive decrease: reduce delay by fixed amount
            self._current_delay_ms = max(
                self._current_delay_ms - self._config.get("delay_decrease_ms", 10),
                self._config.get("min_delay_ms", 50),
            )

    @property
    def current_delay_ms(self) -> float:
        """Get the current delay in milliseconds."""
        return self._current_delay_ms

    @property
    def error_count(self) -> int:
        """Get the total number of 429 errors recorded."""
        return self._429_count


@dataclass
class InFlightRequest:
    """Tracks an in-flight request for orphan recovery."""

    req_id: str
    cost_req: int
    cost_tok: int
    gen_req: int
    gen_tok: int
    start_time: float
    model: str
    account_id: str


@dataclass
class ModelLimits:
    """Per-model rate limits from API."""

    rpm: int
    tpm: int
    cached_at: float = field(default_factory=time.time)


class RedisBackend(BaseBackend):
    """
    A distributed Redis backend for rate limiting.

    This backend uses:
    - Per-account-per-model key scoping with hash tagging
    - Pending gauges for in-flight request tracking
    - Generation counters for window boundary handling
    - Atomic Lua scripts for all operations

    Deployment Requirements:
    - Redis 2.6+ (for Lua bit library)
    - Redis Cluster mode recommended for production multi-key atomicity
    """

    # Class-level connection pools indexed by event loop ID
    _connection_pools: ClassVar[dict[int, Any]] = {}
    _pool_lock: ClassVar[threading.Lock] = threading.Lock()
    _max_pools: ClassVar[int] = 20

    # WeakRef registry for automatic cleanup
    _loop_registry: ClassVar[weakref.WeakValueDictionary[int, Any]] = (
        weakref.WeakValueDictionary()
    )
    _cleanup_callbacks: ClassVar[dict[int, Any]] = {}

    # Class-level Lua scripts loaded from files
    _lua_scripts: ClassVar[dict[str, str]] = {}

    # Conservative fallback limits
    DEFAULT_RPM_LIMIT = 20
    DEFAULT_TPM_LIMIT = 500000
    DEFAULT_WINDOW_SECONDS = 60
    DEFAULT_REQ_MAP_TTL = 1800  # 30 minutes
    DEFAULT_STALE_BUFFER = 10  # seconds

    @classmethod
    def _load_lua_scripts(cls) -> None:
        """Load Lua scripts from files at class level."""
        if cls._lua_scripts:
            return  # Already loaded

        lua_dir = Path(__file__).parent / "lua"

        scripts = [
            "distributed_check_and_reserve",
            "distributed_update_rate_limits",
            "distributed_update_rate_limits_429",
            "distributed_release_capacity",
            "distributed_recover_orphan",
            "distributed_release_streaming",
        ]

        for script_name in scripts:
            script_path = lua_dir / f"{script_name}.lua"
            if script_path.exists():
                cls._lua_scripts[script_name] = script_path.read_text(encoding="utf-8")
            else:
                logger.warning(f"Lua script not found: {script_path}")

    def __init__(
        self,
        redis_url: str | None = None,
        redis_client: Any | None = None,
        account_id: str = "default",
        namespace: str = "rate_limiter",
        key_ttl: int = 86400,  # 24 hours
        max_connections: int = 10,
        req_map_ttl: int = 1800,  # 30 minutes
        stale_buffer: int = 10,  # seconds
        orphan_recovery_interval: int = 30,  # seconds
        max_request_timeout: int = 300,  # 5 minutes
        max_token_delta: int = 120,  # Maximum valid token reset delta (2x 60s window)
        log_validation_failures: bool = True,  # Log when header validation fails
        cluster_mode: bool = False,  # Whether to use Redis Cluster client
        cluster_url: str
        | None = None,  # Cluster URL (env var fallback: REDIS_CLUSTER_URL)
        startup_nodes: list[tuple[str, int]]
        | None = None,  # Multiple cluster startup nodes
    ) -> None:
        """
        Initialize distributed Redis backend.

        Args:
            redis_url: Redis connection URL. If not provided, falls back to REDIS_URL
                environment variable, then to "redis://localhost:6379".
            redis_client: Optional pre-configured Redis client
            account_id: Account ID for key scoping
            namespace: Namespace prefix for keys
            key_ttl: Default TTL for keys in seconds (24h)
            max_connections: Maximum connections per pool
            req_map_ttl: TTL for request mappings (30min)
            stale_buffer: Buffer for stale window detection (10s)
            orphan_recovery_interval: How often to scan for orphans (30s)
            max_request_timeout: Max time before request is considered orphaned (5min)
            max_token_delta: Maximum valid token reset delta in seconds (default 120).
                The 120-second cap provides margin for clock skew (up to 30s),
                network latency, and edge cases near window boundaries.
            log_validation_failures: Whether to log when header validation fails (default True)
            cluster_mode: Whether to use Redis Cluster client (default False)
            cluster_url: Redis Cluster connection URL. If not provided, falls back to
                REDIS_CLUSTER_URL environment variable. Used when cluster_mode=True
                and startup_nodes is not provided.
            startup_nodes: Optional list of (host, port) tuples for cluster mode.
                When provided, enables multi-node discovery for better failover resilience.
                Example: [("localhost", 7001), ("localhost", 7002), ("localhost", 7003)]

        Environment Variables:
            REDIS_URL: Default Redis connection URL when redis_url parameter is not provided.
            REDIS_CLUSTER_URL: Default Redis Cluster URL when cluster_url parameter is not
                provided. Used for cluster_mode=True when startup_nodes is not provided.
        """
        super().__init__(namespace)

        # Use env vars as fallbacks, then hardcoded defaults
        self.redis_url = (
            redis_url or os.environ.get("REDIS_URL") or "redis://localhost:6379"
        )
        self.cluster_url = cluster_url or os.environ.get("REDIS_CLUSTER_URL")
        self.account_id = account_id
        self.key_ttl = key_ttl
        self.max_connections = max_connections
        self.cluster_mode = cluster_mode
        self.startup_nodes = startup_nodes
        self.req_map_ttl = req_map_ttl
        self.stale_buffer = stale_buffer
        self.orphan_recovery_interval = orphan_recovery_interval
        self.max_request_timeout = max_request_timeout
        self.max_token_delta = max_token_delta
        self.log_validation_failures = log_validation_failures

        # Base64 encode account_id for safe key construction
        self._account_b64 = (
            base64.urlsafe_b64encode(account_id.encode()).decode().rstrip("=")
        )

        # Redis client state
        self._redis: Any | None = redis_client
        self._owned_redis = redis_client is None
        self._event_loop_id: int | None = None
        self._connected = False
        self._connection_lock = asyncio.Lock()

        # Lua script SHAs
        self._script_shas: dict[str, str] = {}

        # In-flight request tracking for orphan recovery
        self._in_flight: dict[str, InFlightRequest] = {}
        self._in_flight_lock = asyncio.Lock()

        # Orphan recovery task
        self._orphan_recovery_task: asyncio.Task[None] | None = None

        # Per-model limits cache (in-memory)
        self._model_limits: dict[str, ModelLimits] = {}
        self._model_limits_lock = asyncio.Lock()

        # Failure tracking for circuit breaker
        self._failure_timestamps: list[float] = []
        self._failure_lock = asyncio.Lock()

        # Key prefixes
        self.key_prefix = "rl"
        self.model_limits_key = f"rl:{self._account_b64}:model_limits"

        # === Fallback Mode State ===
        # When Redis is unavailable (circuit breaker open), we fall back to
        # MemoryBackend with conservative limits to prevent abuse protection cascade.
        self._fallback_backend: MemoryBackend | None = None
        self._fallback_start_time: float | None = None
        self._fallback_init_lock = (
            asyncio.Lock()
        )  # Only for initialization, not ongoing requests
        self._fallback_rate_limiter: FallbackRateLimiter | None = None

        # Fallback configuration - per-request rate limiting with AIMD dynamics
        self._fallback_config: dict[str, Any] = {
            # Rate limiting (per-request, not per-activation)
            "min_delay_ms": 50,  # Minimum 50ms between requests per instance
            "max_delay_ms": 5000,  # Maximum 5 seconds between requests under heavy backoff
            "delay_decrease_ms": 10,  # Decrease delay by 10ms on each success (AIMD)
            # Limit divisor (applied to token bucket)
            "limit_divisor": 20,  # Use 1/20th of actual limits
            # Metrics
            "log_interval_seconds": 30,  # Log fallback status every 30s
        }

    def _get_model_b64(self, model: str) -> str:
        """Get Base64-encoded model ID for key construction."""
        return base64.urlsafe_b64encode(model.encode()).decode().rstrip("=")

    def _get_hash_tag(self, model: str) -> str:
        """Get hash tag for Redis Cluster slot consistency."""
        model_b64 = self._get_model_b64(model)
        return f"{{{self._account_b64}|{model_b64}}}"

    def _get_state_key(self, model: str) -> str:
        """Get Redis key for state storage."""
        hash_tag = self._get_hash_tag(model)
        return f"rl:{hash_tag}:state"

    def _get_pending_req_key(self, model: str) -> str:
        """Get Redis key for pending requests gauge."""
        hash_tag = self._get_hash_tag(model)
        return f"rl:{hash_tag}:pending_req"

    def _get_pending_tok_key(self, model: str) -> str:
        """Get Redis key for pending tokens gauge."""
        hash_tag = self._get_hash_tag(model)
        return f"rl:{hash_tag}:pending_tok"

    def _get_req_map_key(self, model: str, req_id: str) -> str:
        """Get Redis key for request mapping."""
        hash_tag = self._get_hash_tag(model)
        return f"rl:{hash_tag}:req_map:{req_id}"

    async def _ensure_connected(self) -> Any:
        """Ensure Redis connection with proper cleanup on loop switches."""
        try:
            current_loop = asyncio.get_running_loop()
            loop_id = id(current_loop)

            # Check if we need to switch pools
            if self._event_loop_id and self._event_loop_id != loop_id:
                old_connection = self._redis
                if old_connection is not None:
                    _ = asyncio.create_task(  # noqa: RUF006
                        self._cleanup_connection(self._event_loop_id, old_connection)
                    )
                self._redis = None
                self._connected = False

            self._event_loop_id = loop_id

            # Return existing connection if valid
            if self._redis and self._connected:
                try:
                    await self._redis.ping()
                    return self._redis
                except (ConnectionError, Exception):
                    logger.warning("Redis connection lost, reconnecting...")
                    # For cluster mode, force close to trigger slot map refresh on reconnect
                    if self.cluster_mode and self._redis is not None:
                        try:
                            await self._redis.close()
                        except Exception as e:
                            logger.debug(
                                f"Error closing cluster client during reconnect: {e}"
                            )
                    self._redis = None
                    self._connected = False

            # Create or reuse connection pool
            async with self._connection_lock:
                # Initialize pool tracking variables before conditional
                # to avoid "possibly unbound" warnings
                pool: Any = None
                created_new_pool = False

                if self.cluster_mode:
                    # Redis Cluster handles its own pooling internally
                    # Note: RedisCluster uses 'retry' and 'retry_on_error' instead of 'retry_on_timeout'
                    if self._redis is None:
                        if self.startup_nodes:
                            # Use multiple startup nodes for better failover resilience
                            # This allows the client to discover the cluster even if
                            # the first node is down during reconnection
                            #
                            # IMPORTANT: Shuffle the startup nodes to distribute load
                            # and avoid always hitting the same (potentially dead) node first.
                            # This is critical for failover recovery when node 1 is down.
                            import random as _random

                            shuffled_nodes = list(self.startup_nodes)
                            _random.shuffle(shuffled_nodes)

                            cluster_nodes = [
                                ClusterNode(host=host, port=port)
                                for host, port in shuffled_nodes
                            ]
                            logger.info(
                                f"Initializing Redis Cluster client with {len(cluster_nodes)} "
                                f"startup nodes for loop {loop_id} "
                                f"(first node: {shuffled_nodes[0][0]}:{shuffled_nodes[0][1]})"
                            )
                            self._redis = RedisCluster(
                                startup_nodes=cluster_nodes,
                                decode_responses=True,
                                socket_connect_timeout=5.0,
                                socket_timeout=5.0,
                                # Failover resilience parameters:
                                # - require_full_coverage=False: Allow operations even when
                                #   not all 16384 slots are covered (during failover)
                                # - reinitialize_steps: Number of times to reinitialize
                                #   cluster topology on errors (default is 5, we use 10)
                                # - cluster_error_retry_attempts: Retries on MOVED/ASK
                                #   redirections (default is 3, we use 5)
                                require_full_coverage=False,
                                reinitialize_steps=10,
                                cluster_error_retry_attempts=5,
                            )
                        else:
                            # Single URL fallback (backward compatible)
                            # Use cluster_url if available, otherwise fall back to redis_url
                            cluster_connection_url = self.cluster_url or self.redis_url
                            logger.info(
                                f"Initializing Redis Cluster client for loop {loop_id}"
                            )
                            self._redis = RedisCluster.from_url(
                                cluster_connection_url,
                                decode_responses=True,
                                socket_connect_timeout=5.0,
                                socket_timeout=5.0,
                                # Failover resilience parameters
                                require_full_coverage=False,
                                reinitialize_steps=10,
                                cluster_error_retry_attempts=5,
                            )
                else:
                    # Standard Redis with manual pooling
                    # Create pool in a temporary variable first - only add to
                    # _connection_pools AFTER successful ping verification to
                    # prevent pool leak on connection failures (redis_002 fix)
                    with self._pool_lock:
                        pool = self._connection_pools.get(loop_id)
                        if pool is None:
                            pool = ConnectionPool.from_url(
                                self.redis_url,
                                max_connections=self.max_connections,
                                decode_responses=True,
                                socket_connect_timeout=5,
                                socket_timeout=5,
                                retry_on_timeout=True,
                                health_check_interval=30,
                            )
                            created_new_pool = True

                    if pool and Redis is not None:
                        self._redis = Redis(connection_pool=pool)
                    elif Redis is not None:
                        self._redis = Redis.from_url(
                            self.redis_url,
                            encoding="utf-8",
                            decode_responses=True,
                            socket_connect_timeout=5,
                            socket_timeout=5,
                            max_connections=self.max_connections,
                        )

                if self._redis:
                    try:
                        # === CLUSTER-AWARE CONNECTION WITH RETRY ===
                        # In cluster mode, the initial ping() may fail if the first
                        # startup node is down. We need to retry with topology refresh
                        # to ensure we connect to available nodes.
                        ping_success = False
                        last_ping_error: Exception | None = None

                        if self.cluster_mode:
                            # Try multiple times to ping with topology refresh between attempts.
                            # The cluster-node-timeout in our docker setup is 5000ms (5 seconds),
                            # so we need to wait at least that long for failover to complete.
                            # We use 12 attempts with 1 second delay = ~12 seconds total timeout,
                            # which gives the cluster time to:
                            # 1. Detect node failure (5s cluster-node-timeout)
                            # 2. Promote replica to master
                            # 3. Update slot mappings across the cluster
                            max_ping_attempts = 12
                            for attempt in range(max_ping_attempts):
                                try:
                                    await asyncio.wait_for(
                                        cast(Awaitable[bool], self._redis.ping()),
                                        timeout=5.0,
                                    )
                                    ping_success = True
                                    break
                                except (
                                    ConnectionError,
                                    TimeoutError,
                                    asyncio.TimeoutError,
                                ) as e:
                                    last_ping_error = e
                                    logger.debug(
                                        f"Cluster ping attempt {attempt + 1}/{max_ping_attempts} failed: {e}. "
                                        f"Waiting for cluster failover and refreshing topology..."
                                    )
                                    # Wait longer to give the cluster time to complete failover.
                                    # The cluster needs ~5 seconds to detect failure, plus time
                                    # for replica promotion. Use exponential backoff capped at 2s.
                                    delay = min(1.0 * (1.5**attempt), 2.0)
                                    await asyncio.sleep(delay)

                                    # Force topology refresh after waiting
                                    if hasattr(self._redis, "nodes_manager"):
                                        try:
                                            await asyncio.wait_for(
                                                self._redis.nodes_manager.initialize(),
                                                timeout=10.0,
                                            )
                                        except Exception as init_err:
                                            logger.debug(
                                                f"Topology refresh failed: {init_err}"
                                            )

                            if not ping_success:
                                # All attempts failed - re-raise the last error
                                raise last_ping_error or ConnectionError(
                                    "All ping attempts failed"
                                )
                        else:
                            # Standard Redis - single ping attempt
                            await asyncio.wait_for(
                                cast(Awaitable[bool], self._redis.ping()),
                                timeout=5.0,
                            )

                        self._connected = True

                        # === CLUSTER SLOT MAP REFRESH ===
                        # After successful connection in cluster mode, ensure the slot map
                        # is fully refreshed to pick up any topology changes (failover, etc.)
                        if self.cluster_mode and hasattr(self._redis, "nodes_manager"):
                            logger.debug(
                                "Refreshing cluster slot map after connection..."
                            )
                            try:
                                # Force the nodes manager to reinitialize from available nodes
                                await asyncio.wait_for(
                                    self._redis.nodes_manager.initialize(), timeout=10.0
                                )
                                logger.debug(
                                    f"Cluster slot map refreshed successfully. "
                                    f"Known nodes: {list(self._redis.nodes_manager.nodes_cache.keys()) if hasattr(self._redis.nodes_manager, 'nodes_cache') else 'N/A'}"
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Failed to refresh cluster slot map: {e}"
                                )
                                # Don't fail the connection - the client may still work
                                # with automatic slot refresh on MOVED/ASK errors

                        await asyncio.wait_for(self._load_scripts(), timeout=10.0)

                        # Only register pool AFTER successful ping and script load
                        # This prevents leaking broken pools (redis_002 fix)
                        # Note: created_new_pool/pool are only set for non-cluster mode
                        if (
                            not self.cluster_mode
                            and created_new_pool
                            and pool is not None
                        ):
                            with self._pool_lock:
                                self._connection_pools[loop_id] = pool
                                logger.info(
                                    f"Created Redis connection pool for loop {loop_id}"
                                )
                    except asyncio.TimeoutError:
                        # Clean up on timeout - don't leave broken pool/connection
                        logger.warning(
                            f"Redis connection/script load timed out for loop {loop_id}"
                        )
                        self._redis = None
                        self._connected = False
                        # Only try to disconnect pool in non-cluster mode
                        if (
                            not self.cluster_mode
                            and created_new_pool
                            and pool is not None
                        ):
                            try:
                                await pool.disconnect()
                            except Exception as e:
                                logger.debug(f"Error disconnecting failed pool: {e}")
                        raise

                    # === FALLBACK RECOVERY ===
                    # If we were in fallback mode and Redis is now available,
                    # clear the fallback state. Fresh API headers will sync state
                    # on the next request.
                    #
                    # redis_005 fix: Log comprehensive fallback state metrics before
                    # discarding to provide operational visibility into fallback behavior
                    if self._fallback_backend is not None:
                        async with self._fallback_init_lock:
                            if self._fallback_backend is not None:
                                duration = time.time() - (
                                    self._fallback_start_time or 0
                                )

                                # Collect fallback metrics before discarding (redis_005 fix)
                                error_count = 0
                                final_delay_ms = 0.0
                                if self._fallback_rate_limiter:
                                    error_count = (
                                        self._fallback_rate_limiter.error_count
                                    )
                                    final_delay_ms = (
                                        self._fallback_rate_limiter.current_delay_ms
                                    )

                                # Get fallback backend stats for visibility
                                fallback_stats = (
                                    await self._fallback_backend.get_all_stats()
                                )

                                # Log at WARNING level for operational visibility (redis_005 fix)
                                # This ensures operators can track fallback impact even if
                                # INFO logging is disabled
                                logger.warning(
                                    f"Redis recovered after {duration:.1f}s fallback period. "
                                    f"Fallback metrics: 429_errors={error_count}, "
                                    f"final_delay_ms={final_delay_ms:.0f}, "
                                    f"fallback_backend_stats={fallback_stats}. "
                                    f"Note: Rate limit state may be briefly stale until "
                                    f"fresh API headers sync on next request."
                                )

                                # Record final fallback duration in Prometheus metrics
                                FALLBACK_DURATION_METRIC.labels(
                                    reason="circuit_breaker"
                                ).observe(duration)

                                # Discard fallback state - do NOT try to reconcile
                                # Fresh API headers will sync state on next request
                                self._fallback_backend = None
                                self._fallback_start_time = None
                                self._fallback_rate_limiter = None
                                FALLBACK_ACTIVE_METRIC.labels(backend="redis").set(0)

                return self._redis

        except RuntimeError as e:
            if "no running event loop" in str(e).lower():
                raise RuntimeError("No running event loop for Redis connection.") from e
            raise

    async def _cleanup_connection(
        self, loop_id: int, connection: Any, timeout: float = 2.5
    ) -> None:
        """Clean up Redis connection with timeout protection."""
        try:
            if hasattr(connection, "aclose"):
                await asyncio.wait_for(connection.aclose(), timeout=timeout)
            elif hasattr(connection, "close"):
                await asyncio.wait_for(connection.close(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Connection cleanup timed out for loop {loop_id}")
        except Exception as e:
            logger.error(f"Error during connection cleanup: {e}")

    async def _load_scripts(self) -> None:
        """Load Lua scripts into Redis."""
        if not self._redis:
            raise RuntimeError("Redis client not initialized")

        self.__class__._load_lua_scripts()

        for script_name, script_source in self._lua_scripts.items():
            self._script_shas[script_name] = await self._redis.script_load(
                script_source
            )

    async def _evalsha_with_reload(
        self,
        redis_client: Any,
        script_name: str,
        num_keys: int,
        *args: Any,
    ) -> Any:
        """
        Execute EVALSHA with automatic script reload on NoScriptError.

        When Redis nodes restart (e.g., during cluster failover), all Lua scripts
        are lost. This method detects the NoScriptError and transparently reloads
        the scripts, then retries the operation once.

        Args:
            redis_client: The Redis client to use
            script_name: Name of the Lua script (key in _lua_scripts)
            num_keys: Number of KEYS arguments
            *args: Keys and arguments for the script

        Returns:
            Result from evalsha

        Raises:
            NoScriptError: If reload and retry also fails
            Other Redis exceptions: Passed through unchanged
        """
        script_sha = self._script_shas.get(script_name)
        if not script_sha:
            await self._load_scripts()
            script_sha = self._script_shas[script_name]

        try:
            return await redis_client.evalsha(script_sha, num_keys, *args)
        except NoScriptError:
            # Redis lost our scripts (node restart, failover, etc.)
            # Clear cached SHAs and reload all scripts
            logger.warning(
                f"Script '{script_name}' not found in Redis (SHA: {script_sha}). "
                f"Reloading all Lua scripts..."
            )
            self._script_shas.clear()
            await self._load_scripts()

            # Retry with new SHA (only once to prevent infinite loop)
            new_sha = self._script_shas[script_name]
            logger.info(f"Scripts reloaded. Retrying with new SHA: {new_sha}")
            return await redis_client.evalsha(new_sha, num_keys, *args)

    # === SDK Initialization Flow ===

    async def fetch_and_cache_model_limits(
        self,
        rate_limits_response: dict[str, Any],
        cache_to_redis: bool = True,
    ) -> dict[str, ModelLimits]:
        """
        Parse and cache per-model rate limits from API response.

        This should be called during SDK initialization with the response from
        a rate limits API endpoint.

        Args:
            rate_limits_response: The API response containing rateLimits array
            cache_to_redis: Whether to also cache in Redis (default True)

        Returns:
            Dictionary mapping model_id to ModelLimits
        """
        limits_by_model: dict[str, ModelLimits] = {}

        data = rate_limits_response.get("data", rate_limits_response)
        rate_limits = data.get("rateLimits", [])

        for entry in rate_limits:
            model_id = entry.get("apiModelId")
            if not model_id:
                continue

            rpm = None
            tpm = None

            for limit in entry.get("rateLimits", []):
                if limit.get("type") == "RPM":
                    rpm = limit.get("amount")
                elif limit.get("type") == "TPM":
                    tpm = limit.get("amount")

            if rpm is not None or tpm is not None:
                limits_by_model[model_id] = ModelLimits(
                    rpm=rpm or self.DEFAULT_RPM_LIMIT,
                    tpm=tpm or self.DEFAULT_TPM_LIMIT,
                )

        # Update in-memory cache
        async with self._model_limits_lock:
            self._model_limits.update(limits_by_model)

        # Optionally cache to Redis
        if cache_to_redis and limits_by_model:
            try:
                redis_client = await self._ensure_connected()
                cache_data = {
                    model_id: {"rpm": lim.rpm, "tpm": lim.tpm}
                    for model_id, lim in limits_by_model.items()
                }
                await redis_client.setex(
                    self.model_limits_key,
                    3600,  # 1 hour TTL
                    json.dumps(cache_data),
                )
            except Exception as e:
                logger.warning(f"Failed to cache model limits to Redis: {e}")

        return limits_by_model

    async def get_model_limits(self, model: str) -> tuple[int, int]:
        """
        Get rate limits for a model with fallback chain.

        Fallback order:
        1. In-memory cache
        2. Redis cache
        3. Conservative defaults (20 RPM, 500K TPM)

        Args:
            model: Model identifier

        Returns:
            Tuple of (rpm_limit, tpm_limit)
        """
        # Try in-memory cache first
        async with self._model_limits_lock:
            if model in self._model_limits:
                limits = self._model_limits[model]
                return (limits.rpm, limits.tpm)

        # Try Redis cache
        try:
            redis_client = await self._ensure_connected()
            cache_json = await redis_client.get(self.model_limits_key)
            if cache_json:
                cache_data = json.loads(cache_json)
                if model in cache_data:
                    limits = cache_data[model]
                    # Update in-memory cache
                    async with self._model_limits_lock:
                        self._model_limits[model] = ModelLimits(
                            rpm=limits.get("rpm", self.DEFAULT_RPM_LIMIT),
                            tpm=limits.get("tpm", self.DEFAULT_TPM_LIMIT),
                        )
                    return (
                        limits.get("rpm", self.DEFAULT_RPM_LIMIT),
                        limits.get("tpm", self.DEFAULT_TPM_LIMIT),
                    )
        except Exception as e:
            logger.warning(f"Failed to get model limits from Redis: {e}")

        # Fall back to conservative defaults
        return (self.DEFAULT_RPM_LIMIT, self.DEFAULT_TPM_LIMIT)

    # === Core Distributed Operations ===

    async def check_and_reserve_capacity(
        self,
        key: str,
        requests: int,
        tokens: int,
        bucket_limits: dict[str, int] | None = None,
        safety_margin: float = 1.0,
        request_id: str | None = None,
    ) -> tuple[bool, str | None]:
        """
        Atomically check and reserve capacity using distributed Lua script.

        When the circuit breaker is open (Redis unavailable), this method
        automatically falls back to MemoryBackend with conservative limits
        (1/20th of actual) and per-request rate limiting to prevent abuse
        protection cascade.

        Args:
            key: The model identifier
            requests: Number of requests to reserve (typically 1)
            tokens: Number of tokens to reserve (typically max_tokens estimate)
            bucket_limits: Optional dict with rpm_limit/tpm_limit
            safety_margin: Safety margin multiplier (not used in distributed mode)
            request_id: Optional pre-generated request ID

        Returns:
            Tuple of (can_proceed, request_id or error_code)
        """
        validate_safety_margin(safety_margin)

        model = key
        req_id = request_id or str(uuid.uuid4())

        # Get per-model limits
        rpm_limit, tpm_limit = await self.get_model_limits(model)
        if bucket_limits:
            rpm_limit = bucket_limits.get("rpm_limit", rpm_limit)
            tpm_limit = bucket_limits.get("tpm_limit", tpm_limit)

        # === FALLBACK MODE: Circuit Breaker Open ===
        # When Redis is unavailable, fall back to MemoryBackend with conservative limits
        if await self.is_circuit_broken():
            # Phase 1: Initialize fallback backend if needed (under init lock, fast)
            if self._fallback_backend is None:
                async with self._fallback_init_lock:
                    # Double-check after acquiring lock
                    if self._fallback_backend is None:
                        self._fallback_backend = MemoryBackend(namespace=self.namespace)
                        self._fallback_start_time = time.time()
                        self._fallback_rate_limiter = FallbackRateLimiter(
                            self._fallback_config
                        )
                        logger.warning(
                            "Circuit breaker open - falling back to local memory backend"
                        )
                        FALLBACK_ACTIVE_METRIC.labels(backend="redis").set(1)

            # Phase 2: Apply per-request rate limiting (OUTSIDE any locks)
            # This is the key fix: rate limiting happens outside the lock,
            # allowing concurrent delay calculation for other requests
            delay_applied = 0.0
            if self._fallback_rate_limiter:
                delay_applied = await self._fallback_rate_limiter.acquire()

            if delay_applied > 0.1:  # Log significant delays
                logger.debug(
                    f"Fallback mode: applied {delay_applied:.3f}s delay before request"
                )

            # Phase 3: Apply conservative limits to prevent abuse protection cascade
            conservative_bucket_limits = None
            if bucket_limits:
                conservative_bucket_limits = {
                    "rpm_limit": max(
                        1,
                        bucket_limits.get("rpm_limit", 100)
                        // self._fallback_config["limit_divisor"],
                    ),
                    "tpm_limit": max(
                        1,
                        bucket_limits.get("tpm_limit", 10000)
                        // self._fallback_config["limit_divisor"],
                    ),
                }
            else:
                # Use conservative defaults if no bucket_limits provided
                conservative_bucket_limits = {
                    "rpm_limit": max(
                        1, rpm_limit // self._fallback_config["limit_divisor"]
                    ),
                    "tpm_limit": max(
                        1, tpm_limit // self._fallback_config["limit_divisor"]
                    ),
                }

            # Track fallback duration for metrics
            duration = time.time() - (self._fallback_start_time or 0.0)
            FALLBACK_DURATION_METRIC.labels(reason="circuit_breaker").observe(duration)

            # Phase 4: Delegate to fallback backend with conservative limits
            # MemoryBackend.check_and_reserve_capacity() DOES enforce limits via token bucket
            return await self._fallback_backend.check_and_reserve_capacity(
                key=key,
                requests=requests,
                tokens=tokens,
                bucket_limits=conservative_bucket_limits,
                safety_margin=safety_margin,
                request_id=req_id,
            )

        try:
            redis_client = await self._ensure_connected()

            # Build keys
            state_key = self._get_state_key(model)
            pend_req_key = self._get_pending_req_key(model)
            pend_tok_key = self._get_pending_tok_key(model)
            req_map_key = self._get_req_map_key(model, req_id)

            result = await self._evalsha_with_reload(
                redis_client,
                "distributed_check_and_reserve",
                4,  # Number of keys
                state_key,
                pend_req_key,
                pend_tok_key,
                req_map_key,
                requests,  # ARGV[1]
                tokens,  # ARGV[2]
                rpm_limit,  # ARGV[3]
                tpm_limit,  # ARGV[4]
                self.DEFAULT_WINDOW_SECONDS,  # ARGV[5]
                self.DEFAULT_WINDOW_SECONDS,  # ARGV[6]
                req_id,  # ARGV[7]
                self.req_map_ttl,  # ARGV[8]
            )

            status_code = int(result[0])

            if status_code == 1:
                # Success - track in-flight request
                gen_req = int(result[4])
                gen_tok = int(result[5])

                async with self._in_flight_lock:
                    self._in_flight[req_id] = InFlightRequest(
                        req_id=req_id,
                        cost_req=requests,
                        cost_tok=tokens,
                        gen_req=gen_req,
                        gen_tok=gen_tok,
                        start_time=time.time(),
                        model=model,
                        account_id=self.account_id,
                    )

                return (True, req_id)

            elif status_code == 0:
                # Rate limited - return wait time as error info
                wait_time = float(result[1])
                logger.debug(f"Rate limited for {model}, wait_time={wait_time:.2f}s")
                return (False, f"RATE_LIMITED:{wait_time}")

            elif status_code == -1:
                return (False, "INVALID_INPUT")

            elif status_code == -2:
                return (False, "COLLISION")

            elif status_code == -3:
                lim_req = int(result[2])
                lim_tok = int(result[3])
                return (False, f"COST_EXCEEDS_LIMIT:req={lim_req},tok={lim_tok}")

            return (False, f"UNKNOWN_ERROR:{status_code}")

        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Redis connection error during check_and_reserve: {e}")
            return (False, None)
        except ResponseError as e:
            logger.error(f"Redis response error during check_and_reserve: {e}")
            return (False, None)
        except RedisError as e:
            logger.error(f"Redis error during check_and_reserve: {e}")
            return (False, None)

    def _select_script(self, status_code: int | None, has_headers: bool) -> str:
        """
        Select the appropriate Lua script based on status code.

        Args:
            status_code: HTTP status code, or None if no HTTP response was received
                (e.g., timeout, connection error, cancellation)
            has_headers: Whether valid rate limit headers are available

        Returns:
            Name of the Lua script to use
        """
        if status_code is None:
            # No HTTP response (timeout, connection error, cancellation)
            return "distributed_release_capacity"

        if status_code == 429:
            # Rate limited: headers are authoritative
            return "distributed_update_rate_limits_429"

        elif 200 <= status_code < 300:
            # Success: sync state and decrement pending
            return "distributed_update_rate_limits"

        elif 400 <= status_code < 500:
            # Client error (non-429): just release pending
            return "distributed_release_capacity"

        elif 500 <= status_code < 600:
            # Server error: sync if headers available, otherwise release
            if has_headers:
                return "distributed_update_rate_limits"
            else:
                return "distributed_release_capacity"

        else:
            # Unknown status: safe default is release only
            return "distributed_release_capacity"

    async def _execute_update_with_tracking(
        self,
        redis_client: Any,
        script_sha: str,
        keys: list[str],
        args: list[Any],
        request_id: str,
        script_name: str = "",
    ) -> int:
        """
        Execute Lua script and clear in-flight tracking atomically.

        Uses asyncio.shield() to ensure both operations complete even
        if the task is cancelled. Note that shield only protects against
        CancelledError - Redis failures will still occur and are handled
        by returning 0.

        Handles NoScriptError by reloading scripts and retrying once.

        Args:
            redis_client: The Redis client to use
            script_sha: SHA of the Lua script to execute
            keys: List of Redis keys for the script
            args: List of arguments for the script
            request_id: The request ID being processed
            script_name: Name of the script (for NoScriptError reload)

        Returns:
            1 if update succeeded, 0 if failed
        """

        async def _shielded_operation() -> int:
            nonlocal script_sha
            try:
                result = await redis_client.evalsha(script_sha, len(keys), *keys, *args)
            except NoScriptError:
                # Redis lost our scripts (node restart, failover, etc.)
                if not script_name:
                    raise  # Can't reload without script name
                logger.warning(
                    f"Script '{script_name}' not found in Redis (SHA: {script_sha}). "
                    f"Reloading all Lua scripts..."
                )
                self._script_shas.clear()
                await self._load_scripts()
                script_sha = self._script_shas[script_name]
                logger.info(f"Scripts reloaded. Retrying with new SHA: {script_sha}")
                result = await redis_client.evalsha(script_sha, len(keys), *keys, *args)

            # Clear tracking AFTER successful Redis call
            async with self._in_flight_lock:
                self._in_flight.pop(request_id, None)

            return int(result)

        try:
            return await asyncio.shield(_shielded_operation())
        except asyncio.CancelledError:
            # Shield completed, now re-raise
            raise
        except (ConnectionError, TimeoutError, ResponseError, RedisError) as e:
            # Don't clear tracking - orphan recovery can still find it
            logger.error(f"Redis error during update_rate_limits: {e}")
            return 0

    async def update_rate_limits(
        self,
        model: str,
        headers: dict[str, str],
        bucket_id: str | None = None,
        request_id: str | None = None,
        status_code: int | None = None,
    ) -> int:
        """
        Update rate limits from API response headers.

        Selects the appropriate Lua script based on response status:
        - 2xx: update_rate_limits (consumed capacity, sync state)
        - 429: update_rate_limits_429 (release pending, sync state)
        - 4xx (non-429): release_capacity (just release pending)
        - 5xx with headers: update_rate_limits
        - 5xx without headers: release_capacity
        - None (timeout/cancellation): release_capacity

        Args:
            model: Model identifier
            headers: Response headers containing rate limit info
            bucket_id: Optional bucket ID (unused in distributed mode)
            request_id: The request ID to update/release
            status_code: HTTP status code of the response, or None if no HTTP
                response was received (timeout, connection error, cancellation)

        Returns:
            1 if state was successfully updated
            0 if update failed (validation error, missing headers, mapping not found,
              or missing request_id)

            For 2xx status codes: return 0 guarantees zero side effects were made
            to Redis state. The caller can safely call release_reservation() as
            fallback without risking double-decrement.

            For 429 status codes: return 0 means mapping was not found (the ONLY
            failure mode). The caller MUST call release_reservation() if a local
            reservation exists.
        """
        if not request_id:
            if self.log_validation_failures:
                logger.warning("update_rate_limits called without request_id")
            return 0

        # Parse headers to determine if we have usable rate limit data
        parsed = self._parse_rate_limit_headers(headers)
        has_headers = parsed.get("rpm_remaining") is not None

        # Select script based on status code and header availability
        script_name = self._select_script(status_code, has_headers)

        try:
            redis_client = await self._ensure_connected()

            # Build keys
            state_key = self._get_state_key(model)
            pend_req_key = self._get_pending_req_key(model)
            pend_tok_key = self._get_pending_tok_key(model)
            req_map_key = self._get_req_map_key(model, request_id)

            keys = [state_key, pend_req_key, pend_tok_key, req_map_key]

            script_sha = self._script_shas.get(script_name)
            if not script_sha:
                await self._load_scripts()
                script_sha = self._script_shas[script_name]

            if script_name == "distributed_release_capacity":
                # Release capacity script has no additional args
                args: list[Any] = []
            else:
                # Update scripts need header values
                # Pass max_token_delta as ARGV[8] for validation in Lua script
                args = [
                    parsed.get("rpm_remaining", 0),
                    parsed.get("tpm_remaining", 0),
                    parsed.get("rpm_limit", self.DEFAULT_RPM_LIMIT),
                    parsed.get("tpm_limit", self.DEFAULT_TPM_LIMIT),
                    parsed.get("rpm_reset", 0),  # Absolute timestamp
                    parsed.get("tpm_reset", 0),  # Relative seconds (delta)
                    self.stale_buffer,
                    self.max_token_delta,  # ARGV[8]: max valid token reset delta
                ]

            # Execute with shielded in-flight tracking cleanup
            result = await self._execute_update_with_tracking(
                redis_client, script_sha, keys, args, request_id, script_name
            )

            # Log validation failures if enabled
            if result == 0 and self.log_validation_failures:
                logger.warning(
                    f"update_rate_limits returned 0 for request {request_id} "
                    f"(status_code={status_code}, script={script_name})"
                )

            return result

        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Redis connection error during update_rate_limits: {e}")
            return 0
        except ResponseError as e:
            logger.error(f"Redis response error during update_rate_limits: {e}")
            return 0
        except RedisError as e:
            logger.error(f"Redis error during update_rate_limits: {e}")
            return 0

    async def release_reservation(self, key: str, reservation_id: str) -> bool:
        """
        Release a capacity reservation atomically.

        Uses asyncio.shield() to ensure both Redis call and tracking cleanup
        complete even if the calling task is cancelled.

        This operation is idempotent - calling it twice for the same reservation_id
        will succeed but log a warning on the second call.
        """
        model = key

        # Check if we're tracking this reservation locally
        async with self._in_flight_lock:
            was_tracked = reservation_id in self._in_flight

        if not was_tracked:
            # This is a double-release or release of an unknown reservation
            # Still call Redis (idempotent), but log a warning for debugging
            logger.warning(
                f"release_reservation called for {reservation_id} which is not in _in_flight. "
                f"This may be a double-release or orphaned request."
            )

        async def _shielded_release() -> bool:
            """Inner shielded operation to ensure completion."""
            try:
                redis_client = await self._ensure_connected()

                script_name = "distributed_release_capacity"
                script_sha = self._script_shas.get(script_name)
                if not script_sha:
                    await self._load_scripts()
                    script_sha = self._script_shas[script_name]

                state_key = self._get_state_key(model)
                pend_req_key = self._get_pending_req_key(model)
                pend_tok_key = self._get_pending_tok_key(model)
                req_map_key = self._get_req_map_key(model, reservation_id)

                try:
                    result = await redis_client.evalsha(
                        script_sha,
                        4,
                        state_key,
                        pend_req_key,
                        pend_tok_key,
                        req_map_key,
                    )
                except NoScriptError:
                    # Redis lost our scripts (node restart, failover, etc.)
                    logger.warning(
                        f"Script '{script_name}' not found in Redis (SHA: {script_sha}). "
                        f"Reloading all Lua scripts..."
                    )
                    self._script_shas.clear()
                    await self._load_scripts()
                    script_sha = self._script_shas[script_name]
                    logger.info(
                        f"Scripts reloaded. Retrying with new SHA: {script_sha}"
                    )
                    result = await redis_client.evalsha(
                        script_sha,
                        4,
                        state_key,
                        pend_req_key,
                        pend_tok_key,
                        req_map_key,
                    )

                # Clear tracking AFTER successful Redis call
                async with self._in_flight_lock:
                    removed = self._in_flight.pop(reservation_id, None)
                    if removed is None and was_tracked:
                        # Edge case: tracking was cleared by another task between check and pop
                        logger.debug(
                            f"Reservation {reservation_id} was cleared by another task"
                        )

                return int(result) == 1

            except (ConnectionError, TimeoutError, ResponseError, RedisError) as e:
                # Don't clear tracking - orphan recovery can still find it
                logger.error(f"Redis error during release_reservation: {e}")
                return False

        try:
            return await asyncio.shield(_shielded_release())
        except asyncio.CancelledError:
            # Shield completed but caller got cancelled
            raise

    async def release_streaming_reservation(
        self,
        key: str,
        reservation_id: str,
        reserved_tokens: int,
        actual_tokens: int,
    ) -> bool:
        """
        Release a streaming capacity reservation with refund-based accounting.

        This method is called by the RateLimitedAsyncIterator when a streaming
        response completes. It uses refund-based accounting where:
        - rem_tok += (reserved_tokens - actual_tokens)
        - Result is clamped to [0, limit] to prevent overflow

        The entire operation is protected with asyncio.shield() to ensure
        completion even if the calling task is cancelled.

        Args:
            key: The bucket/model identifier
            reservation_id: The request ID to release
            reserved_tokens: Tokens that were reserved at request start
            actual_tokens: Actual tokens consumed by the stream

        Returns:
            True if released successfully, False on error
        """
        model = key

        async def _shielded_release() -> bool:
            """Inner shielded operation to ensure completion."""
            try:
                # Remove from in-flight tracking
                async with self._in_flight_lock:
                    self._in_flight.pop(reservation_id, None)

                redis_client = await self._ensure_connected()

                script_name = "distributed_release_streaming"
                script_sha = self._script_shas.get(script_name)
                if not script_sha:
                    await self._load_scripts()
                    script_sha = self._script_shas[script_name]

                state_key = self._get_state_key(model)
                pend_req_key = self._get_pending_req_key(model)
                pend_tok_key = self._get_pending_tok_key(model)
                req_map_key = self._get_req_map_key(model, reservation_id)

                try:
                    result = await redis_client.evalsha(
                        script_sha,
                        4,  # Number of KEYS
                        state_key,
                        pend_req_key,
                        pend_tok_key,
                        req_map_key,
                        reserved_tokens,  # ARGV[1]
                        actual_tokens,  # ARGV[2]
                    )
                except NoScriptError:
                    # Redis lost our scripts (node restart, failover, etc.)
                    logger.warning(
                        f"Script '{script_name}' not found in Redis (SHA: {script_sha}). "
                        f"Reloading all Lua scripts..."
                    )
                    self._script_shas.clear()
                    await self._load_scripts()
                    script_sha = self._script_shas[script_name]
                    logger.info(
                        f"Scripts reloaded. Retrying with new SHA: {script_sha}"
                    )
                    result = await redis_client.evalsha(
                        script_sha,
                        4,  # Number of KEYS
                        state_key,
                        pend_req_key,
                        pend_tok_key,
                        req_map_key,
                        reserved_tokens,  # ARGV[1]
                        actual_tokens,  # ARGV[2]
                    )

                return int(result) == 1

            except (ConnectionError, TimeoutError, ResponseError, RedisError) as e:
                logger.error(f"Redis error during release_streaming_reservation: {e}")
                return False

        try:
            return await asyncio.shield(_shielded_release())
        except asyncio.CancelledError:
            # Shield completed but caller got cancelled
            # Re-raise to propagate cancellation
            raise

    # === Fallback Mode Response Handling ===

    async def record_fallback_429(self) -> None:
        """
        Record a 429 response received while in fallback mode.

        This increases the backoff delay using multiplicative increase (AIMD).
        Should be called by the response handler when a 429 is received
        and the backend is in fallback mode.
        """
        if self._fallback_rate_limiter is not None:
            new_delay = await self._fallback_rate_limiter.record_429()
            logger.warning(
                f"Fallback mode: 429 received, increasing delay to {new_delay:.0f}ms "
                f"(total 429s: {self._fallback_rate_limiter.error_count})"
            )
            FALLBACK_BACKOFF_METRIC.labels(reason="429_received").observe(
                new_delay / 1000.0
            )

    async def record_fallback_success(self) -> None:
        """
        Record a successful response (2xx) received while in fallback mode.

        This decreases the backoff delay using additive decrease (AIMD).
        Should be called by the response handler when a successful response
        is received and the backend is in fallback mode.
        """
        if self._fallback_rate_limiter is not None:
            await self._fallback_rate_limiter.record_success()

    def is_in_fallback_mode(self) -> bool:
        """
        Check if the backend is currently operating in fallback mode.

        Returns:
            True if fallback mode is active, False otherwise.
        """
        return self._fallback_backend is not None

    # === Orphan Recovery ===

    async def start_orphan_recovery(self) -> None:
        """Start the background orphan recovery task."""
        if self._orphan_recovery_task is None or self._orphan_recovery_task.done():
            self._orphan_recovery_task = asyncio.create_task(
                self._orphan_recovery_loop(), name="orphan_recovery"
            )

    async def stop_orphan_recovery(self) -> None:
        """Stop the background orphan recovery task."""
        if self._orphan_recovery_task and not self._orphan_recovery_task.done():
            self._orphan_recovery_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._orphan_recovery_task

    async def _orphan_recovery_loop(self) -> None:
        """Background task to recover orphaned pending reservations."""
        while True:
            try:
                await asyncio.sleep(self.orphan_recovery_interval)
                await self._recover_orphans()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Orphan recovery error: {e}")

    async def _recover_orphans(self) -> None:
        """Scan for and recover orphaned requests."""
        current_time = time.time()
        orphans: list[InFlightRequest] = []

        # Collect orphans but do NOT remove from tracking yet
        # We only remove after successful Redis recovery
        async with self._in_flight_lock:
            for _req_id, req in list(self._in_flight.items()):
                if current_time - req.start_time > self.max_request_timeout:
                    orphans.append(req)

        for orphan in orphans:
            try:
                await self._recover_orphan(orphan)
                # Only remove from tracking AFTER successful Redis recovery
                # This matches the pattern in _execute_update_with_tracking and release_reservation
                async with self._in_flight_lock:
                    self._in_flight.pop(orphan.req_id, None)
                logger.info(f"Recovered orphan request {orphan.req_id}")
            except Exception as e:
                # Keep orphan in _in_flight so it can be retried on next recovery cycle
                logger.error(f"Failed to recover orphan {orphan.req_id}: {e}")

    async def _recover_orphan(self, orphan: InFlightRequest) -> None:
        """Recover a single orphaned request."""
        try:
            redis_client = await self._ensure_connected()

            pend_req_key = self._get_pending_req_key(orphan.model)
            pend_tok_key = self._get_pending_tok_key(orphan.model)
            state_key = self._get_state_key(orphan.model)

            await self._evalsha_with_reload(
                redis_client,
                "distributed_recover_orphan",
                3,  # Number of keys
                pend_req_key,
                pend_tok_key,
                state_key,
                orphan.cost_req,  # ARGV[1]
                orphan.cost_tok,  # ARGV[2]
                orphan.gen_req,  # ARGV[3]
                orphan.gen_tok,  # ARGV[4]
            )
        except (ConnectionError, TimeoutError, ResponseError, RedisError) as e:
            logger.error(f"Redis error during orphan recovery: {e}")

    # === BaseBackend Interface Implementation ===

    async def get_state(self, key: str) -> dict[str, Any] | None:
        """Get state for a key (model)."""
        try:
            redis_client = await self._ensure_connected()
            state_key = self._get_state_key(key)

            state = await redis_client.hgetall(state_key)
            if state:
                return dict(state.items())
            return None
        except (ConnectionError, TimeoutError, ResponseError, RedisError) as e:
            logger.error(f"Redis error getting state for {key}: {e}")
            return None

    async def set_state(self, key: str, state: dict[str, Any]) -> None:
        """Set state for a key (model)."""
        try:
            redis_client = await self._ensure_connected()
            state_key = self._get_state_key(key)

            if state:
                await redis_client.hset(state_key, mapping=state)
                await redis_client.expire(state_key, self.key_ttl)
        except (ConnectionError, TimeoutError, ResponseError, RedisError) as e:
            logger.error(f"Redis error setting state for {key}: {e}")
            raise

    async def get_all_states(self) -> dict[str, dict[str, Any]]:
        """Get all stored states.

        Uses SCAN instead of KEYS to avoid blocking Redis during large keyspace scans.
        KEYS is O(N) and blocks the event loop, which can cause latency spikes.
        SCAN is O(1) per call and iterates incrementally.
        """
        try:
            redis_client = await self._ensure_connected()
            pattern = "rl:*:state"

            result = {}

            if self.cluster_mode:
                # In cluster mode, scan_iter() handles scanning across all nodes
                async for key in redis_client.scan_iter(match=pattern, count=100):
                    state = await redis_client.hgetall(key)
                    if state:
                        result[key] = dict(state.items())
            else:
                # Standard Redis scan
                cursor = 0
                while True:
                    cursor, keys = await redis_client.scan(
                        cursor, match=pattern, count=100
                    )
                    for key in keys:
                        state = await redis_client.hgetall(key)
                        if state:
                            result[key] = dict(state.items())
                    if cursor == 0:
                        break

            return result
        except (ConnectionError, TimeoutError, ResponseError, RedisError) as e:
            logger.error(f"Redis error getting all states: {e}")
            return {}

    async def clear(self) -> None:
        """Clear all stored states."""
        try:
            redis_client = await self._ensure_connected()
            pattern = f"rl:{{{self._account_b64}|*"

            if self.cluster_mode:
                # In cluster mode, scan_iter() is the best way to iterate over keys
                # It handles scanning across all nodes automatically
                keys_to_delete = []
                async for key in redis_client.scan_iter(match=pattern, count=100):
                    keys_to_delete.append(key)
                    if len(keys_to_delete) >= 100:
                        await redis_client.delete(*keys_to_delete)
                        keys_to_delete = []
                if keys_to_delete:
                    await redis_client.delete(*keys_to_delete)
            else:
                # Standard Redis scan
                cursor = 0
                while True:
                    cursor, keys = await redis_client.scan(
                        cursor, match=pattern, count=100
                    )
                    if keys:
                        await redis_client.delete(*keys)
                    if cursor == 0:
                        break
        except (ConnectionError, TimeoutError, ResponseError, RedisError) as e:
            logger.error(f"Redis error during clear: {e}")
            raise

    async def check_capacity(
        self, model: str, request_type: str = "default"
    ) -> tuple[bool, float]:
        """Check if there's capacity for a request."""
        rate_limits = await self.get_rate_limits(model)
        wait_time = self._calculate_wait_time(rate_limits)

        if await self.is_circuit_broken():
            return False, max(wait_time, 30.0)

        can_proceed = wait_time <= 0.0
        return can_proceed, wait_time

    async def record_request(
        self,
        model: str,
        request_type: str = "default",
        tokens_used: int | None = None,
    ) -> None:
        """Record a successful request."""
        # In distributed mode, this is handled by update_rate_limits
        pass

    async def record_failure(self, error_type: str, error_message: str = "") -> None:
        """Record a failure for circuit breaker."""
        async with self._failure_lock:
            self._failure_timestamps.append(time.time())
            # Keep only recent failures
            cutoff = time.time() - 30
            self._failure_timestamps = [
                ts for ts in self._failure_timestamps if ts > cutoff
            ]

    async def get_failure_count(self, window_seconds: int = 30) -> int:
        """Get failure count in time window."""
        async with self._failure_lock:
            cutoff = time.time() - window_seconds
            return len([ts for ts in self._failure_timestamps if ts > cutoff])

    async def is_circuit_broken(self, failure_threshold: int = 20) -> bool:
        """Check if circuit breaker is triggered."""
        failure_count = await self.get_failure_count(30)
        return failure_count >= failure_threshold

    async def get_rate_limits(self, model: str) -> dict[str, Any]:
        """Get current rate limit state for a model."""
        try:
            state = await self.get_state(model)
            if state:
                return {
                    "rpm_limit": int(state.get("lim_req", 0)),
                    "rpm_remaining": int(state.get("rem_req", 0)),
                    "rpm_reset": int(state.get("rst_req", 0)),
                    "tpm_limit": int(state.get("lim_tok", 0)),
                    "tpm_remaining": int(state.get("rem_tok", 0)),
                    "tpm_reset": int(state.get("rst_tok", 0)),
                }
            return {}
        except Exception as e:
            logger.error(f"Error getting rate limits for {model}: {e}")
            return {}

    async def reserve_capacity(
        self, model: str, request_id: str, tokens_estimated: int = 0
    ) -> bool:
        """Reserve capacity for a request."""
        can_proceed, _ = await self.check_and_reserve_capacity(
            key=model,
            requests=1,
            tokens=tokens_estimated,
            request_id=request_id,
        )
        return can_proceed

    async def release_reservation_by_id(self, request_id: str) -> None:
        """Release a capacity reservation by request ID."""
        # Look up the model from in-flight tracking
        async with self._in_flight_lock:
            req = self._in_flight.get(request_id)
            if req:
                model = req.model
                del self._in_flight[request_id]
            else:
                logger.warning(f"Request {request_id} not found in in-flight tracking")
                return

        await self.release_reservation(model, request_id)

    async def cache_bucket_info(
        self, bucket_data: dict[str, Any], ttl_seconds: int = 3600
    ) -> None:
        """Cache bucket information."""
        try:
            redis_client = await self._ensure_connected()
            cache_key = f"rl:{self._account_b64}:bucket_cache"
            await redis_client.setex(cache_key, ttl_seconds, json.dumps(bucket_data))
        except Exception as e:
            logger.error(f"Error caching bucket info: {e}")

    async def get_cached_bucket_info(self) -> dict[str, Any] | None:
        """Get cached bucket information."""
        try:
            redis_client = await self._ensure_connected()
            cache_key = f"rl:{self._account_b64}:bucket_cache"
            cache_json = await redis_client.get(cache_key)
            if cache_json:
                result: dict[str, Any] = json.loads(cache_json)
                return result
            return None
        except Exception as e:
            logger.error(f"Error getting cached bucket info: {e}")
            return None

    async def cache_model_info(
        self, model: str, model_data: dict[str, Any], ttl_seconds: int = 3600
    ) -> None:
        """Cache model information."""
        try:
            redis_client = await self._ensure_connected()
            cache_key = (
                f"rl:{self._account_b64}:model_info:{self._get_model_b64(model)}"
            )
            await redis_client.setex(cache_key, ttl_seconds, json.dumps(model_data))
        except Exception as e:
            logger.error(f"Error caching model info: {e}")

    async def get_cached_model_info(self, model: str) -> dict[str, Any] | None:
        """Get cached model information."""
        try:
            redis_client = await self._ensure_connected()
            cache_key = (
                f"rl:{self._account_b64}:model_info:{self._get_model_b64(model)}"
            )
            cache_json = await redis_client.get(cache_key)
            if cache_json:
                result: dict[str, Any] = json.loads(cache_json)
                return result
            return None
        except Exception as e:
            logger.error(f"Error getting cached model info: {e}")
            return None

    async def health_check(self) -> HealthCheckResult:
        """Perform health check on the backend."""
        try:
            redis_client = await self._ensure_connected()

            test_key = f"health_check_{int(time.time())}"
            await redis_client.set(test_key, "test", ex=60)
            result = await redis_client.get(test_key)
            await redis_client.delete(test_key)

            info = await redis_client.info()

            return HealthCheckResult(
                healthy=result == "test",
                backend_type="redis",
                namespace=self.namespace,
                metadata={
                    "redis_url": self.redis_url,
                    "connected": self._connected,
                    "account_id": self.account_id,
                    "redis_version": info.get("redis_version"),
                    "used_memory": info.get("used_memory_human"),
                    "connected_clients": info.get("connected_clients"),
                    "in_flight_requests": len(self._in_flight),
                },
            )
        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                backend_type="redis",
                namespace=self.namespace,
                error=str(e),
            )

    async def get_all_stats(self) -> dict[str, Any]:
        """Get all statistics from the backend."""
        stats: dict[str, Any] = {
            "backend_type": "redis",
            "account_id": self.account_id,
            "connected": self._connected,
            "in_flight_requests": len(self._in_flight),
            "cached_model_limits": len(self._model_limits),
            "failure_count": await self.get_failure_count(30),
            "circuit_broken": await self.is_circuit_broken(),
            "fallback_mode_active": self._fallback_backend is not None,
        }

        # Add fallback-specific stats if in fallback mode
        if self._fallback_backend is not None:
            stats["fallback_duration_seconds"] = time.time() - (
                self._fallback_start_time or 0
            )
            if self._fallback_rate_limiter is not None:
                stats["fallback_429_count"] = self._fallback_rate_limiter.error_count
                stats["fallback_current_delay_ms"] = (
                    self._fallback_rate_limiter.current_delay_ms
                )

        return stats

    async def cleanup(self) -> None:
        """Clean up backend resources."""
        await self.stop_orphan_recovery()

        if self._redis and self._owned_redis:
            try:
                await self._cleanup_connection(
                    self._event_loop_id or 0, self._redis, timeout=2.5
                )
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
            finally:
                self._redis = None
                self._event_loop_id = None
                self._connected = False

    async def clear_failures(self) -> None:
        """Clear all failure records."""
        async with self._failure_lock:
            self._failure_timestamps.clear()

    async def force_circuit_break(self, duration: float) -> None:
        """Force circuit break for duration."""
        async with self._failure_lock:
            current_time = time.time()
            for _ in range(25):  # More than threshold
                self._failure_timestamps.append(current_time)

        async def clear_after_delay() -> None:
            await asyncio.sleep(duration)
            await self.clear_failures()

        _ = asyncio.create_task(clear_after_delay())  # noqa: RUF006

    async def __aenter__(self) -> "RedisBackend":
        """Async context manager entry."""
        await self._ensure_connected()
        await self.start_orphan_recovery()
        return self

    async def __aexit__(self, exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.cleanup()
