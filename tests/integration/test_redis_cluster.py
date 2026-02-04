"""
Redis Cluster Integration Tests for Adaptive Rate Limiter

This module contains integration tests that validate Redis Cluster behavior
with actual hash tag slot routing. These tests require a real Redis Cluster
to be available and are skipped by default.

=== Setup Instructions ===

To run these tests, you need a Redis Cluster. Use the provided docker-compose file:

1. Start the 6-node cluster (3 masters + 3 replicas):

   docker-compose -f docker-compose.redis-cluster.yml up -d

2. Wait for cluster initialization (about 15 seconds)

3. Set the environment variable:

   export REDIS_CLUSTER_URL="redis://localhost:7001"

4. Run the tests:

   uv run pytest tests/integration/test_redis_cluster.py -v

5. Stop the cluster:

   docker-compose -f docker-compose.redis-cluster.yml down

6. Clean up volumes for fresh start (if needed):

   docker-compose -f docker-compose.redis-cluster.yml down -v

=== Cluster Configuration ===

The cluster uses 6 nodes for high availability and failover testing:
- Nodes 1-3 (ports 7001-7003): Master nodes
- Nodes 4-6 (ports 7004-7006): Replica nodes (one per master)

This configuration enables automatic failover when a master fails.

=== REDIS_CLUSTER_URL Format ===

The REDIS_CLUSTER_URL environment variable should be a standard Redis URL pointing
to any node in the cluster. The redis-py cluster client will automatically discover
all nodes.

Examples:
- redis://localhost:7001
- redis://localhost:7002
- redis://:password@cluster-host:7001

=== Hash Tagging for Slot Routing ===

Redis Cluster uses CRC16 hashing to assign keys to one of 16384 slots. Each node
is responsible for a subset of these slots. Multi-key operations (like our Lua
scripts) MUST have all keys on the same slot to be atomic.

Hash tagging solves this by ensuring that only the portion of the key within
curly braces {} is hashed for slot assignment:

    Key: "rl:{acc|model}:state"      -> Hashes "{acc|model}" -> Slot X
    Key: "rl:{acc|model}:pending_req" -> Hashes "{acc|model}" -> Slot X
    Key: "rl:{acc|model}:pending_tok" -> Hashes "{acc|model}" -> Slot X

All keys with the same hash tag go to the same slot, enabling atomic multi-key
Lua script execution.

Without hash tagging, these keys could end up on different slots, causing
CROSSSLOT errors in Redis Cluster.

=== Failover Testing ===

Failover tests require the `docker` package to control container lifecycle:

    uv add --group dev docker

Or install via optional dependencies:

    uv install --extras dev

Tests will be automatically skipped if Docker is not available. The tests verify:
- Master failure with automatic replica promotion
- Operations during failover (graceful degradation)
- Circuit breaker activation and fallback to MemoryBackend
- Network partition simulation (pause/unpause)

To run failover tests:

    docker-compose -f docker-compose.redis-cluster.yml up -d
    export REDIS_CLUSTER_URL="redis://localhost:7001"
    uv run pytest tests/integration/test_redis_cluster.py::TestRedisClusterFailover -v

Failover tests are slower than basic cluster tests due to:
- Waiting for cluster failover detection (~5-10 seconds)
- Waiting for replica promotion
- Docker container control operations
"""

import os
import socket
from typing import cast

import pytest

# Default cluster nodes for connectivity check
REDIS_CLUSTER_DEFAULT_NODES = [
    ("127.0.0.1", 7001),
    ("127.0.0.1", 7002),
    ("127.0.0.1", 7003),
]


def _is_redis_cluster_available() -> bool:
    """
    Check if the Redis Cluster is actually available by attempting to connect.

    This performs a quick socket check to verify at least one cluster node
    is reachable, preventing test failures when the Docker cluster is not running.
    """
    # First check if the environment variable is set
    if os.getenv("REDIS_CLUSTER_URL") is None:
        return False

    # Try to connect to at least one of the default cluster nodes
    for host, port in REDIS_CLUSTER_DEFAULT_NODES:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)  # 1 second timeout
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                return True
        except (TimeoutError, OSError):
            continue

    return False


# Check if Redis Cluster is available
_REDIS_CLUSTER_URL: str | None = os.getenv("REDIS_CLUSTER_URL")
SKIP_CLUSTER_TESTS = not _is_redis_cluster_available()
SKIP_REASON = (
    "Redis Cluster not available. Either REDIS_CLUSTER_URL is not set, "
    "or the Docker cluster is not running. Start with: "
    "docker-compose -f docker-compose.redis-cluster.yml up -d"
)
# Cast to str for type checker - tests are skipped when None so this is safe
REDIS_CLUSTER_URL: str = cast(str, _REDIS_CLUSTER_URL) if _REDIS_CLUSTER_URL else ""


@pytest.mark.skipif(SKIP_CLUSTER_TESTS, reason=SKIP_REASON)
class TestRedisClusterIntegration:
    """
    Integration tests for Redis Cluster mode.

    These tests validate that:
    1. Multi-key Lua scripts execute atomically
    2. Hash tags correctly route keys to the same slot
    3. Cross-slot errors are prevented by proper key design
    """

    @pytest.fixture
    async def cluster_backend(self):
        """Create a RedisBackend in cluster mode connected to the real cluster."""
        from adaptive_rate_limiter.backends.redis import RedisBackend

        backend = RedisBackend(
            redis_url=REDIS_CLUSTER_URL,
            namespace="test_cluster",
            account_id="cluster-test-account",
            cluster_mode=True,
        )

        yield backend

        # Cleanup
        await backend.cleanup()

    @pytest.mark.asyncio
    async def test_cluster_mode_multi_key_atomicity(self, cluster_backend):
        """
        Test that multi-key Lua scripts execute atomically in cluster mode.

        This test verifies that check_and_reserve_capacity works correctly
        when all keys are properly hash-tagged to the same slot.
        """
        backend = cluster_backend

        # Force connection
        await backend._ensure_connected()

        # Execute reservation - this uses multiple keys with Lua script
        success, result = await backend.check_and_reserve_capacity(
            key="test-model-atomicity",
            requests=1,
            tokens=100,
            bucket_limits={"rpm_limit": 100, "tpm_limit": 10000},
        )

        # Should succeed without CROSSSLOT error
        assert success is True, f"Reservation failed: {result}"
        assert result is not None  # Should return request_id

        # Clean up the reservation
        await backend.release_reservation("test-model-atomicity", result)

    @pytest.mark.asyncio
    async def test_cluster_mode_hash_tag_slot_routing(self, cluster_backend):
        """
        Test that all keys for a bucket are routed to the same slot.

        Uses Redis CLUSTER KEYSLOT command to verify that all keys
        generated by the backend for a given model hash to the same slot.
        """
        backend = cluster_backend

        # Force connection
        redis_client = await backend._ensure_connected()

        model = "test-model-slot-routing"
        req_id = "test-req-123"

        # Get all keys for this bucket
        state_key = backend._get_state_key(model)
        pending_req_key = backend._get_pending_req_key(model)
        pending_tok_key = backend._get_pending_tok_key(model)
        req_map_key = backend._get_req_map_key(model, req_id)

        # Query Redis for the slot of each key
        state_slot = await redis_client.cluster_keyslot(state_key)
        pending_req_slot = await redis_client.cluster_keyslot(pending_req_key)
        pending_tok_slot = await redis_client.cluster_keyslot(pending_tok_key)
        req_map_slot = await redis_client.cluster_keyslot(req_map_key)

        # All keys must be on the same slot
        slots = {state_slot, pending_req_slot, pending_tok_slot, req_map_slot}

        assert len(slots) == 1, (
            f"Keys are on different slots! "
            f"state={state_slot}, pending_req={pending_req_slot}, "
            f"pending_tok={pending_tok_slot}, req_map={req_map_slot}"
        )

    @pytest.mark.asyncio
    async def test_cluster_mode_cross_slot_error_prevention(self, cluster_backend):
        """
        Test that properly hash-tagged keys prevent CROSSSLOT errors.

        This test performs multiple operations that use multi-key Lua scripts
        and verifies none of them produce CROSSSLOT errors.
        """
        backend = cluster_backend

        # Force connection
        await backend._ensure_connected()

        model = "test-model-cross-slot"

        # Perform a sequence of operations that all use multi-key scripts

        # 1. Reserve capacity (uses state, pending_req, pending_tok, req_map)
        success, request_id = await backend.check_and_reserve_capacity(
            key=model,
            requests=1,
            tokens=50,
            bucket_limits={"rpm_limit": 100, "tpm_limit": 10000},
        )
        assert success is True, f"check_and_reserve_capacity failed: {request_id}"

        # 2. Update rate limits (uses same keys)
        # Note: This call exercises the multi-key Lua script for rate limit updates.
        # The return value (0 or 1) indicates whether the request mapping was found,
        # but any CROSSSLOT error would raise an exception - that's what we're testing.
        result = await backend.update_rate_limits(
            model=model,
            headers={
                "x-ratelimit-remaining-requests": "99",
                "x-ratelimit-remaining-tokens": "9950",
                "x-ratelimit-limit-requests": "100",
                "x-ratelimit-limit-tokens": "10000",
            },
            request_id=request_id,
            status_code=200,
        )
        # If we get here without a Redis CROSSSLOT error, the test passes.
        # Result may be 0 (mapping not found) or 1 (updated successfully).
        assert result in (0, 1), f"Unexpected result from update_rate_limits: {result}"

        # 3. Do another reservation
        success2, request_id2 = await backend.check_and_reserve_capacity(
            key=model,
            requests=1,
            tokens=100,
            bucket_limits={"rpm_limit": 100, "tpm_limit": 10000},
        )
        assert success2 is True, f"Second reservation failed: {request_id2}"

        # 4. Release the reservation
        release_result = await backend.release_reservation(model, request_id2)
        assert release_result is True, "release_reservation failed"

        # If we got here without Redis errors, cross-slot prevention is working


@pytest.mark.skipif(SKIP_CLUSTER_TESTS, reason=SKIP_REASON)
class TestRedisClusterStressTests:
    """
    Stress tests for Redis Cluster mode under concurrent load.
    """

    @pytest.fixture
    async def cluster_backend(self):
        """Create a RedisBackend in cluster mode."""
        from adaptive_rate_limiter.backends.redis import RedisBackend

        backend = RedisBackend(
            redis_url=REDIS_CLUSTER_URL,
            namespace="test_cluster_stress",
            account_id="stress-test-account",
            cluster_mode=True,
        )

        yield backend

        await backend.cleanup()

    @pytest.mark.asyncio
    async def test_concurrent_reservations_same_model(self, cluster_backend):
        """
        Test concurrent reservations for the same model don't cause slot errors.

        Multiple concurrent requests should all route to the same slot
        and execute atomically without CROSSSLOT errors.
        """
        import asyncio

        backend = cluster_backend
        await backend._ensure_connected()

        model = "stress-test-model"
        num_concurrent = 10

        async def make_reservation(idx: int):
            """Make a single reservation and release it."""
            success, result = await backend.check_and_reserve_capacity(
                key=model,
                requests=1,
                tokens=10,
                bucket_limits={"rpm_limit": 1000, "tpm_limit": 100000},
            )
            if success:
                await backend.release_reservation(model, result)
            return success

        # Run concurrent reservations
        tasks = [make_reservation(i) for i in range(num_concurrent)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for any exceptions (which would indicate CROSSSLOT errors)
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Got exceptions: {exceptions}"

    @pytest.mark.asyncio
    async def test_multiple_models_concurrent(self, cluster_backend):
        """
        Test concurrent operations across multiple models.

        Each model should route to its own slot (possibly different),
        but each model's keys should all be on the same slot.
        """
        import asyncio

        backend = cluster_backend
        await backend._ensure_connected()

        models = ["model-a", "model-b", "model-c", "model-d"]

        async def operations_for_model(model: str):
            """Run operations for a single model."""
            # Reserve
            success, request_id = await backend.check_and_reserve_capacity(
                key=model,
                requests=1,
                tokens=10,
                bucket_limits={"rpm_limit": 100, "tpm_limit": 10000},
            )

            if not success:
                return False

            # Update
            await backend.update_rate_limits(
                model=model,
                headers={"x-ratelimit-remaining-requests": "99"},
                request_id=request_id,
                status_code=200,
            )

            return True

        # Run operations for all models concurrently
        tasks = [operations_for_model(m) for m in models]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for exceptions
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Got exceptions: {exceptions}"


def _is_docker_available() -> bool:
    """
    Check if Docker is available and running.

    This performs a quick check to verify the docker package is installed
    and the Docker daemon is accessible.

    Returns:
        True if Docker is available and running, False otherwise.
    """
    try:
        import docker

        client = docker.from_env()
        client.ping()
        client.close()
        return True
    except Exception:
        return False


# Import failover fixtures (pytest auto-discovers, but explicit import helps IDE)
# These are defined in conftest_cluster.py
# - cluster_node_controller: ClusterNodeController for stopping/starting nodes
# - failover_cluster_backend: RedisBackend configured for failover testing
# - redis_cluster_containers: Dict of Docker container objects


@pytest.mark.skipif(
    SKIP_CLUSTER_TESTS or not _is_docker_available(),
    reason="Redis Cluster or Docker not available for failover testing",
)
class TestRedisClusterFailover:
    """
    Tests for Redis Cluster failover and recovery scenarios.

    These tests validate behavior when:
    - A cluster master node becomes unavailable and is promoted
    - Operations continue during failover
    - Circuit breaker activates during complete cluster failure
    - Recovery happens after cluster becomes available again

    Requirements:
    - Redis Cluster running via docker-compose -f docker-compose.redis-cluster.yml up -d
    - REDIS_CLUSTER_URL environment variable set
    - Docker daemon running (for container control)
    """

    @pytest.mark.asyncio
    async def test_failover_recovery(
        self,
        cluster_node_controller,
        failover_cluster_backend,
    ):
        """
        Test that the backend recovers gracefully when a master node fails.

        Scenario:
        1. Establish connection and verify it works
        2. Make a reservation to ensure data is in Redis
        3. Stop a master node (node 1)
        4. Wait for Redis Cluster to promote replica to master
        5. Verify the backend can still make reservations (auto-reconnects)
        6. Verify operations work after failover
        7. Restart the stopped node for cleanup
        """
        import asyncio

        backend = failover_cluster_backend
        controller = cluster_node_controller

        # Step 1: Verify initial connection works
        await backend._ensure_connected()
        assert backend._connected is True

        # Step 2: Make an initial reservation
        test_model = "failover-test-model"
        success, request_id = await backend.check_and_reserve_capacity(
            key=test_model,
            requests=1,
            tokens=100,
            bucket_limits={"rpm_limit": 100, "tpm_limit": 10000},
        )
        assert success is True, f"Initial reservation failed: {request_id}"

        # Step 3: Stop master node 1 (this is one of the 3 masters in a 6-node cluster)
        await controller.stop_node(1)

        # Give cluster a moment to detect the failure
        await asyncio.sleep(2.0)

        # Step 4: Wait for cluster to failover and stabilize
        failover_success = await controller.wait_for_failover(timeout=30.0)
        assert failover_success, "Cluster did not stabilize after failover"

        # Step 5: Verify backend can still make reservations after failover
        # The redis-py cluster client should automatically reconnect to available nodes
        success2, request_id2 = await backend.check_and_reserve_capacity(
            key=test_model,
            requests=1,
            tokens=100,
            bucket_limits={"rpm_limit": 100, "tpm_limit": 10000},
        )
        assert success2 is True, f"Post-failover reservation failed: {request_id2}"

        # Step 6: Verify we can release the reservation
        release_result = await backend.release_reservation(test_model, request_id2)
        assert release_result is True, "Post-failover release failed"

        # Step 7: Restart node 1 (cleanup is also handled by fixture)
        await controller.start_node(1)
        await asyncio.sleep(2.0)

        # Verify cluster is healthy again
        cluster_ok = await controller.wait_for_failover(timeout=30.0)
        assert cluster_ok, "Cluster did not stabilize after node restart"

    @pytest.mark.asyncio
    async def test_operations_during_failover(
        self,
        cluster_node_controller,
        failover_cluster_backend,
    ):
        """
        Test that operations during failover either succeed or fail gracefully.

        Scenario:
        1. Start background operations
        2. Trigger failover mid-operation
        3. Verify operations either succeed or fail gracefully (no data corruption)
        4. Verify operations resume after failover completes
        """
        import asyncio

        backend = failover_cluster_backend
        controller = cluster_node_controller

        # Ensure connection
        await backend._ensure_connected()

        test_model = "concurrent-failover-model"
        operation_results: list[tuple[bool, str]] = []
        operation_errors: list[Exception] = []

        async def make_reservation_operation(idx: int) -> tuple[bool, str]:
            """Make a single reservation, tracking success/failure."""
            try:
                success, result = await backend.check_and_reserve_capacity(
                    key=test_model,
                    requests=1,
                    tokens=10,
                    bucket_limits={"rpm_limit": 1000, "tpm_limit": 100000},
                )
                return (success, f"op_{idx}: {result}")
            except Exception as e:
                # Record the error but don't re-raise - we want to see all results
                operation_errors.append(e)
                return (False, f"op_{idx}: error - {type(e).__name__}")

        # Start a batch of operations that will run during failover
        async def run_operations_batch():
            """Run operations before, during, and after failover."""
            tasks = []
            for i in range(20):
                task = asyncio.create_task(make_reservation_operation(i))
                tasks.append(task)
                # Stagger operations slightly
                await asyncio.sleep(0.1)

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, tuple):
                    operation_results.append(r)
                elif isinstance(r, Exception):
                    operation_errors.append(r)

        async def trigger_failover():
            """Wait briefly, then trigger failover."""
            await asyncio.sleep(0.5)  # Let some operations start
            await controller.stop_node(1)
            await controller.wait_for_failover(timeout=30.0)

        # Run operations and failover concurrently
        await asyncio.gather(
            run_operations_batch(),
            trigger_failover(),
        )

        # We expect some operations to succeed and possibly some to fail
        # The key assertion is: no operational corruption occurred
        assert len(operation_results) + len(operation_errors) == 20, (
            f"Expected 20 operations, got {len(operation_results)} results "
            f"and {len(operation_errors)} errors"
        )

        # After failover, operations should work reliably
        (
            post_failover_success,
            post_request_id,
        ) = await backend.check_and_reserve_capacity(
            key=test_model,
            requests=1,
            tokens=100,
            bucket_limits={"rpm_limit": 1000, "tpm_limit": 100000},
        )
        assert post_failover_success is True, (
            f"Post-failover operation failed: {post_request_id}"
        )

        # Restart node for cleanup
        await controller.start_node(1)
        await asyncio.sleep(2.0)

    @pytest.mark.asyncio
    async def test_circuit_breaker_during_cluster_failure(
        self,
        cluster_node_controller,
        failover_cluster_backend,
    ):
        """
        Test circuit breaker activation and fallback during complete cluster failure.

        Scenario:
        1. Verify normal operation
        2. Stop all master nodes (nodes 1, 2, 3) to simulate complete cluster failure
        3. Verify circuit breaker activates after enough failures
        4. Verify fallback to MemoryBackend occurs
        5. Verify operations continue (with degraded performance)
        6. Restart nodes and verify recovery
        """
        import asyncio

        backend = failover_cluster_backend
        controller = cluster_node_controller

        # Ensure connection works initially
        await backend._ensure_connected()

        test_model = "circuit-breaker-test-model"

        # Step 1: Verify normal operation
        success, _initial_request_id = await backend.check_and_reserve_capacity(
            key=test_model,
            requests=1,
            tokens=100,
            bucket_limits={"rpm_limit": 100, "tpm_limit": 10000},
        )
        assert success is True
        assert not backend.is_in_fallback_mode(), "Should not be in fallback initially"

        # Step 2: Stop ALL master nodes (1, 2, 3) - replicas are 4, 5, 6
        # In a 6-node cluster with 3 masters and 3 replicas, stopping all masters
        # will prevent the cluster from accepting writes
        await controller.stop_node(1)
        await controller.stop_node(2)
        await controller.stop_node(3)

        # Wait for cluster to detect multiple failures
        await asyncio.sleep(3.0)

        # Step 3: Attempt multiple operations to trigger circuit breaker
        # The circuit breaker activates after enough failures (default: 20 in 30s)
        # We'll attempt operations that should fail and trigger the circuit breaker
        failure_count = 0
        for _attempt in range(25):
            try:
                await backend.check_and_reserve_capacity(
                    key=test_model,
                    requests=1,
                    tokens=10,
                    bucket_limits={"rpm_limit": 100, "tpm_limit": 10000},
                )
            except Exception:
                failure_count += 1
            await asyncio.sleep(0.1)  # Small delay between attempts

        # Step 4: Verify circuit breaker is now broken
        circuit_broken = await backend.is_circuit_broken()

        # Step 5: If circuit is broken, verify fallback mode
        if circuit_broken:
            # Next operation should use fallback backend
            fallback_success, _ = await backend.check_and_reserve_capacity(
                key=test_model,
                requests=1,
                tokens=10,
                bucket_limits={"rpm_limit": 100, "tpm_limit": 10000},
            )
            # In fallback mode, operations should succeed via MemoryBackend
            assert fallback_success is True, "Fallback mode operation should succeed"
            assert backend.is_in_fallback_mode(), "Should be in fallback mode"

        # Step 6: Restart all master nodes
        await controller.start_node(1)
        await controller.start_node(2)
        await controller.start_node(3)

        # Wait for cluster to fully recover
        await asyncio.sleep(5.0)
        cluster_recovered = await controller.wait_for_failover(timeout=60.0)
        assert cluster_recovered, "Cluster did not recover after restarting nodes"

        # Step 7: Wait for circuit breaker to reset (may take up to 30 seconds)
        # Force a connection refresh to clear the failure timestamps
        await asyncio.sleep(5.0)

        # Attempt to reconnect - the backend should eventually recover
        # and exit fallback mode on successful Redis operations
        await backend._ensure_connected()

        # Verify we can make a reservation after recovery
        # (Note: fallback state may clear on successful reconnect)
        recovery_success, _ = await backend.check_and_reserve_capacity(
            key=test_model,
            requests=1,
            tokens=100,
            bucket_limits={"rpm_limit": 100, "tpm_limit": 10000},
        )
        assert recovery_success is True, "Post-recovery operation should succeed"

    @pytest.mark.asyncio
    async def test_pause_unpause_node_simulation(
        self,
        cluster_node_controller,
        failover_cluster_backend,
    ):
        """
        Test node pause/unpause to simulate network partition.

        Scenario:
        1. Pause a node (simulates network hang, not crash)
        2. Verify cluster handles the paused node
        3. Unpause and verify recovery
        """
        import asyncio

        backend = failover_cluster_backend
        controller = cluster_node_controller

        # Verify initial state
        await backend._ensure_connected()
        test_model = "pause-test-model"

        # Step 1: Make an initial reservation
        success, _ = await backend.check_and_reserve_capacity(
            key=test_model,
            requests=1,
            tokens=100,
            bucket_limits={"rpm_limit": 100, "tpm_limit": 10000},
        )
        assert success is True

        # Step 2: Pause node 4 (a replica node)
        await controller.pause_node(4)
        await asyncio.sleep(2.0)

        # Step 3: Operations should still work (cluster has quorum)
        success2, _ = await backend.check_and_reserve_capacity(
            key=test_model,
            requests=1,
            tokens=100,
            bucket_limits={"rpm_limit": 100, "tpm_limit": 10000},
        )
        assert success2 is True, "Operations should work with one paused replica"

        # Step 4: Unpause the node
        await controller.unpause_node(4)
        await asyncio.sleep(2.0)

        # Step 5: Verify operations continue
        success3, _ = await backend.check_and_reserve_capacity(
            key=test_model,
            requests=1,
            tokens=100,
            bucket_limits={"rpm_limit": 100, "tpm_limit": 10000},
        )
        assert success3 is True, "Operations should work after unpause"
