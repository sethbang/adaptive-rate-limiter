"""
Redis Cluster Failover Testing Fixtures

This module provides pytest fixtures for controlling Docker container lifecycle
to simulate node failures during Redis Cluster failover tests.

Usage:
    These fixtures are automatically available in tests under tests/integration/
    when the docker package is installed and the cluster containers are running.

Requirements:
    - docker package (pip install docker)
    - Redis Cluster running via docker-compose -f docker-compose.redis-cluster.yml up -d
    - REDIS_CLUSTER_URL environment variable set

Example:
    @pytest.mark.asyncio
    async def test_failover(cluster_node_controller, failover_cluster_backend):
        # Stop a master node
        await cluster_node_controller.stop_node(1)

        # Wait for failover to complete
        success = await cluster_node_controller.wait_for_failover()
        assert success

        # Verify backend still works
        result = await failover_cluster_backend.check_and_reserve_capacity(...)

        # Restore the node
        await cluster_node_controller.start_node(1)
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from typing import Any


logger = logging.getLogger(__name__)

# Container name pattern from docker-compose.redis-cluster.yml
CONTAINER_NAME_PREFIX = "redis-cluster-node-"
NODE_COUNT = 6
NODE_PORTS = {i: 7000 + i for i in range(1, NODE_COUNT + 1)}

# Environment variable for cluster URL
REDIS_CLUSTER_URL_ENV = "REDIS_CLUSTER_URL"


def _check_docker_available() -> tuple[bool, str]:
    """
    Check if the docker package is available.

    Returns:
        Tuple of (available, reason) where available is True if docker
        package can be imported, and reason explains why if not.
    """
    try:
        import docker  # noqa: F401

        return True, ""
    except ImportError:
        return False, (
            "docker package not installed. Install with: "
            "pip install docker OR uv add --group dev docker"
        )


@pytest.fixture(scope="session")
def docker_client():
    """
    Create a Docker client from environment.

    This fixture is session-scoped to reuse the client across all tests,
    which is more efficient than creating a new client for each test.

    Skips tests if:
        - docker package is not installed
        - Docker daemon is not running/accessible

    Yields:
        docker.DockerClient: A configured Docker client instance.
    """
    available, reason = _check_docker_available()
    if not available:
        pytest.skip(reason)

    import docker
    from docker.errors import DockerException

    try:
        client = docker.from_env()
        # Verify connection by pinging the daemon
        client.ping()
    except DockerException as e:
        pytest.skip(f"Docker daemon not accessible: {e}")

    yield client

    # Clean up the client
    client.close()


@pytest.fixture(scope="function")
def redis_cluster_containers(docker_client):
    """
    Return a dict mapping node names to container objects.

    Container names follow the pattern: redis-cluster-node-{1-6}

    Skips if:
        - Any of the 6 cluster containers are not found
        - Any container is not in a running state

    Args:
        docker_client: Docker client fixture

    Returns:
        Dict[str, Container]: Mapping of node names (e.g., "node-1") to
            Docker container objects.
    """
    from docker.errors import NotFound

    containers = {}
    missing = []
    not_running = []

    for node_num in range(1, NODE_COUNT + 1):
        container_name = f"{CONTAINER_NAME_PREFIX}{node_num}"
        node_key = f"node-{node_num}"

        try:
            container = docker_client.containers.get(container_name)
            containers[node_key] = container

            # Check if container is running
            container.reload()  # Refresh status
            if container.status != "running":
                not_running.append(f"{container_name} (status: {container.status})")
        except NotFound:
            missing.append(container_name)

    if missing:
        pytest.skip(
            f"Redis Cluster containers not found: {', '.join(missing)}. "
            f"Start with: docker-compose -f docker-compose.redis-cluster.yml up -d"
        )

    if not_running:
        pytest.skip(
            f"Redis Cluster containers not running: {', '.join(not_running)}. "
            f"Start with: docker-compose -f docker-compose.redis-cluster.yml up -d"
        )

    return containers


class ClusterNodeController:
    """
    Controller for managing Redis Cluster node containers during tests.

    This class provides async methods for stopping, starting, pausing,
    and unpausing Redis Cluster containers to simulate various failure
    scenarios during integration tests.

    The Docker SDK is synchronous, so all blocking calls are wrapped
    with asyncio.to_thread() to prevent blocking the event loop.

    Attributes:
        containers: Dict mapping node names to container objects
        docker_client: Docker client instance
    """

    def __init__(self, containers: dict[str, Any], docker_client: Any) -> None:
        """
        Initialize the controller.

        Args:
            containers: Dict from redis_cluster_containers fixture
            docker_client: Docker client from docker_client fixture
        """
        self.containers = containers
        self.docker_client = docker_client
        self._stopped_nodes: set[int] = set()
        self._paused_nodes: set[int] = set()

    def _get_container(self, node_num: int) -> Any:
        """
        Get container object for a node number.

        Args:
            node_num: Node number (1-6)

        Returns:
            Docker container object

        Raises:
            ValueError: If node_num is out of valid range
        """
        if not 1 <= node_num <= NODE_COUNT:
            raise ValueError(
                f"node_num must be between 1 and {NODE_COUNT}, got {node_num}"
            )

        node_key = f"node-{node_num}"
        return self.containers[node_key]

    async def stop_node(self, node_num: int) -> None:
        """
        Stop a specific node container.

        This simulates a hard node failure where the Redis process
        is completely stopped. Other cluster nodes will detect this
        node as failed after the cluster-node-timeout (5 seconds).

        Args:
            node_num: Node number to stop (1-6)

        Raises:
            ValueError: If node_num is out of range
        """
        container = self._get_container(node_num)
        logger.info(f"Stopping Redis Cluster node {node_num}")

        await asyncio.to_thread(container.stop, timeout=10)
        self._stopped_nodes.add(node_num)

        logger.info(f"Node {node_num} stopped successfully")

    async def start_node(self, node_num: int) -> None:
        """
        Start a stopped node container.

        This brings back a previously stopped node. The node will
        attempt to rejoin the cluster automatically.

        Args:
            node_num: Node number to start (1-6)

        Raises:
            ValueError: If node_num is out of range
        """
        container = self._get_container(node_num)
        logger.info(f"Starting Redis Cluster node {node_num}")

        await asyncio.to_thread(container.start)
        self._stopped_nodes.discard(node_num)

        # Give the node a moment to start Redis
        await asyncio.sleep(1.0)

        logger.info(f"Node {node_num} started successfully")

    async def pause_node(self, node_num: int) -> None:
        """
        Pause (freeze) a container to simulate network hang.

        This uses Docker's pause functionality which suspends all
        processes in the container. This simulates a network partition
        or a node that is unresponsive but not crashed.

        Unlike stopping, paused nodes don't lose their state.

        Args:
            node_num: Node number to pause (1-6)

        Raises:
            ValueError: If node_num is out of range
        """
        container = self._get_container(node_num)
        logger.info(f"Pausing Redis Cluster node {node_num}")

        await asyncio.to_thread(container.pause)
        self._paused_nodes.add(node_num)

        logger.info(f"Node {node_num} paused successfully")

    async def unpause_node(self, node_num: int) -> None:
        """
        Unpause a container.

        Resumes a previously paused container. The node should
        automatically recover and resync with the cluster.

        Args:
            node_num: Node number to unpause (1-6)

        Raises:
            ValueError: If node_num is out of range
        """
        container = self._get_container(node_num)
        logger.info(f"Unpausing Redis Cluster node {node_num}")

        await asyncio.to_thread(container.unpause)
        self._paused_nodes.discard(node_num)

        logger.info(f"Node {node_num} unpaused successfully")

    async def get_cluster_info(self, node_num: int) -> dict[str, str]:
        """
        Execute CLUSTER INFO on a node and return parsed results.

        This queries the Redis CLUSTER INFO command and parses
        the response into a dictionary.

        Args:
            node_num: Node number to query (1-6)

        Returns:
            Dict with cluster info fields, e.g.:
            {
                "cluster_state": "ok",
                "cluster_slots_assigned": "16384",
                "cluster_known_nodes": "6",
                ...
            }

        Raises:
            ValueError: If node_num is out of range
            Exception: If the node is not accessible
        """
        container = self._get_container(node_num)
        port = NODE_PORTS[node_num]

        # Execute redis-cli CLUSTER INFO inside the container
        cmd = f"redis-cli -p {port} CLUSTER INFO"
        exec_result = await asyncio.to_thread(container.exec_run, cmd, demux=True)

        exit_code = exec_result.exit_code
        stdout, stderr = exec_result.output

        if exit_code != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            raise RuntimeError(f"CLUSTER INFO failed on node {node_num}: {error_msg}")

        # Parse the output into a dict
        result = {}
        if stdout:
            for line in stdout.decode().strip().split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    result[key.strip()] = value.strip()

        return result

    async def wait_for_failover(self, timeout: float = 30.0) -> bool:
        """
        Wait for cluster to stabilize after a failure.

        This polls the cluster state from available nodes until
        the cluster reports "ok" status, indicating that failover
        has completed and the cluster is healthy again.

        Args:
            timeout: Maximum time to wait in seconds (default: 30.0)

        Returns:
            True if cluster stabilized within timeout, False otherwise
        """
        logger.info(f"Waiting for cluster failover (timeout: {timeout}s)")

        start_time = asyncio.get_event_loop().time()
        check_interval = 0.5  # Check every 500ms

        while (asyncio.get_event_loop().time() - start_time) < timeout:
            # Try to get cluster info from any running node
            for node_num in range(1, NODE_COUNT + 1):
                if node_num in self._stopped_nodes or node_num in self._paused_nodes:
                    continue

                try:
                    info = await self.get_cluster_info(node_num)
                    cluster_state = info.get("cluster_state", "")

                    if cluster_state == "ok":
                        elapsed = asyncio.get_event_loop().time() - start_time
                        logger.info(
                            f"Cluster stabilized after {elapsed:.2f}s "
                            f"(checked node {node_num})"
                        )
                        return True

                    logger.debug(
                        f"Node {node_num} reports cluster_state: {cluster_state}"
                    )
                except Exception as e:
                    logger.debug(f"Could not query node {node_num}: {e}")
                    continue

            await asyncio.sleep(check_interval)

        logger.warning(f"Cluster did not stabilize within {timeout}s")
        return False

    async def restore_all_nodes(self) -> None:
        """
        Restore all stopped and paused nodes.

        This is useful for cleanup in test fixtures to ensure
        all nodes are back to running state.
        """
        # Unpause all paused nodes
        for node_num in list(self._paused_nodes):
            try:
                await self.unpause_node(node_num)
            except Exception as e:
                logger.warning(f"Failed to unpause node {node_num}: {e}")

        # Start all stopped nodes
        for node_num in list(self._stopped_nodes):
            try:
                await self.start_node(node_num)
            except Exception as e:
                logger.warning(f"Failed to start node {node_num}: {e}")


@pytest.fixture(scope="function")
async def cluster_node_controller(docker_client, redis_cluster_containers):
    """
    Return a ClusterNodeController instance for managing cluster nodes.

    This fixture provides methods to stop, start, pause, and unpause
    Redis Cluster containers during tests to simulate various failure
    scenarios.

    The controller automatically restores any stopped/paused nodes
    during fixture cleanup.

    Args:
        docker_client: Docker client fixture
        redis_cluster_containers: Containers fixture

    Yields:
        ClusterNodeController: Controller instance for managing nodes
    """
    controller = ClusterNodeController(redis_cluster_containers, docker_client)

    yield controller

    # Cleanup: restore any stopped/paused nodes and wait for cluster to stabilize
    await controller.restore_all_nodes()

    # Wait for cluster to become healthy again before allowing next test to run
    cluster_ok = await controller.wait_for_failover(timeout=60.0)
    if not cluster_ok:
        logger.warning(
            "Cluster did not stabilize after restoring nodes. "
            "Subsequent tests may fail."
        )


# Default startup nodes for multi-node cluster discovery during failover
# These correspond to the 6-node cluster from docker-compose.redis-cluster.yml
CLUSTER_STARTUP_NODES = [
    ("localhost", 7001),
    ("localhost", 7002),
    ("localhost", 7003),
    ("localhost", 7004),
    ("localhost", 7005),
    ("localhost", 7006),
]


@pytest.fixture(scope="function")
async def failover_cluster_backend():
    """
    Create a RedisBackend configured for failover testing.

    This fixture creates a RedisBackend in cluster mode with **multiple startup
    nodes** for proper failover resilience. When node 1 (7001) goes down during
    failover tests, the client can still discover the cluster topology through
    the other startup nodes.

    Skips if:
        - REDIS_CLUSTER_URL environment variable is not set
        - Redis Cluster is not accessible

    Yields:
        RedisBackend: Configured backend instance for failover testing
    """
    cluster_url = os.getenv(REDIS_CLUSTER_URL_ENV)
    if not cluster_url:
        pytest.skip(
            f"{REDIS_CLUSTER_URL_ENV} environment variable not set. "
            f"Set it to a cluster node URL, e.g., redis://localhost:7001"
        )

    from adaptive_rate_limiter.backends.redis import RedisBackend

    # Use multiple startup nodes for failover resilience
    # This ensures the client can rediscover the cluster even if the
    # initial connection node (e.g., 7001) goes down
    backend = RedisBackend(
        redis_url=cluster_url,
        namespace="test_cluster_failover",
        account_id="failover-test-account",
        cluster_mode=True,
        startup_nodes=CLUSTER_STARTUP_NODES,
    )

    try:
        # Verify connection
        await backend._ensure_connected()
    except Exception as e:
        pytest.skip(f"Could not connect to Redis Cluster: {e}")

    yield backend

    # Cleanup
    await backend.cleanup()
