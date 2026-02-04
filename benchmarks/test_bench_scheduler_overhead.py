"""
Benchmark: Scheduler Overhead

Measures the overhead introduced by the rate limiter scheduler
when processing requests. This benchmark isolates the scheduling
machinery from actual API call latency.

Usage:
    uv run pytest benchmarks/test_bench_scheduler_overhead.py -v --no-cov
"""

import pytest
import asyncio
import time

from adaptive_rate_limiter.strategies.modes.intelligent import IntelligentModeStrategy
from adaptive_rate_limiter.types.request import RequestMetadata


class TestSchedulerOverhead:
    """Benchmark scheduler overhead for different modes."""

    @pytest.mark.asyncio
    async def test_intelligent_mode_single_request_overhead(
        self,
        memory_backend,
        benchmark_config,
        state_manager,
        mock_scheduler,
        benchmark_client,
        benchmark_provider,
        benchmark_classifier,
    ):
        """
        Measure overhead of a single request through intelligent mode.

        This isolates the scheduling overhead from actual request execution.
        The mock request returns instantly, so any measured time is pure overhead.
        """
        strategy = IntelligentModeStrategy(
            scheduler=mock_scheduler,
            config=benchmark_config,
            client=benchmark_client,
            provider=benchmark_provider,
            classifier=benchmark_classifier,
            state_manager=state_manager,
        )

        await strategy.start()

        try:
            metadata = RequestMetadata(
                request_id="overhead-1",
                model_id="benchmark-model",
                resource_type="text",
                estimated_tokens=50,
            )

            async def instant_request():
                return {"result": "instant"}

            # Warmup
            for _ in range(10):
                await strategy.submit_request(metadata, instant_request)

            # Benchmark
            iterations = 100
            start = time.perf_counter()
            for i in range(iterations):
                metadata = RequestMetadata(
                    request_id=f"overhead-{i}",
                    model_id="benchmark-model",
                    resource_type="text",
                    estimated_tokens=50,
                )
                await strategy.submit_request(metadata, instant_request)
            elapsed = time.perf_counter() - start

            avg_ms = (elapsed / iterations) * 1000
            ops_per_sec = iterations / elapsed

            print("\n--- Intelligent Mode Single Request Overhead ---")
            print(f"Iterations: {iterations}")
            print(f"Total time: {elapsed:.4f}s")
            print(f"Average latency: {avg_ms:.3f}ms per request")
            print(f"Throughput: {ops_per_sec:.1f} ops/sec")

            # Assert reasonable overhead (should be < 10ms per request)
            assert avg_ms < 10, f"Overhead too high: {avg_ms:.3f}ms"

        finally:
            await strategy.stop()

    @pytest.mark.asyncio
    async def test_intelligent_mode_burst_overhead(
        self,
        memory_backend,
        benchmark_config,
        state_manager,
        mock_scheduler,
        benchmark_client,
        benchmark_provider,
        benchmark_classifier,
    ):
        """
        Measure overhead when submitting a burst of concurrent requests.

        Tests how the scheduler handles many simultaneous submissions.
        """
        strategy = IntelligentModeStrategy(
            scheduler=mock_scheduler,
            config=benchmark_config,
            client=benchmark_client,
            provider=benchmark_provider,
            classifier=benchmark_classifier,
            state_manager=state_manager,
        )

        await strategy.start()

        try:

            async def create_and_submit(i: int):
                metadata = RequestMetadata(
                    request_id=f"burst-{i}",
                    model_id="benchmark-model",
                    resource_type="text",
                    estimated_tokens=50,
                )

                async def instant_request():
                    return {"result": f"burst-{i}"}

                return await strategy.submit_request(metadata, instant_request)

            # Warmup
            await asyncio.gather(*[create_and_submit(i) for i in range(10)])

            # Benchmark burst of 50 concurrent requests
            burst_size = 50
            start = time.perf_counter()
            results = await asyncio.gather(
                *[create_and_submit(i) for i in range(burst_size)]
            )
            elapsed = time.perf_counter() - start

            avg_ms = (elapsed / burst_size) * 1000
            ops_per_sec = burst_size / elapsed

            print(f"\n--- Intelligent Mode Burst Overhead ({burst_size} concurrent) ---")
            print(f"Total time: {elapsed:.4f}s")
            print(f"Average latency: {avg_ms:.3f}ms per request")
            print(f"Throughput: {ops_per_sec:.1f} ops/sec")

            # All requests should complete
            assert len(results) == burst_size
            # Burst should complete in reasonable time
            assert elapsed < 5.0, f"Burst took too long: {elapsed:.2f}s"

        finally:
            await strategy.stop()


class TestBackendOverhead:
    """Benchmark backend operation overhead."""

    @pytest.mark.asyncio
    async def test_memory_backend_reservation_cycle(self, memory_backend):
        """
        Measure the cost of a complete reservation cycle:
        check_and_reserve -> release_reservation
        """
        bucket_id = "bench-bucket"
        bucket_limits = {"rpm_limit": 10000, "tpm_limit": 1000000}

        # Initialize state
        await memory_backend.set_state(
            bucket_id,
            {
                "remaining_requests": 10000,
                "remaining_tokens": 1000000,
                "request_limit": 10000,
                "token_limit": 1000000,
            },
        )

        # Warmup
        for _ in range(10):
            success, res_id = await memory_backend.check_and_reserve_capacity(
                key=bucket_id,
                requests=1,
                tokens=100,
                bucket_limits=bucket_limits,
            )
            if success and res_id:
                await memory_backend.release_reservation(bucket_id, res_id)

        # Refresh state for benchmark
        await memory_backend.set_state(
            bucket_id,
            {
                "remaining_requests": 10000,
                "remaining_tokens": 1000000,
                "request_limit": 10000,
                "token_limit": 1000000,
            },
        )

        # Benchmark
        iterations = 500
        start = time.perf_counter()
        for _ in range(iterations):
            success, res_id = await memory_backend.check_and_reserve_capacity(
                key=bucket_id,
                requests=1,
                tokens=100,
                bucket_limits=bucket_limits,
            )
            if success and res_id:
                await memory_backend.release_reservation(bucket_id, res_id)
        elapsed = time.perf_counter() - start

        avg_us = (elapsed / iterations) * 1_000_000
        ops_per_sec = iterations / elapsed

        print("\n--- Memory Backend Reservation Cycle ---")
        print(f"Iterations: {iterations}")
        print(f"Total time: {elapsed:.4f}s")
        print(f"Average latency: {avg_us:.1f}μs per cycle")
        print(f"Throughput: {ops_per_sec:.1f} cycles/sec")

        # Should be very fast (< 1ms per cycle)
        assert avg_us < 1000, f"Backend too slow: {avg_us:.1f}μs"

    @pytest.mark.asyncio
    async def test_memory_backend_state_operations(self, memory_backend):
        """Measure get/set state operation latency."""
        bucket_id = "state-bench"

        state = {
            "remaining_requests": 100,
            "remaining_tokens": 10000,
            "request_limit": 100,
            "token_limit": 10000,
        }

        # Warmup
        for _ in range(10):
            await memory_backend.set_state(bucket_id, state)
            await memory_backend.get_state(bucket_id)

        # Benchmark set operations
        iterations = 1000
        start = time.perf_counter()
        for i in range(iterations):
            state["remaining_requests"] = 100 - (i % 100)
            await memory_backend.set_state(bucket_id, state)
        set_elapsed = time.perf_counter() - start

        # Benchmark get operations
        start = time.perf_counter()
        for _ in range(iterations):
            await memory_backend.get_state(bucket_id)
        get_elapsed = time.perf_counter() - start

        set_avg_us = (set_elapsed / iterations) * 1_000_000
        get_avg_us = (get_elapsed / iterations) * 1_000_000

        print("\n--- Memory Backend State Operations ---")
        print(f"Iterations: {iterations}")
        print(f"Set state: {set_avg_us:.1f}μs avg ({iterations/set_elapsed:.1f} ops/sec)")
        print(f"Get state: {get_avg_us:.1f}μs avg ({iterations/get_elapsed:.1f} ops/sec)")

        # Both operations should be very fast
        assert set_avg_us < 500, f"Set too slow: {set_avg_us:.1f}μs"
        assert get_avg_us < 500, f"Get too slow: {get_avg_us:.1f}μs"


class TestReservationTrackerOverhead:
    """Benchmark reservation tracker operations."""

    @pytest.mark.asyncio
    async def test_reservation_tracker_store_and_retrieve(
        self,
        memory_backend,
        benchmark_config,
        state_manager,
        mock_scheduler,
        benchmark_client,
        benchmark_provider,
        benchmark_classifier,
    ):
        """Measure overhead of reservation tracking operations."""
        strategy = IntelligentModeStrategy(
            scheduler=mock_scheduler,
            config=benchmark_config,
            client=benchmark_client,
            provider=benchmark_provider,
            classifier=benchmark_classifier,
            state_manager=state_manager,
        )

        await strategy.start()

        try:
            tracker = strategy._reservation_tracker

            # Warmup
            for i in range(10):
                await tracker.store(
                    request_id=f"warmup-{i}",
                    bucket_id="benchmark-model",
                    reservation_id=f"res-warmup-{i}",
                    estimated_tokens=100,
                )
                await tracker.get_and_clear(
                    request_id=f"warmup-{i}",
                    bucket_id="benchmark-model",
                )

            # Benchmark store operations
            iterations = 500
            start = time.perf_counter()
            for i in range(iterations):
                await tracker.store(
                    request_id=f"bench-{i}",
                    bucket_id="benchmark-model",
                    reservation_id=f"res-{i}",
                    estimated_tokens=100,
                )
            store_elapsed = time.perf_counter() - start

            # Benchmark retrieve operations
            start = time.perf_counter()
            for i in range(iterations):
                await tracker.get(
                    request_id=f"bench-{i}",
                    bucket_id="benchmark-model",
                )
            get_elapsed = time.perf_counter() - start

            # Cleanup
            for i in range(iterations):
                await tracker.get_and_clear(
                    request_id=f"bench-{i}",
                    bucket_id="benchmark-model",
                )

            store_avg_us = (store_elapsed / iterations) * 1_000_000
            get_avg_us = (get_elapsed / iterations) * 1_000_000

            print("\n--- Reservation Tracker Operations ---")
            print(f"Iterations: {iterations}")
            print(
                f"Store: {store_avg_us:.1f}μs avg ({iterations/store_elapsed:.1f} ops/sec)"
            )
            print(
                f"Get: {get_avg_us:.1f}μs avg ({iterations/get_elapsed:.1f} ops/sec)"
            )

            # Operations should be fast
            assert store_avg_us < 1000, f"Store too slow: {store_avg_us:.1f}μs"
            assert get_avg_us < 1000, f"Get too slow: {get_avg_us:.1f}μs"

        finally:
            await strategy.stop()
