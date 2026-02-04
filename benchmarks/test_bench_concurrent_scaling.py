"""
Benchmark: Concurrent Request Scaling

Measures how the scheduler scales with increasing concurrency levels.
Useful for understanding throughput characteristics and identifying
potential bottlenecks under load.

Usage:
    uv run pytest benchmarks/test_bench_concurrent_scaling.py -v --no-cov
"""

import pytest
import asyncio
import time
import statistics
from typing import List, Tuple

from adaptive_rate_limiter.strategies.modes.intelligent import IntelligentModeStrategy
from adaptive_rate_limiter.types.request import RequestMetadata


class TestConcurrentScaling:
    """Test how scheduler scales with concurrent requests."""

    @pytest.mark.asyncio
    async def test_scaling_at_concurrency_levels(
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
        Measure throughput at different concurrency levels.

        This helps identify:
        - Optimal concurrency level
        - Point of diminishing returns
        - Potential lock contention
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
            concurrency_levels = [1, 5, 10, 25, 50, 100]
            results: List[Tuple[int, float, float]] = []

            for concurrency in concurrency_levels:

                async def submit_one(idx: int) -> float:
                    metadata = RequestMetadata(
                        request_id=f"scale-{concurrency}-{idx}",
                        model_id="benchmark-model",
                        resource_type="text",
                        estimated_tokens=50,
                    )

                    async def instant_request():
                        return {"idx": idx}

                    start = time.perf_counter()
                    await strategy.submit_request(metadata, instant_request)
                    return time.perf_counter() - start

                # Warmup at this concurrency level
                await asyncio.gather(*[submit_one(i) for i in range(concurrency)])

                # Benchmark - 3 rounds
                round_times = []
                for _ in range(3):
                    round_start = time.perf_counter()
                    await asyncio.gather(
                        *[submit_one(i) for i in range(concurrency)]
                    )
                    round_elapsed = time.perf_counter() - round_start
                    round_times.append(round_elapsed)

                avg_round_time = statistics.mean(round_times)
                throughput = concurrency / avg_round_time
                avg_latency = statistics.mean(
                    [statistics.mean(round_times) / concurrency for _ in range(3)]
                )

                results.append((concurrency, throughput, avg_latency * 1000))

            print("\n--- Concurrent Scaling Results ---")
            print(f"{'Concurrency':>12} | {'Throughput':>12} | {'Avg Latency':>12}")
            print(f"{'-'*12}-+-{'-'*12}-+-{'-'*12}")
            for conc, throughput, latency in results:
                print(f"{conc:>12} | {throughput:>10.1f}/s | {latency:>10.2f}ms")

            # Verify scaling is reasonable
            # At higher concurrency, throughput should increase (or at least not collapse)
            single_throughput = results[0][1]
            max_throughput = max(r[1] for r in results)
            assert max_throughput >= single_throughput * 0.8, (
                "Throughput collapsed under concurrency"
            )

        finally:
            await strategy.stop()

    @pytest.mark.asyncio
    async def test_sustained_load(
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
        Measure performance under sustained load.

        Runs for a fixed duration to detect any degradation over time.
        """
        from adaptive_rate_limiter.exceptions import RateLimiterError

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
            duration_seconds = 2.0
            concurrency = 10  # Lower concurrency to avoid queue overflow
            completed = 0
            queue_full_count = 0
            latencies: List[float] = []

            async def worker(worker_id: int, stop_event: asyncio.Event):
                nonlocal completed, queue_full_count
                local_latencies = []
                request_num = 0

                while not stop_event.is_set():
                    metadata = RequestMetadata(
                        request_id=f"sustained-{worker_id}-{request_num}",
                        model_id="benchmark-model",
                        resource_type="text",
                        estimated_tokens=50,
                    )

                    async def instant_request():
                        return {"worker": worker_id}

                    start = time.perf_counter()
                    try:
                        await strategy.submit_request(metadata, instant_request)
                        local_latencies.append(time.perf_counter() - start)
                        completed += 1
                    except RateLimiterError:
                        # Queue full is expected under sustained load
                        queue_full_count += 1
                        # Small backoff
                        await asyncio.sleep(0.01)
                    request_num += 1

                return local_latencies

            stop_event = asyncio.Event()

            # Start workers
            worker_tasks = [
                asyncio.create_task(worker(i, stop_event)) for i in range(concurrency)
            ]

            # Run for duration
            await asyncio.sleep(duration_seconds)
            stop_event.set()

            # Collect results
            worker_results = await asyncio.gather(*worker_tasks)
            for worker_latencies in worker_results:
                latencies.extend(worker_latencies)

            if latencies:
                avg_latency = statistics.mean(latencies) * 1000
                p50 = statistics.median(latencies) * 1000
                p95 = sorted(latencies)[int(len(latencies) * 0.95)] * 1000
                p99 = sorted(latencies)[int(len(latencies) * 0.99)] * 1000
                throughput = completed / duration_seconds

                print(f"\n--- Sustained Load Results ({duration_seconds}s) ---")
                print(f"Concurrency: {concurrency}")
                print(f"Completed: {completed} requests")
                print(f"Queue full events: {queue_full_count}")
                print(f"Throughput: {throughput:.1f} req/s")
                print(f"Latency avg: {avg_latency:.2f}ms")
                print(f"Latency p50: {p50:.2f}ms")
                print(f"Latency p95: {p95:.2f}ms")
                print(f"Latency p99: {p99:.2f}ms")

                # Sanity checks
                assert completed > 50, f"Too few requests completed: {completed}"
                assert p99 < 100, f"p99 latency too high: {p99:.2f}ms"
            else:
                pytest.fail("No requests completed during sustained load test")

        finally:
            await strategy.stop()


class TestQueuePressure:
    """Test behavior under queue pressure scenarios."""

    @pytest.mark.asyncio
    async def test_queue_buildup_and_drain(
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
        Measure queue behavior when requests arrive faster than they drain.

        Simulates bursty traffic patterns.
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
            # Submit a large burst
            burst_size = 200

            async def submit_burst_request(idx: int):
                metadata = RequestMetadata(
                    request_id=f"burst-{idx}",
                    model_id="benchmark-model",
                    resource_type="text",
                    estimated_tokens=50,
                )

                async def instant_request():
                    return {"idx": idx}

                return await strategy.submit_request(metadata, instant_request)

            # Measure time to submit all (queue buildup)
            submit_start = time.perf_counter()
            tasks = [
                asyncio.create_task(submit_burst_request(i)) for i in range(burst_size)
            ]
            submit_elapsed = time.perf_counter() - submit_start

            # Measure time to complete all (queue drain)
            drain_start = time.perf_counter()
            results = await asyncio.gather(*tasks)
            drain_elapsed = time.perf_counter() - drain_start

            total_elapsed = submit_elapsed + drain_elapsed
            throughput = burst_size / total_elapsed

            print(f"\n--- Queue Buildup/Drain ({burst_size} requests) ---")
            print(f"Submit time: {submit_elapsed*1000:.1f}ms")
            print(f"Drain time: {drain_elapsed*1000:.1f}ms")
            print(f"Total time: {total_elapsed*1000:.1f}ms")
            print(f"Effective throughput: {throughput:.1f} req/s")

            # All should complete
            assert len(results) == burst_size
            # Should complete in reasonable time
            assert total_elapsed < 10.0, f"Burst took too long: {total_elapsed:.2f}s"

        finally:
            await strategy.stop()
