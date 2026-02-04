# Benchmarks

Performance benchmarks for the adaptive-rate-limiter library.

## Purpose

These benchmarks measure the overhead introduced by the rate limiting machinery. They help:

1. **Detect Performance Regressions**: Track overhead changes between releases
2. **Validate Scaling**: Ensure throughput scales with concurrency
3. **Identify Bottlenecks**: Find hot spots under load

## Running Benchmarks

### Quick Validation
Run all benchmarks with default settings:

```bash
uv run pytest benchmarks/ -v
```

### Single Benchmark File
```bash
uv run pytest benchmarks/test_bench_scheduler_overhead.py -v --no-cov
uv run pytest benchmarks/test_bench_concurrent_scaling.py -v --no-cov
```

### With pytest-benchmark (if installed)
```bash
uv run pytest benchmarks/ -v --benchmark-only
```

## Benchmark Suites

### `test_bench_scheduler_overhead.py`

Measures the core overhead of the rate limiter:

| Test | Description |
|------|-------------|
| `test_intelligent_mode_single_request_overhead` | Single request latency through intelligent mode |
| `test_intelligent_mode_burst_overhead` | Handling concurrent request bursts |
| `test_memory_backend_reservation_cycle` | Backend reserve/release cycle time |
| `test_memory_backend_state_operations` | Get/set state operation latency |
| `test_reservation_tracker_store_and_retrieve` | Tracking operation overhead |

### `test_bench_concurrent_scaling.py`

Tests scalability under load:

| Test | Description |
|------|-------------|
| `test_scaling_at_concurrency_levels` | Throughput at 1, 5, 10, 25, 50, 100 concurrent requests |
| `test_sustained_load` | Performance over time under constant load |
| `test_queue_buildup_and_drain` | Behavior during traffic bursts |

## Expected Results

All benchmarks include assertions for reasonable performance:

| Metric | Expected |
|--------|----------|
| Single request overhead | < 10ms |
| Backend reservation cycle | < 1ms (1000μs) |
| State operations | < 0.5ms (500μs) |
| Burst handling (50 req) | < 5s total |
| p99 latency under load | < 100ms |

## Interpreting Results

Sample output from `test_scaling_at_concurrency_levels`:

```
--- Concurrent Scaling Results ---
 Concurrency |   Throughput |  Avg Latency
-------------|--------------|-------------
           1 |       2500/s |       0.40ms
           5 |       5800/s |       0.86ms
          10 |       7200/s |       1.39ms
          25 |       8500/s |       2.94ms
          50 |       9100/s |       5.49ms
         100 |       8900/s |      11.24ms
```

Key observations:
- **Throughput should increase** with concurrency (up to a point)
- **Latency will increase** with concurrency (expected due to queueing)
- **Throughput plateau** indicates optimal concurrency level
- **Throughput collapse** would indicate contention problems

## Note on Results Variability

Benchmark results depend on:
- Hardware (CPU, memory speed, disk I/O)
- System load during test
- Python version and async implementation

Run multiple times and look for consistent patterns rather than absolute numbers.
