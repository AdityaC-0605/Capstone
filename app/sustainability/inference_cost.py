import time
import numpy as np

def measure_block_latency(fn, X, runs=50, warmup=5):
    times = []

    for _ in range(warmup):
        fn(X)

    for _ in range(runs):
        start = time.perf_counter()
        fn(X)
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean_ms": float(np.mean(times) * 1000),
        "p95_ms": float(np.percentile(times, 95) * 1000)
    }
