from __future__ import annotations

import argparse
import time

import numpy as np
import tensorflow as tf

from vnet.inference import load_vnet, predict


def generate_volume(shape: tuple[int, int, int]) -> np.ndarray:
    """Generate a random 3D volume for benchmarking."""
    d, h, w = shape
    return np.random.rand(d, h, w, 1).astype(np.float32)


def benchmark(
    model: tf.keras.Model,
    volume: np.ndarray,
    warmup: int = 3,
    runs: int = 10,
) -> None:
    """Benchmark inference latency and throughput."""
    # Warmup
    for _ in range(warmup):
        _ = predict(model, volume)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = predict(model, volume)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg = sum(times) / len(times)
    p50 = np.percentile(times, 50)
    p90 = np.percentile(times, 90)
    p99 = np.percentile(times, 99)

    print("\n=== VNet Inference Benchmark ===")
    print(f"Volume shape: {volume.shape}")
    print(f"Runs: {runs}")
    print(f"Average latency: {avg*1000:.2f} ms")
    print(f"P50 latency:    {p50*1000:.2f} ms")
    print(f"P90 latency:    {p90*1000:.2f} ms")
    print(f"P99 latency:    {p99*1000:.2f} ms")
    print(f"Throughput:     {1/avg:.2f} volumes/sec\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark VNet inference performance")
    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        default=[64, 64, 64],
        help="Volume shape: D H W",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of timed runs",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup iterations",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Load model with dynamic spatial dims
    model = load_vnet(
        weights_path=None,
        input_shape=(None, None, None, 1),
        num_classes=1,
    )

    print("Device:", tf.config.list_physical_devices("GPU"))

    volume = generate_volume(tuple(args.shape))
    benchmark(model, volume, warmup=args.warmup, runs=args.runs)


if __name__ == "__main__":
    main()
