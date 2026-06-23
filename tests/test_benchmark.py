from __future__ import annotations

import numpy as np

from bench.benchmark_vnet import benchmark
from vnet.inference import load_vnet


def test_benchmark_runs(capsys):
    model = load_vnet(
        weights_path=None,
        input_shape=(None, None, None, 1),
        num_classes=1,
    )

    volume = np.zeros((16, 16, 16, 1), dtype=np.float32)

    benchmark(model, volume, warmup=1, runs=3)

    out = capsys.readouterr().out

    assert "VNet Inference Benchmark" in out
    assert "Average latency" in out
    assert "Throughput" in out
