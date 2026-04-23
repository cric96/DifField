#!/bin/bash
set -e

echo "=== Running CUDA Benchmarks ==="
uv sync --extra cuda
uv run python examples/channel/benchmark_scaling.py \
  --node-counts 1000,2000,4000,8000,16000,32000 \
  --k-neighbors 4,8,12,16,32 \
  --repetitions 32 \
  --rounds 100 \
  --out-dir generated/results/channel_scaling_multi_cuda

echo "=== Running CPU Benchmarks ==="
uv sync --extra cpu
uv run python examples/channel/benchmark_scaling.py \
  --node-counts 1000,2000,4000,8000,16000,32000 \
  --k-neighbors 4,8,12,16,32 \
  --repetitions 32 \
  --rounds 100 \
  --out-dir generated/results/channel_scaling_multi_cpu

echo "=== All benchmarks completed ==="
