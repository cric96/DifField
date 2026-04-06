# Field Calculus Examples & Training Workflows

This directory contains examples of applying differentiable Field Calculus, focusing on learning coordination parameters from data (imitation learning) or objective functions.

## Overview

All examples utilize the `autofield` library to implement collective behaviors and `torch` for gradient-based optimization. The architecture follows a clean separation of concerns:
- `domain/`: Pure aggregate programs and geometry logic.
- `model/`: PyTorch modules parameterizing the aggregate programs.
- `training/`: Optimization loops, curriculum schedules, and supervision logic.
- `visualization/`: Rendering tools for snapshots, trajectories, and GIFs.

## Requirements

The project uses `uv` for dependency management. You can run any example directly using:
```bash
uv run examples/<example_name>/learnable.py [arguments]
```

## Example Catalog

### 1. Boids (Flocking Control)
Learn separation, alignment, and cohesion weights by imitating a teacher simulation.
- **Run Training**: 
  ```bash
  uv run examples/boids/learnable.py --epochs 100 --num-nodes 32
  ```
- **Evaluation**: Compare learned parameters against ground truth across multiple seeds.
  ```bash
  uv run examples/boids/evaluation.py --seeds "11,13,17"
  ```

### 2. Territories (Multi-Sink Partitioning)
Learn cost weights and local surcharge policies (via Neural Networks) for optimal territory formation on grid-based scenarios.
- **Run Training (Scalar mode)**:
  ```bash
  uv run examples/territories/learnable.py --mode scalars --epochs 120
  ```
- **Run Training (Hybrid Neural mode)**:
  ```bash
  uv run examples/territories/learnable.py --mode hybrid --epochs 120
  ```

### 3. Gradients (Self-Healing Distance Fields)
Optimize hop-cost parameters and attention-based aggregators for distance field estimation.
- **Learnable Weights**:
  ```bash
  uv run examples/gradients/learnable.py
  ```
- **Attention-based Aggregation**:
  ```bash
  uv run examples/gradients/attention.py
  ```
- **Moving Nodes Gradient**:
  ```bash
  uv run examples/gradients/moving_nodes_learnable.py
  ```

## Shared Utilities

Common logic for plotting, diagnostics, and experiment orchestration is located in the `shared/` package.
- `shared.plotting`: 2D grid and moving node visualization.
- `shared.diagnostics`: Training history export and metric plotting.
- `shared.experiment`: Checkpoint management and visualization pipelines.

## Artifacts

Results (summaries, plots, and GIFs) are saved by default in the `generated/` directory relative to the project root.
