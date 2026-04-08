# AutoField Examples

Application examples and training workflows built on top of the **autofield** library.

These examples demonstrate how to apply differentiable Field Calculus to real problems — from distance fields to flocking control and territory partitioning — with a focus on learning coordination parameters from data via imitation learning or objective functions.

## Prerequisites

Install the parent project with its dependencies:

```bash
# From the project root
uv sync --extras "cpu"   # or "cuda"
```

Every example can be run directly with:

```bash
uv run python examples/<path>.py [arguments]
```

## Architecture

Each example family follows a clean separation of concerns:

| Package | Responsibility |
|---------|---------------|
| `domain/` | Pure aggregate programs and geometry logic |
| `model/` | PyTorch modules parameterizing the aggregate programs |
| `training/` | Optimization loops, curriculum schedules, supervision |
| `visualization/` | Rendering tools for snapshots, trajectories, GIFs |
| `reporting/` | Summary builders, metrics, parameter recovery |

Shared utilities live in `shared/`:
- `shared.plotting` — 2D grid and moving node visualization
- `shared.diagnostics` — Training history export, CSV, metric plotting
- `shared.experiment` — Checkpoint management, visualization pipelines
- `shared.training` — Training utils, history tracking

---

## Example Catalog

### Gradients

Self-healing distance fields — learn hop-cost parameters and attention-based aggregators.

**Fixed distance field:**
```bash
uv run python examples/gradients/fixed.py
```

**Learnable weights:**
```bash
uv run python examples/gradients/learnable.py --rows 5 --cols 5 --epochs 200
```

**Attention-based aggregation:**
```bash
uv run python examples/gradients/attention.py
```

**Moving nodes:**
```bash
uv run python examples/gradients/moving_nodes.py
```

**Moving nodes (learnable):**
```bash
uv run python examples/gradients/moving_nodes_learnable.py
```

**Moving source:**
```bash
uv run python examples/gradients/moving_source.py
```

---

### Boids

Flocking control — learn separation, alignment, and cohesion weights by imitating a teacher simulation.

**Pure aggregate boids (no learning):**
```bash
uv run python examples/boids/simple.py --rounds 80 --num-nodes 60 --radius 0.23
```

**Learnable boids:**
```bash
uv run python examples/boids/learnable.py --epochs 40 --rounds 24 --num-nodes 24
```

**Multi-seed evaluation:**
```bash
uv run python examples/boids/evaluation.py \
  --out-dir generated/results/evaluation \
  --epochs 40 --rounds 24 --num-nodes 24 \
  --seeds 11,13,17,19,23 --eval-seeds 101,103,107 --skip-viz
```

---

### Territories

Multi-sink territory partitioning — learn cost weights and local surcharge policies (including neural network policies) for optimal territory formation on grid-based scenarios.

**Learnable territories:**
```bash
uv run python examples/territories/learnable.py --epochs 120 --rows 18 --cols 18
```

**Evaluation:**
```bash
uv run python examples/territories/evaluation.py
```

---

### Collects

Collect/gradient-cast building blocks — visual demonstrations of payload collection on spatial layouts.

**Large spatial collect:**
```bash
uv run python examples/collects/large.py --rows 40 --cols 40
```

**Fully connected topology:**
```bash
uv run python examples/collects/large.py --rows 40 --cols 40 --topology full
```

**k-NN topology:**
```bash
uv run python examples/collects/large.py --rows 40 --cols 40 --topology knn --k-neighbors 8
```

**Small collect:**
```bash
uv run python examples/collects/small.py
```

---

### Channel

Shortest-path channel examples — combine multiple fields to route around obstacles.

**Small channel:**
```bash
uv run python examples/channel/small.py
```

**Large channel:**
```bash
uv run python examples/channel/large.py
```

**Channel visualization:**
```bash
uv run python examples/channel/viz.py
```

---

## Artifacts

Results (summaries, plots, GIFs, checkpoints) are saved by default in the `generated/` directory relative to the project root.
