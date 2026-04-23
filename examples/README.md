# DifField Examples

Application examples and training workflows built on top of the **diffield** library.

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


Shared utilities live in `shared/`:
- `shared.plotting` — 2D grid and moving node visualization
- `shared.diagnostics` — Training history export, CSV, metric plotting
- `shared.experiment` — Checkpoint management, visualization pipelines
- `shared.training` — Training utils, history tracking
