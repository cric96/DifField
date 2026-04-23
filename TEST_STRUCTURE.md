# DifField Test Structure

## Overview

Tests are organized by **testing level** and **domain**. We use **pytest** as the test runner and **hypothesis** for property-based testing.

## Test Levels

| Level | Location | Marker | Purpose |
|-------|----------|--------|---------|
| Unit | `tests/` | `unit` | Verify individual modules and framework components in isolation |
| Integration | `examples/` | `integration` | Validate complete training pipelines and real-world use cases |

## Unit Test Domains

The `tests/` directory is split into subpackages, each covering a distinct layer of the framework:

| Domain | What It Covers |
|--------|----------------|
| **aggregate** | DSL primitives, layers, composition, control flow, autodiff, field helpers, and naming conventions |
| **device_context** | Decentralized (node-centric) execution — ensures single-device execution matches global computation |
| **functional** | Low-level functional backend including PyG scatter operators |
| **spatial** | Continuous spatial simulation: kinematics, boundaries, boids, radius graphs |
| **runtime** | Simulation engine, event scheduling, and snapshot archiving |

## Property-Based Testing

Critical vectorized functions (e.g., `scatter_aggr`) are tested with **hypothesis**, which generates thousands of random inputs and verifies that optimized computations match naive pure-Python reference implementations.

## Execution

```bash
# Run all tests
uv run pytest

# Run only unit tests
uv run pytest -m unit

# Run only integration tests
uv run pytest -m integration

# Skip slow tests
uv run pytest -m "not slow"
```
