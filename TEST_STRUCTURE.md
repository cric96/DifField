# Autofield Test Structure

This document describes the organization and philosophy of the test suite for the `differentiable-field-calculus` (Autofield) repository, refactored to adhere to the highest software standards and Test-Driven Development (TDD) principles.

## General Organization

Tests are logically divided based on the type of code being verified. We use **pytest** as the primary test runner and **hypothesis** for property-based testing. Tests are automatically tagged by pytest:

- **Unit tests** (`pytest -m unit`): Located in the `tests/` directory, they verify the behavior of individual modules or framework components.
- **Integration tests** (`pytest -m integration`): Located within `examples/` to ensure the framework works correctly on concrete use cases and complete training pipelines (boids, gradients, territories).

## `tests/` Directory Structure

Tests are grouped by the layer they exercise.

**`tests/aggregate/`** — the DSL itself.

- `test_alignment.py`: occurrence identity. Two uses of the same block, two uses
  of the same reduction, and repeated calls on one source line must each get
  their own store slot; branch partitions must stay disjoint and contained;
  keys must be identical across processes.
- `test_core_primitives.py`: `iterate`, `scatter`, `gather` and their reductions.
- `test_control_flow.py`: `branch` and `mux`, including partition isolation and
  the hard/soft correspondence.
- `test_building_blocks.py`: `gradient`, `gradient_cast`, `broadcast`, `collect_cast`.
- `test_election.py`: `elect` (leader uniqueness, grain spacing, re-election after a
  leader dies), `descend`, and the neighbour-degree helpers.
- `test_composition.py`: multi-primitive programs and nesting.
- `test_implicit_naming.py`: constructs used without an explicit `name=`.
- `test_link_fields.py`: link-domain expressions and `include_self`.
- `test_field_helpers.py`: `field`, `const`, `mid`.
- `test_layers.py`: the `nn.Module` API (`GatherLayer`, `IterateLayer`, ...).
- `test_autodiff.py`: every path stays differentiable.

**`tests/device_context/`** — decentralized (node-centric) execution, verifying
that single-device execution reproduces the globally computed state for
programs within the local contract (see *Local execution* in
`docs/architecture.md`).

**`tests/functional/`** — the bare functional backend: PyG scatter operators and
property-based tests on the vectorized functions.

**`tests/spatial/`** — continuous spatial domains (kinematics, boundaries, mask
generators, boids, radius graphs).

**`tests/runtime/`** — the simulation engine, event scheduling, and snapshot
recording.

## Methodological Approach

### TDD and Quality
- **Isolation**: We use pytest fixtures located in `conftest.py` (e.g., `triangle_topology`, `line_topology`) to instantiate shared graphs, avoiding repetitive tests and boilerplate that mask the test's business logic. Fixtures keep tests DRY and clean.
- **Parameterization**: Where possible, we use `@pytest.mark.parametrize` to validate multiple input configurations through the same basic assertion.
- **Differentiability**: Much of the test suite evaluates that a pipeline produces a tensor whose `grad` is not `None` and is finite (`isfinite`), ensuring that no operation breaks the backpropagation chain.

### Border Case Handling
The suite verifies the framework's behavior in problematic scenarios. In `conftest.py`, we have standardized some types of degenerate graphs:
- `empty_topology`: A graph with zero nodes.
- `isolated_topology`: A graph consisting only of disconnected nodes.
- `disconnected_topology`: Multiple connected components sharing the same tensor to ensure messages do not erroneously "leak" between separate components.

### Property-Based Testing
Thanks to the integration of `hypothesis`, we apply property-based testing to critical functions (e.g., `scatter_aggr` in `test_functional_api.py`) by automatically generating thousands of random input instances, checking that the vectorized computation yields exactly the same result as a naive iterative loop in pure Python.

## Execution

```bash
# Run all tests (unit and integration)
uv run pytest 

# Run only unit tests
uv run pytest -m unit

# Run only integration tests
uv run pytest -m integration

# Run fast tests ignoring any slow modules (if 'slow' marks are present in the future)
uv run pytest -m "not slow"
```

## SEAMS experiment invariants

`tests/test_space_fluid.py` verifies lexicographic elections, strict cumulative
radius, delayed samples, partition/rejoin and leader removal, isolated nodes,
metric symmetry/positivity, oracle isolation and edge-order independence. It checks
exact hard training forwards, numerical derivatives of the continuous surrogate,
identical synchronous IDs, asynchronous message delivery, batch isolation,
deterministic checkpoint resumption, data splits and honest incomplete reports.

The 38 selected Boids/distributed regression tests remain in `test_boids.py`,
`test_boids_flocking.py` and `decentralized/test_mesa_equivalence.py`. The retired
sensor experiment tests were removed; general runtime tests were retained.
