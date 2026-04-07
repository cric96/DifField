# Autofield Test Structure

This document describes the organization and philosophy of the test suite for the `differentiable-field-calculus` (Autofield) repository, refactored to adhere to the highest software standards and Test-Driven Development (TDD) principles.

## General Organization

Tests are logically divided based on the type of code being verified. We use **pytest** as the primary test runner and **hypothesis** for property-based testing. Tests are automatically tagged by pytest:

- **Unit tests** (`pytest -m unit`): Located in the `tests/` directory, they verify the behavior of individual modules or framework components.
- **Integration tests** (`pytest -m integration`): Located within `examples/` to ensure the framework works correctly on concrete use cases and complete training pipelines (boids, gradients, territories).

## `tests/` Directory Structure

1. `test_dsl_primitives.py`: Contains tests for aggregate computing primitives (`rep`, `nbr`, `branch`, `mux`). The tests verify branch isolation, message passing, and the differentiability of all execution paths.
2. `test_device_context.py`: Contains unit tests for decentralized (node-centric) execution, verifying that single-device execution exactly reproduces the state calculated globally.
3. `test_building_blocks.py`: Exercises and ensures the functionality of high-level DSL layers such as `gradient_cast` and `collect_cast`.
4. `test_layers_and_composition.py`: Verifies the Object-Oriented API based on `nn.Module` (e.g., `NbrLayer`, `RepLayer`) and their correct interfacing with the functional DSL.
5. `test_functional_api.py`: Contains tests for the bare functional backend, such as PyG scatter operators and property-based testing on vectorized functions.
6. `test_spatial_sim.py`: Specific to continuous spatial simulation domains (kinematics, boundaries, mask generators, boids, radius graphs).
7. `test_runtime_public_api.py`: Tests the simulation engine, integration of timed event scheduling, and snapshot data archiving.

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