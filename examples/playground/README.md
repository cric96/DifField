# AutoField Playground

This directory contains an interactive Jupyter Notebook to step-by-step explore and play around with `autofield`.

## Running the Playground

1. Open the project in VS Code or any IDE that supports Jupyter Notebooks.
2. Open `examples/playground/playground.ipynb`.
3. Select your `.venv` as the active Python Kernel.
4. Run the cells sequentially!

## What you will learn
- How to setup a random `SpatialScenario` and access edge/position data.
- How to visualize a network topology.
- **How to inspect the inner message exchanges** across edges using the `full_repr()` / message grid views.
- How to write an aggregate program with primitives like `iterate`, `mux`, `scatter`, `gather_min`.
- How to execute it using `SimulationEngine.run()`.

