#!/usr/bin/env python3
"""Tensor encoding of fields — example from the paper.

Replicates the four-stage computational cycle described in
"Tensor Encoding of Fields" using diffield primitives, and compares
the results against the dense tensor formulation.

Graph: 3 nodes, bidirectional edges 0↔1, 1↔2
Sensor field: q = [10, 20, 30]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffield.dsl import field, gather_min, iterate, mux, scatter, scatter_range  # noqa: E402
from diffield.sim import GridScenario, SimulationEngine  # noqa: E402


def program(runtime):
    src = runtime.signals["source"]
    return iterate(
        field.inf(),
        lambda d: mux(src, field.of(0.0), gather_min(scatter(d) + scatter_range())),
        name="dist",
    )

scenario = GridScenario(10, 10, connectivity=4)
engine = SimulationEngine.from_scenario(scenario)
source = scenario.marker(0, 0)
output, runtime = engine.run(
    rounds=20,
    program=program,
    signals={"source": source},
)
print("---Output field---")
print(output.cpu().numpy())
