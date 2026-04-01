## aggregate-gnn-equivalence

Aggregate Computing (AC) and Message Passing Neural Networks (MPNN) equivalence experiments and reference implementations.

### Install

```bash
uv sync
```

### Run Tests

```bash
uv run pytest
```

### Reusable Simulation Base

This project now includes a reusable simulation layer in `src/aggregate_gnn/sim` to avoid hand-crafted per-example execution loops.

- `GridScenario`: grid topology, markers, mask builders, index/position helpers.
- `SimulationEngine`: round stepping (`run` / `step`) on top of `AggregateContext`.
- `EventSchedule` + `ScheduledEvent`: deterministic round-based dynamic updates (movement, toggles, etc.).
- `SnapshotRecorder`: structured snapshot capture for states/exports/output.

You can import these from top-level `aggregate_gnn`.

### Simulation Example

```python
from aggregate_gnn import (
	GridScenario,
	SimulationEngine,
	EventSchedule,
	ScheduledEvent,
	SnapshotRecorder,
	rep,
	nbr,
	mux,
)
from aggregate_gnn.dsl import field

scenario = GridScenario(10, 10, connectivity=4)
engine = SimulationEngine.from_scenario(scenario)

source = scenario.marker(0, 0)

def move_source(runtime):
	runtime.signals["source"] = scenario.marker(9, 9)

schedule = EventSchedule([
	ScheduledEvent(round_idx=20, callback=move_source, name="move"),
])

recorder = SnapshotRecorder(state_fields=["dist"], capture_output=True, record_rounds={0, 20, 39})

def program(runtime):
	src = runtime.signals["source"]
	return rep("dist", float("inf"), lambda d: mux(src, field.of(0.0), nbr(d + 1.0, aggr="min")))

output, runtime = engine.run(
	rounds=40,
	program=program,
	signals={"source": source},
	schedule=schedule,
	recorder=recorder,
)
```

### Refactored Examples

- `examples/gradient_fixed.py`
- `examples/gradient_learnable.py`
- `examples/gradient_attention.py`
- `examples/gradient_large.py`
- `examples/gradient_local.py`
- `examples/channel.py`
- `examples/channel_large.py`
- `examples/gradient_moving_source.py` (new dynamic scenario demo)

### Notes

- Existing DSL semantics (`rep`, `nbr`, `branch`, `mux`) are unchanged.
- Plotting remains optional and example-level.
- `docs/equivalence.md` contains the formal AC↔GNN treatment.
