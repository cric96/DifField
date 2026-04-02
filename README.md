## aggregate-gnn-equivalence

Aggregate Computing (AC) and Message Passing Neural Networks (MPNN) equivalence experiments and reference implementations.

### Install
If you have an nvdia gpu, then write
```bash
uv sync --extras "cuda"
```
If you have a rocm (amd) gpu, then write
```bash
uv sync --extras "rocm"
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

- `examples/boids/learnable.py` (aggregate `rep`/`nbr` boids with learnable weights + attention)
- `examples/boids/simple.py` (pure aggregate boids using only `rep` + `nbr` dynamics)
- `examples/boids/evaluation.py`
- `examples/boids/tune_optuna.py`
- `examples/gradients/fixed.py`
- `examples/gradients/learnable.py`
- `examples/gradients/attention.py`
- `examples/gradients/large.py`
- `examples/gradients/local.py`
- `examples/gradients/moving_nodes.py`
- `examples/gradients/moving_nodes_learnable.py`
- `examples/gradients/moving_source.py`
- `examples/channel/small.py`
- `examples/channel/large.py`
- `examples/channel/viz.py`

The examples tree is organized by family:

- `examples/boids/` contains reusable boids modules and focused entrypoints.
- `examples/gradients/` contains the reusable gradient examples.
- `examples/channel/` contains the reusable channel examples and visualization helpers.
- `examples/shared/` contains plotting and diagnostics helpers reused across families.
- The example entrypoints now live only inside these subfolders.

### Aggregate Boids (Learnable)

Run a short training session:

```bash
uv run python examples/boids/learnable.py --epochs 60 --rounds 30 --num-nodes 40 --mode joint
```

Export predicted snapshots + trajectories + gif:

```bash
uv run python examples/boids/learnable.py --epochs 60 --rounds 30 --num-nodes 40 --mode joint --viz-prefix examples/boids/learnable --gif-fps 8
```

This also exports a side-by-side predicted vs teacher panel at
`<viz-prefix>_compare_panel.png` (disable with `--no-compare-panel`).

Modes:

- `--mode weights`: learn only flocking/damping/speed parameters
- `--mode attention`: learn only neighbor attention aggregators
- `--mode joint`: learn both together

Teacher defaults: `w_sep=1.4`, `w_align=0.8`, `w_cohesion=0.6`, `damping=0.96`, `max_speed=0.014`.
Override with `--teacher-w-sep`, `--teacher-w-align`, `--teacher-w-cohesion`, `--teacher-damping`, `--teacher-max-speed`.

Connectivity defaults:

- `--init-connectivity hybrid` (default) builds a connected round-0 graph and enforces a minimum degree target.

### Boids Evaluation

Run a multi-seed evaluation across the three learnable modes (weights / attention / joint):

```bash
uv run python examples/boids/evaluation.py \
	--out-dir examples/results/evaluation \
	--epochs 120 \
	--rounds 40 \
	--num-nodes 40 \
	--seeds 11,13,17,19,23 \
	--eval-seeds 101,103,107 \
	--skip-viz
```

Then optionally tune learning rate and regularization with Optuna:

```bash
uv run python examples/boids/tune_optuna.py \
	--out-dir examples/results/optuna \
	--trials 40 \
	--epochs 120 \
	--rounds 40 \
	--num-nodes 40 \
	--eval-seeds 101,103,107 \
	--eval-every 10 \
	--skip-viz
```

Useful artifacts:

- per-run `summary.json` with parameter recovery and rollout connectivity health
- `evaluation_summary.csv` with mean±std per mode
- `best_trial.json`, `trials.csv`, `param_importance.json`

### Pure Aggregate Boids

Run fixed boids dynamics expressed only with aggregate state + neighborhood operators:

```bash
uv run python examples/boids/simple.py --rounds 80 --num-nodes 60 --radius 0.23
```

Export snapshots + trajectories + gif:

```bash
uv run python examples/boids/simple.py --rounds 80 --num-nodes 60 --radius 0.23 --viz-prefix examples/boids/simple --gif-fps 8
```

### Movement Physics Utilities

Shared movement helpers now live in `src/aggregate_gnn/sim/physics.py` and are re-exported by `aggregate_gnn`:

- `normalize_vectors`
- `limit_speed`
- `bounce_in_box`
- `boids_acceleration_dense`

### Notes

- Existing DSL semantics (`rep`, `nbr`, `branch`, `mux`) are unchanged.
- Plotting remains optional and example-level.
- `docs/equivalence.md` contains the formal AC↔GNN treatment.
