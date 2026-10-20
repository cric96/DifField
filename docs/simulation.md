# Simulation

`diffield.sim` runs a program for many rounds on a network: a **scenario** holds the
topology, the **engine** steps the rounds, an **event schedule** changes the world
between them, and a **recorder** keeps what you want to look at.

## Scenarios

| Scenario | Topology |
|----------|----------|
| `GridScenario(rows, cols, connectivity=4)` | A grid, with 4 or 8 neighbours |
| `SpatialScenario(positions, edge_radius=...)` | Devices in the plane, linked within a radius (or to their `k_neighbors`) |
| `RelaxedRadiusScenario(positions, edge_radius=...)` | Fully connected, links longer than the radius cost a smooth penalty: for learning connectivity |
| `FullyConnectedScenario(num_nodes)` | Everyone with everyone |

Scenarios build the fields a program needs: `zeros()`, `full(value)`, `marker(...)` for
a single node, `mask_from_predicate` / `mask_from_positions` for regions. Spatial
scenarios move: `step_positions(velocities)` and `update_positions(positions)` rebuild
the links, and the engine hands the new graph to the program before the next round.

### Link lengths

On a spatial scenario a link can weigh one hop or its length. With
`edge_weight_mode="distance"`, `scatter_range()` is the Euclidean length of each link,
so `gradient` measures distances coherent with the positions:

```python
from diffield.dsl import gradient
from diffield.sim import SpatialScenario

scenario = SpatialScenario(positions=positions, edge_radius=0.2, edge_weight_mode="distance")

def program(runtime):
    return gradient(runtime.signals["source"], name="dist")
```

A custom cost is a link field: `gradient(source, weight=scatter(cost) + scatter_range())`,
and the same `weight` goes to `gradient_cast`, `broadcast` and `collect_cast`.

## The engine

```python
from diffield.sim import GridScenario, SimulationEngine

scenario = GridScenario(10, 10, connectivity=4)
engine = SimulationEngine.from_scenario(scenario)
output, runtime = engine.run(rounds=40, program=program, signals={"source": scenario.marker(0, 0)})
```

`run` returns the output of the last round and the `runtime`: its `signals` (the inputs
of the program), its `metadata` (anything the events need) and the round number. The
state of the program lives in `engine.ctx` and survives between calls, so a second `run`
continues the first; `engine.ctx.reset()` starts again from round zero. `step` runs a
single round, for loops of your own.

## Events and recording

An `EventSchedule` runs callbacks before given rounds: here the source jumps to the
opposite corner halfway, and the distances heal. A `SnapshotRecorder` keeps states,
exports and outputs at the rounds you ask for.

```python
from diffield.dsl import field, gather_min, iterate, mux, scatter
from diffield.sim import EventSchedule, ScheduledEvent, SnapshotRecorder

def move_source(runtime):
    runtime.signals["source"] = scenario.marker(9, 9)

schedule = EventSchedule([ScheduledEvent(round_idx=20, callback=move_source, name="move")])
recorder = SnapshotRecorder(state_fields=["dist"], capture_output=True, record_rounds={0, 20, 39})

def program(runtime):
    source = runtime.signals["source"]
    return iterate(
        field.inf(),
        lambda d: mux(source, field.of(0.0), gather_min(scatter(d + 1.0))),
        name="dist",
    )

output, runtime = engine.run(
    rounds=40,
    program=program,
    signals={"source": scenario.marker(0, 0)},
    schedule=schedule,
    recorder=recorder,
)
recorder.records[20]["output"]   # the field when the source moved
```

## Moving networks without a scenario

Outside the engine, `AggregateContext.update_topology(edge_index, positions=...)` swaps
the graph between rounds while the nodes keep their state: links appear and disappear,
and stateful fields (`gradient`, `elect`, `iterate`) heal instead of restarting. The
VMAS experiments in `examples/vmas_diffield/` drive their agents this way, one physics
step per round.
