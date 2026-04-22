# Architecture & Design

High-level conceptual model of DifField — focusing on **what** each component represents and **how** they compose, not on implementation details.

---

## System Overview

DifField implements a **field calculus** runtime on top of PyTorch. The core idea is that programs are written as **field operations** — computations that happen simultaneously across all nodes of a graph — and are executed through rounds of message passing.

```mermaid
graph TB
    subgraph User["User Code"]
        P[Aggregate Program]
    end

    subgraph DSL["DSL Layer"]
        R[iterate]
        N[scatter]
        B[branch]
        M[mux]
        G[gradient]
        C[collect_cast]
    end

    subgraph Layers["PyTorch Layers"]
        RL[IterateLayer]
        NL[GatherLayer]
        BL[BranchLayer]
        ML[MuxLayer]
    end

    subgraph Functional["Functional Primitives"]
        SA[scatter_aggr]
        SW[soft_where]
        MK[masking]
        FL[folding]
    end

    subgraph Core["Core Runtime"]
        AC[AggregateContext]
        SM[StateManager]
        RC[RoundContext]
    end

    subgraph Sim["Simulation Layer"]
        SC[Scenario]
        SE[SimulationEngine]
        EV[EventSchedule]
        SR[SnapshotRecorder]
    end

    P --> DSL
    R --> RL
    N --> NL
    B --> BL
    M --> ML
    G --> R
    G --> N
    C --> R
    C --> N
    RL --> SA
    NL --> SA
    BL --> SW
    ML --> SW
    SA --> AC
    SW --> AC
    AC --> SM
    AC --> RC
    SE --> SC
    SE --> AC
    SE --> EV
    SE --> SR
```

---

## Layer Architecture

### 1. Core Runtime

The foundation — manages graph topology, per-node state, and round execution.

```mermaid
classDiagram
    class AggregateContext {
        +edge_index: Tensor
        +num_nodes: int
        +edge_weight: Tensor?
        +round() ContextManager
        +reset()
    }

    class RoundContext {
        +edge_index: Tensor
        +edge_weight: Tensor
        +num_nodes: int
        +round_num: int
        +state: StateManager
        +exports: dict
        +round() ContextManager
        +reset()
    }

    class StateManager {
        +num_nodes: int
        +get_or_init(name, init) Tensor
        +update(name, value)
        +track_branch(name, cond) mask
        +reset_states_for_nodes(mask, inits)
        +snapshot() dict
        +restore(snapshot)
    }

    class DeviceContext {
        +num_neighbors: int
        +round(neighbor_exports, neighbor_messages, neighbor_ranges)
        +local_field(own, scatter) Tensor
        +result(tensor) Tensor
        +get_state(name) float
    }

    AggregateContext *-- RoundContext
    RoundContext *-- StateManager
    DeviceContext --> AggregateContext : wraps for local execution
```

**Concepts:**

- **AggregateContext** — the global execution environment for a graph. One context per simulation.
- **RoundContext** — the per-round snapshot of topology, state, and exports. Created fresh each round.
- **StateManager** — persistent per-node memory across rounds. Handles `iterate` state, branch-switch resets, and snapshots.
- **DeviceContext** — a local view of execution from a single device's perspective, useful for debugging and didactic purposes.

### 2. Functional Utilities

Low-level differentiable building blocks — scatter operations, soft conditionals, masking, and folding.

```mermaid
graph LR
    subgraph Aggregation
        SA[scatter_aggr] --> |"sum, min, mean, max"| WV[weighted variants]
    end

    subgraph Conditionals
        SW[soft_where] --> |hard| BS[boolean selection]
        SW --> |soft| DB[differentiable blending]
    end

    subgraph Masking
        ME[mask_edges] --> FEC[filter edges by condition]
        MEP[mask_edges_for_partition] --> PAM[partition-aware masking]
    end

    subgraph Folding
        SBF[scatter_binary_fold] --> ABO[accumulate with binary op]
        SFM[scatter_min_by_first] --> LM[lexicographic minimum]
    end
```

**Concepts:**

- **scatter_aggr** — differentiable neighborhood aggregation with multiple modes (hard/soft).
- **soft_where** — differentiable conditional: `soft_where(cond, a, b)` blends between `a` and `b` based on `cond`.
- **mask_edges** — filter edges based on node/edge conditions.
- **scatter_binary_fold** — fold values from neighbors using a binary accumulation function.

### 3. PyTorch Layers

`nn.Module` wrappers that make DSL primitives compatible with PyTorch's autograd and parameter management.

```mermaid
classDiagram
    class IterateLayer {
        +name: str
        +init: float
        +fn: Callable
        +forward(x) Tensor
    }

    class GatherLayer {
        +aggr: str
        +mode: str (hard|soft)
        +tau: float
        +forward(expr) Tensor
    }

    class BranchLayer {
        +true_mod: Module
        +false_mod: Module
        +branch_name: str
        +mode: str (hard|soft)
        +forward(x, cond) Tensor
    }

    class MuxLayer {
        +forward(cond, if_true, if_false) Tensor
    }

    note for IterateLayer "Per-node recurrent state with branch-aware reset"
    note for GatherLayer "Neighborhood message passing with aggregation"
    note for BranchLayer "Domain restriction with communication isolation"
    note for MuxLayer "Pointwise conditional without topology change"
```

**Concepts:**

- **IterateLayer** — wraps `iterate`: manages named recurrent state, automatically resets on branch switches.
- **GatherLayer** — wraps `gather`: sends messages along edges, aggregates with configurable strategy.
- **BranchLayer** — wraps `branch`: splits execution domain, isolates communication between branches.
- **MuxLayer** — wraps `mux`: pointwise selection without affecting communication topology.

### 4. DSL Primitives

The user-facing API — composable field operators.

```mermaid
graph TB
    subgraph Core["Core Primitives"]
        iterate["iterate(init, fn, name)"]
        scatter["scatter(expr, aggr)"]
        branch["branch(cond, if_true, if_false)"]
        mux["mux(cond, if_true, if_false)"]
    end

    subgraph Blocks["Building Blocks"]
        gradient["gradient(source)"]
        gradient_cast["gradient_cast(source, center, acc)"]
        broadcast["broadcast(mask, value)"]
        collect_cast["collect_cast(potential, local, null, acc)"]
    end

    subgraph Helpers["Field Helpers"]
        field_of["field.of(value)"]
        field_zeros["field.zeros()"]
        field_ones["field.ones()"]
        field_inf["field.inf()"]
        const["const(value)"]
        mid["mid()"]
        scatter_range["scatter_range()"]
    end

    gradient --> iterate
    gradient --> scatter
    gradient --> mux

    gradient_cast --> iterate
    gradient_cast --> scatter

    broadcast --> gradient_cast

    collect_cast --> iterate
    collect_cast --> scatter
```

**Conceptual semantics:**

| Operator | Meaning |
|----------|---------|
| `iterate` | "Remember this value across rounds, evolving it with `fn`" |
| `scatter` | "Send this expression to neighbors and aggregate their messages" |
| `branch` | "Split the domain: nodes where `cond` is true run `if_true`, others run `if_false`, with no communication between branches" |
| `mux` | "Select between two values per-node, without changing communication" |
| `gradient` | "Compute minimum-cost distance from source nodes" |
| `gradient_cast` | "Route payloads along gradient paths toward sources" |
| `broadcast` | "Spread a value from root nodes to the rest of the network" |
| `collect_cast` | "Gather payloads from children toward local minima of a potential field" |

### 5. Simulation Layer

Orchestrates multi-round executions with topology management, events, and recording.

```mermaid
classDiagram
    class Scenario {
        <<interface>>
        +edge_index: Tensor
        +edge_weight: Tensor
        +num_nodes: int
        +sync_context(round_ctx)
        +zeros() Tensor
        +full(value) Tensor
        +marker(idx, value) Tensor
    }

    class GridScenario {
        +rows: int
        +cols: int
        +connectivity: int
        +pos_to_idx(row, col) int
        +marker(row, col, value) Tensor
        +mask_from_positions(positions) Tensor
        +mask_from_predicate(predicate) Tensor
    }

    class SpatialScenario {
        +positions: Tensor
        +edge_radius: float?
        +k_neighbors: int?
        +edge_weight_mode: str
        +update_positions(positions)
        +step_positions(velocities, dt)
        +refresh_topology()
    }

    class FullyConnectedScenario {
        +num_nodes: int
    }

    class RelaxedRadiusScenario {
        +positions: Tensor
        +edge_radius: Tensor
        +relaxation_tau: float
        +penalty_strength: float
    }

    class SimulationEngine {
        +ctx: AggregateContext
        +num_nodes: int
        +from_scenario(scenario) Engine
        +run(rounds, program, signals) output, runtime
        +step(runtime, program) output
    }

    class SimulationRuntime {
        +scenario: Scenario?
        +signals: dict
        +metadata: dict
        +round_idx: int
    }

    class EventSchedule {
        +add(event)
        +at(round_idx) list
        +apply(round_idx, runtime)
    }

    class ScheduledEvent {
        +round_idx: int
        +callback(runtime)
        +name: str
    }

    class SnapshotRecorder {
        +state_fields: list
        +capture_output: bool
        +record_rounds: set?
        +record(round_idx, ctx, output)
        +snapshots: dict
    }

    Scenario <|.. GridScenario
    Scenario <|.. SpatialScenario
    Scenario <|.. FullyConnectedScenario
    Scenario <|.. RelaxedRadiusScenario

    SimulationEngine --> AggregateContext : owns
    SimulationEngine --> Scenario : optional
    SimulationEngine --> SimulationRuntime : creates
    SimulationEngine --> EventSchedule : optional
    SimulationEngine --> SnapshotRecorder : optional

    EventSchedule *-- ScheduledEvent
    SimulationRuntime --> Scenario : reference
```

**Concepts:**

- **Scenario** — defines the graph topology (nodes, edges, weights) and provides helper methods for creating fields, markers, and masks. Different scenarios model different spatial abstractions.
- **SimulationEngine** — the main execution loop. Runs a program for N rounds, optionally applying events and recording snapshots.
- **SimulationRuntime** — mutable state shared between the engine, events, and the program. Carries signals, metadata, and the current round index.
- **EventSchedule** — a registry of callbacks triggered at specific rounds. Enables dynamic behavior (moving nodes, changing sources, topology updates).
- **SnapshotRecorder** — captures the state of fields, exports, and outputs at specified rounds for later analysis.

---

## Execution Lifecycle

How a simulation executes, round by round:

```mermaid
sequenceDiagram
    participant User
    participant Engine as SimulationEngine
    participant Schedule as EventSchedule
    participant Scenario
    participant ACtx as AggregateContext
    participant Program
    participant Recorder as SnapshotRecorder

    User->>Engine: run(rounds, program, signals, schedule, recorder)
    Engine->>Engine: init_runtime(signals)

    loop for each round
        Engine->>Schedule: apply(round_idx, runtime)
        Schedule-->>Engine: events mutate runtime.signals

        Engine->>Scenario: sync_context(round_ctx)
        Scenario-->>Engine: topology pushed to context

        Engine->>ACtx: round()
        ACtx->>Program: program(runtime)
        Program->>Program: execute DSL primitives
        Program-->>ACtx: output field
        ACtx-->>Engine: round complete

        Engine->>Recorder: record(round_idx, ctx, output)
        Recorder-->>Engine: snapshot stored

        Engine->>Engine: round_idx += 1
    end

    Engine-->>User: output, runtime
```

### Round Execution Detail

Within a single round, the DSL primitives compose as follows:

```mermaid
sequenceDiagram
    participant DSL as DSL Program
    participant Iterate as IterateLayer
    participant Gather as GatherLayer
    participant Branch as BranchLayer
    participant Func as Functional
    participant State as StateManager

    DSL->>Iterate: iterate(init, fn, name)
    Iterate->>State: get_or_init(init, name)
    State-->>Iterate: current_state

    Iterate->>DSL: fn(current_state)

    DSL->>Gather: gather_min(scatter(expr))
    Gather->>Func: scatter_aggr(messages, aggr)
    Func-->>Gather: aggregated values
    Gather-->>DSL: neighbor contribution

    DSL->>Branch: branch(cond, if_true, if_false)
    Branch->>State: track_branch(name, cond)
    State-->>Branch: switched_mask
    Branch->>State: reset_states_for_nodes(mask, resets)

    alt hard mode
        Branch->>DSL: execute if_true or if_false per node
    else soft mode
        Branch->>Func: soft_where(cond, true_result, false_result)
        Func-->>Branch: blended result
    end

    Branch-->>DSL: branch result
    DSL-->>Iterate: new_state
    Iterate->>State: update(name, new_state)
```

---

## Conceptual Model: Fields and Aggregate Computing

### What is a Field?

A **field** is a function from nodes to values. Every DSL expression produces a field — a tensor of shape `(num_nodes,)` or `(num_nodes, d)`.

```mermaid
graph TB
    subgraph "Field Types"
        Scalar["Scalar Field\nshape: (N,)"]
        Vector["Vector Field\nshape: (N, d)"]
        Bool["Boolean Field\nshape: (N,) dtype: bool"]
    end

    subgraph "Field Operations"
        Local["Local Ops\npointwise arithmetic"]
        Neighbor["Neighbor Ops\nscatter() + aggregation"]
        Temporal["Temporal Ops\niterate() + state evolution"]
        Conditional["Conditional Ops\nmux(), branch()"]
    end

    Scalar --> Local
    Scalar --> Neighbor
    Scalar --> Temporal
    Scalar --> Conditional
    Vector --> Local
    Vector --> Neighbor
    Bool --> Conditional
```

### Aggregate Computing Paradigm

DifField implements the **aggregate computing** model:

```mermaid
graph LR
    subgraph "Aggregate Computing Model"
        Devices[Devices / Nodes]
        Neighbors[Neighbor Relations]
        LocalComp[Local Computation]
        NeighborComm[Neighbor Communication]
        State[State Evolution]
    end

    Devices --> Neighbors
    Devices --> LocalComp
    Neighbors --> NeighborComm
    LocalComp --> State
    NeighborComm --> State
    State --> LocalComp
```

1. **Devices** are nodes in a graph
2. **Neighbor relations** define who can communicate with whom
3. **Local computation** happens independently on each device
4. **Neighbor communication** exchanges messages along edges
5. **State evolution** (`iterate`) carries information across rounds

### Hard vs Soft Semantics

Every operator supports two modes:

| Mode | Behavior | Use Case |
|------|----------|----------|
| **Hard** | Discrete selection, exact aggregation | Classical aggregate computing, exact algorithms |
| **Soft** | Differentiable blending, smooth aggregation | Gradient-based learning, optimization |

```mermaid
graph TB
    subgraph "Hard Mode"
        H1[branch: boolean split]
        H2[scatter: exact min/sum/max]
        H3[mux: exact selection]
    end

    subgraph "Soft Mode"
        S1[branch: sigmoid-weighted blend]
        S2[scatter: softmax-weighted aggregation]
        S3[mux: soft_where with tau]
    end

    H1 -. tau .-> S1
    H2 -. tau .-> S2
    H3 -. tau .-> S3

    N1["tau → 0 recovers hard semantics"]
    N2["tau controls smoothness"]
```

---

## Scenario Hierarchy

Scenarios model different spatial abstractions:

```mermaid
graph TB
    subgraph "Topology Types"
        Grid["GridScenario\nRegular grid, 4/8 connectivity"]
        Spatial["SpatialScenario\n2D positions, radius/k-NN graph"]
        Full["FullyConnectedScenario\nComplete graph"]
        Relaxed["RelaxedRadiusScenario\nSmooth radius penalty"]
    end

    subgraph "Use Cases"
        UC1["Fixed topology\n(Grid, Full)"]
        UC2["Dynamic topology\n(Spatial)"]
        UC3["Learnable topology\n(Relaxed)"]
    end

    Grid --> UC1
    Full --> UC1
    Spatial --> UC2
    Relaxed --> UC3
```

| Scenario | Topology | Dynamic | Learnable | Best For |
|----------|----------|---------|-----------|----------|
| `GridScenario` | Fixed grid | No | No | Structured spatial domains |
| `FullyConnectedScenario` | Complete graph | No | No | Small-scale, all-to-all |
| `SpatialScenario` | Radius / k-NN | Yes | No | Moving nodes, dynamic graphs |
| `RelaxedRadiusScenario` | Fully connected + penalty | Yes | Yes | Differentiable connectivity learning |

---

## DSL Composition Patterns

Complex behaviors emerge from composing primitives:

### Gradient Pattern

```
gradient(source) = iterate(field.inf(), λd. mux(source, field.of(0), gather_min(scatter(d) + step)), name="dist")
```

```mermaid
graph LR
    Source[source field] --> Mux
    Iterate[iterate state d] --> Add[d + step]
    Add --> Gather[scatter aggr=min]
    Gather --> Mux[mux]
    Mux --> Iterate
    Iterate --> Output[distance field]
```

### Collect-Cast Pattern

```
collect_cast(potential, local, null, acc) = iterate("collect", local, λc. acc(local, collect_from_children(c)))
```

```mermaid
graph TB
    Potential[potential field] --> Parent[find parent nodes]
    Local[local payload] --> Acc
    Parent --> Filter[filter children edges]
    Filter --> Gather[gather child values]
    Gather --> Acc[accumulate]
    Acc --> Iterate[iterate state]
    Iterate --> Output[collected field]
```

### Broadcast Pattern

```
broadcast(mask, value) = gradient_cast(source=mask, center=value, accumulation=identity)
```

```mermaid
graph LR
    Mask[root mask] --> GC[gradient_cast]
    Value[source value] --> GC
    GC --> Output[spread value from roots]
```

---

## Key Design Decisions

1. **PyTorch-native** — All operations are differentiable; the entire simulation can be optimized end-to-end.
2. **Hard/soft duality** — Every operator supports both discrete and smooth semantics, controlled by `mode` and `tau`.
3. **Round-based execution** — Computation happens in discrete rounds, enabling stateful programs and dynamic topology.
4. **Scenario abstraction** — Topology is decoupled from programs; the same program runs on any graph.
5. **Event-driven dynamics** — `EventSchedule` enables complex temporal behaviors without modifying the program.
6. **State isolation** — `branch` provides communication isolation between sub-domains, enabling spatial partitioning.
