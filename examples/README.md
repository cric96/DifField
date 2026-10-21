# DifField experiments

The experiments built on DifField: distance fields, channels, Boids, the SEAMS campaigns
and the VMAS controllers. They use only the public API of `diffield` and will move to a
repository of their own; until then, their tests run in their own CI job
(`uv run pytest examples`), apart from the library's.

Run commands from the repository root after `uv sync --extra cpu`. Add
`--extra decentralized` for Mesa examples or `--extra vmas` for VMAS experiments.

## SEAMS campaign

The [Space-Fluid suite](seams/README.md) compares fixed and offline learned
coordination metrics on moving spatial phenomena, with hard elections, frozen
checkpoints and batched/distributed execution. Boids remains a separate comparison
of the three-weight program with GNN32/GNN64 using observed states.

```bash
uv run python -m examples.seams space-fluid --profile smoke --stage all
uv run python -m examples.seams space-fluid --profile paper-cpu --stage all \
  --out generated/seams/space-fluid-paper-cpu
uv run python -m examples.seams boids --profile paper-cpu \
  --out generated/seams/boids-paper-cpu
```

## Other examples

These remain available independently of the SEAMS campaign.

```bash
# Distance fields and learning
uv run python examples/gradients/fixed.py
uv run python examples/gradients/learnable.py --rows 5 --cols 5 --epochs 200
uv run python examples/gradients/attention.py
uv run python examples/gradients/moving_nodes.py
uv run python examples/gradients/moving_nodes_learnable.py
uv run python examples/gradients/moving_source.py

# Collection and shortest-path channels
uv run python examples/collect_cast/spatial.py --node-counts 256 --rounds 100
uv run python examples/channel/small.py
uv run python examples/channel/large.py
uv run python examples/channel/viz.py

# Independent devices
uv run --extra decentralized python examples/decentralized/channel_obstacles.py
uv run --extra decentralized python examples/decentralized/moving_devices.py

```

## VMAS experiments

[`vmas_diffield/`](vmas_diffield/) trains differentiable field-calculus controllers inside
[VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator), comparing a
hand-tuned aggregate program against a learned-θ version of the same program, a
GNN-gated hybrid, and a pure GNN — under SHAC (analytic-gradient policy learning)
and behaviour cloning. Five scenarios: `flocking`, `flocking_beacon`, `navigation`,
`discovery`, `sampling`.

Requires the `vmas` extra:

```bash
uv sync --extra cpu --extra vmas
```

The programs in `vmas_diffield/programs.py` are written entirely in the DSL —
`elect` picks a flock leader, `descend` reads out a steering direction from a `gradient`
field, and `collect_cast`/`broadcast` run the discovery recruit quota. Agents move each
physics step, so the rollout calls `AggregateContext.update_topology(edge_index,
positions=pos)` to swap the graph in place while the recurrent field state persists and
self-heals.

```bash
# one scenario, quick pass
uv run --extra cpu --extra vmas python examples/vmas_diffield/main.py \
    --mode shac --scenarios flocking_beacon --updates 20 --seeds 0 --num-envs 32

# behaviour cloning instead
uv run --extra cpu --extra vmas python examples/vmas_diffield/main.py \
    --mode imitation --scenarios flocking,navigation

# cross-scenario figures and master CSV
uv run --extra cpu --extra vmas python examples/vmas_diffield/summarize.py

# the whole suite (~6-7 h; `--pilot` for a ~15 min sanity pass)
bash examples/vmas_diffield/run_all.sh --pilot
```

Artifacts land under `generated/vmas-<scenario>-{shac,imitation}/` and
`generated/vmas-comparison/`. Training is checkpointed per `(policy, seed)`, so a
re-run resumes rather than restarting.

Shared helpers live in `shared/`: artifacts and provenance, the harness of resumable
campaigns, a resumable training loop, a network that runs a program device by device,
metrics and plotting.
Generated artifacts are saved under `generated/`.
