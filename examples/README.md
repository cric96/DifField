# DifField examples

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

# Other robotic tasks, outside the SEAMS campaign
uv run --extra vmas python examples/vmas_diffield/main.py \
  --mode imitation --scenarios flocking,navigation
uv run --extra vmas python examples/vmas_diffield/summarize.py
```

Shared plotting, diagnostics, checkpoint and training utilities live in `shared/`.
Generated artifacts are saved under `generated/`.
