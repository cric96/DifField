#!/usr/bin/env python3
"""Large-scale channel with multiple obstacles.

Same AC program as ``channel.py`` but on a configurable grid (default 50×80,
4 000 devices) with a maze-like obstacle layout: two vertical walls with
staggered gaps that force the channel through a winding corridor.

8-connected grid → Chebyshev (chessboard) distance.

Produces three PNG figures:
  1. channel_large_setup.png     — grid layout
  2. channel_large_evolution.png — field snapshots at selected rounds
  3. channel_large_final.png     — converged overlay (channel path on grid)
"""

import sys, time, argparse

sys.path.insert(0, "src")

import torch

from aggregate_gnn import GridScenario, SimulationEngine, SnapshotRecorder, rep, nbr, mux, branch, broadcast
from aggregate_gnn.dsl import field
from channel_viz import plot_channel_large_evolution, plot_channel_large_final, plot_channel_large_setup
# ── Constants ──────────────────────────────────────────────────────────────

CHANNEL_THRESHOLD = 0.5    # classify node as "on channel" when value > threshold


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Large-scale AC Channel")
    parser.add_argument("--rows",      type=int,   default=50,  help="Grid rows")
    parser.add_argument("--cols",      type=int,   default=80,  help="Grid columns")
    parser.add_argument("--rounds",    type=int,   default=300, help="Number of compute rounds")
    parser.add_argument("--tolerance", type=float, default=0.5, help="Path tolerance")
    return parser.parse_args()

# ── Obstacle builder ────────────────────────────────────────────────────

def build_obstacles(rows, cols):
    """Two vertical walls with staggered gaps → Z-shaped corridor.

    Wall 1: col 25, rows  0–39  (gap at rows 40–49, bottom)
    Wall 2: col 55, rows 10–49  (gap at rows  0–9,  top)

    Forces: left → down → right → up → right.
    """
    obstacle = torch.zeros(rows * cols, dtype=torch.bool)

    # Wall 1
    for r in range(0, 40):
        obstacle[r * cols + 25] = True

    # Wall 2
    for r in range(10, rows):
        obstacle[r * cols + 55] = True

    return obstacle


# ── Channel body ────────────────────────────────────────────────────────

def channel_body(source, dest, tolerance):
    dist_src = rep("dist_src", float("inf"), lambda d:
        mux(source, field.of(0.0), nbr(d + 1.0, aggr="min")))
    dist_dst = rep("dist_dst", float("inf"), lambda d:
        mux(dest, field.of(0.0), nbr(d + 1.0, aggr="min")))
    dist_sd = broadcast(source > 0.5, dist_dst, name="dist_channel")

    on_path = (dist_src + dist_dst <= dist_sd + tolerance)
    finite  = torch.isfinite(dist_src) & torch.isfinite(dist_dst) & torch.isfinite(dist_sd)
    return (on_path & finite).float()


# ── Main ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rows, cols = args.rows, args.cols
    T = args.rounds
    N = rows * cols
    print(f"Building {rows}×{cols} grid ({N} devices) …")
    scenario = GridScenario(args.rows, args.cols, connectivity=8)
    N = scenario.num_nodes
    
    # Source: middle-left;  Destination: middle-right
    src_pos = (rows // 2, 5)
    dst_pos = (rows // 2, cols - 6)
    source = scenario.marker(src_pos[0], src_pos[1])
    dest = scenario.marker(dst_pos[0], dst_pos[1])

    obstacle = build_obstacles(rows, cols)
    print(f"Obstacle cells: {obstacle.sum().item()}")

    # ── Run ─────────────────────────────────────────────────────────────
    engine = SimulationEngine.from_scenario(scenario)
    runtime = engine.init_runtime(signals={"source": source, "dest": dest, "obstacle": obstacle})
    recorder = SnapshotRecorder(
        state_fields=["dist_src", "dist_dst", "_bc_dist_channel"],
        capture_output=True,
        record_rounds=None,
    )
    snapshot_at_seconds = [0.1, 0.3, 0.7, 1.5]
    snapshot_steps: list[int] = []
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    next_snap_idx = 0

    t0 = time.time()
    def program(_runtime):
        return branch(
            ~obstacle,
            lambda: channel_body(source, dest, args.tolerance),
            lambda: field.of(0.0),
            branch_name="obstacle",
        )

    for t in range(T):
        ch = engine.step(runtime=runtime, program=program, recorder=recorder)

        elapsed_now = time.time() - t0

        # Time-based snapshots
        if next_snap_idx < len(snapshot_at_seconds) and elapsed_now >= snapshot_at_seconds[next_snap_idx]:
            snapshot_steps.append(t)
            next_snap_idx += 1

        if (t + 1) % 100 == 0:
            print(f"  round {t + 1}/{T}  ({elapsed_now:.1f}s)")

    # Always include last round
    if T - 1 not in snapshot_steps:
        snapshot_steps.append(T - 1)

    # Keep a stable evolution layout even when execution is very fast.
    if len(snapshot_steps) < 3:
        fallback_steps = [max(0, T // 4), max(0, T // 2), T - 1]
        snapshot_steps.extend(fallback_steps)

    snapshot_steps = sorted(set(snapshot_steps))

    for t, payload in recorder.records.items():
        ds = payload.get("dist_src", torch.full((N,), float("inf")))
        dd = payload.get("dist_dst", torch.full((N,), float("inf")))
        dsd = payload.get("_bc_dist_channel", torch.full((N,), float("inf")))
        snapshots[t] = {
            "dist_src": ds,
            "dist_dst": dd,
            "sum": ds + dd,
            "dist_sd": dsd,
            "channel": payload["output"],
        }

    elapsed = time.time() - t0
    final = snapshots[T - 1]
    sd_dist = final["dist_src"][dst_pos[0] * cols + dst_pos[1]].item()
    n_ch = (final["channel"] > 0.5).sum().item()
    print(f"\nDone in {elapsed:.1f}s  |  distance = {sd_dist:.0f}  |  channel nodes = {n_ch}")

    sec_per_round = elapsed / T
    snap_labels = []
    for s in snapshot_steps:
        t_sec = (s + 1) * sec_per_round
        snap_labels.append(f"t={s+1}  ({t_sec:.1f}s)")

    plot_channel_large_setup(rows, cols, N, src_pos, dst_pos, obstacle)
    plot_channel_large_evolution(rows, cols, snapshots, snapshot_steps, snap_labels, N, T, elapsed, obstacle)
    plot_channel_large_final(rows, cols, final, src_pos, dst_pos, obstacle, CHANNEL_THRESHOLD, N, T, elapsed, sd_dist, n_ch)

    print("\nDone.")


if __name__ == "__main__":
    main()