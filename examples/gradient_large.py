#!/usr/bin/env python3
"""Large-scale gradient example — 500x500 grid (250,000 nodes).

Demonstrates the performance and scalability of the Aggregate GNN DSL
on a larger network than typical small examples.
"""

import sys
import time
import argparse
import torch
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

sys.path.insert(0, "src")

from aggregate_gnn import GridScenario, SimulationEngine, SnapshotRecorder, rep, nbr, mux
from aggregate_gnn.dsl import field


def parse_args():
    parser = argparse.ArgumentParser(description="Large-scale Gradient")
    parser.add_argument("--rows", type=int, default=500, help="Grid rows")
    parser.add_argument("--cols", type=int, default=500, help="Grid columns")
    parser.add_argument("--rounds", type=int, default=0, help="Number of compute rounds (0 = auto)")
    parser.add_argument("--device", type=str, default="", help="Device (cuda/cpu) [auto if empty]")
    return parser.parse_args()


def setup_data(args, device):
    """Build large grid and source at center."""
    scenario = GridScenario(args.rows, args.cols, connectivity=4, device=device)

    # Source: center of the grid
    source = torch.zeros(scenario.num_nodes, dtype=torch.float32, device=device)
    center_r = args.rows // 2
    center_c = args.cols // 2
    center_idx = center_r * args.cols + center_c
    source[center_idx] = 1.0

    return scenario, source, center_r, center_c


def run_gradient(args, scenario, source, device):
    engine = SimulationEngine.from_scenario(scenario)
    T = args.rounds if args.rounds > 0 else (args.rows + args.cols)
    w = torch.tensor(1.0, device=device, requires_grad=True)

    print(f"Running {T} rounds of aggregate computation...")
    start_time = time.time()
    
    record_at = [max(1, T//10), max(1, T//2), T]
    recorder = SnapshotRecorder(
        state_fields=[],
        capture_output=True,
        record_rounds={t - 1 for t in record_at},
    )

    def program(_runtime):
        return rep("dist", float("inf"), lambda d_old:
            mux(source, field.of(0.0), nbr(d_old + w, aggr="min"))
        )

    d, _ = engine.run(
        rounds=T,
        program=program,
        signals={"source": source},
        recorder=recorder,
    )

    snapshots = {
        round_idx + 1: payload["output"].detach().cpu().view(args.rows, args.cols).clone()
        for round_idx, payload in recorder.records.items()
    }
            
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    compute_time = time.time() - start_time
    print(f"Computation finished in {compute_time:.4f}s ({compute_time/T:.6f}s per round)")

    return d, w, snapshots, compute_time


def plot_results(args, dist, snapshots):
    if plt is None:
        return
        
    fig, axes = plt.subplots(1, len(snapshots), figsize=(20, 4))
    for i, (round_t, snapshot) in enumerate(sorted(snapshots.items())):
        im = axes[i].imshow(snapshot.numpy(), cmap="magma")
        axes[i].set_title(f"Round {round_t}")
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.savefig("examples/gradient_large_evolution.png")
    print("Evolution visualization saved to examples/gradient_large_evolution.png")

    plt.figure(figsize=(8, 6))
    plt.imshow(dist.numpy(), cmap="magma")
    plt.colorbar(label="Distance")
    plt.title(f"Final Gradient field on {args.rows}x{args.cols} grid (Source at center)")
    plt.savefig("examples/gradient_large.png")
    print("Final visualization saved to examples/gradient_large.png")


def main():
    args = parse_args()
    
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    print(f"=== Large-scale Gradient ({args.rows}x{args.cols} grid, {args.rows*args.cols} nodes) ===")
    print(f"Device: {device}")

    print("Building graph...")
    start_time = time.time()
    scenario, source, center_r, center_c = setup_data(args, device)
    graph_time = time.time() - start_time
    print(f"Graph built in {graph_time:.4f}s (Edges: {scenario.edge_index.shape[1]})")

    d, w, snapshots, _ = run_gradient(args, scenario, source, device)

    # Validate a few points
    dist = d.detach().cpu().view(args.rows, args.cols)
    center_val = dist[center_r, center_c].item()
    print(f"Distance at center: {center_val:.1f} (expected 0.0)")
    
    corner_val = dist[0, 0].item()
    expected_corner = center_r + center_c
    print(f"Distance at corner (0,0): {corner_val:.1f} (expected {expected_corner}.0)")

    print("Computing gradient w.r.t. weight w...")
    start_time = time.time()
    loss = d.sum()
    loss.backward()
    backward_time = time.time() - start_time
    print(f"Backward pass in {backward_time:.4f}s")
    print(f"d(loss)/dw = {w.grad.item():.1f}")

    plot_results(args, dist, snapshots)


if __name__ == "__main__":
    main()
