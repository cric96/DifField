#!/usr/bin/env python3
"""Gradient (hop-distance) with FIXED weights — classic AC algorithm.

AC program:
    def gradient(source):
        rep("dist", inf)(d => mux(source, 0, nbr(d + w, min)))

GNN equivalent (Bellman-Ford MPNN with min-aggregation):
    h_i^(t) = source_i * 0 + (1 - source_i) * min_{j in N(i)} (h_j^(t-1) + w)

With w = 1 this computes shortest-path (hop) distance from source nodes.
"""

import sys
import argparse
import torch
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

sys.path.insert(0, "src")

from aggregate_gnn import GridScenario, SimulationEngine, rep, nbr, mux
from aggregate_gnn.dsl import field
from aggregate_gnn.utils import get_grid_distances


def parse_args():
    parser = argparse.ArgumentParser(description="Fixed Gradient AC program")
    parser.add_argument("--rows", type=int, default=5, help="Grid rows")
    parser.add_argument("--cols", type=int, default=5, help="Grid columns")
    parser.add_argument("--rounds", type=int, default=0, help="Number of compute rounds (0 = auto)")
    parser.add_argument("--weight", type=float, default=1.0, help="Fixed edge weight")
    return parser.parse_args()


def setup_data(args):
    """Build grid and expected Manhattan distances."""
    scenario = GridScenario(args.rows, args.cols, connectivity=4)

    # Source = top-left corner (node 0)
    source = torch.zeros(scenario.num_nodes, dtype=torch.float32)
    source[0] = 1.0

    expected = get_grid_distances(args.rows, args.cols, src_r=0, src_c=0, connectivity=4)

    return scenario, source, expected


def run_gradient(scenario, source, w_val, rounds):
    """Run the gradient program with a fixed weight."""
    w = torch.tensor(w_val, requires_grad=True)
    engine = SimulationEngine.from_scenario(scenario)

    def program(_runtime):
        return rep("dist", float("inf"), lambda d:
            mux(source, field.of(0.0), nbr(d + w, aggr="min"))
        )

    d, _ = engine.run(rounds=rounds, program=program, signals={"source": source})
            
    return d, w


def plot_results(args, dist, expected):
    """Plot computed vs expected distances."""
    if plt is None:
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(dist.numpy(), cmap="viridis")
    axes[0].set_title("Computed (rep + mux + nbr)")
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(expected.numpy(), cmap="viridis")
    axes[1].set_title("Expected (Manhattan)")
    plt.colorbar(im1, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig("examples/gradient_fixed.png", dpi=150)
    print("Saved figure to examples/gradient_fixed.png")


def main():
    args = parse_args()
    
    # Auto-calculate rounds if not provided
    T = args.rounds if args.rounds > 0 else (args.rows + args.cols)
    
    scenario, source, expected = setup_data(args)
    d, w = run_gradient(scenario, source, args.weight, T)

    print(f"=== Gradient (fixed w={args.weight}) on {args.rows}×{args.cols} grid ===")
    print(f"Rounds: {T}")
    print()

    dist = d.detach().view(args.rows, args.cols)
    print("Distance field:")
    print(dist.numpy())
    print()

    expected_2d = expected.view(args.rows, args.cols)
    print("Expected (Manhattan):")
    print(expected_2d.numpy())
    print()

    max_err = (dist - expected_2d).abs().max().item()
    print(f"Max error: {max_err}")

    # Differentiability check
    loss = d.sum()
    loss.backward()
    print(f"d(loss)/dw = {w.grad}")
    print(f"(expected: sum of all distances = {expected.sum().item()})")
    print()

    plot_results(args, dist, expected_2d)


if __name__ == "__main__":
    main()
