#!/usr/bin/env python3
"""Same AC program executed **globally** (full graph) and **locally** (single device).

Demonstrates that:
1. The program is device-count agnostic (no reference to N or num_nodes).
2. Local execution on a single device produces the same result as global.
"""

import sys
import argparse
sys.path.insert(0, "src")

import torch
from aggregate_gnn import AggregateContext, rep, nbr, mux
from aggregate_gnn.dsl import field, DeviceContext
from aggregate_gnn.utils import make_grid_graph


def parse_args():
    parser = argparse.ArgumentParser(description="Local vs Global Gradient AC")
    parser.add_argument("--rows", type=int, default=5, help="Grid rows")
    parser.add_argument("--cols", type=int, default=5, help="Grid columns")
    parser.add_argument("--device-row", type=int, default=2, help="Row of local device to simulate")
    parser.add_argument("--device-col", type=int, default=2, help="Column of local device to simulate")
    return parser.parse_args()


# ── Device-agnostic AC program (no reference to N) ─────────────────────

def gradient(source, w):
    """AC gradient: shortest-path distance from source."""
    return rep("dist", float("inf"), lambda d:
        mux(source, field.of(0.0), nbr(d + w, aggr="min")))


# ── Graph helpers ────────────────────────────────────────────────────────

def neighbors_of(node, edge_index):
    """Distinct neighbours of *node* (excluding self-loop)."""
    mask = (edge_index[1] == node) & (edge_index[0] != node)
    return sorted(edge_index[0, mask].tolist())


def setup_data(args):
    """Build graph and global source."""
    edge_index, N = make_grid_graph(args.rows, args.cols, connectivity=4)
    source_global = torch.zeros(N)
    source_global[0] = 1.0  # Node 0 is always the source
    return edge_index, N, source_global


def run_global(edge_index, N, source_global, w, T):
    """Run globally across all nodes."""
    ctx = AggregateContext(edge_index, N)
    global_states: list[torch.Tensor] = []
    
    for _ in range(T):
        with ctx.round():
            d = gradient(source_global, w)
        global_states.append(d.detach().clone())
        
    return d, global_states


def run_local(args, edge_index, source_global, global_states, w, T):
    """Simulate execution for a single device using previous global states for neighbors."""
    device_id = args.device_row * args.cols + args.device_col
    nbrs = neighbors_of(device_id, edge_index)
    K = len(nbrs)

    print(f"\n=== Local execution for device {device_id}"
          f" (K={K} neighbours: {nbrs}) ===")

    device = DeviceContext(num_neighbors=K)
    source_local = device.local_field(own=source_global[device_id].item(), nbr=0.0)

    for t in range(T):
        if t == 0:
            nbr_exports = None
        else:
            nbr_exports = {
                "dist": [global_states[t - 1][j].item() for j in nbrs],
            }

        with device.round(neighbor_exports=nbr_exports):
            d_local = gradient(source_local, w)

        my_d = device.result(d_local).item()
        global_d = global_states[t][device_id].item()
        same = abs(my_d - global_d) < 1e-6 or (my_d == global_d == float("inf"))
        tag = "ok" if same else "MISMATCH"
        my_str = f"{my_d:6.2f}" if my_d != float("inf") else "   inf"
        gl_str = f"{global_d:6.2f}" if global_d != float("inf") else "   inf"
        print(f"  Round {t + 1:2d}: local={my_str}  global={gl_str}  [{tag}]")

    return device_id, device.result(d_local).item()


def main():
    args = parse_args()
    
    edge_index, N, source_global = setup_data(args)
    T = args.rows + args.cols
    w = torch.tensor(1.0)

    # 1. Global execution
    print(f"=== Global execution ({args.rows}x{args.cols} grid) ===")
    d, global_states = run_global(edge_index, N, source_global, w, T)

    print("Distance field:")
    print(d.detach().view(args.rows, args.cols).numpy())

    # 2. Local execution
    device_id, final = run_local(args, edge_index, source_global, global_states, w, T)

    expected = args.device_row + args.device_col
    print(f"\nFinal: device {device_id} distance = {final:.1f}"
          f"  (expected: {expected}.0)")


if __name__ == "__main__":
    main()
