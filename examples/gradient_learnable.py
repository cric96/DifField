#!/usr/bin/env python3
"""Gradient with LEARNABLE weight — demonstrates end-to-end differentiability.

Same AC program as gradient_fixed, but w is a learnable nn.Parameter.
We train it via MSE loss against the true Manhattan distances.
w should converge to ~1.0.
"""

import sys
import argparse
sys.path.insert(0, "src")

import torch
import torch.nn as nn

from aggregate_gnn import AggregateContext, rep, nbr, mux
from aggregate_gnn.dsl import field
from aggregate_gnn.utils import make_grid_graph, get_grid_distances


def parse_args():
    parser = argparse.ArgumentParser(description="Learnable Gradient AC program")
    parser.add_argument("--rows", type=int, default=5, help="Grid rows")
    parser.add_argument("--cols", type=int, default=5, help="Grid columns")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--initial-weight", type=float, default=3.0, help="Initial w value")
    return parser.parse_args()


class GradientModel(nn.Module):
    """Wraps the AC gradient program so that w is a learnable parameter."""

    def __init__(self, edge_index: torch.Tensor, num_nodes: int, T: int, init_w: float = 3.0):
        super().__init__()
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.T = T
        # Learnable edge weight
        self.w = nn.Parameter(torch.tensor(init_w))

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        ctx = AggregateContext(self.edge_index, self.num_nodes)
        w = self.w

        for t in range(self.T):
            with ctx.round():
                d = rep("dist", float("inf"), lambda d:
                    mux(source,
                        field.of(0.0),
                        nbr(d + w, aggr="min"),
                    )
                )
        return d


def setup_data(args):
    """Build grid and expected Manhattan distances."""
    edge_index, num_nodes = make_grid_graph(args.rows, args.cols, connectivity=4)

    source = torch.zeros(num_nodes, dtype=torch.float32)
    source[0] = 1.0

    target = get_grid_distances(args.rows, args.cols, src_r=0, src_c=0, connectivity=4)

    return edge_index, num_nodes, source, target


def train_model(args, model, source, target):
    """Train the edge weight w against target Manhattan distances."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("=== Gradient (learnable w) training ===")
    print(f"Initial w = {model.w.item():.4f}")
    print()

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        pred = model(source)
        # Only compute loss on reachable nodes (non-inf)
        mask = pred.isfinite()
        loss = nn.functional.mse_loss(pred[mask], target[mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}  loss={loss.item():.6f}  w={model.w.item():.4f}")

    print()
    print(f"Final w = {model.w.item():.4f}  (expected: 1.0)")


def main():
    args = parse_args()
    
    edge_index, num_nodes, source, target = setup_data(args)
    T = args.rows + args.cols
    
    model = GradientModel(edge_index, num_nodes, T, init_w=args.initial_weight)
    
    train_model(args, model, source, target)

    # Final prediction
    with torch.no_grad():
        pred = model(source)
    
    print()
    print("Predicted distances:")
    print(pred.view(args.rows, args.cols).numpy())
    print()
    print("Target (Manhattan):")
    print(target.view(args.rows, args.cols).numpy())


if __name__ == "__main__":
    main()
