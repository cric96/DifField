#!/usr/bin/env python3
"""Gradient (hop-distance) with fixed weights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

try:
    from .common import auto_rounds, build_corner_source_grid, run_gradient_program
except ImportError:
    from common import auto_rounds, build_corner_source_grid, run_gradient_program


def parse_args():
    parser = argparse.ArgumentParser(description="Fixed Gradient AC program")
    parser.add_argument("--rows", type=int, default=5, help="Grid rows")
    parser.add_argument("--cols", type=int, default=5, help="Grid cols")
    parser.add_argument("--rounds", type=int, default=0, help="Number of compute rounds (0 = auto)")
    parser.add_argument("--weight", type=float, default=1.0, help="Fixed edge weight")
    return parser.parse_args()


def plot_results(dist: torch.Tensor, expected: torch.Tensor) -> None:
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
    rounds = auto_rounds(args.rows, args.cols, args.rounds)
    scenario, source, expected = build_corner_source_grid(args.rows, args.cols, connectivity=4)

    weight = torch.tensor(args.weight, requires_grad=True)
    output, _ = run_gradient_program(scenario, source, rounds=rounds, weight=weight)

    print(f"=== Gradient (fixed w={args.weight}) on {args.rows}x{args.cols} grid ===")
    print(f"Rounds: {rounds}")
    print()

    dist = output.detach().view(args.rows, args.cols)
    expected_2d = expected.view(args.rows, args.cols)

    print("Distance field:")
    print(dist.numpy())
    print()
    print("Expected (Manhattan):")
    print(expected_2d.numpy())
    print()

    max_err = (dist - expected_2d).abs().max().item()
    print(f"Max error: {max_err}")

    loss = output.sum()
    loss.backward()
    print(f"d(loss)/dw = {weight.grad}")
    print(f"(expected: sum of all distances = {expected.sum().item()})")
    print()

    plot_results(dist, expected_2d)


if __name__ == "__main__":
    main()