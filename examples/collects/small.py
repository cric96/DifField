#!/usr/bin/env python3
"""Small collect-cast example on a 1xN line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch

from autofield import GridScenario, SimulationEngine, collect_cast, gradient_cast
from autofield.dsl import field
from autofield.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Collect-cast example on a line")
    parser.add_argument(
        "--length", type=int, default=8, help="Number of nodes in the line"
    )
    parser.add_argument(
        "--rounds", type=int, default=0, help="Compute rounds (0 = auto)"
    )
    parser.add_argument(
        "--payload-mode",
        type=str,
        default="ones",
        choices=["ones", "ids"],
        help="Local payloads to collect toward the root",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument(
        "--device", type=str, default="", help="Device (cuda/cpu) [auto if empty]"
    )
    return parser.parse_args()


def auto_rounds(length: int, rounds: int) -> int:
    # One wave computes the potential, a second wave carries the collection.
    return rounds if rounds > 0 else 2 * length


def make_payload(num_nodes: int, mode: str, device: torch.device) -> torch.Tensor:
    if mode == "ids":
        return torch.arange(1, num_nodes + 1, dtype=torch.float32, device=device)
    return torch.ones(num_nodes, dtype=torch.float32, device=device)


def expected_root_total(length: int, mode: str) -> float:
    if mode == "ids":
        return float(length * (length + 1) // 2)
    return float(length)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    scenario = GridScenario(1, args.length, connectivity=4, device=device)
    rounds = auto_rounds(args.length, args.rounds)

    source = scenario.marker(0, 0)
    local = make_payload(scenario.num_nodes, args.payload_mode, device)

    engine = SimulationEngine.from_scenario(scenario)

    def program(_runtime):
        potential = gradient_cast(
            source,
            field.zeros(),
            lambda value: value + 1.0,
            name="potential",
        )
        collected = collect_cast(
            potential,
            local,
            field.zeros(),
            lambda acc, value: acc + value,
            name="mass",
        )
        return torch.stack((potential, collected), dim=-1)

    output, _ = engine.run(
        rounds=rounds,
        program=program,
        signals={"source": source, "local": local},
    )

    potential = output[:, 0]
    collected = output[:, 1]
    root_total = collected[0].item()
    expected = expected_root_total(args.length, args.payload_mode)

    torch.set_printoptions(precision=2, sci_mode=False)
    print("=== Collect-cast example ===")
    print(f"Device: {device}")
    print(
        f"Nodes: {scenario.num_nodes}  Rounds: {rounds}  Payload mode: {args.payload_mode}"
    )
    print("Potential field:")
    print(potential.detach().cpu())
    print("Collected field:")
    print(collected.detach().cpu())
    print(f"Root total: {root_total:.2f}  (expected: {expected:.2f})")


if __name__ == "__main__":
    main()
