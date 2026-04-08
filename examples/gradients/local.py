#!/usr/bin/env python3
"""Local vs global gradient execution example."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from autofield import DeviceContext, GridScenario, SimulationEngine, gradient, nbr
from autofield.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Local vs Global Gradient AC")
    parser.add_argument("--rows", type=int, default=5, help="Grid rows")
    parser.add_argument("--cols", type=int, default=5, help="Grid cols")
    parser.add_argument(
        "--device-row", type=int, default=2, help="Row of local device to simulate"
    )
    parser.add_argument(
        "--device-col", type=int, default=2, help="Column of local device to simulate"
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument(
        "--device", type=str, default="", help="Device (cuda/cpu) [auto if empty]"
    )
    return parser.parse_args()


def neighbors_of(node, edge_index):
    mask = (edge_index[1] == node) & (edge_index[0] != node)
    return sorted(edge_index[0, mask].tolist())


def setup_data(args, device: torch.device):
    scenario = GridScenario(args.rows, args.cols, connectivity=4, device=device)
    source_global = torch.zeros(scenario.num_nodes, device=device)
    source_global[0] = 1.0
    return scenario, source_global


def run_global(scenario, source_global, weight, rounds):
    engine = SimulationEngine.from_scenario(scenario)
    global_states = []

    def program(_runtime):
        return gradient(source_global, nbr(weight), name="dist")

    runtime = engine.init_runtime(signals={"source": source_global})
    for _ in range(rounds):
        output = engine.step(runtime=runtime, program=program)
        global_states.append(output.detach().clone())
    return output, global_states


def run_local(args, edge_index, source_global, global_states, weight, rounds):
    device_id = args.device_row * args.cols + args.device_col
    neighbor_ids = neighbors_of(device_id, edge_index)
    device = DeviceContext(num_neighbors=len(neighbor_ids))
    source_local = device.local_field(own=source_global[device_id].item(), nbr=0.0)
    local_weight = torch.full_like(source_local, float(weight.item()))

    print(
        f"\n=== Local execution for device {device_id} (K={len(neighbor_ids)} neighbours: {neighbor_ids}) ==="
    )

    for round_idx in range(rounds):
        if round_idx == 0:
            neighbor_exports = None
        else:
            neighbor_exports = {
                "_grad_dist": [
                    global_states[round_idx - 1][node].item() for node in neighbor_ids
                ]
            }

        with device.round(neighbor_exports=neighbor_exports):
            d_local = gradient(source_local, nbr(local_weight), name="dist")

        local_value = device.result(d_local).item()
        global_value = global_states[round_idx][device_id].item()
        same = abs(local_value - global_value) < 1e-6 or (
            local_value == global_value == float("inf")
        )
        tag = "ok" if same else "MISMATCH"
        local_str = f"{local_value:6.2f}" if local_value != float("inf") else "   inf"
        global_str = (
            f"{global_value:6.2f}" if global_value != float("inf") else "   inf"
        )
        print(
            f"  Round {round_idx + 1:2d}: local={local_str}  global={global_str}  [{tag}]"
        )

    return device_id, device.result(d_local).item()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    scenario, source_global = setup_data(args, device)
    rounds = args.rows + args.cols
    weight = torch.tensor(1.0, device=device)

    print(f"=== Global execution ({args.rows}x{args.cols} grid) ===")
    print(f"Device: {device}")
    output, global_states = run_global(scenario, source_global, weight, rounds)
    print("Distance field:")
    print(output.detach().cpu().view(args.rows, args.cols).numpy())

    device_id, final = run_local(
        args, scenario.edge_index, source_global, global_states, weight, rounds
    )
    expected = args.device_row + args.device_col
    print(
        f"\nFinal: device {device_id} distance = {final:.1f}  (expected: {expected}.0)"
    )


if __name__ == "__main__":
    main()
