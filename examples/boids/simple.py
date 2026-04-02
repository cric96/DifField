#!/usr/bin/env python3
"""Simple boids entrypoint using the new boids package layout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import torch

from aggregate_gnn import SpatialScenario, bounce_in_box, limit_speed, nbr, normalize_vectors, rep
from aggregate_gnn.dsl import AggregateContext
from aggregate_gnn.utils import get_device
from shared.plotting import export_moving_gif, plot_moving_snapshots, plot_node_trajectories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pure aggregate boids (rep/nbr)")
    parser.add_argument("--num-nodes", type=int, default=60)
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--radius", type=float, default=0.23)
    parser.add_argument("--init-connectivity", choices=["radius", "knn", "hybrid"], default="hybrid")
    parser.add_argument("--init-k-neighbors", type=int, default=8)
    parser.add_argument("--init-min-degree", type=int, default=2)
    parser.add_argument("--sep", type=float, default=0.06, help="Distance threshold for separation influence")
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--w-sep", type=float, default=1.0)
    parser.add_argument("--w-align", type=float, default=0.7)
    parser.add_argument("--w-cohesion", type=float, default=0.6)
    parser.add_argument("--damping", type=float, default=0.95)
    parser.add_argument("--speed", type=float, default=0.014)
    parser.add_argument("--record-every", type=int, default=10)
    parser.add_argument("--highlight-node", type=int, default=0)
    parser.add_argument("--viz-prefix", type=str, default="generated/boids/simple")
    parser.add_argument("--gif-fps", type=int, default=8)
    parser.add_argument("--no-viz", action="store_true", help="Disable figure export")
    parser.add_argument("--no-gif", action="store_true", help="Disable gif export")
    parser.add_argument("--hide-links", action="store_true", help="Do not draw graph links in visual outputs")
    parser.add_argument("--links-alpha", type=float, default=0.15)
    parser.add_argument("--links-width", type=float, default=0.6)
    parser.add_argument("--device", type=str, default="", help="Device (cuda/cpu) [auto if empty]")
    return parser.parse_args()


def aggregate_boids_velocity(
    vel: torch.Tensor,
    pos: torch.Tensor,
    *,
    dt: float,
    sep: float,
    w_sep: float,
    w_align: float,
    w_cohesion: float,
    damping: float,
    max_speed: float,
) -> torch.Tensor:
    neigh_vel = nbr(vel, aggr="mean")
    neigh_pos = nbr(pos, aggr="mean")
    align_force = neigh_vel - vel
    cohesion_force = neigh_pos - pos
    dx = pos[:, 0].unsqueeze(1) - pos[:, 0].unsqueeze(0)
    dy = pos[:, 1].unsqueeze(1) - pos[:, 1].unsqueeze(0)
    dist = torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-9)
    sep_mask = (dist <= sep) & (dist > 0)
    sep_force = torch.stack([(dx * sep_mask).sum(dim=1), (dy * sep_mask).sum(dim=1)], dim=1)
    acc = w_sep * normalize_vectors(sep_force) + w_align * align_force + w_cohesion * cohesion_force
    return limit_speed(damping * vel + dt * acc, max_speed)


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    torch.manual_seed(args.seed)
    positions = torch.rand(args.num_nodes, 2, device=device)
    velocities0 = normalize_vectors(torch.rand(args.num_nodes, 2, device=device) * 2.0 - 1.0) * args.speed

    scenario = SpatialScenario(
        positions=positions,
        edge_radius=args.radius if args.init_connectivity in {"radius", "hybrid"} else None,
        k_neighbors=args.init_k_neighbors if args.init_connectivity == "knn" else None,
        ensure_init_connected=args.init_connectivity == "hybrid",
        init_min_degree=args.init_min_degree,
        init_k_neighbors=args.init_k_neighbors,
        device=device,
    )
    ctx = AggregateContext(scenario.edge_index, scenario.num_nodes, edge_weight=scenario.edge_weight)
    positions_by_round: dict[int, torch.Tensor] = {}
    edge_index_by_round: dict[int, torch.Tensor] = {}
    values_by_round: dict[int, torch.Tensor] = {}
    record_rounds = set(range(0, args.rounds, max(1, args.record_every))) | {args.rounds - 1}

    for step in range(args.rounds):
        scenario.sync_context(ctx._ctx)
        pos_t = scenario.positions
        with ctx.round():
            vel = rep(
                "vel",
                velocities0,
                lambda prev: aggregate_boids_velocity(
                    prev,
                    pos_t,
                    dt=args.dt,
                    sep=args.sep,
                    w_sep=args.w_sep,
                    w_align=args.w_align,
                    w_cohesion=args.w_cohesion,
                    damping=args.damping,
                    max_speed=args.speed,
                ),
            )
        new_pos = pos_t + args.dt * vel
        new_pos, vel = bounce_in_box(new_pos, vel)
        scenario.update_positions(new_pos, refresh_topology=True)
        ctx._ctx.state.update("vel", vel)
        if step in record_rounds:
            positions_by_round[step] = scenario.positions.detach().cpu().clone()
            edge_index_by_round[step] = scenario.edge_index.detach().cpu().clone()
            values_by_round[step] = vel.norm(dim=1).detach().cpu().clone()

    center = scenario.positions.mean(dim=0)
    spread = (scenario.positions - center).norm(dim=1).mean().item()
    print("=== Pure Aggregate Boids ===")
    print(f"nodes={args.num_nodes} rounds={args.rounds} radius={args.radius} sep={args.sep} spread={spread:.4f} edges={scenario.edge_index.shape[1]}")
    print(
        f"init_connectivity={args.init_connectivity} init_components={int(scenario.init_graph_stats['num_components'])} "
        f"init_min_degree={scenario.init_graph_stats['min_degree']:.0f} init_edges={int(scenario.init_graph_stats['num_edges'])}"
    )

    if not args.no_viz:
        highlight_idx = int(max(0, min(args.highlight_node, args.num_nodes - 1)))
        plot_moving_snapshots(
            positions_by_round=positions_by_round,
            values_by_round=values_by_round,
            source_idx=highlight_idx,
            output_path=f"{args.viz_prefix}_snapshots.png",
            title="Pure Aggregate Boids speed snapshots",
            edge_index_by_round=edge_index_by_round,
            show_links=not args.hide_links,
            links_alpha=args.links_alpha,
            links_width=args.links_width,
        )
        plot_node_trajectories(
            positions_over_time=[positions_by_round[idx] for idx in sorted(positions_by_round.keys())],
            source_idx=highlight_idx,
            output_path=f"{args.viz_prefix}_trajectories.png",
            title="Pure Aggregate Boids trajectories",
        )
        if not args.no_gif:
            export_moving_gif(
                positions_by_round=positions_by_round,
                values_by_round=values_by_round,
                source_idx=highlight_idx,
                output_path=f"{args.viz_prefix}.gif",
                title="Pure Aggregate Boids speed",
                fps=max(1, args.gif_fps),
                edge_index_by_round=edge_index_by_round,
                show_links=not args.hide_links,
                links_alpha=args.links_alpha,
                links_width=args.links_width,
            )


if __name__ == "__main__":
    main()