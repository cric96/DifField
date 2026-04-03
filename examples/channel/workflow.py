"""Workflow orchestration for channel examples."""

from __future__ import annotations

import time

import torch

from autofield import GridScenario, SimulationEngine, SnapshotRecorder, branch
from autofield.dsl import field

try:
    from .core import CHANNEL_THRESHOLD, build_snapshot_payloads, channel_body, count_channel_nodes, distance_src_to_dst
    from .specs import LargeChannelSpec, SmallChannelSpec
    from .viz import (
        plot_channel_evolution,
        plot_channel_final_fields,
        plot_channel_large_evolution,
        plot_channel_large_final,
        plot_channel_large_setup,
        plot_channel_overlay,
        plot_channel_setup,
        plot_channel_gif,
    )
except ImportError:
    from core import CHANNEL_THRESHOLD, build_snapshot_payloads, channel_body, count_channel_nodes, distance_src_to_dst
    from specs import LargeChannelSpec, SmallChannelSpec
    from viz import (
        plot_channel_evolution,
        plot_channel_final_fields,
        plot_channel_large_evolution,
        plot_channel_large_final,
        plot_channel_large_setup,
        plot_channel_overlay,
        plot_channel_setup,
        plot_channel_gif,
    )


class SmallChannelWorkflow:
    def __init__(self, spec: SmallChannelSpec):
        self.spec = spec

    def run(
        self,
        device: torch.device | None = None,
        gif: bool = False,
        viz: bool = True,
        viz_prefix: str = "generated/channel_small",
        gif_fps: int = 10,
    ) -> None:
        scenario, source, dest, obstacle, noise, src_pos, dst_pos, wall_col = self._build_scenario(device=device)
        recorder = self._run_simulation(scenario, source, dest, obstacle, noise, record_all=gif)
        snapshots = build_snapshot_payloads(recorder.records, num_nodes=scenario.num_nodes)

        final_step = self.spec.program.rounds - 1
        final = snapshots[final_step]
        sd_dist = distance_src_to_dst(final, dst_pos, self.spec.grid.cols)

        print("=== Channel with Obstacles ===")
        print(f"Grid: {self.spec.grid.rows}x{self.spec.grid.cols}   Source: {src_pos}   Dest: {dst_pos}")
        print(f"Wall: column {wall_col}")
        print(f"Distance source->dest: {sd_dist:.2f}")
        print(f"Channel nodes: {count_channel_nodes(final)}")

        if viz:
            self._plot_results(snapshots, final, src_pos, dst_pos, obstacle, sd_dist, gif=gif, viz_prefix=viz_prefix, gif_fps=gif_fps)

    def _build_scenario(self, device: torch.device | None = None):
        rows, cols = self.spec.grid.rows, self.spec.grid.cols
        scenario = GridScenario(rows, cols, connectivity=8, device=device)

        src_pos = (rows // 2, 2)
        dst_pos = (rows // 2, cols - 3)
        source = scenario.marker(src_pos[0], src_pos[1])
        dest = scenario.marker(dst_pos[0], dst_pos[1])

        wall_col = cols // 2
        obstacle = scenario.mask_from_predicate(lambda row, col: col == wall_col and row < rows - 2)
        torch.manual_seed(self.spec.seed)
        noise = torch.rand(scenario.num_nodes, device=scenario.device) * self.spec.noise_scale
        return scenario, source, dest, obstacle, noise, src_pos, dst_pos, wall_col

    def _run_simulation(
        self,
        scenario: GridScenario,
        source: torch.Tensor,
        dest: torch.Tensor,
        obstacle: torch.Tensor,
        noise: torch.Tensor,
        record_all: bool = False,
    ) -> SnapshotRecorder:
        engine = SimulationEngine.from_scenario(scenario)
        snapshot_steps = [5, 15, 30, 60, self.spec.program.rounds - 1]
        record_rounds = None if record_all else set(snapshot_steps)
        recorder = SnapshotRecorder(
            state_fields=["dist_src", "dist_dst", "_bc_dist_channel"],
            capture_output=True,
            record_rounds=record_rounds,
        )

        def program(_runtime):
            return branch(
                ~obstacle,
                lambda: channel_body(source, dest, self.spec.program.tolerance, noise),
                lambda: field.of(0.0),
                branch_name="obstacle",
            )

        engine.run(
            rounds=self.spec.program.rounds,
            program=program,
            signals={"source": source, "dest": dest, "obstacle": obstacle},
            recorder=recorder,
        )
        return recorder

    def _plot_results(
        self,
        snapshots,
        final,
        src_pos,
        dst_pos,
        obstacle,
        sd_dist,
        gif: bool = False,
        viz_prefix: str = "generated/channel_small",
        gif_fps: int = 10,
    ):
        plot_channel_setup(self.spec.grid.rows, self.spec.grid.cols, src_pos, dst_pos, obstacle, viz_prefix=viz_prefix)
        plot_channel_final_fields(
            self.spec.grid.rows,
            self.spec.grid.cols,
            final,
            src_pos,
            dst_pos,
            obstacle,
            self.spec.program.rounds,
            viz_prefix=viz_prefix,
        )
        plot_channel_overlay(
            self.spec.grid.rows, self.spec.grid.cols, final, src_pos, dst_pos, obstacle, sd_dist, viz_prefix=viz_prefix
        )

        # Only plot evolution for the selected snapshot rounds
        evolution_snapshots = snapshots
        if gif:
            plot_channel_gif(
                self.spec.grid.rows,
                self.spec.grid.cols,
                snapshots,
                src_pos,
                dst_pos,
                obstacle,
                output_path=f"{viz_prefix}_evolution.gif",
                fps=gif_fps,
            )
            # Filter snapshots for the evolution plot to avoid overcrowding
            snapshot_steps = [5, 15, 30, 60, self.spec.program.rounds - 1]
            evolution_snapshots = {k: v for k, v in snapshots.items() if k in snapshot_steps}

        plot_channel_evolution(
            self.spec.grid.rows, self.spec.grid.cols, evolution_snapshots, src_pos, dst_pos, obstacle, viz_prefix=viz_prefix
        )


class LargeChannelWorkflow:
    def __init__(self, spec: LargeChannelSpec):
        self.spec = spec

    def run(
        self,
        device: torch.device | None = None,
        gif: bool = False,
        viz: bool = True,
        viz_prefix: str = "generated/channel_large",
        gif_fps: int = 10,
    ) -> None:
        scenario, source, dest, obstacle, src_pos, dst_pos = self._build_scenario(device=device)
        recorder = self._run_simulation(scenario, source, dest, obstacle, record_all=gif)
        snapshots = build_snapshot_payloads(recorder.records, num_nodes=scenario.num_nodes)
        
        final_step = self.spec.program.rounds - 1
        final = snapshots[final_step]
        sd_dist = distance_src_to_dst(final, dst_pos, self.spec.grid.cols)

        print("=== Large-Scale Channel ===")
        print(f"Grid: {self.spec.grid.rows}x{self.spec.grid.cols}   Source: {src_pos}   Dest: {dst_pos}")
        print(f"Obstacle cells: {obstacle.sum().item()}")
        print(f"Distance source->dest: {sd_dist:.2f}")
        print(f"Channel nodes: {count_channel_nodes(final)}")

        if viz:
            self._plot_results(snapshots, final, src_pos, dst_pos, obstacle, sd_dist, gif=gif, viz_prefix=viz_prefix, gif_fps=gif_fps)

    def _build_scenario(self, device: torch.device | None = None):
        rows, cols = self.spec.grid.rows, self.spec.grid.cols
        scenario = GridScenario(rows, cols, connectivity=8, device=device)
        src_pos = (rows // 2, 5)
        dst_pos = (rows // 2, cols - 6)
        source = scenario.marker(src_pos[0], src_pos[1])
        dest = scenario.marker(dst_pos[0], dst_pos[1])
        obstacle = self._build_obstacles(rows, cols, device=device)
        print(f"Obstacle cells: {obstacle.sum().item()}")

        return scenario, source, dest, obstacle, src_pos, dst_pos

    def _build_obstacles(self, rows: int, cols: int, device: torch.device | None = None) -> torch.Tensor:
        obstacle = torch.zeros(rows * cols, dtype=torch.bool, device=device)
        for row in range(0, min(40, rows)):
            if 25 < cols:
                obstacle[row * cols + 25] = True
        for row in range(10, rows):
            if 55 < cols:
                obstacle[row * cols + 55] = True
        return obstacle

    def _run_simulation(
        self,
        scenario: GridScenario,
        source: torch.Tensor,
        dest: torch.Tensor,
        obstacle: torch.Tensor,
        record_all: bool = False,
    ) -> SnapshotRecorder:
        engine = SimulationEngine.from_scenario(scenario)
        snapshot_steps = [10, 50, 100, 200, self.spec.program.rounds - 1]
        record_rounds = None if record_all else set(snapshot_steps)
        recorder = SnapshotRecorder(
            state_fields=["dist_src", "dist_dst", "_bc_dist_channel"],
            capture_output=True,
            record_rounds=record_rounds,
        )

        def program(_runtime):
            return branch(
                ~obstacle,
                lambda: channel_body(source, dest, self.spec.program.tolerance, None),
                lambda: field.of(0.0),
                branch_name="obstacle",
            )

        engine.run(
            rounds=self.spec.program.rounds,
            program=program,
            signals={"source": source, "dest": dest, "obstacle": obstacle},
            recorder=recorder,
        )
        return recorder

    def _plot_results(
        self,
        snapshots,
        final,
        src_pos,
        dst_pos,
        obstacle,
        sd_dist,
        gif: bool = False,
        viz_prefix: str = "generated/channel_large",
        gif_fps: int = 10,
    ):
        plot_channel_large_setup(
            self.spec.grid.rows, self.spec.grid.cols, self.spec.grid.num_nodes, src_pos, dst_pos, obstacle, viz_prefix=viz_prefix
        )
        plot_channel_large_final(
            self.spec.grid.rows,
            self.spec.grid.cols,
            final,
            src_pos,
            dst_pos,
            obstacle,
            CHANNEL_THRESHOLD,
            self.spec.grid.num_nodes,
            self.spec.program.rounds,
            0.0,  # elapsed time not tracked in this workflow
            sd_dist,
            count_channel_nodes(final),
            viz_prefix=viz_prefix,
        )
        
        # Only plot evolution for the selected snapshot rounds
        evolution_snapshots = snapshots
        snapshot_steps = [10, 50, 100, 200, self.spec.program.rounds - 1]
        snapshot_steps = [s for s in snapshot_steps if s in snapshots]
        snapshot_labels = [f"t={s + 1}" for s in snapshot_steps]
        
        if gif:
            plot_channel_gif(
                self.spec.grid.rows,
                self.spec.grid.cols,
                snapshots,
                src_pos,
                dst_pos,
                obstacle,
                output_path=f"{viz_prefix}_evolution.gif",
                fps=gif_fps,
            )
            # Filter snapshots for the evolution plot to avoid overcrowding
            evolution_snapshots = {k: v for k, v in snapshots.items() if k in snapshot_steps}
            
        plot_channel_large_evolution(
            self.spec.grid.rows,
            self.spec.grid.cols,
            evolution_snapshots,
            snapshot_steps,
            snapshot_labels,
            self.spec.grid.num_nodes,
            self.spec.program.rounds,
            0.0,  # elapsed time not tracked
            obstacle,
            viz_prefix=viz_prefix,
        )
