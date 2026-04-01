"""Workflow orchestration for channel examples."""

from __future__ import annotations

import time

import torch

from aggregate_gnn import GridScenario, SimulationEngine, SnapshotRecorder, branch
from aggregate_gnn.dsl import field

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
    )


class SmallChannelWorkflow:
    def __init__(self, spec: SmallChannelSpec):
        self.spec = spec

    def run(self) -> None:
        scenario, source, dest, obstacle, noise, src_pos, dst_pos, wall_col = self._build_scenario()
        recorder = self._run_simulation(scenario, source, dest, obstacle, noise)
        snapshots = build_snapshot_payloads(recorder.records, num_nodes=scenario.num_nodes)
        final = snapshots[self.spec.program.rounds - 1]
        sd_dist = distance_src_to_dst(final, dst_pos, self.spec.grid.cols)

        print("=== Channel with Obstacles ===")
        print(f"Grid: {self.spec.grid.rows}x{self.spec.grid.cols}   Source: {src_pos}   Dest: {dst_pos}")
        print(f"Wall: column {wall_col}")
        print(f"Distance source->dest: {sd_dist:.2f}")
        print(f"Channel nodes: {count_channel_nodes(final)}")

        self._plot_results(snapshots, final, src_pos, dst_pos, obstacle, sd_dist)

    def _build_scenario(self):
        rows, cols = self.spec.grid.rows, self.spec.grid.cols
        scenario = GridScenario(rows, cols, connectivity=8)

        src_pos = (rows // 2, 2)
        dst_pos = (rows // 2, cols - 3)
        source = scenario.marker(src_pos[0], src_pos[1])
        dest = scenario.marker(dst_pos[0], dst_pos[1])

        wall_col = cols // 2
        obstacle = scenario.mask_from_predicate(lambda row, col: col == wall_col and row < rows - 2)
        torch.manual_seed(self.spec.seed)
        noise = torch.rand(scenario.num_nodes) * self.spec.noise_scale
        return scenario, source, dest, obstacle, noise, src_pos, dst_pos, wall_col

    def _run_simulation(
        self,
        scenario: GridScenario,
        source: torch.Tensor,
        dest: torch.Tensor,
        obstacle: torch.Tensor,
        noise: torch.Tensor,
    ) -> SnapshotRecorder:
        engine = SimulationEngine.from_scenario(scenario)
        snapshot_steps = [5, 15, 30, 60, self.spec.program.rounds - 1]
        recorder = SnapshotRecorder(
            state_fields=["dist_src", "dist_dst", "_bc_dist_channel"],
            capture_output=True,
            record_rounds=set(snapshot_steps),
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
        snapshots: dict[int, dict[str, torch.Tensor]],
        final: dict[str, torch.Tensor],
        src_pos: tuple[int, int],
        dst_pos: tuple[int, int],
        obstacle: torch.Tensor,
        sd_dist: float,
    ) -> None:
        rows, cols = self.spec.grid.rows, self.spec.grid.cols
        plot_channel_setup(rows, cols, src_pos, dst_pos, obstacle)
        plot_channel_evolution(rows, cols, snapshots, src_pos, dst_pos, obstacle)
        plot_channel_final_fields(rows, cols, final, src_pos, dst_pos, obstacle, self.spec.program.rounds)
        plot_channel_overlay(rows, cols, final, src_pos, dst_pos, obstacle, sd_dist)


class LargeChannelWorkflow:
    def __init__(self, spec: LargeChannelSpec):
        self.spec = spec

    def run(self) -> None:
        rows, cols, rounds = self.spec.grid.rows, self.spec.grid.cols, self.spec.program.rounds
        print(f"Building {rows}x{cols} grid ({self.spec.grid.num_nodes} devices) ...")

        scenario = GridScenario(rows, cols, connectivity=8)
        source_pos = (rows // 2, 5)
        dest_pos = (rows // 2, cols - 6)
        source = scenario.marker(source_pos[0], source_pos[1])
        dest = scenario.marker(dest_pos[0], dest_pos[1])
        obstacle = self._build_obstacles()
        print(f"Obstacle cells: {obstacle.sum().item()}")

        engine = SimulationEngine.from_scenario(scenario)
        runtime = engine.init_runtime(signals={"source": source, "dest": dest, "obstacle": obstacle})
        recorder = SnapshotRecorder(
            state_fields=["dist_src", "dist_dst", "_bc_dist_channel"],
            capture_output=True,
            record_rounds=None,
        )

        snapshot_at_seconds = [0.1, 0.3, 0.7, 1.5]
        snapshot_steps: list[int] = []
        next_snapshot_index = 0
        start_time = time.time()

        def program(_runtime):
            return branch(
                ~obstacle,
                lambda: channel_body(source, dest, self.spec.program.tolerance, None),
                lambda: field.of(0.0),
                branch_name="obstacle",
            )

        for round_idx in range(rounds):
            engine.step(runtime=runtime, program=program, recorder=recorder)
            elapsed = time.time() - start_time
            if next_snapshot_index < len(snapshot_at_seconds) and elapsed >= snapshot_at_seconds[next_snapshot_index]:
                snapshot_steps.append(round_idx)
                next_snapshot_index += 1
            if (round_idx + 1) % 100 == 0:
                print(f"  round {round_idx + 1}/{rounds}  ({elapsed:.1f}s)")

        if rounds - 1 not in snapshot_steps:
            snapshot_steps.append(rounds - 1)
        if len(snapshot_steps) < 3:
            snapshot_steps.extend([max(0, rounds // 4), max(0, rounds // 2), rounds - 1])
        snapshot_steps = sorted(set(snapshot_steps))

        snapshots = build_snapshot_payloads(recorder.records, num_nodes=scenario.num_nodes)
        elapsed = time.time() - start_time
        final = snapshots[rounds - 1]
        sd_dist = distance_src_to_dst(final, dest_pos, cols)
        channel_nodes = count_channel_nodes(final, CHANNEL_THRESHOLD)
        print(f"\nDone in {elapsed:.1f}s  |  distance = {sd_dist:.0f}  |  channel nodes = {channel_nodes}")

        sec_per_round = elapsed / rounds
        snapshot_labels = [f"t={step + 1}  ({(step + 1) * sec_per_round:.1f}s)" for step in snapshot_steps]

        plot_channel_large_setup(rows, cols, scenario.num_nodes, source_pos, dest_pos, obstacle)
        plot_channel_large_evolution(rows, cols, snapshots, snapshot_steps, snapshot_labels, scenario.num_nodes, rounds, elapsed, obstacle)
        plot_channel_large_final(
            rows,
            cols,
            final,
            source_pos,
            dest_pos,
            obstacle,
            CHANNEL_THRESHOLD,
            scenario.num_nodes,
            rounds,
            elapsed,
            sd_dist,
            channel_nodes,
        )
        print("\nDone.")

    def _build_obstacles(self) -> torch.Tensor:
        rows, cols = self.spec.grid.rows, self.spec.grid.cols
        obstacle = torch.zeros(self.spec.grid.num_nodes, dtype=torch.bool)
        for row in range(0, min(40, rows)):
            if 25 < cols:
                obstacle[row * cols + 25] = True
        for row in range(10, rows):
            if 55 < cols:
                obstacle[row * cols + 55] = True
        return obstacle
