"""Tests for dynamic spatial simulation support."""

import torch

from diffield.sim import (
    EventSchedule,
    RelaxedRadiusScenario,
    ScheduledEvent,
    SimulationEngine,
    SpatialScenario,
    boids_acceleration_dense,
    bounce_in_box,
    build_spatial_graph,
    limit_speed,
    normalize_vectors,
)
from diffield.dsl import (
    gradient,
    mux,
    scatter,
    iterate,
    gather_min,
    gather_max,
    gather_sum,
    gather_avg,
    field,
)


class TestBuildSpatialGraph:
    def test_radius_edges_and_weights(self):
        positions = torch.tensor(
            [
                [0.0, 0.0],
                [0.2, 0.0],
                [0.8, 0.0],
            ],
            dtype=torch.float32,
        )

        edge_index, edge_weight = build_spatial_graph(
            positions, edge_radius=0.25, self_loops=False
        )

        edges = {tuple(e) for e in edge_index.t().tolist()}
        assert edges == {(0, 1), (1, 0)}
        assert torch.allclose(edge_weight, torch.tensor([1.0, 1.0]), atol=1e-6)

    def test_inverse_distance_mode(self):
        positions = torch.tensor(
            [
                [0.0, 0.0],
                [0.2, 0.0],
                [0.8, 0.0],
            ],
            dtype=torch.float32,
        )
        _, edge_weight = build_spatial_graph(
            positions,
            edge_radius=0.25,
            self_loops=False,
            edge_weight_mode="inverse_distance",
        )
        assert torch.allclose(edge_weight, torch.tensor([5.0, 5.0]), atol=1e-3)

    def test_distance_mode(self):
        positions = torch.tensor(
            [
                [0.0, 0.0],
                [0.2, 0.0],
                [0.8, 0.0],
            ],
            dtype=torch.float32,
        )
        _, edge_weight = build_spatial_graph(
            positions,
            edge_radius=0.25,
            self_loops=False,
            edge_weight_mode="distance",
        )
        assert torch.allclose(edge_weight, torch.tensor([0.2, 0.2]), atol=1e-6)


class TestSpatialScenarioGradients:
    def test_distance_weights_support_position_gradients_on_fixed_topology(self):
        positions = torch.tensor(
            [
                [0.0, 0.0],
                [0.35, 0.1],
                [0.9, 0.6],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        scenario = SpatialScenario(
            positions=positions,
            fully_connected=True,
            edge_weight_mode="distance",
        )
        engine = SimulationEngine.from_scenario(scenario)
        source = scenario.marker(0)

        def program(_runtime):
            return gradient(source, name="geo")

        output, _ = engine.run(rounds=3, program=program, signals={"source": source})

        loss = output[output.isfinite()].sum()
        loss.backward()

        assert positions.grad is not None
        assert torch.isfinite(positions.grad).all()
        assert positions.grad.abs().sum().item() > 0.0

    def test_relaxed_radius_scenario_supports_radius_gradients(self):
        radius = torch.tensor(0.35, requires_grad=True)
        positions = torch.tensor(
            [
                [0.0, 0.0],
                [0.36, 0.0],
                [0.80, 0.0],
            ],
            dtype=torch.float32,
        )
        scenario = RelaxedRadiusScenario(
            positions=positions,
            edge_radius=radius,
            relaxation_tau=0.03,
            penalty_strength=15.0,
        )
        engine = SimulationEngine.from_scenario(scenario)
        source = scenario.marker(0)

        def program(_runtime):
            return gradient(source, name="relaxed")

        output, _ = engine.run(rounds=3, program=program, signals={"source": source})

        loss = output[2]
        loss.backward()

        assert radius.grad is not None
        assert torch.isfinite(radius.grad)
        assert radius.grad.item() < 0.0


class TestSpatialScenarioTopology:
    def test_engine_uses_refreshed_topology_each_round(self):
        positions = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.9, 0.0],
            ],
            dtype=torch.float32,
        )
        scenario = SpatialScenario(positions=positions, edge_radius=0.25)
        engine = SimulationEngine.from_scenario(scenario)

        source = scenario.marker(0)

        def move_node_2(runtime):
            new_pos = runtime.scenario.positions.clone()
            new_pos[2] = torch.tensor([0.34, 0.0])
            runtime.scenario.update_positions(new_pos, refresh_topology=True)

        schedule = EventSchedule(
            [
                ScheduledEvent(round_idx=1, callback=move_node_2, name="move_node_2"),
            ]
        )

        def program(_runtime):
            return iterate(
                field.inf(),
                lambda d: mux(source, field.of(0.0), gather_min(scatter(d + 1.0))),
                name="dist",
            )

        output, _ = engine.run(
            rounds=3, program=program, signals={"source": source}, schedule=schedule
        )

        assert torch.isfinite(output[2])
        assert output[2] > 0.0

    def test_sync_context_updates_edges(self):
        positions = torch.tensor(
            [
                [0.0, 0.0],
                [0.2, 0.0],
                [0.8, 0.0],
            ],
            dtype=torch.float32,
        )
        scenario = SpatialScenario(positions=positions, edge_radius=0.25)
        engine = SimulationEngine.from_scenario(scenario)

        old_e = engine.ctx._ctx.edge_index.shape[1]

        new_positions = positions.clone()
        new_positions[2] = torch.tensor([0.35, 0.0])
        scenario.update_positions(new_positions, refresh_topology=True)
        scenario.sync_context(engine.ctx._ctx)

        new_e = engine.ctx._ctx.edge_index.shape[1]
        assert new_e > old_e


class TestSpatialScenarioInit:
    def test_hybrid_init_enforces_connected_graph(self):
        torch.manual_seed(4)
        positions = torch.rand(120, 2)
        scenario = SpatialScenario(
            positions=positions,
            edge_radius=0.05,
            ensure_init_connected=True,
            init_min_degree=2,
            init_k_neighbors=6,
        )

        assert scenario.init_graph_stats["num_components"] == 1.0
        assert scenario.init_graph_stats["min_degree"] >= 2.0

    def test_knn_mode_keeps_init_stats_available(self):
        torch.manual_seed(7)
        positions = torch.rand(50, 2)
        scenario = SpatialScenario(
            positions=positions,
            edge_radius=None,
            k_neighbors=5,
        )

        assert set(scenario.init_graph_stats.keys()) == {
            "num_edges",
            "min_degree",
            "num_components",
        }


class TestVectorHelpers:
    def test_limit_speed_and_normalize_vectors(self):
        vel = torch.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=torch.float32)
        normalized = normalize_vectors(vel)
        assert torch.isfinite(normalized).all()

        limited = limit_speed(vel, max_speed=2.0)
        speeds = limited.norm(dim=1)
        assert speeds[0] <= 2.0 + 1e-6
        assert speeds[1] <= 1e-6


class TestBounceInBox:
    def test_reflects_velocity(self):
        pos = torch.tensor([[-0.1, 0.5], [1.2, -0.3]], dtype=torch.float32)
        vel = torch.tensor([[1.0, 0.5], [-2.0, 1.5]], dtype=torch.float32)
        new_pos, new_vel = bounce_in_box(pos, vel)

        assert ((new_pos >= 0.0) & (new_pos <= 1.0)).all()
        assert new_vel[0, 0] < 0.0
        assert new_vel[1, 0] > 0.0
        assert new_vel[1, 1] < 0.0


class TestBoidsAcceleration:
    def test_dense_shape_and_finiteness(self):
        pos = torch.tensor(
            [[0.1, 0.1], [0.15, 0.1], [0.2, 0.12], [0.8, 0.8]],
            dtype=torch.float32,
        )
        vel = torch.tensor(
            [[0.01, 0.0], [0.0, 0.01], [-0.01, 0.0], [0.0, -0.01]],
            dtype=torch.float32,
        )
        acc = boids_acceleration_dense(
            pos,
            vel,
            radius=0.2,
            sep=0.08,
            w_sep=1.4,
            w_align=0.8,
            w_cohesion=0.6,
        )
        assert acc.shape == vel.shape
        assert torch.isfinite(acc).all()
