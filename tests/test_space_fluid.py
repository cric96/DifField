"""Scientific invariants of hard Space-Fluid, the surrogate and resumable experiments."""

from dataclasses import replace

import pytest
import torch

from diffield import with_mode
from examples.seams.artifacts import Budget, read_json
from examples.seams.metrics import interval
from examples.seams.space_fluid.campaign import (
    campaign,
    evaluation_jobs,
    initialization,
    methods,
    output_lock,
)
from examples.seams.space_fluid.config import CONDITIONS, protocol
from examples.seams.space_fluid.data import (
    RegionEpisode,
    combine,
    make_episode,
    normalization,
    symmetric_edges,
    training_bank,
)
from examples.seams.space_fluid.execution import central, decentralized, equivalence, kmeans
from examples.seams.space_fluid.metrics import fragmentation, recovery, summarize
from examples.seams.space_fluid.program import (
    COUNT,
    DISTANCE,
    ELECTED,
    LEADER,
    SAMPLE,
    TIMESTAMP,
    TOTAL,
    WIRE_FLOATS,
    EdgeMetric,
    LeaderStrength,
    make_program,
)
from examples.seams.space_fluid.training import (
    load_program,
    sensitivity,
    surrogate_objective,
    train_job,
)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def line(rounds=16):
    points = torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]])
    edges = symmetric_edges(((0, 1), (1, 2), (2, 3)), 4)
    observations = torch.arange(4).float().expand(rounds, -1).clone()
    return RegionEpisode(
        "line",
        points,
        observations.clone(),
        observations,
        torch.tensor([1.0, 0.7, 0.4, 0.1]),
        torch.ones(rounds, 4, dtype=torch.bool),
        [edges] * rounds,
        4,
        10,
        {"condition": "clean"},
    )


def tiny():
    return replace(
        protocol("smoke"),
        train_rounds=5,
        eval_rounds=8,
        nodes=6,
        train_episodes=1,
        validation_episodes=1,
        batch_size=2,
        updates=3,
    )


def test_compact_protocol_has_full_prespecified_inventory():
    c = protocol("compact-cpu")
    assert (c.seeds, c.updates, c.batch_size, c.nodes, c.train_rounds) == (
        (0, 1, 2),
        400,
        4,
        64,
        48,
    )
    assert (c.train_episodes, c.validation_episodes, c.test_episodes) == (16, 4, 4)
    assert c.main_lambda in c.lambdas
    jobs = list(evaluation_jobs(c))
    assert len(jobs) == 5264
    for spec, method, seed in jobs:
        if method["lambda"] not in (None, c.main_lambda):
            assert spec["panel"] == "main" and spec["condition"] == "clean"
        if spec["executor"] == "sync":
            assert seed == c.seeds[0]
        if spec["panel"] == "distributed":
            assert method["method"] != "kmeans"
    assert {m["method"] for m in methods(c)} == {
        "spatial",
        "combined",
        "value",
        "variance",
        "search",
        "parametric",
        "neural",
        "gnn",
        "kmeans",
    }
    assert [m["lambda"] for m in methods(c) if m["method"] == "gnn"] == [c.main_lambda]


@pytest.mark.parametrize("method", ["fixed", "search", "parametric", "neural"])
def test_metric_is_positive_symmetric_and_finite(method):
    metric = EdgeMetric(method).double()
    a = torch.tensor([0.0, 2.0, -1.0], dtype=torch.double)
    b = torch.tensor([0.0, -1.0, 2.0], dtype=torch.double)
    d = torch.tensor([0.0, 0.4, 0.4], dtype=torch.double)
    assert bool((metric(d, a, b) > 0).all())
    torch.testing.assert_close(metric(d, a, b), metric(d, b, a), atol=0, rtol=0)


def test_metric_knots_are_a_piecewise_linear_curve_of_the_level():
    a, b, d = torch.randn(3, 10)
    d, delta = d.abs(), (a - b).abs()
    one = EdgeMetric(weights=(1.5, 0.7, 0.2))  # one knot: the main-study metric
    expected = torch.stack((d, delta, d * delta), -1) @ torch.tensor((1.5, 0.7, 0.2)) + 1e-4
    torch.testing.assert_close(one(d, a, b), expected)
    metric = EdgeMetric(knots=5)
    with torch.no_grad():
        metric.log_weights.copy_(torch.randn(15))
    for knot, level in enumerate(torch.linspace(-1, 3, 5)):  # knots in normalized units
        value = level.expand(10)
        weight = metric.weights.view(5, 3)[knot, 0]
        torch.testing.assert_close(metric(d, value, value), d * weight + 1e-4)


def test_neural_initialization_preserves_parametric_metric():
    a, b, d = torch.randn(3, 10)
    d = d.abs()
    torch.testing.assert_close(EdgeMetric("neural")(d, a, b), EdgeMetric()(d, a, b), atol=0, rtol=0)
    priority = torch.rand(10)
    torch.testing.assert_close(
        LeaderStrength("neural")(a, b, d, priority), priority, atol=0, rtol=0
    )
    network = EdgeMetric("neural").network
    assert [
        (layer.in_features, layer.out_features)
        for layer in network
        if isinstance(layer, torch.nn.Linear)
    ] == [(3, 16), (16, 16), (16, 1)]


def test_hard_scr_election_collect_and_broadcast_regional_mean():
    ep = line()
    trace = central(ep, make_program("spatial"))
    assert trace.leaders[-1].tolist() == [0, 0, 0, 3]
    assert bool((trace.fields[..., DISTANCE] < 1).all())
    # Converge-cast toward the leader inside its region, then broadcast the region mean.
    assert trace.fields[-1, :, COUNT].tolist() == [3, 2, 1, 1]
    assert trace.fields[-1, :, SAMPLE].tolist() == [1, 1, 1, 3]
    assert trace.fields[-1, :, TIMESTAMP].tolist() == [15, 14, 13, 15]


def test_collect_conserves_devices_once_regions_settle():
    ep = make_episode(protocol("smoke"), family="gaussian")
    # A moving phenomenon reshapes the tree (transient omission is measured, not tested).
    ep.observations = ep.observations[:1].expand(ep.rounds, -1).clone()
    fields = central(ep, make_program("combined")).fields
    leaders = fields[-1, :, ELECTED] > 0.5
    assert fields[-1, leaders, COUNT].sum() == ep.nodes


def test_radius_boundary_is_excluded_and_self_candidacy_is_unconditional():
    ep = line()
    # cost = 1 exactly for the first link (including the positive floor).
    model = make_program("search", weights=((1 - 1e-4) / 0.1, 0, 0))
    trace = central(ep, model)
    assert trace.leaders[-1, 1].item() == 1
    ep.edges = [torch.empty(2, 0, dtype=torch.long)] * ep.rounds
    trace = central(ep, model)
    assert torch.equal(trace.leaders, torch.arange(4).expand(ep.rounds, -1))


def test_lexicographic_distance_then_id_and_edge_order():
    ep = line()
    ep.positions = torch.tensor([[-0.1, 0.0], [0.0, 0.0], [0.1, 0.0], [0.25, 0.0]])
    ep.priorities = torch.tensor([1.0, 0.0, 1.0, 0.0])
    # Equal strength: a local candidate always wins over a distant equal-strength candidate.
    expected = central(ep, make_program("spatial"))
    assert expected.leaders[-1].tolist() == [0, 0, 2, 2]
    ep.edges = [e.flip(1) for e in ep.edges]
    actual = central(ep, make_program("spatial"))
    torch.testing.assert_close(expected.fields, actual.fields, atol=0, rtol=0)


def test_partition_rejoin_and_removed_leader_heal_without_reset():
    ep = line(rounds=30)
    ep.edges = [ep.edges[0].clone() for _ in ep.edges]
    for t in range(6, 16):
        a, b = ep.edges[t]
        ep.edges[t] = ep.edges[t][:, (a != 0) & (b != 0)]
        ep.active[t, 0] = False
    model = make_program("spatial")
    central_trace, device_trace = central(ep, model), decentralized(ep, model)
    assert central_trace.leaders[15, 1:].tolist() == [1, 1, 1]
    assert central_trace.leaders[-1].tolist() == [0, 0, 0, 3]
    # Paused local state/output are retained, including their timestamp.
    torch.testing.assert_close(
        central_trace.fields[6:16, 0], central_trace.fields[5, 0].expand(10, -1)
    )
    assert equivalence(central_trace, device_trace)["values_close"]


@pytest.mark.parametrize("condition", CONDITIONS)
@pytest.mark.parametrize("method", ["combined", "value", "parametric", "neural", "gnn"])
def test_synchronous_identifiers_values_and_actual_message_bytes(condition, method):
    ep = make_episode(protocol("smoke"), condition=condition)
    model = make_program(method)
    a, b = central(ep, model), decentralized(ep, model)
    check = equivalence(a, b)
    assert all(
        check[key]
        for key in ("identifiers_equal", "elections_equal", "values_close", "traffic_equal")
    )
    assert a.traffic == [edges.shape[1] * WIRE_FLOATS * 4 for edges in ep.edges]


def test_remote_observations_are_delayed_and_oracle_is_not_visible():
    ep = line()
    ep.observations[:, 0] = torch.arange(ep.rounds).float()
    model = make_program("spatial")
    result = central(ep, model)
    # Node 1 receives the leader's published previous-round mean, never today's reading;
    # the leader folds its children's previous-round subtree sums into its own reading.
    fields = result.fields
    assert fields[5, 1, SAMPLE] == fields[4, 0, SAMPLE]
    assert fields[5, 2, SAMPLE] == fields[3, 0, SAMPLE]
    assert fields[5, 0, TOTAL] == ep.observations[5, 0] + fields[4, 1, TOTAL]
    ep.truth.fill_(9999)
    torch.testing.assert_close(result.fields, central(ep, model).fields)
    assert set(ep.signals(0)) == {"observation", "priority", "time"}


@pytest.mark.parametrize("probability", [0.5, 0.8])
def test_async_is_reproducible_with_stale_samples_and_fewer_transmissions(probability):
    ep = line()
    ep.observations[:, 0] = torch.arange(ep.rounds).float()
    model = make_program("spatial")
    a = decentralized(ep, model, activation_probability=probability, seed=7)
    b = decentralized(ep, model, activation_probability=probability, seed=7)
    torch.testing.assert_close(a.fields, b.fields)
    assert sum(a.activations) < ep.nodes * ep.rounds
    assert sum(a.traffic) < sum(central(ep, model).traffic)
    for t in range(ep.rounds):
        assert bool((a.fields[t, :, TIMESTAMP] <= t).all())
        if a.leaders[t, 2] == 0:
            # Several actual activations may propagate through multiple hops in
            # one wall-clock round. The mean must be the one the leader sent then.
            sent = int(a.fields[t, 2, TIMESTAMP])
            assert a.fields[t, 2, SAMPLE] == a.fields[sent, 0, SAMPLE]
    assert bool((a.fields[1:, 2, TIMESTAMP] < torch.arange(1, ep.rounds)).any())


def test_soft_training_gradient_reaches_every_scr_block_but_not_ids():
    ep = make_episode(protocol("smoke"), "train")  # several admissible parents per device
    model = make_program("neural")
    hard = central(ep, model)
    with with_mode("soft", tau=1e-6):  # a cold relaxation reproduces the hard program
        cold = central(ep, model)
    assert torch.equal(cold.fields[..., LEADER], hard.fields[..., LEADER])
    torch.testing.assert_close(cold.fields[..., SAMPLE], hard.fields[..., SAMPLE])
    model.probe = []
    with with_mode("soft", tau=0.1):
        trained = central(ep, model)
    parameters = (model.metric.log_weights, model.strength.weights)
    for grad in torch.autograd.grad(
        trained.fields[..., LEADER].sum(), parameters, retain_graph=True, allow_unused=True
    ):
        assert grad is None or torch.equal(grad, torch.zeros_like(grad))
    loss = (trained.fields[..., SAMPLE] - ep.truth).square().mean()
    for name in ("S", "G", "C", "B"):  # every block's output carries gradient
        outputs = [blocks[name] for blocks in model.probe]
        grads = torch.autograd.grad(loss, outputs, retain_graph=True, allow_unused=True)
        assert sum(g.abs().sum() for g in grads if g is not None) > 0, name
    model.probe = None
    surrogate_objective(ep, model, 0.2, 0.1, tiny()).backward()
    for parameter in parameters:
        assert torch.isfinite(parameter.grad).all() and parameter.grad.norm() > 0


def test_batch_matches_independent_episodes_including_discrete_ids():
    episodes = [make_episode(tiny(), "train", index=i) for i in range(2)]
    model = make_program("neural")
    batched = central(combine(episodes), model).fields
    for i, ep in enumerate(episodes):
        single = central(ep, model).fields.clone()
        single[..., LEADER] += i * ep.nodes
        torch.testing.assert_close(batched[:, i * ep.nodes : (i + 1) * ep.nodes], single)


def test_splits_geometry_and_training_only_normalization():
    c = tiny()
    bank = training_bank(c)
    keys = [e.key for episodes in bank.values() for e in episodes]
    assert len(keys) == len(set(keys))
    for family in ("ring", "front"):
        with pytest.raises(ValueError, match="reserved"):
            make_episode(c, "train", family)
    intervals = [
        [make_episode(c, split, index=i).metadata["geometry"]["width"] for i in range(3)]
        for split in ("train", "validation", "test")
    ]
    assert max(intervals[0]) < min(intervals[1]) < max(intervals[1]) < min(intervals[2])
    norm = normalization(bank["train"])
    for ep in bank["validation"]:
        ep.truth *= 1000
    assert normalization(bank["train"]) == norm


def test_condition_streams_share_truth_priorities_and_keep_edges_symmetric():
    clean = make_episode(tiny())
    for condition in CONDITIONS:
        changed = make_episode(tiny(), condition=condition)
        torch.testing.assert_close(changed.truth, clean.truth)
        torch.testing.assert_close(changed.priorities, clean.priorities)
        for edges in changed.edges:
            values = set(map(tuple, edges.T.tolist()))
            assert values == {(b, a) for a, b in values}


@pytest.mark.parametrize("method", ["parametric", "neural", "search", "cem", "gnn"])
def test_resume_matches_uninterrupted_optimizer_batches_and_hard_selection(tmp_path, method):
    c = replace(tiny(), search_candidates=30)  # CEM refits after 24
    bank = training_bank(c)
    norm = normalization(bank["train"])

    class Interrupt:
        calls = 0

        def check(self):
            self.calls += 1
            if self.calls == 3:
                raise TimeoutError("interrupted")

    with pytest.raises(TimeoutError):
        train_job(tmp_path / "resumed", c, bank, norm, method, 0, 0.1, Interrupt())
    for name in ("full", "resumed"):
        train_job(tmp_path / name, c, bank, norm, method, 0, 0.1, Budget(60))
    a, b = [
        torch.load(tmp_path / name / "last.pt", weights_only=True) for name in ("full", "resumed")
    ]
    assert a["history"] == b["history"] and a["selected_update"] == b["selected_update"]
    if method == "cem":
        assert a["cem"] == b["cem"] and a["cem"]["mean"] != [0.5] * 6  # refitted once
    assert a["best_score"] == min(
        row["validation_hard"] for row in a["history"] if "validation_hard" in row
    )
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)
    loaded = load_program(tmp_path / "full/best.pt")
    assert all(not p.requires_grad for p in loaded.parameters())


def test_sensitivity_preserves_checkpoint_and_compares_with_hard_differences():
    c = tiny()
    model = make_program("parametric")
    original = {k: v.clone() for k, v in model.state_dict().items()}
    result = sensitivity(model, make_episode(c, "validation"), {"scale": 0.2}, 0.1, c)
    assert [r["parameter"] for r in result["named"]] == [
        *(f"metric.log_weights[{i}]" for i in range(3)),
        *(f"strength.weights[{i}]" for i in range(3)),
    ]
    assert result["gradient_norm"] > 0 and len(result["descent"]) == 3
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key])


def test_fragmentation_recovery_and_constant_normalization():
    labels = torch.tensor([0, 1, 0, 1])
    assert fragmentation(labels, line().edges[0], torch.ones(4, dtype=torch.bool)) == 1
    assert recovery([0.1] * 5 + [0.9] * 2 + [0.1] * 5, 5)["rounds"] == 2
    assert recovery([0.1] * 5 + [0.9] * 6, 5)["status"] == "no_recovery"
    assert recovery([0.9] * 5 + [0.1] * 5, 5)["status"] == "attained"
    ep = make_episode(tiny(), family="constant")
    metrics, _ = summarize(ep, central(ep, make_program("combined")), 0.2, tiny())
    assert metrics["nrmse"] >= 0 and metrics["homogeneity"] < 1e-12
    assert metrics["mean_region_std"] < 1e-6 and metrics["region_mean_std"] < 1e-6
    metrics, _ = summarize(ep, central(ep, make_program("gnn")), 0.2, tiny())
    assert metrics["nrmse"] >= 0 and metrics["regions"] == ep.nodes  # every device samples


def test_kmeans_uses_observations_not_truth_and_has_no_fake_message_cost():
    c = tiny()
    ep = make_episode(c)
    norm = {"mean": 0.0, "scale": 0.2}
    a = kmeans(ep, 3, norm)
    ep.truth *= 100
    b = kmeans(ep, 3, norm)
    assert torch.equal(a.fields, b.fields)
    metrics, _ = summarize(ep, a, 0.2, c)
    assert metrics["regions"] == 3 and metrics["message_bytes"] is None


def test_paired_intervals_reject_incomplete_and_duplicate_cells():
    rows = [{"seed": s, "episode": e, "metrics": {"x": s + e}} for s in range(2) for e in range(2)]
    assert interval(rows, "x", reference=rows)["ci95"] == [0.0, 0.0]
    assert interval(rows[:-1], "x", reference=rows)["ci95"] is None
    with pytest.raises(ValueError, match="Duplicate"):
        interval(rows + rows[:1], "x")


def test_budget_records_pending_work_without_reducing_protocol(tmp_path):
    c = tiny()
    assert campaign(tmp_path / "run", c, seconds=1e-9) == 2
    state = read_json(tmp_path / "run/progress.json")
    assert state["status"] == "incomplete" and state["evaluation"]["complete"] == 0
    assert read_json(tmp_path / "run/manifest.json")["config"]["updates"] == 3
    assert (tmp_path / "run/REPORT.md").exists()
    with pytest.raises(ValueError, match="Configuration mismatch"):
        initialization(tmp_path / "run", replace(c, updates=1))


def test_cli_has_space_fluid_and_rejects_retired_commands(monkeypatch, tmp_path):
    from examples.seams.__main__ import main  # noqa: PLC0415 -- test CLI import isolation

    calls = []
    monkeypatch.setattr(
        "examples.seams.space_fluid.campaign.campaign", lambda *a: calls.append(a) or 0
    )
    assert main(["space-fluid", "--stage", "evaluate", "--out", str(tmp_path)]) == 0
    assert calls[0][0] == tmp_path and calls[0][2] == "evaluate"
    for command in ("prepare-intel", "learning-v2", "collection-controls"):
        with pytest.raises(SystemExit):
            main([command])


def test_output_lock_excludes_concurrent_runs_and_releases_after_failure(tmp_path):
    with output_lock(tmp_path):  # noqa: SIM117 -- the first invocation must retain its lock
        with pytest.raises(RuntimeError, match="Another Space-Fluid process"):
            with output_lock(tmp_path):
                pytest.fail("Concurrent output access must not be admitted")
    with pytest.raises(ValueError, match="simulated"), output_lock(tmp_path):
        raise ValueError("simulated failure")
    with output_lock(tmp_path):
        assert (tmp_path / ".run.lock").exists()


def test_zone_clusters_levels_ari_and_peak_crash():
    from examples.seams.space_fluid.clusters import (  # noqa: PLC0415
        GAP,
        cluster_episode,
        cluster_metrics,
    )

    c = replace(protocol("smoke"), nodes=36, eval_rounds=12)
    ep = cluster_episode(c, "test", 0, count=3, condition="crash")
    labels = torch.tensor(ep.metadata["labels"])
    assert set(labels.tolist()) == {0, 1, 2}
    zone_means = torch.stack([ep.observations[0, labels == k].mean() for k in range(3)])
    assert float(zone_means.sort().values.diff().min()) > GAP / 2  # zones carry distinct levels
    torch.testing.assert_close(ep.truth, ep.observations)
    metrics, _ = cluster_metrics(ep, labels.expand(ep.rounds, -1))
    assert metrics["ari_final"] == 1 and metrics["count_error"] == 0
    assert int((~ep.active[-1]).sum()) == 3 and bool(ep.active[ep.fault_at - 1].all())


def test_hotspot_campaign_scales_knots_and_keeps_equivalence(tmp_path):
    from examples.seams.space_fluid.hotspot import STUDIES, campaign  # noqa: PLC0415

    assert campaign(tmp_path, "smoke", 600) == 0
    (study,) = STUDIES["smoke"]
    summary = read_json(tmp_path / study.name / "summary.json")
    parameters = {name: row["parameters"] for name, row in summary["table"].items()}
    assert [parameters[f"cem-k{k}"] for k in study.knots] == [3 * k + 3 for k in study.knots]
    assert parameters["fixed-combined"] == 6 and 500 < parameters["neural"] < parameters["gnn"]
    checks = summary["equivalence"]
    assert [c["identifiers_equal"] and c["values_close"] for c in checks] == [True, True]
    assert (tmp_path / "REPORT.md").exists()
    assert (tmp_path / study.name / "figures/hotspot-curves.png").exists()


def test_static_and_moving_scenarios_and_crash():
    from examples.seams.space_fluid.clusters import cluster_episode, truth_labels  # noqa: PLC0415

    c = replace(protocol("smoke"), nodes=36, eval_rounds=12)
    frozen = make_episode(c, "test", "gaussian", 0, static=True, condition="crash")
    torch.testing.assert_close(frozen.truth[0], frozen.truth[-1])
    assert int((~frozen.active[-1]).sum()) == 4 and bool(frozen.active[frozen.fault_at - 1].all())
    moving = make_episode(c, "test", "gaussian", 0)
    assert not torch.allclose(moving.truth[0], moving.truth[-1])
    still = cluster_episode(c, "test", 0, count=3)
    drifting = cluster_episode(c, "test", 0, count=3, moving=True)
    torch.testing.assert_close(drifting.truth[0], still.truth[0])  # the drift starts in place
    assert not torch.allclose(drifting.truth[0], drifting.truth[-1])
    assert truth_labels(drifting).shape == (12, 36) == truth_labels(still).shape
