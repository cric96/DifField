"""Scientific and numerical invariants of the Boids learning comparison."""

from dataclasses import fields
from pathlib import Path

import pytest
import torch

from examples.seams.artifacts import Budget, checksum, json_write, read_json, tensor_write
from examples.seams.boids import CampaignBudget, configuration, evaluate_job, initialize, traces
from examples.seams.boids_dynamics import (
    TEACHER,
    Boids,
    dense_step,
    radius_edges,
)
from examples.seams.boids_learning import (
    GraphBoids,
    ObservedBatch,
    combine,
    free_rollout,
    make_model,
    objective,
    train_job,
    transitions,
)
from examples.seams.boids_report import METHODS, complete_panel, report
from examples.seams.metrics import interval


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def episode(seed=0, rounds=3, nodes=8):
    gen = torch.Generator().manual_seed(seed)
    p = torch.rand(nodes, 2, generator=gen)
    v = torch.randn(nodes, 2, generator=gen) * 0.006
    with torch.no_grad():
        tp, tv = free_rollout(None, p, v, rounds)
    return {
        "positions": p,
        "velocities": v,
        "target_pos": tp,
        "target_vel": tv,
        "key": f"episode-{seed}",
    }


@pytest.mark.parametrize(
    "positions,velocities",
    [
        ([[0.5, 0.5]], [[0.001, -0.002]]),
        ([[0.2, 0.2], [0.2, 0.2], [0.24, 0.21]], [[0.0, 0.001], [0.002, 0.0], [0.001, -0.002]]),
        ([[0.001, 0.5], [0.04, 0.5], [0.98, 0.99]], [[-0.02, 0.0], [0.0, 0.01], [0.02, 0.02]]),
    ],
)
def test_dense_dsl_step_including_isolation_coincidence_and_bounce(positions, velocities):
    p, v = torch.tensor(positions), torch.tensor(velocities)
    model = Boids()
    with torch.no_grad():
        model.log_weights.copy_(torch.tensor(TEACHER).log())
        expected = dense_step(p, v)
        actual = model.step(p, v, radius_edges(p))
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_three_weight_checkpoint_stays_compatible():
    model = Boids()
    assert set(model.state_dict()) == {"log_weights"}
    saved = {"model": {"log_weights": torch.tensor(TEACHER).log()}, "selected_update": 100}
    model.load_state_dict(saved["model"])
    torch.testing.assert_close(model.log_weights.exp(), torch.tensor(TEACHER))
    e = episode(seed=7)
    with torch.no_grad():
        actual = model.step(e["positions"], e["velocities"], radius_edges(e["positions"]))
    torch.testing.assert_close(actual.positions, e["target_pos"][0], atol=1e-6, rtol=1e-5)


def test_gradients_match_finite_differences_away_from_boundaries():
    model = Boids().double()
    p = torch.tensor(
        [[0.2, 0.3], [0.25, 0.32], [0.4, 0.35]], dtype=torch.float64, requires_grad=True
    )
    v = torch.tensor(
        [[0.001, 0.003], [-0.002, 0.003], [0.001, -0.001]], dtype=torch.float64, requires_grad=True
    )
    edges = radius_edges(p)
    assert torch.autograd.gradcheck(
        lambda a, b: model.step(a, b, edges), (p, v), eps=1e-6, atol=2e-4
    )


@pytest.mark.parametrize("variant", ["parametric", "gnn32", "gnn64"])
def test_disjoint_batch_matches_independent_transitions(variant):
    e1, e2 = episode(2), episode(3, nodes=5)
    a, b = transitions(e1), transitions(e2)
    batch = combine([a, b])
    model = make_model(variant, 0)
    with torch.no_grad():
        actual = model.step(batch.positions, batch.velocities, batch.edge_index)
        separate = [model.step(x.positions, x.velocities, x.edge_index) for x in (a, b)]
    for i in range(3):
        torch.testing.assert_close(
            actual[i], torch.cat([x[i] for x in separate]), atol=2e-7, rtol=1e-5
        )
    source, target = batch.edge_index
    assert torch.equal(source < len(a.positions), target < len(a.positions))


@pytest.mark.parametrize("hidden", [32, 64])
def test_gnn_permutation_and_empty_graph(hidden):
    e = episode(5)
    p, v = e["positions"], e["velocities"]
    model = GraphBoids(hidden)
    order = torch.randperm(len(p))
    with torch.no_grad():
        a = model.step(p, v, radius_edges(p))
        b = model.step(p[order], v[order], radius_edges(p[order]))
        empty = model.step(p[:1], v[:1], torch.empty((2, 0), dtype=torch.long))
    for x, y in zip(a, b, strict=True):
        torch.testing.assert_close(x[order], y, atol=2e-7, rtol=1e-5)
    assert all(torch.isfinite(x).all() and x.shape == (1, 2) for x in empty)
    assert sum(x.numel() for x in model.parameters()) == (2498 if hidden == 32 else 9090)


def test_training_uses_only_observed_states(monkeypatch, tmp_path):
    e = episode(4)
    batch = transitions(e)
    assert "preclip" not in {f.name for f in fields(ObservedBatch)}

    def forbidden(*args, **kwargs):
        raise AssertionError("Observed training accessed the teacher")

    monkeypatch.setattr("examples.seams.boids_learning.dense_step", forbidden)
    model = Boids()
    observed, _ = objective(model, batch)
    observed.backward()
    assert torch.isfinite(model.log_weights.grad).all()
    config = configuration("smoke")
    assert config["regimes"] == ["observed"]
    train_job(tmp_path, config, [batch], batch, "parametric", 0, 0.03, Budget(60))


def test_parameter_recovery_with_observations_only():
    batch = combine([transitions(e) for e in traces(configuration("paper-cpu"), "train", 4)])
    model = Boids()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    for _ in range(500):
        optimizer.zero_grad()
        loss = objective(model, batch)[0]
        loss.backward()
        optimizer.step()
    torch.testing.assert_close(model.log_weights.exp(), torch.tensor(TEACHER), rtol=0.02, atol=0)


@pytest.mark.parametrize("variant", ["parametric", "gnn32"])
def test_resume_preserves_optimizer_batches_and_selected_checkpoint(tmp_path, variant):
    config = {**configuration("smoke"), "updates": 4, "validate_every": 1}
    train = [transitions(episode(i)) for i in range(4)]
    validation = transitions(episode(10))

    class Interrupt:
        calls = 0

        def check(self):
            self.calls += 1
            if self.calls == 3:
                raise TimeoutError("simulated interruption")

    arguments = (config, train, validation, variant, 0, 0.003)
    with pytest.raises(TimeoutError):
        train_job(tmp_path / "resumed", *arguments, Interrupt())
    for name in ("full", "resumed"):
        train_job(tmp_path / name, *arguments, Budget(60))
    a, b = [
        torch.load(tmp_path / name / "last.pt", weights_only=True) for name in ("full", "resumed")
    ]
    assert a["history"] == b["history"]
    assert a["best_checkpoint"]["selected_update"] == b["best_checkpoint"]["selected_update"]
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key])


def test_minibatches_are_shared_across_variants(tmp_path):
    config = configuration("smoke")
    episodes = [episode(i) for i in range(6)]
    train = [transitions(e) for e in episodes]
    histories = []
    for variant in ("parametric", "gnn32", "gnn64"):
        _, result = train_job(
            tmp_path / variant,
            config,
            train,
            train[0],
            variant,
            2,
            0.003,
            Budget(60),
        )
        histories.append([r["batch_indices"] for r in result["history"]])
    assert histories[0] == histories[1] == histories[2]


def test_zero_update_is_selectable_and_budget_is_cumulative(tmp_path):
    config = {**configuration("smoke"), "updates": 0}
    batch = transitions(episode())
    _, result = train_job(
        tmp_path / "job", config, [batch], batch, "parametric", 0, 0.03, Budget(60)
    )
    assert result["selected_update"] == 0
    budget = CampaignBudget(tmp_path, 1)
    budget.used = 2
    budget.save()
    resumed = CampaignBudget(tmp_path, 14400)
    with pytest.raises(TimeoutError):
        resumed.check()
    assert read_json(tmp_path / "budget.json")["limit_seconds"] == 1


def test_missing_seed_cannot_yield_a_complete_confidence_interval():
    config = {"seeds": [0, 1]}
    data = {"test": [episode(0), episode(1)]}
    rows = [
        {"seed": 0, "episode": e["key"], "metrics": {"position_rmse": 0.1}} for e in data["test"]
    ]
    stats = interval(complete_panel(rows, config, data), "position_rmse")
    assert stats["ci95"] is None and stats["mean"] is None
    assert stats["n"] == 4 and stats["invalid"] == 2


def test_evaluation_repairs_missing_replay_and_records_nonfinite_failure(tmp_path):
    data = {"test": [episode()]}
    config = {"rounds": 3}
    out = tmp_path / "out"
    json_write(out / "manifest.json", {"data_sha256": "fixture"})
    args = (out, tmp_path / "source", data, config, "observed", "fixed", 0)
    evaluate_job(*args, Boids(), {}, Budget(60))
    replay = out / "replays/observed/fixed/episode-0.pt"
    replay.unlink()
    evaluate_job(*args, Boids(), {}, Budget(60))
    assert replay.exists()
    model = GraphBoids(32)
    with torch.no_grad():
        next(model.parameters()).fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="Nonfinite Boids rollout"):
        evaluate_job(
            out, tmp_path / "source", data, config, "observed", "gnn32", 0, model, {}, Budget(60)
        )
    failed = read_json(out / "episodes/observed/gnn32/seed0/episode-0.json")
    assert failed["status"] == "failed" and not failed["metrics"]


@pytest.mark.parametrize("historical", [False, True])
def test_report_ignores_artifacts_outside_observed_comparison(tmp_path, historical):
    config = configuration("smoke")
    if historical:
        config["methods"].insert(2, "legacy")
    data = {"test": [episode(0), episode(1)]}
    for method in config["methods"]:
        for ep in data["test"]:
            row = {
                "regime": "observed",
                "method": method,
                "seed": 0,
                "episode": ep["key"],
                "status": "complete",
                "metrics": {"position_rmse": 0.1, "velocity_rmse": 0.01},
                "parameters": 3,
                "weights": list(TEACHER),
                "selected_update": 0,
                "training_seconds": 0,
            }
            json_write(tmp_path / "episodes/observed" / method / "seed0" / f"{ep['key']}.json", row)
    json_write(
        tmp_path / "episodes/retired/parametric/seed0/stale.json", {**row, "regime": "retired"}
    )
    assert report(tmp_path, data, config, None, render=False)
    results = read_json(tmp_path / "results.json")
    expected = len(config["methods"]) * 2
    assert len(results) == expected and {r["regime"] for r in results} == {"observed"}
    document = (tmp_path / "REPORT.md").read_text()
    assert f"{expected}/{expected}" in document and "retired" not in document
    assert ("Legacy" in document) == historical


def test_standalone_dataset_is_deterministic_and_resume_checks_hashes(tmp_path):
    config = configuration("smoke")
    out = tmp_path / "run"
    data = initialize(None, out, config)
    manifest = read_json(out / "manifest.json")
    assert manifest["source"] is None and manifest["source_sha256"] == {}
    assert manifest["config"]["methods"] == list(METHODS)
    resumed = initialize(None, out, config)
    repeated = initialize(None, tmp_path / "repeat", config)
    keys = []
    for split in ("train", "validation", "test"):
        for a, b, c in zip(data[split], resumed[split], repeated[split], strict=True):
            keys.append(a["key"])
            for field in ("positions", "velocities", "target_pos", "target_vel"):
                assert torch.equal(a[field], b[field]) and torch.equal(a[field], c[field])
    assert len(keys) == len(set(keys)) == 7
    data["test"][0]["target_pos"][0, 0, 0] += 0.1
    tensor_write(out / "data/traces.pt", data)
    with pytest.raises(ValueError, match="resume mismatch: data_sha256"):
        initialize(None, out, config)


@pytest.mark.parametrize("historical", [False, True])
def test_source_import_is_optional_and_read_only(tmp_path, historical):
    config = configuration("smoke")
    source = tmp_path / "archive"
    data = initialize(None, tmp_path / "fresh", config)
    tensor_write(source / "boids/traces.pt", data)
    json_write(source / "manifest.json", {"archived": True})
    if historical:
        tensor_write(source / "boids/seed0.pt", {"model": Boids().state_dict()})
        json_write(source / "boids/results.json", [{"historical": True}])
    hashes = {p: checksum(p) for p in source.rglob("*") if p.is_file()}
    out = tmp_path / "imported"
    initialize(source, out, config)
    manifest = read_json(out / "manifest.json")
    assert ("legacy" in manifest["config"]["methods"]) == historical
    assert (out / "historical-results.json").exists() == historical
    initialize(source, out, config)
    assert hashes == {p: checksum(p) for p in hashes}
    if historical:
        with pytest.raises(ValueError, match="checkpoint for every seed"):
            initialize(source, tmp_path / "partial", {**config, "seeds": [0, 1]})


def test_cli_has_only_boids_and_no_source_requirement(monkeypatch, tmp_path, capsys):
    from examples.seams.__main__ import main

    calls = []
    monkeypatch.setattr("examples.seams.boids.campaign", lambda *args: calls.append(args) or 0)
    assert main(["boids", "--out", str(tmp_path)]) == 0
    assert calls == [(None, tmp_path, "smoke", 14400)]
    assert main(["boids"]) == 0
    assert calls[-1] == (None, Path("generated/seams/boids-smoke"), "smoke", 14400)
    with pytest.raises(SystemExit) as error:
        main(["boids-v2"])
    assert error.value.code == 2
    assert "invalid choice: 'boids-v2'" in capsys.readouterr().err
