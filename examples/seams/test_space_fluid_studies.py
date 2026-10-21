"""The scenarios and clusters studies of Space-Fluid: their campaigns end to end."""

import pytest
import torch

from examples.seams.space_fluid import cluster_gnn, clusters, scenarios
from examples.seams.space_fluid.config import protocol
from examples.shared.artifacts import json_write, read_json
from examples.shared.campaign import Budget


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("study", [scenarios, clusters])
def test_study_campaign_stops_resumably_when_the_budget_runs_out(tmp_path, study):
    assert study.campaign(tmp_path, "smoke", 1e-9) == 2

    assert read_json(tmp_path / "manifest.json")["profile"] == "smoke"


@pytest.mark.parametrize("study", [scenarios, clusters])
def test_interrupted_study_campaign_is_resumable(tmp_path, monkeypatch, study):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(study, "train", interrupt)

    assert study.campaign(tmp_path, "smoke", 60) == 2


@pytest.mark.parametrize("study", [scenarios, clusters])
def test_study_refuses_to_resume_a_run_whose_configuration_changed(tmp_path, study):
    study.campaign(tmp_path, "smoke", 1e-9)
    manifest = read_json(tmp_path / "manifest.json")
    manifest["config"]["updates"] += 1
    json_write(tmp_path / "manifest.json", manifest)

    with pytest.raises(ValueError, match="Resume mismatch: config"):
        study.campaign(tmp_path, "smoke", 1e-9)


NORM = {"mean": 0.5, "scale": 0.3}


class Countdown:
    """A budget that runs out after a number of checks."""

    def __init__(self, checks):
        self.checks = checks

    def check(self):
        self.checks -= 1
        if self.checks < 0:
            raise TimeoutError("budget")


def small_bank():
    config = protocol("smoke")
    return config, clusters.bank(config, clusters.SIZES["smoke"], "static")


def test_central_gnn_keeps_the_model_of_lowest_hard_validation(tmp_path):
    config, bank = small_bank()

    cluster_gnn.train(tmp_path, config, bank, NORM, seed=0, penalty=0.3, budget=Budget(60))

    record = read_json(tmp_path / "training.json")
    validated = [r for r in record["history"] if "validation_hard" in r]
    scores = [r["validation_hard"] for r in validated]
    assert record["history"][0]["step"] == 0 and record["complete"]
    assert record["best_score"] == min(scores)
    assert record["selected_update"] == validated[scores.index(min(scores))]["step"]
    assert set(torch.load(tmp_path / "best.pt", weights_only=True)) == {"normalization", "model"}


def test_central_gnn_labels_each_cluster_by_its_smallest_active_device():
    _, bank = small_bank()
    episode = bank["validation"][0]

    def one_slot(graph):
        logits = torch.zeros(len(graph["values"]), cluster_gnn.SLOTS)
        logits[:, 0] = 1
        return logits

    labels = cluster_gnn.labels(one_slot, episode)

    for t in range(episode.rounds):
        active = episode.active[t]
        assert (labels[t, ~active] == -1).all()
        for label in labels[t, active].unique():
            members = (labels[t] == label).nonzero().flatten()
            assert label == members.min() and active[label]


def test_interrupted_central_gnn_training_resumes_exactly(tmp_path):
    config, bank = small_bank()
    arguments = (config, bank, NORM)
    with pytest.raises(TimeoutError):
        cluster_gnn.train(tmp_path / "resumed", *arguments, seed=0, penalty=1, budget=Countdown(2))

    for name in ("resumed", "full"):
        cluster_gnn.train(tmp_path / name, *arguments, seed=0, penalty=1, budget=Budget(60))

    a, b = [read_json(tmp_path / name / "training.json") for name in ("full", "resumed")]
    assert a["history"] == b["history"] and a["selected_update"] == b["selected_update"]
    best = [torch.load(tmp_path / n / "best.pt", weights_only=True) for n in ("full", "resumed")]
    for key, value in best[0]["model"].items():
        assert torch.equal(value, best[1]["model"][key])


def test_clusters_refuse_a_synchronous_run_whose_messages_differ_from_batched(
    tmp_path, monkeypatch
):
    def other_traffic(expected, actual):
        return {
            "identifiers_equal": True,
            "values_close": True,
            "elections_equal": True,
            "traffic_equal": False,
        }

    monkeypatch.setattr("examples.seams.space_fluid.execution.equivalence", other_traffic)
    config = protocol("smoke")
    spec = {"panel": "distributed", "moving": False, "count": 4, "separation": 0.3,
            "index": 0, "condition": "clean", "executor": "sync"}  # fmt: skip
    job = (tmp_path, config, NORM, spec, "fixed-combined", 0, "sync", tmp_path / "row.json")

    with pytest.raises(AssertionError, match="Synchronous mismatch"):
        clusters.evaluate_one(job)
