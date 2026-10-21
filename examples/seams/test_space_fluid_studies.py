"""The scenarios and clusters studies of Space-Fluid: their campaigns end to end."""

import pytest
import torch

from examples.seams.space_fluid import clusters, scenarios
from examples.shared.artifacts import json_write, read_json


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
