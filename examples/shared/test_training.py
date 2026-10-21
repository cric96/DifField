"""Resumable training with selection on validation."""

import pytest
import torch
from torch import nn

from examples.shared.campaign import Budget
from examples.shared.training import fit


class Countdown:
    """A budget that runs out after a number of checks."""

    def __init__(self, checks):
        self.checks = checks

    def check(self):
        self.checks -= 1
        if self.checks < 0:
            raise TimeoutError("budget")


def job(last, budget, *, updates=6, scores=None):
    """A seeded linear regression; ``scores`` replaces its validation loss, update by update."""
    torch.manual_seed(0)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    x = torch.randn(16, 2, generator=torch.Generator().manual_seed(1))
    y = x.sum(1, keepdim=True)
    weights = []

    def update(step):
        optimizer.zero_grad()
        loss = (model(x) - y).square().mean()
        loss.backward()
        optimizer.step()
        weights.append(model.weight.detach().clone())
        return {"step": step, "loss": float(loss.detach())}

    def validate():
        score = float((model(x) - y).square().mean()) if scores is None else scores[len(weights)]
        return score, {"validation": score}

    state = fit(
        model, optimizer, update, validate, updates=updates, validate_every=2, last=last,
        budget=budget,
    )  # fmt: skip
    return state, model, weights


def test_resumed_training_repeats_the_uninterrupted_one_exactly(tmp_path):
    full, full_model, _ = job(tmp_path / "full.pt", Budget(60))
    with pytest.raises(TimeoutError):
        job(tmp_path / "resumed.pt", Countdown(3))

    resumed, resumed_model, _ = job(tmp_path / "resumed.pt", Budget(60))

    assert resumed.history == full.history and resumed.complete
    assert (resumed.best_score, resumed.selected_update) == (full.best_score, full.selected_update)
    for a, b in zip(full_model.parameters(), resumed_model.parameters(), strict=True):
        assert torch.equal(a, b)


def test_selection_keeps_the_earliest_lowest_score_including_the_untrained_model(tmp_path):
    untrained, _, _ = job(tmp_path / "untrained.pt", Budget(60), updates=0)
    tied, _, _ = job(tmp_path / "tied.pt", Budget(60), scores=[1.0] * 7)

    assert untrained.selected_update == 0 and untrained.complete
    assert tied.selected_update == 0 and tied.history[1]["validation"] == 1.0


def test_nonfinite_validation_stops_training(tmp_path):
    with pytest.raises(FloatingPointError):
        job(tmp_path / "last.pt", Budget(60), scores=[1.0, 1.0, float("nan")] + [1.0] * 4)


def test_trained_model_holds_the_selected_weights(tmp_path):
    state, model, weights = job(tmp_path / "last.pt", Budget(60), scores=[3, 3, 2, 3, 1, 3, 2])

    assert state.selected_update == 4 and state.best_score == 1
    assert torch.equal(model.weight, weights[3])
