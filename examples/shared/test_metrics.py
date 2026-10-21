"""Paired intervals and censored recovery."""

import numpy as np
import pytest

from examples.shared.metrics import bootstrap_band, interval, recovery


def test_paired_intervals_reject_incomplete_and_duplicate_cells():
    rows = [{"seed": s, "episode": e, "metrics": {"x": s + e}} for s in range(2) for e in range(2)]
    assert interval(rows, "x", reference=rows)["ci95"] == [0.0, 0.0]
    assert interval(rows[:-1], "x", reference=rows)["ci95"] is None
    with pytest.raises(ValueError, match="Duplicate"):
        interval(rows + rows[:1], "x")


def test_recovery_counts_rounds_from_the_event_and_censors_runs_that_never_recover():
    assert recovery([0.1] * 5 + [0.9] * 2 + [0.1] * 5, 5)["rounds"] == 2
    assert recovery([0.1] * 5 + [0.9] * 6, 5)["status"] == "no_recovery"
    assert recovery([0.9] * 5 + [0.1] * 5, 5)["status"] == "attained"


def test_bootstrap_band_holds_the_mean_curve_and_collapses_on_identical_runs():
    curves = np.array([[0.0, 1.0], [2.0, 3.0]])

    mean, low, high = bootstrap_band(curves)

    assert mean.tolist() == [1.0, 2.0]
    assert (low <= mean).all() and (mean <= high).all()
    assert bootstrap_band(np.ones((3, 2)))[1].tolist() == [1.0, 1.0]
