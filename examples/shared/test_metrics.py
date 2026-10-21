"""Paired intervals and censored recovery."""

import numpy as np
import pytest

from examples.shared.metrics import (
    bootstrap_band,
    event_windows,
    interval,
    interval_cell,
    recovery,
    recovery_summary,
    relative_deviation,
)


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


def test_interval_cell_shows_the_mean_its_interval_or_a_dash():
    assert interval_cell({"mean": 0.25, "ci95": [0.125, 0.5]}) == "0.250 [0.125, 0.500]"
    assert interval_cell({"mean": 0.25, "ci95": None}, 1, sign=True) == "+0.2"
    assert interval_cell({"mean": None, "ci95": None}) == "—"


def test_event_windows_run_from_each_event_to_the_next_and_the_last_to_the_end():
    assert event_windows([10, 20, 30], 40) == [(10, 20), (20, 30), (30, 40)]


def test_recovery_summary_leaves_censored_runs_out_of_the_rounds():
    assert recovery_summary([4, None, 2, 9]) == {
        "recovered": 3,
        "median_rounds": 4.0,
        "max_rounds": 9,
    }
    assert recovery_summary([None]) == {"recovered": 0, "median_rounds": None, "max_rounds": None}


def test_relative_deviation_is_the_mean_gap_over_the_mean_of_the_reference():
    assert relative_deviation([1.0, 3.0], [2.0, 2.0]) == 0.5
