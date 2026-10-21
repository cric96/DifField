"""The output lock and the time budget of resumable campaigns."""

import pytest

from examples.shared.artifacts import json_write, read_json
from examples.shared.campaign import Budget, output_lock, resumable, status_file


def test_output_lock_excludes_concurrent_runs_and_releases_after_failure(tmp_path):
    with output_lock(tmp_path):  # noqa: SIM117 -- the first invocation must retain its lock
        with pytest.raises(RuntimeError, match="Another process"):
            with output_lock(tmp_path):
                pytest.fail("Concurrent output access must not be admitted")
    with pytest.raises(ValueError, match="simulated"), output_lock(tmp_path):
        raise ValueError("simulated failure")
    with output_lock(tmp_path):
        assert (tmp_path / ".run.lock").exists()


def test_recorded_budget_accumulates_across_invocations_and_never_grows(tmp_path):
    record = tmp_path / "budget.json"
    Budget(1, record=record)
    json_write(record, {**read_json(record), "used_seconds": 2})  # spent by that invocation

    resumed = Budget(14400, record=record)

    with pytest.raises(TimeoutError):
        resumed.check()
    assert read_json(record)["limit_seconds"] == 1


def test_budget_rejects_a_limit_that_is_not_positive():
    with pytest.raises(ValueError, match="positive"):
        Budget(0)


def test_status_file_marks_a_stopped_stage_incomplete_and_a_broken_one_failed(tmp_path):
    path = tmp_path / "stage.json"

    with pytest.raises(TimeoutError), status_file(path, stage="train"):
        raise TimeoutError("budget")
    stopped = read_json(path)
    with pytest.raises(ValueError, match="broken"), status_file(path):
        raise ValueError("broken")

    assert stopped == {"status": "incomplete", "stage": "train", "reason": "budget"}
    assert read_json(path) == {"status": "failed", "reason": "ValueError: broken"}


def test_resumable_writes_the_partial_report_and_exits_with_2_when_stopped(tmp_path):
    reports = []

    def interrupted():
        raise KeyboardInterrupt

    code = resumable(tmp_path, interrupted, partial=lambda: reports.append("partial"))

    assert code == 2 and reports == ["partial"]
    assert resumable(tmp_path, lambda: 0) == 0
