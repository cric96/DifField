"""The output lock and the time budget of resumable campaigns."""

import pytest

from examples.shared.artifacts import json_write, read_json
from examples.shared.campaign import (
    Budget,
    check_resume,
    open_run,
    output_lock,
    require_new,
    resumable,
    status_file,
)


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


def test_check_resume_compares_the_sources_of_the_software_only():
    saved = {"config": {"rounds": 3}, "software": {"source_sha256": {"a.py": "1"}, "cpu": "x"}}

    check_resume(saved, {"config": {"rounds": 3}, "software": {"source_sha256": {"a.py": "1"}}})

    with pytest.raises(ValueError, match="Resume mismatch: software"):
        check_resume(saved, {"software": {"source_sha256": {"a.py": "2"}}})
    with pytest.raises(ValueError, match="Resume mismatch: config"):
        check_resume(saved, {"config": {"rounds": 4}})


def test_require_new_refuses_a_directory_with_files_or_inside_another_run(tmp_path):
    with output_lock(tmp_path / "run"):
        require_new(tmp_path / "run")
    json_write(tmp_path / "run/manifest.json", {})

    with pytest.raises(ValueError, match="empty"):
        require_new(tmp_path / "run")
    with pytest.raises(ValueError, match="nested"):
        require_new(tmp_path / "run/inner")


def test_open_run_writes_a_new_manifest_and_resumes_only_an_unchanged_one(tmp_path):
    open_run(tmp_path / "run", {"config": {"rounds": 3}})

    open_run(tmp_path / "run", {"config": {"rounds": 3}})

    assert read_json(tmp_path / "run/manifest.json") == {"config": {"rounds": 3}}
    with pytest.raises(ValueError, match="Resume mismatch: config"):
        open_run(tmp_path / "run", {"config": {"rounds": 4}})
