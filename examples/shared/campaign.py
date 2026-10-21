"""Long, resumable experiment campaigns: lock, torch setup, time budget and stop handling."""

import fcntl
import json
import time
from contextlib import contextmanager

import torch

from .artifacts import json_write, read_json


@contextmanager
def output_lock(out):
    """Kernel-owned lock survives daemon restarts and releases on process exit.

    A PID file cannot distinguish processes in separate sandbox namespaces.
    flock also prevents report/checkpoint temporary files from racing.
    """
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is using this output directory") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def require_new(out):
    """Refuse to start a run in a directory that holds files, or inside another run.

    Raises:
        ValueError: ``out`` holds anything but its lock, or a parent holds a manifest.
    """
    if out.exists() and any(p.name != ".run.lock" for p in out.iterdir()):
        raise ValueError("New output directory must be empty")
    if any((p / "manifest.json").exists() for p in out.resolve().parents):
        raise ValueError("Experiment output directories must not be nested")


def check_resume(saved, expected):
    """Refuse to resume a run whose configuration, data or sources changed.

    Args:
        saved: The manifest of the run.
        expected: What the manifest would hold now; values compare as JSON stores them, and
            of ``software`` only the hashes of the sources.

    Raises:
        ValueError: Naming the first key whose value changed.
    """
    for key, value in expected.items():
        old, new = saved.get(key), json.loads(json.dumps(value))
        if key == "software":
            old, new = (old or {}).get("source_sha256"), new["source_sha256"]
        if old != new:
            raise ValueError(f"Resume mismatch: {key}; use a new output directory")


@contextmanager
def status_file(path, **fields):
    """Mark a stage as running in ``path``, then as incomplete or failed if it stops early.

    An exhausted budget or an interrupt leaves the stage ``incomplete``, any other error
    ``failed``; both record the reason next to ``fields`` and propagate.
    """
    json_write(path, {"status": "running", **fields})
    try:
        yield
    except (TimeoutError, KeyboardInterrupt) as error:
        json_write(path, {"status": "incomplete", **fields, "reason": str(error)})
        raise
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
        json_write(path, {"status": "failed", **fields, "reason": reason})
        raise


def resumable(out, run, *, partial=None):
    """The exit code of ``run()``, or 2 when it stops early, after a ``partial()`` report.

    Only an exhausted budget or an interrupt stops a run early: the same command resumes it
    from the artifacts it left in ``out``.
    """
    try:
        return run()
    except (TimeoutError, KeyboardInterrupt):
        if partial is not None:
            partial()
        print(f"Incomplete, resumable: {out}", flush=True)
        return 2


def deterministic(threads=1):
    """Make torch repeat a run exactly: a fixed number of threads, deterministic kernels."""
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


class Budget:
    """Soft deadline checked between atomic work units, never inside a recurrence.

    With a ``record`` file, the time used accumulates across invocations, and a resumed run
    can lower the limit but never raise it.
    """

    def __init__(self, seconds: float, record=None):
        if seconds <= 0:
            raise ValueError("Budget must be positive")
        old = read_json(record) if record is not None and record.exists() else {}
        self.seconds = min(seconds, old.get("limit_seconds", seconds))
        self.used = old.get("used_seconds", 0.0)
        self.record = record
        self.started = time.monotonic()
        self.save()

    def save(self) -> float:
        """The seconds used so far, written to the record if there is one."""
        used = self.used + time.monotonic() - self.started
        if self.record is not None:
            json_write(
                self.record,
                {
                    "limit_seconds": self.seconds,
                    "used_seconds": used,
                    "remaining_seconds": max(0, self.seconds - used),
                },
            )
        return used

    def check(self):
        if self.save() >= self.seconds:
            raise TimeoutError(
                "Budget exhausted; completed work is saved and the run is incomplete"
            )
