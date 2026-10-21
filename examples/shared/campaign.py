"""Long, resumable experiment campaigns: their output lock, torch setup and time budget."""

import fcntl
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
