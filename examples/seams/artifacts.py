"""Atomic, resumable artifacts with configuration and software provenance."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[2]


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def tensor_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text())


def software():
    def git(*args):
        return subprocess.run(  # noqa: S603 -- fixed read-only git commands below
            ["git", *args],
            cwd=PROJECT,
            capture_output=True,
            text=True,
            check=False,  # noqa: S607
        ).stdout.strip()

    files = sorted((PROJECT / "src/diffield").rglob("*.py"))
    files += sorted((PROJECT / "examples/seams").rglob("*.py"))
    return {
        "revision": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "source_sha256": {str(p.relative_to(PROJECT)): checksum(p) for p in files},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torch-geometric", "numpy", "scikit-learn")
        },
    }


class Budget:
    """Soft deadline checked between atomic work units, never inside a recurrence."""

    def __init__(self, seconds: float):
        self.started = time.monotonic()
        self.seconds = seconds

    def check(self):
        if time.monotonic() - self.started >= self.seconds:
            raise TimeoutError(
                "Budget exhausted; completed work is saved and the run is incomplete"
            )
