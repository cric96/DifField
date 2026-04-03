"""Pytest path bootstrap for the local src-layout project."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

for path in (ROOT, ROOT / "src"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)