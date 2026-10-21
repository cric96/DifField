"""Put the example folders on the path, as when an example runs from this directory."""

import sys
from pathlib import Path

EXAMPLES = str(Path(__file__).resolve().parent)
if EXAMPLES not in sys.path:
    sys.path.insert(0, EXAMPLES)
