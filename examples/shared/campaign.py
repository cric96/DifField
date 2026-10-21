"""Long, resumable experiment campaigns: their time budget."""

import time


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
