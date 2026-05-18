from __future__ import annotations

import time
from dataclasses import dataclass


class Clock:
    def now(self) -> float:
        raise NotImplementedError


class WallClock(Clock):
    def now(self) -> float:
        return time.time() / 86400.0


@dataclass
class DatasetClock(Clock):
    """Clock advanced by a benchmark driver.

    Times are expressed in days to match existing LongMemEval scripts and keep
    beta values interpretable as per-day decay rates.
    """

    current_time: float = 0.0

    def set(self, value: float) -> None:
        self.current_time = float(value)

    def advance_to(self, value: float) -> None:
        self.current_time = max(self.current_time, float(value))

    def now(self) -> float:
        return float(self.current_time)
