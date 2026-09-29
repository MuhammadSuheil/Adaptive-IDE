"""Time-based throughput measurements shared by capture and inference."""
from collections import deque
import os


class TimerResolution:
    """Scoped Windows timer request, paired with timeEndPeriod on close."""
    def __init__(self, milliseconds=None):
        self.milliseconds = milliseconds
        self.active = False

    def start(self):
        if self.milliseconds is not None and os.name == 'nt' and not self.active:
            import ctypes
            self.api = ctypes.WinDLL('winmm')
            self.active = self.api.timeBeginPeriod(self.milliseconds) == 0
        return self

    def close(self):
        if self.active:
            self.api.timeEndPeriod(self.milliseconds)
            self.active = False


class FrameRate:
    def __init__(self, window_seconds=2.0):
        self.window_seconds = window_seconds
        self.times = deque()

    def add(self, timestamp):
        self.times.append(timestamp)
        self._trim(timestamp)

    def _trim(self, now):
        while self.times and self.times[0] < now - self.window_seconds:
            self.times.popleft()

    def fps(self, now):
        self._trim(now)
        if len(self.times) < 2:
            return 0.0
        return (len(self.times) - 1) / max(now - self.times[0], 1e-9)


def timing_summary(values):
    import numpy as np
    if not values:
        return {key: None for key in ("mean", "p50", "p95", "p99", "max")}
    return {"mean": float(np.mean(values)), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
            "max": float(max(values))}
