import time
from collections import deque


class BlinkDetector:
    """
    Eye Aspect Ratio (EAR) based real-time blink detector and counter.

    State Machine Logic:
    - Eye Open (EAR >= ear_threshold): closed_frames reset to 0.
    - Eye Closing/Closed (EAR < ear_threshold): closed_frames incremented.
    - Eye Reopening (EAR >= ear_threshold after being closed):
      If min_blink_frames <= closed_frames <= max_blink_frames,
      a valid natural blink is registered and blink_count increments.
    """

    def __init__(self, cfg):
        self.enabled = getattr(cfg, "blink_enabled", True)
        self.ear_threshold = getattr(cfg, "blink_ear_threshold", 0.20)
        self.min_blink_frames = getattr(cfg, "blink_min_frames", 1)
        self.max_blink_frames = getattr(cfg, "blink_max_frames", 10)

        self.blink_count = 0
        self.closed_frames = 0
        self.is_blinking = False
        self.blink_timestamps = deque()  # Timestamps of registered blinks (last 60s)

    def process(self, ear, current_time_sec=None):
        """
        Process current frame EAR and update blink state.
        Returns: (is_blinking: bool, total_blinks: int, blink_rate_bpm: float)
        """
        if not self.enabled or ear is None or ear <= 0.0:
            return False, self.blink_count, self.get_blink_rate_bpm(current_time_sec)

        if current_time_sec is None:
            current_time_sec = time.time()

        if ear < self.ear_threshold:
            self.closed_frames += 1
            self.is_blinking = True
        else:
            if self.min_blink_frames <= self.closed_frames <= self.max_blink_frames:
                self.blink_count += 1
                self.blink_timestamps.append(current_time_sec)
            self.closed_frames = 0
            self.is_blinking = False

        bpm = self.get_blink_rate_bpm(current_time_sec)
        return self.is_blinking, self.blink_count, bpm

    def get_blink_rate_bpm(self, current_time_sec=None, window_sec=60.0):
        """Calculates blinks per minute based on a rolling window."""
        if current_time_sec is None:
            current_time_sec = time.time()

        while self.blink_timestamps and (current_time_sec - self.blink_timestamps[0]) > window_sec:
            self.blink_timestamps.popleft()

        return float(len(self.blink_timestamps))

    def reset(self):
        self.blink_count = 0
        self.closed_frames = 0
        self.is_blinking = False
        self.blink_timestamps.clear()
