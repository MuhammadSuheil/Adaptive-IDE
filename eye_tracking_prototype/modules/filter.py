import math
import numpy as np
from collections import deque

try:
    from filterpy.kalman import KalmanFilter
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False


class GazeFilter:
    """
    Gaze smoothing filter supporting EMA (fixed or adaptive), Median, and Kalman modes.

    Adaptive EMA mode (enabled via cfg.adaptive_ema = True):
    - Low velocity  (<= low_v threshold):  uses low alpha  (heavy smoothing for stable dwell)
    - Mid velocity  (low..saccade range):  uses mid alpha  (current default)
    - High velocity (>= saccade threshold): uses high alpha (fast follow, minimal lag on intentional saccade)
    """

    def __init__(self, cfg):
        self.type = cfg.filter_type
        if self.type == "kalman" and not KALMAN_AVAILABLE:
            print("[WARN] filterpy not found. Falling back to EMA.")
            self.type = "ema"

        # Standard EMA params
        self.ema_alpha = cfg.ema_alpha
        self.median_window = cfg.median_window

        # Adaptive EMA params
        self.adaptive_ema = getattr(cfg, "adaptive_ema", False)
        self.adaptive_low_alpha = getattr(cfg, "adaptive_ema_low_v_alpha", 0.12)
        self.adaptive_high_alpha = getattr(cfg, "adaptive_ema_high_v_alpha", 0.70)
        self.saccade_threshold_px = getattr(cfg, "adaptive_ema_saccade_threshold_px", 80.0)
        # Legacy thresholds are converted from reference-frame units to px/second.
        self.low_velocity_threshold_px = self.saccade_threshold_px * 0.15

        self.last_val = None
        self.last_velocity = 0.0          # Exposed so tracking layer can inspect motion state
        self.history = deque(maxlen=self.median_window)
        options = getattr(cfg, 'data', {}).get('filter', {})
        self.reference_fps = float(options.get('reference_fps', 40))
        self.reset_gap = float(options.get('reset_gap_seconds', .3))
        self.jump_px = float(options.get('jump_guard_px', 250))
        self.jump_speed = float(options.get('jump_guard_speed_px_sec', 12000))
        self.confirm_seconds = float(options.get('jump_confirm_seconds', .035))
        self.last_timestamp = None
        self.last_raw = None
        self.pending_jump = None
        self.sample_accepted = True

        if self.type == "kalman":
            self.kf = KalmanFilter(dim_x=4, dim_z=2)  # state: [x, y, dx, dy]
            self.kf.x = np.array([0., 0., 0., 0.])
            self.kf.F = np.array([[1., 0., 1., 0.],
                                   [0., 1., 0., 1.],
                                   [0., 0., 1., 0.],
                                   [0., 0., 0., 1.]])
            self.kf.H = np.array([[1., 0., 0., 0.],
                                   [0., 1., 0., 0.]])
            self.kf.P *= 1000.
            self.kf.R = np.eye(2) * cfg.kalman_m_noise
            self.kf.Q = np.eye(4) * cfg.kalman_p_noise

    def _adaptive_alpha(self, x, y, dt):
        """Return reference alpha based on velocity between accepted raw samples."""
        if self.last_val is None:
            return self.ema_alpha
        lx, ly = self.last_raw if self.last_raw is not None else self.last_val
        velocity = math.hypot(x - lx, y - ly) / dt
        self.last_velocity = velocity
        high = self.saccade_threshold_px * self.reference_fps
        low = self.low_velocity_threshold_px * self.reference_fps
        if velocity >= high:
            return self.adaptive_high_alpha
        elif velocity <= low:
            return self.adaptive_low_alpha
        else:
            # Linear interpolation between low and high alpha
            t = (velocity - low) / max(high - low, 1e-6)
            return self.adaptive_low_alpha + t * (self.adaptive_high_alpha - self.adaptive_low_alpha)

    def update(self, x, y, timestamp=None):
        now = timestamp if timestamp is not None else (self.last_timestamp or 0) + 1 / self.reference_fps
        dt = 1 / self.reference_fps if self.last_timestamp is None else max(now - self.last_timestamp, 1e-6)
        if dt > self.reset_gap:
            self.reset()
            dt = 1 / self.reference_fps
        self.sample_accepted = True
        # Confirm a large discontinuity for a short duration before accepting it.
        if self.last_raw is not None:
            jump = math.dist((x, y), self.last_raw)
            if self.pending_jump is not None:
                point, since = self.pending_jump
                if jump < self.jump_px:
                    self.pending_jump = None
                elif math.dist(point, (x, y)) <= self.jump_px:
                    if now - since < self.confirm_seconds:
                        self.sample_accepted = False
                        return self.last_val
                    self.pending_jump = None
                else:
                    self.pending_jump = ((x, y), now)
                    self.sample_accepted = False
                    return self.last_val
            elif jump > self.jump_px and jump / dt > self.jump_speed:
                self.pending_jump = ((x, y), now)
                self.sample_accepted = False
                return self.last_val
        self.last_timestamp = now
        if self.type == "none":
            self.last_raw = self.last_val = (x, y)
            return x, y

        elif self.type == "ema":
            if self.last_val is None:
                self.last_val = (x, y)
                self.last_velocity = 0.0
            else:
                if self.adaptive_ema:
                    a = self._adaptive_alpha(x, y, dt)
                else:
                    lx, ly = self.last_val
                    self.last_velocity = math.hypot(x - lx, y - ly) / dt
                    a = self.ema_alpha
                a = 1 - (1 - a) ** (dt * self.reference_fps)
                lx, ly = self.last_val
                self.last_val = (lx * (1 - a) + x * a, ly * (1 - a) + y * a)
            self.last_raw = (x, y)
            return self.last_val

        elif self.type == "median":
            self.history.append((x, y))
            xs = [v[0] for v in self.history]
            ys = [v[1] for v in self.history]
            self.last_velocity = 0.0
            self.last_raw = (x, y)
            self.last_val = (float(np.median(xs)), float(np.median(ys)))
            return self.last_val

        elif self.type == "kalman":
            if self.last_val is None:
                self.kf.x = np.array([x, y, 0., 0.])
                self.last_val = (x, y)
                self.last_raw = (x, y)
                self.last_velocity = 0.0
                return x, y

            self.kf.F[0, 2] = self.kf.F[1, 3] = dt * self.reference_fps
            self.last_raw = (x, y)
            self.kf.predict()
            self.kf.update(np.array([x, y]))
            lx, ly = self.last_val
            self.last_velocity = math.hypot(x - lx, y - ly)
            self.last_val = (float(self.kf.x[0]), float(self.kf.x[1]))
            return self.last_val

        return x, y

    def reset(self):
        self.last_val = None
        self.last_velocity = 0.0
        self.history.clear()
        self.last_timestamp = None
        self.last_raw = None
        self.pending_jump = None
        self.sample_accepted = True
        if self.type == 'kalman':
            self.kf.P = np.eye(4) * 1000.0

    def invalidate(self, timestamp):
        self.pending_jump = None
        if self.last_timestamp is not None and timestamp - self.last_timestamp >= self.reset_gap:
            self.reset()


def eyes_are_open(cfg, ear_left, ear_right):
    """Both eyes must be finite, open and geometrically plausible, independent of blink counting."""
    threshold = max(cfg.min_ear, cfg.blink_ear_threshold)
    return all(value is not None and math.isfinite(value) and threshold <= value <= cfg.max_ear
               for value in (ear_left, ear_right))
