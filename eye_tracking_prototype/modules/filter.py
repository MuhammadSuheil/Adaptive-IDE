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
        # Low velocity threshold is 15% of saccade threshold (in same px units)
        self.low_velocity_threshold_px = self.saccade_threshold_px * 0.15

        self.last_val = None
        self.last_velocity = 0.0          # Exposed so tracking layer can inspect motion state
        self.history = deque(maxlen=self.median_window)

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

    def _adaptive_alpha(self, x, y):
        """Return the appropriate EMA alpha based on instantaneous velocity from last position."""
        if self.last_val is None:
            return self.ema_alpha
        lx, ly = self.last_val
        velocity = math.hypot(x - lx, y - ly)
        self.last_velocity = velocity
        if velocity >= self.saccade_threshold_px:
            return self.adaptive_high_alpha
        elif velocity <= self.low_velocity_threshold_px:
            return self.adaptive_low_alpha
        else:
            # Linear interpolation between low and high alpha
            t = (velocity - self.low_velocity_threshold_px) / max(
                self.saccade_threshold_px - self.low_velocity_threshold_px, 1e-6
            )
            return self.adaptive_low_alpha + t * (self.adaptive_high_alpha - self.adaptive_low_alpha)

    def update(self, x, y):
        if self.type == "none":
            return x, y

        elif self.type == "ema":
            if self.last_val is None:
                self.last_val = (x, y)
                self.last_velocity = 0.0
            else:
                if self.adaptive_ema:
                    a = self._adaptive_alpha(x, y)
                else:
                    lx, ly = self.last_val
                    self.last_velocity = math.hypot(x - lx, y - ly)
                    a = self.ema_alpha
                lx, ly = self.last_val
                self.last_val = (lx * (1 - a) + x * a, ly * (1 - a) + y * a)
            return self.last_val

        elif self.type == "median":
            self.history.append((x, y))
            xs = [v[0] for v in self.history]
            ys = [v[1] for v in self.history]
            self.last_velocity = 0.0
            return float(np.median(xs)), float(np.median(ys))

        elif self.type == "kalman":
            if self.last_val is None:
                self.kf.x = np.array([x, y, 0., 0.])
                self.last_val = (x, y)
                self.last_velocity = 0.0
                return x, y

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