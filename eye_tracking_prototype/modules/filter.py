import numpy as np
from collections import deque

try:
    from filterpy.kalman import KalmanFilter
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False

class GazeFilter:
    def __init__(self, cfg):
        self.type = cfg.filter_type
        if self.type == "kalman" and not KALMAN_AVAILABLE:
            print("[WARN] filterpy not found. Falling back to EMA.")
            self.type = "ema"
            
        self.ema_alpha = cfg.ema_alpha
        self.median_window = cfg.median_window
        
        self.last_val = None
        self.history = deque(maxlen=self.median_window)
        
        if self.type == "kalman":
            self.kf = KalmanFilter(dim_x=4, dim_z=2) # state: [x, y, dx, dy]
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

    def update(self, x, y):
        if self.type == "none":
            return x, y
            
        elif self.type == "ema":
            if self.last_val is None:
                self.last_val = (x, y)
            else:
                lx, ly = self.last_val
                a = self.ema_alpha
                self.last_val = (lx * (1-a) + x * a, ly * (1-a) + y * a)
            return self.last_val
            
        elif self.type == "median":
            self.history.append((x, y))
            xs = [v[0] for v in self.history]
            ys = [v[1] for v in self.history]
            return float(np.median(xs)), float(np.median(ys))
            
        elif self.type == "kalman":
            if self.last_val is None:
                self.kf.x = np.array([x, y, 0., 0.])
                self.last_val = (x, y)
                return x, y
            
            self.kf.predict()
            self.kf.update(np.array([x, y]))
            self.last_val = (float(self.kf.x[0]), float(self.kf.x[1]))
            return self.last_val
            
        return x, y

    def reset(self):
        self.last_val = None
        self.history.clear()