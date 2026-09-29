"""Time-based head alignment, independent of camera/UI refresh rate."""
from collections import deque
import math
import numpy as np


class HeadGate:
    pose_keys = ('pitch', 'yaw', 'roll', 'face_width_ratio', 'center_x_ratio', 'center_y_ratio')

    def __init__(self, cfg):
        self.cfg = cfg
        options = cfg.data.get('head_positioning', {})
        self.tau = float(options.get('smoothing_seconds', .12))
        self.window = float(options.get('stability_seconds', .5))
        self.grace = float(options.get('invalid_grace_seconds', .35))
        self.hysteresis = float(options.get('hysteresis_ratio', .08))
        self.angle_range = float(options.get('stable_angle_range_deg', 3.0))
        self.position_range = float(options.get('stable_position_range', .08))
        self.last_time = None
        self.filtered = None
        self.latched = False
        self.history = deque()
        self.baseline_samples = deque(maxlen=1200)
        self.invalid_since = None
        self.hold = 0.0
        self.previous_stable = False
        self.record = {'phase': 'head_gate', 'decision': 'incomplete', 'resets': 0,
                       'rejection_seconds': {}, 'samples': 0}

    def update(self, now, anchors=None, pose=None):
        dt = max(0.0, now - self.last_time) if self.last_time is not None else 0.0
        self.last_time = now
        reasons = []
        checks = [False, False, False, False, False]
        if dt > self.grace:
            self.filtered = None
            self.history.clear()
            self._reset_hold()
        if anchors is None or pose is None:
            reasons = ['face_missing']
        else:
            vector = np.concatenate((np.asarray(anchors).ravel(), [pose[k] for k in self.pose_keys]))
            if not np.isfinite(vector).all():
                reasons = ['pose_invalid']
            else:
                self.record['samples'] += 1
                alpha = 1.0 - math.exp(-dt / self.tau)
                self.filtered = vector if self.filtered is None else self.filtered + alpha * (vector - self.filtered)
                points = self.filtered[:8].reshape(4, 2)
                pitch, yaw, roll = self.filtered[8:11]
                h = self.hysteresis if self.latched else 0.0
                radii = np.linalg.norm(points, axis=1)
                inside = bool(np.all(radii <= self.cfg.hp_max_outside_ratio + h))
                filled = bool(np.all(radii[:2] >= self.cfg.hp_min_vertical_fill_ratio - h)
                              and np.all(radii[2:] >= self.cfg.hp_min_horizontal_fill_ratio - h))
                centered = bool(np.linalg.norm(points.mean(axis=0)) <= self.cfg.hp_center_tolerance_ratio + h)
                angle_ok = (abs(yaw) <= self.cfg.hp_max_yaw_deg + h * 20
                            and abs(pitch) <= self.cfg.hp_max_pitch_deg + h * 20
                            and abs(roll) <= 10 + h * 20)
                checks[:4] = [True, inside and filled, centered, bool(angle_ok)]
                if not inside: reasons.append('face_outside_oval')
                if not filled: reasons.append('face_too_small')
                if not centered: reasons.append('face_not_centered')
                if not angle_ok: reasons.append('head_angle')
                self.latched = all(checks[:4])
                # Stability uses RAW pose/anchors so smoothing cannot conceal motion.
                self.history.append((now, vector))
                while len(self.history) > 1 and self.history[1][0] <= now - self.window:
                    self.history.popleft()
                values = np.array([v for _, v in self.history])
                covered = now - self.history[0][0] >= self.window - 1e-6
                # Ignore isolated landmark outliers, but reject sustained movement.
                low, high = np.percentile(values, [10, 90], axis=0)
                spread = high - low
                movement_ok = (np.max(spread[:8]) <= self.position_range
                               and np.max(spread[8:11]) <= self.angle_range)
                self.record['last_measurement'] = {'radii': radii.tolist(),
                    'center_offset': float(np.linalg.norm(points.mean(axis=0))),
                    'angles_deg': [float(pitch), float(yaw), float(roll)],
                    'raw_angle_p10_p90_range_deg': float(np.max(spread[8:11])),
                    'raw_anchor_p10_p90_range': float(np.max(spread[:8]))}
                checks[4] = bool(covered and movement_ok and self.latched)
                if self.latched and not checks[4]:
                    reasons.append('collecting_stability' if not covered else 'head_moving')
                if checks[4]:
                    self.baseline_samples.append(vector[8:].copy())
        stable = checks[4]
        if stable:
            self.invalid_since = None
            if self.previous_stable:
                self.hold += dt
        else:
            if self.invalid_since is None:
                self.invalid_since = now
            if now - self.invalid_since > self.grace:
                self._reset_hold()
                if anchors is None:
                    self.history.clear()
                    self.filtered = None
        self.previous_stable = stable
        for reason in reasons:
            counts = self.record['rejection_seconds']
            counts[reason] = counts.get(reason, 0.0) + dt
        passed = stable and self.hold >= self.cfg.hp_countdown_seconds
        baseline = None
        if passed:
            baseline = dict(zip(self.pose_keys, np.median(self.baseline_samples, axis=0).tolist()))
            self.record.update(decision='passed', baseline=baseline, valid_hold_seconds=self.hold)
        return {'checks': checks, 'reasons': reasons, 'remaining': max(0., self.cfg.hp_countdown_seconds - self.hold),
                'passed': passed, 'baseline': baseline}

    def _reset_hold(self):
        if self.hold or self.baseline_samples:
            self.record['resets'] += 1
        self.hold = 0.0
        self.previous_stable = False
        self.baseline_samples.clear()
        self.latched = False
