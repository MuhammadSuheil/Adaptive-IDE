import cv2
import numpy as np


class GazeMapper:
    """Regularized gaze mapper with normalized targets and held-out diagnostics."""

    def __init__(self, method="hybrid"):
        self.method = method
        self.model_x = None
        self.model_y = None
        self.affine_M = None
        self.screen_w = 1.0
        self.screen_h = 1.0
        self.diagnostics = {}

    @staticmethod
    def _poly(feats):
        return np.column_stack([
            np.ones(len(feats)), feats[:, 0], feats[:, 1],
            feats[:, 0] ** 2, feats[:, 1] ** 2, feats[:, 0] * feats[:, 1]
        ])

    @staticmethod
    def _ridge_solve(design, target, strength):
        penalty = strength * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        return np.linalg.solve(design.T @ design + penalty, design.T @ target)

    def _fit_models(self, feats, targets):
        if self.method == "hybrid":
            x_design = self._poly(feats)
            y_design = np.column_stack([
                np.ones(len(feats)), feats[:, 1], feats[:, 2], feats[:, 1] * feats[:, 2]
            ])
            self.model_x = self._ridge_solve(x_design, targets[:, 0], 1e-6)
            self.model_y = self._ridge_solve(y_design, targets[:, 1], 1e-5)
        elif self.method == "polynomial":
            design = self._poly(feats)
            self.model_x = self._ridge_solve(design, targets[:, 0], 1e-6)
            self.model_y = self._ridge_solve(design, targets[:, 1], 1e-6)
        else:
            self.affine_M, _ = cv2.estimateAffinePartial2D(feats[:, :2], targets)

    def _predict_normalized(self, norm_x, norm_y, ear):
        if self.method == "hybrid" and self.model_x is not None:
            fx = np.array([1.0, norm_x, norm_y, norm_x ** 2, norm_y ** 2, norm_x * norm_y])
            fy = np.array([1.0, norm_y, ear, norm_y * ear])
            return float(fx @ self.model_x), float(fy @ self.model_y)
        if self.method == "polynomial" and self.model_x is not None:
            feat = np.array([1.0, norm_x, norm_y, norm_x ** 2, norm_y ** 2, norm_x * norm_y])
            return float(feat @ self.model_x), float(feat @ self.model_y)
        if self.affine_M is not None:
            src = np.array([[[norm_x, norm_y]]], dtype=np.float32)
            dst = cv2.transform(src, self.affine_M)
            return float(dst[0, 0, 0]), float(dst[0, 0, 1])
        return 0.0, 0.0

    def fit(self, eye_features, screen_pts, screen_size=None):
        feats = np.asarray(eye_features, dtype=np.float64)
        points = np.asarray(screen_pts, dtype=np.float64)
        if len(feats) < 4 or len(feats) != len(points):
            raise ValueError("Calibration requires matching feature/target sets with at least four points")

        if screen_size is None:
            self.screen_w = max(float(np.max(points[:, 0])), 1.0)
            self.screen_h = max(float(np.max(points[:, 1])), 1.0)
        else:
            self.screen_w, self.screen_h = map(float, screen_size)
        targets = points / np.array([self.screen_w, self.screen_h])

        held_out_errors = []
        for held_out in range(len(feats)):
            train = np.arange(len(feats)) != held_out
            candidate = GazeMapper(self.method)
            candidate._fit_models(feats[train], targets[train])
            pred = np.array(candidate._predict_normalized(*feats[held_out]))
            held_out_errors.append(float(np.linalg.norm(pred - targets[held_out])))

        self._fit_models(feats, targets)
        median_error = float(np.median(held_out_errors))
        p95_error = float(np.percentile(held_out_errors, 95))
        # Median captures typical accuracy; half of P95 prevents one catastrophic
        # target from being hidden by an otherwise acceptable median.
        score_error = max(median_error, p95_error * 0.5)
        quality = max(0.0, min(1.0, 1.0 - score_error / 0.25))
        diagonal = float(np.hypot(self.screen_w, self.screen_h))
        self.diagnostics = {
            "validation_median_error_screen_fraction": median_error,
            "validation_p95_error_screen_fraction": p95_error,
            "validation_median_error_px": median_error * diagonal,
            "validation_p95_error_px": p95_error * diagonal,
            "quality": quality,
        }
        return quality

    def predict(self, norm_x, norm_y, ear):
        x, y = self._predict_normalized(norm_x, norm_y, ear)
        return x * self.screen_w, y * self.screen_h
