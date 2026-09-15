import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator


class GazeMapper:
    """
    Regularized gaze mapper with:
    - Normalized targets (screen-independent)
    - Bounded RBF (Thin-Plate Spline) Interpolation
    - Center-weighted Leave-One-Out Cross-Validation quality score
      (edge/corner points get reduced weight so their natural LOO-CV
       extrapolation artifacts don't unfairly penalize a valid calibration)
    - predict_with_confidence(): returns (px, py, conf) based on distance
      from nearest training point in feature space
    """

    # Normalized screen-space distance thresholds for confidence calculation
    _CONF_NEAR_DIST = 0.05   # within 5% of screen diagonal → confidence ~1.0
    _CONF_FAR_DIST  = 0.40   # beyond 40% → confidence ~0.0

    def __init__(self, method="rbf", rbf_kernel="thin_plate_spline",
                 rbf_smoothing=0.0, output_clamp=True):
        self.method = method
        self.rbf_kernel = rbf_kernel
        self.rbf_smoothing = rbf_smoothing
        self.output_clamp = output_clamp

        self.model_x = None
        self.model_y = None
        self.rbf_model = None
        self.affine_M = None
        self.screen_w = 1.0
        self.screen_h = 1.0
        self.diagnostics = {}

        # Training feature points (normalized), stored for confidence calc
        self._train_feats = None

    # ─── Internal helpers ────────────────────────────────────────────────────

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
        if self.method == "rbf":
            self.rbf_model = RBFInterpolator(
                feats, targets,
                kernel=self.rbf_kernel,
                smoothing=self.rbf_smoothing
            )
        elif self.method == "hybrid":
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
        else:  # affine fallback
            self.affine_M, _ = cv2.estimateAffinePartial2D(feats[:, :2], targets)

    def _predict_normalized(self, norm_x, norm_y, ear):
        if self.method == "rbf" and self.rbf_model is not None:
            inp = np.array([[norm_x, norm_y, ear]])
            out = self.rbf_model(inp)[0]
            px, py = float(out[0]), float(out[1])
            if self.output_clamp:
                px = float(np.clip(px, -0.10, 1.10))
                py = float(np.clip(py, -0.10, 1.10))
            return px, py

        if self.method == "hybrid" and self.model_x is not None:
            fx = np.array([1.0, norm_x, norm_y, norm_x ** 2, norm_y ** 2, norm_x * norm_y])
            fy = np.array([1.0, norm_y, ear, norm_y * ear])
            px, py = float(fx @ self.model_x), float(fy @ self.model_y)
        elif self.method == "polynomial" and self.model_x is not None:
            feat = np.array([1.0, norm_x, norm_y, norm_x ** 2, norm_y ** 2, norm_x * norm_y])
            px, py = float(feat @ self.model_x), float(feat @ self.model_y)
        elif self.affine_M is not None:
            src = np.array([[[norm_x, norm_y]]], dtype=np.float32)
            dst = cv2.transform(src, self.affine_M)
            px, py = float(dst[0, 0, 0]), float(dst[0, 0, 1])
        else:
            px, py = 0.5, 0.5

        if self.output_clamp:
            px = float(np.clip(px, -0.10, 1.10))
            py = float(np.clip(py, -0.10, 1.10))

        return px, py

    @staticmethod
    def _point_edge_weight(target_norm):
        """
        Returns a weight in [0.5, 1.0] for a calibration point based on how
        close to the centre of the screen it is (in normalized screen space).
        Corner/edge points that naturally suffer from LOO-CV extrapolation
        artifacts are down-weighted so they don't unfairly drag down quality.
        """
        cx = abs(target_norm[0] - 0.5) * 2.0  # 0 = center, 1 = edge
        cy = abs(target_norm[1] - 0.5) * 2.0
        edge_dist = max(cx, cy)                # Chebyshev distance from centre
        # Weight: 1.0 at centre, 0.5 at the extreme corners
        return 1.0 - 0.5 * edge_dist

    # ─── Public API ──────────────────────────────────────────────────────────

    def fit(self, eye_features, screen_pts, screen_size=None):
        feats = np.asarray(eye_features, dtype=np.float64)
        points = np.asarray(screen_pts, dtype=np.float64)
        if len(feats) < 4 or len(feats) != len(points):
            raise ValueError(
                "Calibration requires matching feature/target sets with at least four points"
            )

        if screen_size is None:
            self.screen_w = max(float(np.max(points[:, 0])), 1.0)
            self.screen_h = max(float(np.max(points[:, 1])), 1.0)
        else:
            self.screen_w, self.screen_h = map(float, screen_size)
        targets = points / np.array([self.screen_w, self.screen_h])

        # Centre-weighted Leave-One-Out Cross-Validation
        held_out_errors = []
        loo_weights = []
        for held_out in range(len(feats)):
            train_mask = np.arange(len(feats)) != held_out
            candidate = GazeMapper(
                method=self.method,
                rbf_kernel=self.rbf_kernel,
                rbf_smoothing=self.rbf_smoothing,
                output_clamp=self.output_clamp
            )
            candidate._fit_models(feats[train_mask], targets[train_mask])
            pred = np.array(candidate._predict_normalized(*feats[held_out]))
            err = float(np.linalg.norm(pred - targets[held_out]))
            held_out_errors.append(err)
            loo_weights.append(self._point_edge_weight(targets[held_out]))

        # Weighted median & weighted P95 for quality score
        errors_arr = np.array(held_out_errors)
        weights_arr = np.array(loo_weights)
        weights_norm = weights_arr / weights_arr.sum()

        # Weighted percentile via sorted cumulative weights
        sort_idx = np.argsort(errors_arr)
        sorted_errors = errors_arr[sort_idx]
        sorted_weights = weights_norm[sort_idx]
        cum_weights = np.cumsum(sorted_weights)
        median_error = float(sorted_errors[np.searchsorted(cum_weights, 0.50)])
        p95_error = float(sorted_errors[np.searchsorted(cum_weights, 0.95)])

        # Quality: median * 0.8 + p95 * 0.2 (same proven formula, now weighted)
        score_error = median_error * 0.8 + p95_error * 0.2
        quality = max(0.0, min(1.0, 1.0 - score_error / 0.35))

        self._fit_models(feats, targets)
        self._train_feats = feats.copy()

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
        px = x * self.screen_w
        py = y * self.screen_h
        if self.output_clamp:
            px = float(np.clip(px, -0.10 * self.screen_w, 1.10 * self.screen_w))
            py = float(np.clip(py, -0.10 * self.screen_h, 1.10 * self.screen_h))
        return px, py

    def predict_with_confidence(self, norm_x, norm_y, ear):
        """
        Returns (px, py, confidence) where confidence ∈ [0.0, 1.0].
        Confidence is based on the minimum distance in feature space between
        the current eye feature vector and the nearest calibration training point.
        Points near well-sampled regions → confidence ≈ 1.0.
        Points far from any training sample (extrapolation) → confidence ≈ 0.0.
        """
        px, py = self.predict(norm_x, norm_y, ear)
        if self._train_feats is None:
            return px, py, 1.0

        query = np.array([norm_x, norm_y, ear])
        dists = np.linalg.norm(self._train_feats - query, axis=1)
        min_dist = float(np.min(dists))

        # Map distance to confidence: near → 1.0, far → 0.0
        conf = 1.0 - np.clip(
            (min_dist - self._CONF_NEAR_DIST) / max(
                self._CONF_FAR_DIST - self._CONF_NEAR_DIST, 1e-6
            ),
            0.0, 1.0
        )
        return px, py, float(conf)
