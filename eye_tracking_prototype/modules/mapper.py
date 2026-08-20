import cv2
import numpy as np

class GazeMapper:
    def __init__(self, method="hybrid"):
        self.method = method
        self.model_x = None
        self.model_y = None
        self.affine_M = None

    def fit(self, eye_features, screen_pts):
        """
        eye_features: Nx3 array [norm_x, norm_y, ear]
        screen_pts: Nx2 array [target_x, target_y]
        """
        feats = np.array(eye_features, dtype=np.float32)
        screen_pts = np.array(screen_pts, dtype=np.float32)

        if self.method == "hybrid":
            # X Model: Polynomial 2nd degree on [norm_x, norm_y]
            X_poly = np.column_stack([
                np.ones(len(feats)),
                feats[:, 0],
                feats[:, 1],
                feats[:, 0]**2,
                feats[:, 1]**2,
                feats[:, 0] * feats[:, 1]
            ])
            ridge_x = 1e-3 * np.eye(X_poly.shape[1])
            self.model_x, _, _, _ = np.linalg.lstsq(X_poly.T @ X_poly + ridge_x, X_poly.T @ screen_pts[:, 0], rcond=None)

            # Y Model: Ridge Linear on [norm_y, ear] + interaction
            Y_ridge_feats = np.column_stack([
                np.ones(len(feats)),
                feats[:, 1],          # norm_y
                feats[:, 2],          # EAR
                feats[:, 1] * feats[:, 2] # Interaction term Y * EAR
            ])
            ridge_y = 1e-2 * np.eye(Y_ridge_feats.shape[1])
            self.model_y, _, _, _ = np.linalg.lstsq(Y_ridge_feats.T @ Y_ridge_feats + ridge_y, Y_ridge_feats.T @ screen_pts[:, 1], rcond=None)

            pred_x = X_poly @ self.model_x
            pred_y = Y_ridge_feats @ self.model_y
            errors = np.hypot(pred_x - screen_pts[:, 0], pred_y - screen_pts[:, 1])
            return max(0.0, 1.0 - (np.mean(errors) / 300.0))

        elif self.method == "polynomial":
            X_poly = np.column_stack([
                np.ones(len(feats)),
                feats[:, 0],
                feats[:, 1],
                feats[:, 0]**2,
                feats[:, 1]**2,
                feats[:, 0] * feats[:, 1]
            ])
            ridge = 1e-4 * np.eye(X_poly.shape[1])
            self.model_x, _, _, _ = np.linalg.lstsq(X_poly.T @ X_poly + ridge, X_poly.T @ screen_pts[:, 0], rcond=None)
            self.model_y, _, _, _ = np.linalg.lstsq(X_poly.T @ X_poly + ridge, X_poly.T @ screen_pts[:, 1], rcond=None)
            
            pred_x = X_poly @ self.model_x
            pred_y = X_poly @ self.model_y
            errors = np.hypot(pred_x - screen_pts[:, 0], pred_y - screen_pts[:, 1])
            return max(0.0, 1.0 - (np.mean(errors) / 300.0))

        else: # "affine"
            iris_pts = feats[:, :2]
            M, inliers = cv2.estimateAffinePartial2D(iris_pts, screen_pts)
            self.affine_M = M
            return float(np.sum(inliers)) / len(inliers) if inliers is not None else 0.0

    def predict(self, norm_x, norm_y, ear):
        if self.method == "hybrid" and self.model_x is not None:
            feat_x = np.array([1.0, norm_x, norm_y, norm_x**2, norm_y**2, norm_x * norm_y])
            feat_y = np.array([1.0, norm_y, ear, norm_y * ear])
            sx = np.dot(feat_x, self.model_x)
            sy = np.dot(feat_y, self.model_y)
            return float(sx), float(sy)

        elif self.method == "polynomial" and self.model_x is not None:
            feat = np.array([1.0, norm_x, norm_y, norm_x**2, norm_y**2, norm_x * norm_y])
            sx = np.dot(feat, self.model_x)
            sy = np.dot(feat, self.model_y)
            return float(sx), float(sy)

        elif self.affine_M is not None:
            src = np.array([[[norm_x, norm_y]]], dtype=np.float32)
            dst = cv2.transform(src, self.affine_M)
            return float(dst[0][0][0]), float(dst[0][0][1])

        return 0.0, 0.0