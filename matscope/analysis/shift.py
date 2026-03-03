"""
Distribution shift analysis at the representation level.

Answers the question: "If I train on domain A and deploy on domain B,
where in the network does the shift hit hardest?"

This is directly useful for scientific FMs where training on bulk
crystals but deploying on catalysis surfaces, or training on small
molecules but deploying on polymers, creates systematic shift.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


class ShiftAnalyzer:
    """
    Detect and quantify distribution shift in representation space.

    Parameters
    ----------
    method : str
        "mmd" - Maximum Mean Discrepancy (default)
        "fisher" - Fisher divergence approximation
        "cosine_drift" - Mean cosine distance between distribution centers
    """

    def __init__(self, method: Literal["mmd", "fisher", "cosine_drift"] = "mmd"):
        self.method = method

    def compute_shift(self, X_train: np.ndarray, X_deploy: np.ndarray) -> float:
        """
        Compute shift magnitude between two representation distributions.

        Parameters
        ----------
        X_train : np.ndarray
            Representations from training distribution, shape (n, d).
        X_deploy : np.ndarray
            Representations from deployment distribution, shape (m, d).

        Returns
        -------
        float
            Shift magnitude (higher = more shift).
        """
        if self.method == "mmd":
            return self._mmd(X_train, X_deploy)
        elif self.method == "fisher":
            return self._fisher(X_train, X_deploy)
        elif self.method == "cosine_drift":
            return self._cosine_drift(X_train, X_deploy)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    @staticmethod
    def _mmd(X: np.ndarray, Y: np.ndarray, gamma: float = None) -> float:
        """
        Maximum Mean Discrepancy with RBF kernel.

        Unbiased estimator of MMD^2 using the quadratic-time U-statistic.
        """
        n = X.shape[0]
        m = Y.shape[0]

        if gamma is None:
            # Median heuristic
            combined = np.vstack([X[:min(500, n)], Y[:min(500, m)]])
            dists = np.sum((combined[:, None] - combined[None, :]) ** 2, axis=-1)
            gamma = 1.0 / (np.median(dists) + 1e-8)

        def rbf_kernel(A, B):
            dists = np.sum((A[:, None] - B[None, :]) ** 2, axis=-1)
            return np.exp(-gamma * dists)

        K_xx = rbf_kernel(X, X)
        K_yy = rbf_kernel(Y, Y)
        K_xy = rbf_kernel(X, Y)

        # Unbiased MMD^2
        np.fill_diagonal(K_xx, 0)
        np.fill_diagonal(K_yy, 0)

        mmd2 = (K_xx.sum() / (n * (n - 1)) + K_yy.sum() / (m * (m - 1)) - 2 * K_xy.mean())

        return float(max(0, mmd2))

    @staticmethod
    def _fisher(X: np.ndarray, Y: np.ndarray) -> float:
        """
        Approximate Fisher divergence via difference in mean and covariance.

        Lightweight alternative to MMD for high-dimensional representations.
        """
        mean_x = X.mean(axis=0)
        mean_y = Y.mean(axis=0)
        mean_diff = mean_x - mean_y

        # Pooled covariance (regularized)
        cov_x = np.cov(X.T) + 1e-6 * np.eye(X.shape[1])
        cov_y = np.cov(Y.T) + 1e-6 * np.eye(Y.shape[1])
        cov_pooled = 0.5 * (cov_x + cov_y)

        # Mahalanobis-like distance
        try:
            cov_inv = np.linalg.inv(cov_pooled)
            fisher_dist = float(mean_diff @ cov_inv @ mean_diff)
        except np.linalg.LinAlgError:
            fisher_dist = float(np.sum(mean_diff**2))

        return fisher_dist

    @staticmethod
    def _cosine_drift(X: np.ndarray, Y: np.ndarray) -> float:
        """
        Cosine distance between distribution centroids.

        Fast and interpretable, but insensitive to higher-order
        distribution changes.
        """
        mean_x = X.mean(axis=0)
        mean_y = Y.mean(axis=0)

        dot = np.dot(mean_x, mean_y)
        norm_x = np.linalg.norm(mean_x)
        norm_y = np.linalg.norm(mean_y)

        if norm_x < 1e-10 or norm_y < 1e-10:
            return 1.0

        cosine_sim = dot / (norm_x * norm_y)
        return float(1.0 - cosine_sim)
