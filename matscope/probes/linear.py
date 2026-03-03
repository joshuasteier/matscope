"""
Linear probing for representation analysis.

Linear probes test whether a property is linearly decodable from
a model's representations. High linear probe accuracy at layer L
implies that the information is explicitly represented in a
linearly accessible form at that depth.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import numpy as np


class LinearProbe:
    """
    Sklearn-based linear probe for classification or regression.

    Uses LogisticRegression (classification) or Ridge (regression)
    with cross-validation for robust evaluation.

    Parameters
    ----------
    task : str
        "classification" or "regression".
    regularization : float
        Regularization strength (C for LogReg, alpha for Ridge).
    max_iter : int
        Maximum iterations for solver.
    """

    def __init__(
        self,
        task: Literal["classification", "regression"] = "classification",
        regularization: float = 1.0,
        max_iter: int = 5000,
    ):
        self.task = task
        self.regularization = regularization
        self.max_iter = max_iter
        self.model_ = None

    def _build_model(self):
        from sklearn.linear_model import LogisticRegression, Ridge

        if self.task == "classification":
            return LogisticRegression(
                C=self.regularization,
                max_iter=self.max_iter,
                solver="lbfgs",
            )
        else:
            return Ridge(alpha=self.regularization)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearProbe":
        """Fit the probe."""
        from sklearn.preprocessing import StandardScaler

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        self.model_ = self._build_model()
        self.model_.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with fitted probe."""
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict(X_scaled)

    def fit_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5,
    ) -> Dict[str, float]:
        """
        Fit and evaluate with cross-validation.

        Returns
        -------
        dict
            Metrics: accuracy/f1 for classification, r2/mse for regression.
        """
        from sklearn.model_selection import cross_validate
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        pipeline = make_pipeline(StandardScaler(), self._build_model())

        if self.task == "classification":
            scoring = ["accuracy", "f1_weighted"]
        else:
            scoring = ["r2", "neg_mean_squared_error"]

        cv_results = cross_validate(
            pipeline, X, y, cv=cv_folds, scoring=scoring, return_train_score=False
        )

        if self.task == "classification":
            return {
                "accuracy": float(np.mean(cv_results["test_accuracy"])),
                "accuracy_std": float(np.std(cv_results["test_accuracy"])),
                "f1_weighted": float(np.mean(cv_results["test_f1_weighted"])),
                "f1_std": float(np.std(cv_results["test_f1_weighted"])),
            }
        else:
            return {
                "r2": float(np.mean(cv_results["test_r2"])),
                "r2_std": float(np.std(cv_results["test_r2"])),
                "mse": float(-np.mean(cv_results["test_neg_mean_squared_error"])),
                "mse_std": float(np.std(cv_results["test_neg_mean_squared_error"])),
            }


class SelectivityProbe(LinearProbe):
    """
    Probe with selectivity analysis.

    Beyond accuracy, measures how *selectively* a property is
    encoded — i.e., does the probe rely on a small number of
    dimensions (high selectivity) or is the information distributed?

    This is useful for understanding representation structure in
    equivariant models where physical symmetries constrain
    how information can be encoded.
    """

    def fit_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5,
    ) -> Dict[str, float]:
        metrics = super().fit_evaluate(X, y, cv_folds=cv_folds)

        # Fit once on full data for selectivity analysis
        self.fit(X, y)

        if self.task == "classification" and hasattr(self.model_, "coef_"):
            coef = self.model_.coef_
            # Selectivity: ratio of L1 to L2 norm (lower = more selective)
            l1 = np.mean(np.abs(coef), axis=0)
            l2 = np.sqrt(np.mean(coef**2, axis=0))
            selectivity = np.mean(l1 / (l2 + 1e-8))
            metrics["selectivity"] = float(selectivity)
            # Effective dimensionality via participation ratio
            singular_values = np.linalg.svd(coef, compute_uv=False)
            sv_normalized = singular_values / singular_values.sum()
            participation_ratio = 1.0 / np.sum(sv_normalized**2)
            metrics["effective_dim"] = float(participation_ratio)

        return metrics
