"""
Nonlinear (MLP) probing for representation analysis.

When a linear probe fails but an MLP probe succeeds, the property
is encoded nonlinearly — present but not in a linearly accessible
form. This distinction matters for understanding representation
quality: linearly decodable properties are more robustly encoded.
"""

from __future__ import annotations

from typing import Dict, Literal

import numpy as np


class MLPProbe:
    """
    One-hidden-layer MLP probe.

    Deliberately shallow to avoid the probe itself learning
    complex transformations — we want to measure what the
    *model* has learned, not what the *probe* can learn.

    Parameters
    ----------
    task : str
        "classification" or "regression".
    hidden_dim : int
        Hidden layer size.
    regularization : float
        L2 regularization strength (alpha).
    max_iter : int
        Max training iterations.
    """

    def __init__(
        self,
        task: Literal["classification", "regression"] = "classification",
        hidden_dim: int = 128,
        regularization: float = 1e-4,
        max_iter: int = 5000,
    ):
        self.task = task
        self.hidden_dim = hidden_dim
        self.regularization = regularization
        self.max_iter = max_iter

    def _build_model(self):
        from sklearn.neural_network import MLPClassifier, MLPRegressor

        if self.task == "classification":
            return MLPClassifier(
                hidden_layer_sizes=(self.hidden_dim,),
                alpha=self.regularization,
                max_iter=self.max_iter,
                early_stopping=True,
                validation_fraction=0.1,
            )
        else:
            return MLPRegressor(
                hidden_layer_sizes=(self.hidden_dim,),
                alpha=self.regularization,
                max_iter=self.max_iter,
                early_stopping=True,
                validation_fraction=0.1,
            )

    def fit_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5,
    ) -> Dict[str, float]:
        """Fit and evaluate with cross-validation."""
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


def linear_nonlinear_gap(
    X: np.ndarray,
    y: np.ndarray,
    task: Literal["classification", "regression"] = "classification",
    cv_folds: int = 5,
) -> Dict[str, float]:
    """
    Compute the gap between linear and nonlinear probe accuracy.

    A large gap indicates the property is encoded but not linearly
    accessible — the model has learned a nonlinear representation.

    Returns
    -------
    dict
        linear_acc, mlp_acc, gap
    """
    from matscope.probes.linear import LinearProbe

    linear = LinearProbe(task=task)
    mlp = MLPProbe(task=task)

    linear_metrics = linear.fit_evaluate(X, y, cv_folds=cv_folds)
    mlp_metrics = mlp.fit_evaluate(X, y, cv_folds=cv_folds)

    metric_key = "accuracy" if task == "classification" else "r2"
    gap = mlp_metrics[metric_key] - linear_metrics[metric_key]

    return {
        f"linear_{metric_key}": linear_metrics[metric_key],
        f"mlp_{metric_key}": mlp_metrics[metric_key],
        "gap": gap,
    }
