"""
Composition Projection Decomposition (CPD).

The central methodology from "Information Routing in Atomistic Foundation Models"
(Steier, 2026). CPD separates composition-encoded and geometry-encoded information
in atomistic model representations by:

1. Fitting a Ridge regression from composition features to layer representations
2. Computing the residual (geometry-specific component)
3. Probing the residual to measure "geometric R²" — information accessible
   only through the geometric pathway

The key insight: equivariant models (MACE) produce linearly disentangled
composition/geometry representations, while invariant models (ANI-2x) require
nonlinear access to geometric information.

Reference:
    Steier, J. (2026). Information Routing in Atomistic Foundation Models:
    A Composition Projection Decomposition Analysis. arXiv preprint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CPDResult:
    """Container for CPD analysis results.

    Attributes
    ----------
    layer_name : str
        Which layer was analyzed.
    composition_r2 : float
        R² of Ridge regression from composition → representation.
        Measures how much of the representation is explained by composition alone.
    residual_linear_r2 : float
        R² of linear probe on the residual (composition-removed) representation.
        High value = geometric information is linearly accessible.
    residual_mlp_r2 : float
        R² of MLP probe on the residual representation.
        High value = geometric information is present but possibly nonlinear.
    linear_nonlinear_gap : float
        residual_mlp_r2 - residual_linear_r2.
        Large gap = geometric information requires nonlinear decoding.
    target_property : str
        Which property was being probed.
    composition_variance_explained : float
        Fraction of total variance in representations explained by composition.
    """

    layer_name: str
    composition_r2: float
    residual_linear_r2: float
    residual_mlp_r2: float
    linear_nonlinear_gap: float
    target_property: str
    composition_variance_explained: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def disentanglement_score(self) -> float:
        """Disentanglement score: high residual_linear_r2 + low gap = clean separation.

        Range: [-1, 1]. Higher = more linearly disentangled.
        """
        return self.residual_linear_r2 - abs(self.linear_nonlinear_gap)


@dataclass
class CPDProfile:
    """Full CPD profile across all layers of a model.

    Collects CPDResult for each layer, enabling the "disentanglement gradient"
    analysis across model depth.
    """

    model_name: str
    results: Dict[str, CPDResult]  # layer_name -> CPDResult
    target_property: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def disentanglement_trajectory(self) -> List[float]:
        """Disentanglement score across layers in order."""
        return [r.disentanglement_score for r in self.results.values()]

    @property
    def best_layer(self) -> str:
        """Layer with highest residual linear R²."""
        return max(self.results, key=lambda k: self.results[k].residual_linear_r2)

    @property
    def most_nonlinear_layer(self) -> str:
        """Layer with largest linear-nonlinear gap."""
        return max(self.results, key=lambda k: self.results[k].linear_nonlinear_gap)

    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        import pandas as pd

        rows = []
        for layer_name, r in self.results.items():
            rows.append(
                {
                    "layer": layer_name,
                    "composition_r2": r.composition_r2,
                    "residual_linear_r2": r.residual_linear_r2,
                    "residual_mlp_r2": r.residual_mlp_r2,
                    "gap": r.linear_nonlinear_gap,
                    "disentanglement": r.disentanglement_score,
                    "composition_variance": r.composition_variance_explained,
                }
            )
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"CPD Profile: {self.model_name}",
            f"Target: {self.target_property}",
            f"Layers analyzed: {len(self.results)}",
            f"Best geometric layer: {self.best_layer} "
            f"(R²_resid={self.results[self.best_layer].residual_linear_r2:.3f})",
            f"Most nonlinear layer: {self.most_nonlinear_layer} "
            f"(gap={self.results[self.most_nonlinear_layer].linear_nonlinear_gap:.3f})",
        ]
        return "\n".join(lines)


class CompositionProjectionDecomposition:
    """
    CPD: Separate composition-encoded from geometry-encoded information.

    The core diagnostic from the CPD paper. Given:
    - H: model representations at some layer (n_samples × d)
    - C: composition features (n_samples × n_elements), e.g., fractional composition
    - y: target property values (n_samples,)

    CPD computes:
    1. H_comp = Ridge(C → H).predict(C)  — composition-predicted component
    2. H_resid = H - H_comp               — residual (geometric) component
    3. R²_resid = LinearProbe(H_resid → y) — geometric information accessibility

    Parameters
    ----------
    composition_alpha : float
        Ridge regularization for composition projection. Default 1.0.
    probe_type : str
        "both" (default) runs linear and MLP probes on residual.
        "linear" or "mlp" for only one.
    cv_folds : int
        Cross-validation folds for probing.
    mlp_hidden_dim : int
        Hidden layer size for MLP probe.
    random_state : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        composition_alpha: float = 1.0,
        probe_type: Literal["both", "linear", "mlp"] = "both",
        cv_folds: int = 5,
        mlp_hidden_dim: int = 128,
        random_state: Optional[int] = 42,
    ):
        self.composition_alpha = composition_alpha
        self.probe_type = probe_type
        self.cv_folds = cv_folds
        self.mlp_hidden_dim = mlp_hidden_dim
        self.random_state = random_state

    def decompose(
        self,
        representations: np.ndarray,
        compositions: np.ndarray,
        targets: np.ndarray,
        layer_name: str = "unknown",
        target_property: str = "unknown",
    ) -> CPDResult:
        """
        Run CPD on a single layer's representations.

        Parameters
        ----------
        representations : np.ndarray
            Shape (n_samples, hidden_dim). Layer activations.
        compositions : np.ndarray
            Shape (n_samples, n_elements). Fractional composition vectors.
        targets : np.ndarray
            Shape (n_samples,). Target property values.
        layer_name : str
            Name of the layer (for labeling).
        target_property : str
            Name of the target property (for labeling).

        Returns
        -------
        CPDResult
        """
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        n_samples = representations.shape[0]
        assert compositions.shape[0] == n_samples
        assert targets.shape[0] == n_samples

        # Step 1: Project out composition
        scaler_c = StandardScaler()
        C_scaled = scaler_c.fit_transform(compositions)

        ridge = Ridge(alpha=self.composition_alpha)
        ridge.fit(C_scaled, representations)
        H_comp = ridge.predict(C_scaled)
        H_resid = representations - H_comp

        # Composition variance explained
        total_var = np.var(representations, axis=0).sum()
        resid_var = np.var(H_resid, axis=0).sum()
        comp_var_explained = 1.0 - (resid_var / (total_var + 1e-10))

        # Composition R²: how well does composition predict the representation?
        from sklearn.metrics import r2_score

        comp_r2 = r2_score(
            representations, H_comp, multioutput="variance_weighted"
        )

        # Step 2: Probe the residual
        resid_linear_r2 = 0.0
        resid_mlp_r2 = 0.0

        if self.probe_type in ("both", "linear"):
            from matscope.probes.linear import LinearProbe

            probe = LinearProbe(task="regression", regularization=1.0)
            metrics = probe.fit_evaluate(H_resid, targets, cv_folds=self.cv_folds)
            resid_linear_r2 = metrics["r2"]

        if self.probe_type in ("both", "mlp"):
            from matscope.probes.nonlinear import MLPProbe

            probe = MLPProbe(
                task="regression",
                hidden_dim=self.mlp_hidden_dim,
                max_iter=5000,
            )
            metrics = probe.fit_evaluate(H_resid, targets, cv_folds=self.cv_folds)
            resid_mlp_r2 = metrics["r2"]

        gap = resid_mlp_r2 - resid_linear_r2

        return CPDResult(
            layer_name=layer_name,
            composition_r2=float(comp_r2),
            residual_linear_r2=float(resid_linear_r2),
            residual_mlp_r2=float(resid_mlp_r2),
            linear_nonlinear_gap=float(gap),
            target_property=target_property,
            composition_variance_explained=float(comp_var_explained),
            metadata={
                "n_samples": n_samples,
                "hidden_dim": representations.shape[1],
                "n_elements": compositions.shape[1],
                "composition_alpha": self.composition_alpha,
                "cv_folds": self.cv_folds,
            },
        )

    def profile(
        self,
        representations: Dict[str, np.ndarray],
        compositions: np.ndarray,
        targets: np.ndarray,
        model_name: str = "unknown",
        target_property: str = "unknown",
    ) -> CPDProfile:
        """
        Run CPD across all layers to produce a full disentanglement profile.

        Parameters
        ----------
        representations : dict
            {layer_name: array of shape (n_samples, hidden_dim)}
        compositions : np.ndarray
            Shape (n_samples, n_elements).
        targets : np.ndarray
            Shape (n_samples,).
        model_name : str
            Model identifier for labeling.
        target_property : str
            Target property name.

        Returns
        -------
        CPDProfile
        """
        results = {}
        for layer_name, H in representations.items():
            logger.info(f"CPD: analyzing layer '{layer_name}' ({H.shape})")
            results[layer_name] = self.decompose(
                representations=H,
                compositions=compositions,
                targets=targets,
                layer_name=layer_name,
                target_property=target_property,
            )

        return CPDProfile(
            model_name=model_name,
            results=results,
            target_property=target_property,
        )

    @staticmethod
    def compare_profiles(
        profiles: Dict[str, CPDProfile],
    ) -> Dict[str, Any]:
        """
        Compare CPD profiles across multiple models.

        This produces the "disentanglement gradient" — the core figure
        from the paper showing how different architectures organize
        composition vs geometry information differently.

        Parameters
        ----------
        profiles : dict
            {model_name: CPDProfile}

        Returns
        -------
        dict
            Comparison statistics including per-model disentanglement
            scores and cross-model rankings.
        """
        comparison = {}
        for model_name, profile in profiles.items():
            best = profile.results[profile.best_layer]
            comparison[model_name] = {
                "best_residual_r2": best.residual_linear_r2,
                "best_gap": best.linear_nonlinear_gap,
                "best_disentanglement": best.disentanglement_score,
                "mean_composition_r2": np.mean(
                    [r.composition_r2 for r in profile.results.values()]
                ),
                "trajectory": profile.disentanglement_trajectory,
            }

        # Rank by disentanglement
        ranked = sorted(
            comparison.items(),
            key=lambda x: x[1]["best_disentanglement"],
            reverse=True,
        )
        for rank, (name, stats) in enumerate(ranked, 1):
            stats["disentanglement_rank"] = rank

        return comparison
