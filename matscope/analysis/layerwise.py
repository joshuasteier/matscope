"""
Layerwise representation analysis.

Builds "property emergence maps" — visualizations showing at which
layer each physical property becomes linearly decodable. This is
the core diagnostic for understanding what a scientific FM has learned.

For atomistic models:
- Layer 1 might encode element identity
- Layer 2 might encode coordination number
- Layer 3 might encode bond types
- Final layers might encode complex properties like formation energy

The emergence order tells us about the model's inductive hierarchy.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


class LayerwiseAnalyzer:
    """
    Analyze how representation quality evolves across model depth.

    Computes per-layer statistics including:
    - Effective dimensionality (participation ratio)
    - Isotropy (uniformity of singular value spectrum)
    - Cluster separability (for labeled data)
    - Representation entropy
    """

    @staticmethod
    def effective_dimensionality(X: np.ndarray) -> float:
        """
        Participation ratio of singular value spectrum.

        High values indicate information distributed across many
        dimensions. Low values indicate a low-dimensional manifold.
        """
        X_centered = X - X.mean(axis=0)
        _, s, _ = np.linalg.svd(X_centered, full_matrices=False)
        s_normalized = s**2 / np.sum(s**2)
        pr = 1.0 / np.sum(s_normalized**2)
        return float(pr)

    @staticmethod
    def isotropy(X: np.ndarray) -> float:
        """
        Isotropy score: how uniformly spread the representations are.

        Score of 1.0 = perfectly isotropic (all directions equally used).
        Score near 0 = representations collapsed to a low-dim subspace.

        Uses the ratio of min to max singular value (condition number proxy).
        """
        X_centered = X - X.mean(axis=0)
        _, s, _ = np.linalg.svd(X_centered, full_matrices=False)
        s = s[s > 1e-10]
        if len(s) < 2:
            return 0.0
        return float(s[-1] / s[0])

    @staticmethod
    def cluster_separability(X: np.ndarray, labels: np.ndarray) -> float:
        """
        Fisher criterion: ratio of between-class to within-class variance.

        Higher values indicate better class separation in representation
        space — the model has learned to distinguish the classes.
        """
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return 0.0

        global_mean = X.mean(axis=0)
        between_var = 0.0
        within_var = 0.0

        for label in unique_labels:
            mask = labels == label
            X_class = X[mask]
            n_class = X_class.shape[0]
            class_mean = X_class.mean(axis=0)

            between_var += n_class * np.sum((class_mean - global_mean) ** 2)
            within_var += np.sum((X_class - class_mean) ** 2)

        if within_var < 1e-10:
            return float("inf")
        return float(between_var / within_var)

    @staticmethod
    def representation_entropy(X: np.ndarray, n_bins: int = 50) -> float:
        """
        Approximate entropy of the representation distribution.

        Uses histogram-based estimation on the first few principal
        components. Higher entropy = more diverse representations.
        """
        X_centered = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)

        n_components = min(10, Vt.shape[0])
        X_projected = X_centered @ Vt[:n_components].T

        total_entropy = 0.0
        for i in range(n_components):
            counts, _ = np.histogram(X_projected[:, i], bins=n_bins)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            total_entropy -= np.sum(probs * np.log(probs))

        return float(total_entropy / n_components)

    def analyze_all_layers(
        self,
        representations: Dict[str, np.ndarray],
        labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run all layerwise analyses.

        Parameters
        ----------
        representations : dict
            {layer_name: array of shape (n_samples, hidden_dim)}
        labels : np.ndarray, optional
            If provided, computes cluster separability.

        Returns
        -------
        dict
            {layer_name: {metric_name: value}}
        """
        results = {}
        for layer_name, X in representations.items():
            layer_metrics = {
                "effective_dim": self.effective_dimensionality(X),
                "isotropy": self.isotropy(X),
                "entropy": self.representation_entropy(X),
                "n_samples": X.shape[0],
                "hidden_dim": X.shape[1],
            }
            if labels is not None:
                layer_metrics["separability"] = self.cluster_separability(X, labels)

            results[layer_name] = layer_metrics
        return results
