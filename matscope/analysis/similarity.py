"""
Representation similarity analysis.

Methods for comparing representations across models, layers,
or training stages. Key tool for answering:
- Do two models learn the same things?
- How do representations evolve across layers?
- Does fine-tuning change what is encoded?

Implements:
- CKA (Centered Kernel Alignment) — Kornblith et al., 2019
- CCA (Canonical Correlation Analysis)
- Procrustes distance
"""

from __future__ import annotations

from typing import Dict, Literal

import numpy as np


class RepresentationSimilarity:
    """
    Compute pairwise similarity between sets of representations.

    Parameters
    ----------
    method : str
        "cka" (default), "cca", or "procrustes".
    kernel : str
        Kernel for CKA. "linear" (default) or "rbf".
    """

    def __init__(
        self,
        method: Literal["cka", "cca", "procrustes"] = "cka",
        kernel: str = "linear",
    ):
        self.method = method
        self.kernel = kernel

    def compute(
        self,
        representations_a: Dict[str, np.ndarray],
        representations_b: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Compute similarity matrix between two sets of representations.

        Parameters
        ----------
        representations_a : dict
            {layer_name: array of shape (n_samples, dim_a)}
        representations_b : dict
            {layer_name: array of shape (n_samples, dim_b)}

        Returns
        -------
        np.ndarray
            Similarity matrix of shape (n_layers_a, n_layers_b).
        """
        layers_a = list(representations_a.keys())
        layers_b = list(representations_b.keys())
        sim_matrix = np.zeros((len(layers_a), len(layers_b)))

        for i, la in enumerate(layers_a):
            for j, lb in enumerate(layers_b):
                X = representations_a[la]
                Y = representations_b[lb]
                assert X.shape[0] == Y.shape[0], (
                    f"Sample count mismatch: {X.shape[0]} vs {Y.shape[0]}"
                )

                if self.method == "cka":
                    sim_matrix[i, j] = self._linear_cka(X, Y)
                elif self.method == "cca":
                    sim_matrix[i, j] = self._cca(X, Y)
                elif self.method == "procrustes":
                    sim_matrix[i, j] = self._procrustes(X, Y)
                else:
                    raise ValueError(f"Unknown method: {self.method}")

        return sim_matrix

    def pairwise_across_layers(
        self,
        representations: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Compute self-similarity across layers of a single model."""
        return self.compute(representations, representations)

    @staticmethod
    def _center_gram(K: np.ndarray) -> np.ndarray:
        """Center a Gram matrix."""
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    def _linear_cka(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Linear CKA (Kornblith et al., 2019).

        Measures similarity of two representations invariant to
        orthogonal transformations and isotropic scaling.
        """
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)

        # Gram matrices
        KX = X @ X.T
        KY = Y @ Y.T

        # HSIC
        hsic_xy = np.sum(self._center_gram(KX) * self._center_gram(KY))
        hsic_xx = np.sum(self._center_gram(KX) * self._center_gram(KX))
        hsic_yy = np.sum(self._center_gram(KY) * self._center_gram(KY))

        denom = np.sqrt(hsic_xx * hsic_yy)
        if denom < 1e-10:
            return 0.0
        return float(hsic_xy / denom)

    @staticmethod
    def _cca(X: np.ndarray, Y: np.ndarray, n_components: int = 10) -> float:
        """
        Mean CCA correlation.

        Returns the mean of the top-k canonical correlations.
        """
        from sklearn.cross_decomposition import CCA

        k = min(n_components, X.shape[1], Y.shape[1])
        cca = CCA(n_components=k)
        X_c, Y_c = cca.fit_transform(X, Y)

        correlations = []
        for i in range(k):
            corr = np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
            correlations.append(abs(corr))

        return float(np.mean(correlations))

    @staticmethod
    def _procrustes(X: np.ndarray, Y: np.ndarray) -> float:
        """
        Procrustes similarity (1 - normalized Procrustes distance).

        Values near 1 indicate similar representations.
        """
        from scipy.spatial import procrustes as scipy_procrustes

        # Ensure same dimensionality by padding
        max_dim = max(X.shape[1], Y.shape[1])
        if X.shape[1] < max_dim:
            X = np.pad(X, ((0, 0), (0, max_dim - X.shape[1])))
        if Y.shape[1] < max_dim:
            Y = np.pad(Y, ((0, 0), (0, max_dim - Y.shape[1])))

        _, _, disparity = scipy_procrustes(X, Y)
        return float(1.0 - disparity)
