"""
MatScope core orchestrator.

This is the primary user-facing class. It wraps model loading,
hook registration, representation extraction, probing, and reporting
into a clean, composable API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """Container for probing results."""

    probe_type: str
    target_property: str
    layer_results: Dict[str, Dict[str, float]]  # layer_name -> {metric: value}
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def best_layer(self) -> str:
        """Return the layer with highest accuracy / lowest error."""
        metric = "accuracy" if "accuracy" in next(iter(self.layer_results.values())) else "r2"
        return max(self.layer_results, key=lambda k: self.layer_results[k].get(metric, 0.0))

    @property
    def summary(self) -> Dict[str, float]:
        """Return best-layer metrics."""
        return self.layer_results[self.best_layer]

    def to_dataframe(self):
        """Convert to pandas DataFrame for easy analysis."""
        import pandas as pd

        rows = []
        for layer, metrics in self.layer_results.items():
            rows.append({"layer": layer, **metrics})
        return pd.DataFrame(rows)


@dataclass
class SimilarityResult:
    """Container for representation similarity results."""

    method: str  # "cka", "cca", "procrustes"
    similarity_matrix: np.ndarray
    layer_names_a: List[str]
    layer_names_b: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShiftResult:
    """Container for distribution shift analysis."""

    layers_analyzed: List[str]
    shift_scores: Dict[str, float]  # layer -> shift magnitude
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def most_affected_layer(self) -> str:
        return max(self.shift_scores, key=self.shift_scores.get)


class MatScope:
    """
    Main diagnostic toolkit for scientific foundation models.

    MatScope extracts intermediate representations from any model
    and runs standardized diagnostic analyses: linear probes,
    representation similarity, distribution shift detection,
    and layerwise property emergence.

    Parameters
    ----------
    model : Any
        A PyTorch model, or a model wrapper that implements
        the MatScope model interface.
    representation_hooks : dict, optional
        Mapping of layer_name -> hook_fn for custom extraction.
        If None, MatScope auto-discovers layers.
    device : str
        Device for computation.

    Examples
    --------
    >>> pk = MatScope.from_model("mace-mp-0")
    >>> results = pk.probe("bond_type", dataset=my_data)
    >>> print(results.best_layer, results.summary)

    >>> sim = pk.compare(model_a, model_b, dataset=my_data)
    >>> sim.similarity_matrix  # CKA matrix across layers

    >>> shift = pk.detect_shift(train_data, deploy_data)
    >>> print(shift.most_affected_layer)
    """

    def __init__(
        self,
        model: Any,
        representation_hooks: Optional[Dict[str, Callable]] = None,
        device: str = "cpu",
    ):
        self.model = model
        self.device = device
        self._hooks = representation_hooks or {}
        self._representations: Dict[str, np.ndarray] = {}
        self._registered_hooks: List[Any] = []

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_model(
        cls,
        model_name_or_path: str,
        device: str = "cpu",
        **kwargs,
    ) -> "MatScope":
        """
        Load a scientific FM by name or path.

        Supported models (v0.1):
            - "mace-mp-0"      : MACE foundation model
            - "mace-off"       : MACE organic force field
            - "chgnet"         : CHGNet
            - "orb"            : Orb models
            - Any local .pt checkpoint

        Additional backends planned for v0.2:
            - EquiformerV2, GNoME, SchNet, DimeNet++

        Parameters
        ----------
        model_name_or_path : str
            Model identifier or local path.
        device : str
            Target device.
        """
        from matscope.models.registry import load_model

        model, hooks = load_model(model_name_or_path, device=device, **kwargs)
        return cls(model=model, representation_hooks=hooks, device=device)

    @classmethod
    def from_torch(cls, model, layer_names: Optional[List[str]] = None, device: str = "cpu") -> "MatScope":
        """
        Wrap any PyTorch nn.Module.

        If layer_names is None, MatScope auto-discovers all named
        modules and registers forward hooks.
        """
        from matscope.models.torch_wrapper import wrap_torch_model

        wrapped, hooks = wrap_torch_model(model, layer_names=layer_names)
        return cls(model=wrapped, representation_hooks=hooks, device=device)

    # ------------------------------------------------------------------
    # Representation extraction
    # ------------------------------------------------------------------

    def extract(
        self,
        dataset: Any,
        layers: Optional[List[str]] = None,
        batch_size: int = 32,
        max_samples: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extract representations from specified layers.

        Parameters
        ----------
        dataset : Any
            An iterable yielding model inputs (e.g., ASE Atoms,
            PyG Data, or raw tensors).
        layers : list of str, optional
            Layer names to extract. None = all registered hooks.
        batch_size : int
            Batch size for extraction.
        max_samples : int, optional
            Cap on total samples extracted.

        Returns
        -------
        dict
            Mapping layer_name -> np.ndarray of shape (n_samples, hidden_dim).
        """
        from matscope.utils.extraction import extract_representations

        target_layers = layers or list(self._hooks.keys())
        self._representations = extract_representations(
            model=self.model,
            hooks=self._hooks,
            dataset=dataset,
            layers=target_layers,
            batch_size=batch_size,
            max_samples=max_samples,
            device=self.device,
        )
        return self._representations

    # ------------------------------------------------------------------
    # Probing
    # ------------------------------------------------------------------

    def probe(
        self,
        target_property: str,
        dataset: Any = None,
        representations: Optional[Dict[str, np.ndarray]] = None,
        labels: Optional[np.ndarray] = None,
        probe_type: Literal["linear", "mlp"] = "linear",
        task: Literal["classification", "regression"] = "classification",
        cv_folds: int = 5,
        layers: Optional[List[str]] = None,
        **probe_kwargs,
    ) -> ProbeResult:
        """
        Train probes to predict a target property from representations.

        This is the core diagnostic: if a linear probe can recover
        a physical property (e.g., bond type, oxidation state,
        coordination number) from a layer's representation, that
        property is encoded at that layer.

        Parameters
        ----------
        target_property : str
            Name of the property being probed (for labeling).
        dataset : Any, optional
            If representations haven't been extracted yet, extract from this.
        representations : dict, optional
            Pre-extracted representations {layer: array}.
        labels : np.ndarray, optional
            Target labels. If dataset is a MatScope dataset, labels
            are extracted automatically.
        probe_type : str
            "linear" for LogisticRegression/Ridge, "mlp" for 1-hidden-layer MLP.
        task : str
            "classification" or "regression".
        cv_folds : int
            Number of cross-validation folds.
        layers : list of str, optional
            Subset of layers to probe.

        Returns
        -------
        ProbeResult
        """
        from matscope.probes.linear import LinearProbe
        from matscope.probes.nonlinear import MLPProbe

        reps = representations or self._representations
        if not reps:
            if dataset is None:
                raise ValueError("Provide either dataset or pre-extracted representations.")
            reps = self.extract(dataset, layers=layers)

        target_layers = layers or list(reps.keys())
        probe_cls = LinearProbe if probe_type == "linear" else MLPProbe

        layer_results = {}
        for layer_name in target_layers:
            if layer_name not in reps:
                logger.warning(f"Layer '{layer_name}' not in representations, skipping.")
                continue

            X = reps[layer_name]
            y = labels  # TODO: auto-extract from dataset if MatScope dataset

            probe = probe_cls(task=task, **probe_kwargs)
            metrics = probe.fit_evaluate(X, y, cv_folds=cv_folds)
            layer_results[layer_name] = metrics

        return ProbeResult(
            probe_type=probe_type,
            target_property=target_property,
            layer_results=layer_results,
            metadata={"cv_folds": cv_folds, "task": task},
        )

    # ------------------------------------------------------------------
    # Representation similarity
    # ------------------------------------------------------------------

    def compare(
        self,
        other: Union["MatScope", Dict[str, np.ndarray]],
        dataset: Any = None,
        method: Literal["cka", "cca", "procrustes"] = "cka",
        layers_self: Optional[List[str]] = None,
        layers_other: Optional[List[str]] = None,
    ) -> SimilarityResult:
        """
        Compare representations between two models (or two sets of representations).

        Uses Centered Kernel Alignment (CKA), Canonical Correlation
        Analysis (CCA), or Procrustes distance.

        Parameters
        ----------
        other : MatScope or dict
            Second model or pre-extracted representations.
        dataset : Any
            Common dataset to extract representations from both models.
        method : str
            Similarity method.
        """
        from matscope.analysis.similarity import RepresentationSimilarity

        reps_a = self._representations
        if isinstance(other, MatScope):
            reps_b = other._representations
            names_b = layers_other or list(reps_b.keys())
        else:
            reps_b = other
            names_b = layers_other or list(reps_b.keys())

        names_a = layers_self or list(reps_a.keys())

        analyzer = RepresentationSimilarity(method=method)
        sim_matrix = analyzer.compute(
            {k: reps_a[k] for k in names_a},
            {k: reps_b[k] for k in names_b},
        )

        return SimilarityResult(
            method=method,
            similarity_matrix=sim_matrix,
            layer_names_a=names_a,
            layer_names_b=names_b,
        )

    # ------------------------------------------------------------------
    # Distribution shift
    # ------------------------------------------------------------------

    def detect_shift(
        self,
        train_data: Any,
        deploy_data: Any,
        layers: Optional[List[str]] = None,
        method: Literal["mmd", "fisher", "cosine_drift"] = "mmd",
    ) -> ShiftResult:
        """
        Detect representation-level distribution shift between
        training and deployment data.

        This answers: "where in the network does shift hit hardest?"

        Parameters
        ----------
        train_data : Any
            Training distribution data.
        deploy_data : Any
            Deployment / test distribution data.
        method : str
            Shift detection method.
        """
        from matscope.analysis.shift import ShiftAnalyzer

        reps_train = self.extract(train_data, layers=layers)
        reps_deploy = self.extract(deploy_data, layers=layers)

        analyzer = ShiftAnalyzer(method=method)
        shift_scores = {}
        target_layers = layers or list(reps_train.keys())

        for layer_name in target_layers:
            if layer_name in reps_train and layer_name in reps_deploy:
                score = analyzer.compute_shift(reps_train[layer_name], reps_deploy[layer_name])
                shift_scores[layer_name] = score

        return ShiftResult(
            layers_analyzed=list(shift_scores.keys()),
            shift_scores=shift_scores,
            metadata={"method": method},
        )

    # ------------------------------------------------------------------
    # Layerwise analysis
    # ------------------------------------------------------------------

    def layerwise_analysis(
        self,
        dataset: Any,
        properties: List[str],
        labels_dict: Dict[str, np.ndarray],
        probe_type: Literal["linear", "mlp"] = "linear",
    ) -> Dict[str, ProbeResult]:
        """
        Run probing for multiple properties across all layers.

        Produces a "property emergence map" — showing which
        physical properties are learned at which depth.

        Parameters
        ----------
        dataset : Any
            Input data.
        properties : list of str
            Property names to probe.
        labels_dict : dict
            {property_name: labels_array}.

        Returns
        -------
        dict
            {property_name: ProbeResult}
        """
        reps = self.extract(dataset)
        results = {}
        for prop in properties:
            if prop not in labels_dict:
                logger.warning(f"No labels for '{prop}', skipping.")
                continue
            results[prop] = self.probe(
                target_property=prop,
                representations=reps,
                labels=labels_dict[prop],
                probe_type=probe_type,
            )
        return results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(
        self,
        results: Union[ProbeResult, Dict[str, ProbeResult], SimilarityResult, ShiftResult],
        save: Optional[str] = None,
        show: bool = True,
    ):
        """
        Generate publication-quality diagnostic report.

        Parameters
        ----------
        results : ProbeResult, dict, SimilarityResult, or ShiftResult
            Output from any MatScope analysis.
        save : str, optional
            Path to save figure(s).
        show : bool
            Whether to display interactively.
        """
        from matscope.utils.reporting import generate_report

        generate_report(results, save=save, show=show)
