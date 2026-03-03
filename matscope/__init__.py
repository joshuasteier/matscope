"""
MatScope: A diagnostic toolkit for scientific foundation models.

MatScope brings mechanistic interpretability to scientific ML —
atomistic foundation models, molecular FMs, and beyond.

Understand what your model learned, where it will fail,
and how representations evolve across layers.

New in v0.2.0:
    - Composition Projection Decomposition (CPD)
    - Model backends for SchNet, DimeNet++, PaiNN, ANI-2x
    - Dataset utilities (QM9 loading, composition features)
    - linear_nonlinear_gap convenience function

Example:
    >>> from matscope import MatScope
    >>> pk = MatScope.from_model("mace-mp-0")
    >>> results = pk.probe("bond_type", dataset=my_data)
    >>> pk.report(results, save="probing_report.pdf")

CPD Example:
    >>> from matscope import CompositionProjectionDecomposition
    >>> from matscope.datasets import load_qm9, composition_features
    >>> cpd = CompositionProjectionDecomposition()
    >>> result = cpd.decompose(representations, compositions, targets)
    >>> print(result.residual_linear_r2, result.linear_nonlinear_gap)
"""

__version__ = "0.2.0"
__author__ = "Joshua Steier"

from matscope.core import MatScope
from matscope.probes.linear import LinearProbe, SelectivityProbe
from matscope.probes.nonlinear import MLPProbe, linear_nonlinear_gap
from matscope.analysis.similarity import RepresentationSimilarity
from matscope.analysis.shift import ShiftAnalyzer
from matscope.analysis.layerwise import LayerwiseAnalyzer
from matscope.analysis.cpd import (
    CompositionProjectionDecomposition,
    CPDResult,
    CPDProfile,
)

__all__ = [
    # Core
    "MatScope",
    # Probes
    "LinearProbe",
    "SelectivityProbe",
    "MLPProbe",
    "linear_nonlinear_gap",
    # Analysis
    "RepresentationSimilarity",
    "ShiftAnalyzer",
    "LayerwiseAnalyzer",
    # CPD (paper methodology)
    "CompositionProjectionDecomposition",
    "CPDResult",
    "CPDProfile",
]
