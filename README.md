# MatScope

**Diagnostic toolkit for scientific foundation models.**

MatScope brings mechanistic interpretability to atomistic ML. Understand what your model learned, where it will fail, and how representations evolve across layers.

**Paper:** [Information Routing in Atomistic Foundation Models: A Composition Projection Decomposition Analysis](https://arxiv.org/) (Steier, 2026)
<!-- TODO: Update with arXiv ID once assigned -->

## What MatScope Does

Most atomistic foundation model benchmarks test *performance* — energies, forces, stability. MatScope tests *representations*: what information is encoded, at which layer, and in what form.

The central method is **Composition Projection Decomposition (CPD)**, which separates composition-encoded from geometry-encoded information in model representations. Key finding: equivariant models (MACE) produce linearly disentangled composition/geometry representations, while invariant models (ANI-2x) require nonlinear access to geometric information.

## Installation

```bash
pip install matscope
```

With model backends:

```bash
pip install matscope[mace]      # MACE support
pip install matscope[pyg]       # SchNet, DimeNet++, PaiNN
pip install matscope[ani]       # ANI-2x (TorchANI)
pip install matscope[full]      # Everything
```

## Quick Start

### CPD Analysis

```python
from matscope import CompositionProjectionDecomposition
from matscope.datasets import composition_features
import numpy as np

# Your model representations at each layer (n_samples × hidden_dim)
representations = {
    "interaction_0": layer_0_reps,
    "interaction_1": layer_1_reps,
    "readout": readout_reps,
}

# Composition features from atomic structures
compositions = composition_features(structures, method="fractional")

# Run CPD
cpd = CompositionProjectionDecomposition()
profile = cpd.profile(
    representations, compositions, targets,
    model_name="MACE-MP-0",
    target_property="HOMO-LUMO gap"
)

print(profile.summary())
# CPD Profile: MACE-MP-0
# Target: HOMO-LUMO gap
# Best geometric layer: interaction_2 (R²_resid=0.782)
# Most nonlinear layer: interaction_0 (gap=0.031)
```

### Compare Architectures

```python
# Run CPD on multiple models
profiles = {
    "MACE": mace_profile,
    "SchNet": schnet_profile,
    "ANI-2x": ani_profile,
}

comparison = CompositionProjectionDecomposition.compare_profiles(profiles)
# MACE: rank 1 (high residual R², low gap — linearly disentangled)
# ANI-2x: rank 3 (negative residual R², high gap — nonlinearly entangled)
```

### Probing

```python
from matscope import MatScope

# Load model and probe
ms = MatScope.from_model("mace-mp-0")
reps = ms.extract(dataset, layers=["interaction_0", "interaction_1", "readout"])
result = ms.probe("coordination_number", representations=reps, labels=labels)
ms.report(result, save="probing_results.pdf")
```

### Representation Comparison

```python
# Compare what two models learn
sim = ms_mace.compare(ms_chgnet, dataset=structures, method="cka")
ms_mace.report(sim, save="cka_matrix.pdf")
```

### Distribution Shift Detection

```python
shift = ms.detect_shift(
    train_data=bulk_crystals,
    deploy_data=surface_slabs,
    method="mmd"
)
print(f"Shift hits hardest at: {shift.most_affected_layer}")
```

## Supported Models

| Model | Backend | Architecture Type |
|-------|---------|------------------|
| MACE-MP-0 | `mace-torch` | Equivariant (E(3)) |
| CHGNet | `chgnet` | Graph neural network |
| SchNet | `torch-geometric` | Learned invariant |
| DimeNet++ | `torch-geometric` | Directional message passing |
| PaiNN | `torch-geometric` | Equivariant (E(3)) |
| ANI-2x | `torchani` | Handcrafted invariant |
| Any PyTorch model | built-in | Generic hook registration |

## API Reference

### Core Classes

- **`MatScope`** — Main orchestrator. Load models, extract representations, probe, compare, detect shift.
- **`CompositionProjectionDecomposition`** — CPD methodology. Separates composition/geometry in representations.
- **`LinearProbe` / `MLPProbe`** — Probing classifiers/regressors with cross-validation.
- **`SelectivityProbe`** — Linear probe with feature selectivity analysis.
- **`RepresentationSimilarity`** — CKA, CCA, Procrustes comparison across models/layers.
- **`ShiftAnalyzer`** — MMD, Fisher, cosine drift detection.
- **`LayerwiseAnalyzer`** — Effective dimensionality, isotropy, entropy per layer.

### Dataset Utilities

- **`composition_features(structures)`** — Extract fractional/count/onehot composition vectors.
- **`load_qm9()`** — Load QM9 with all 19 properties.
- **`load_from_ase(atoms_list)`** — Wrap ASE Atoms for MatScope.
- **`train_test_split(dataset_dict)`** — Split with consistent indexing.

## Relationship to Other Tools

- **mlipx** (BASF): Benchmarks *performance* (energy curves, MD stability). Complementary — mlipx asks "how accurate?", MatScope asks "what did it learn?"
- **LAMBench / MLIP-Arena**: Leaderboard comparisons. MatScope goes deeper into representations.
- **TransformerLens / SAELens**: LLM interpretability. MatScope is the atomistic equivalent.

## Citation

```bibtex
@article{steier2026information,
  title={Information Routing in Atomistic Foundation Models: A Composition Projection Decomposition Analysis},
  author={Steier, Joshua},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

Apache 2.0
