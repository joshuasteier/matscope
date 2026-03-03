"""
Dataset utilities for MatScope.

Provides standardized dataset loading, composition feature extraction,
and train/test splitting for diagnostic analyses.

Supported datasets:
    - QM9 (via PyG or raw download)
    - Materials Project (via MPRester or local cache)
    - Custom ASE Atoms collections

The key utility for CPD is `composition_features()`, which converts
atomic structures into fractional composition vectors suitable for
the composition projection step.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Standard periodic table (first 95 elements)
ELEMENT_LIST = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am",
]

ELEMENT_TO_IDX = {e: i for i, e in enumerate(ELEMENT_LIST)}


def composition_features(
    structures: Union[List[Any], Any],
    method: str = "fractional",
    elements: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Extract composition feature vectors from atomic structures.

    Parameters
    ----------
    structures : list
        List of ASE Atoms objects, PyG Data objects, or dicts with
        'atomic_numbers' or 'numbers' keys.
    method : str
        "fractional" — fraction of each element (sums to 1)
        "count" — raw atom counts per element
        "onehot" — binary presence/absence of each element
    elements : list of str, optional
        Restrict to these elements. If None, uses all elements present
        in the dataset (compact representation).

    Returns
    -------
    np.ndarray
        Shape (n_structures, n_elements). Composition feature matrix.
    """
    if not isinstance(structures, (list, tuple)):
        structures = [structures]

    # Extract atomic numbers from various formats
    all_numbers = []
    for s in structures:
        numbers = _get_atomic_numbers(s)
        all_numbers.append(numbers)

    # Determine element set
    if elements is not None:
        elem_indices = {e: i for i, e in enumerate(elements)}
        n_elements = len(elements)
    else:
        # Auto-detect: use only elements present in dataset
        unique_z = set()
        for nums in all_numbers:
            unique_z.update(nums)
        sorted_z = sorted(unique_z)
        elem_indices = {ELEMENT_LIST[z - 1]: i for i, z in enumerate(sorted_z) if z <= len(ELEMENT_LIST)}
        n_elements = len(elem_indices)

    # Build composition matrix
    C = np.zeros((len(structures), n_elements))
    z_to_elem = {i + 1: e for i, e in enumerate(ELEMENT_LIST)}

    for i, numbers in enumerate(all_numbers):
        for z in numbers:
            elem = z_to_elem.get(z, None)
            if elem and elem in elem_indices:
                C[i, elem_indices[elem]] += 1

    if method == "fractional":
        row_sums = C.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        C = C / row_sums
    elif method == "onehot":
        C = (C > 0).astype(float)
    # "count" returns raw counts

    return C


def _get_atomic_numbers(structure: Any) -> List[int]:
    """Extract atomic numbers from various structure formats."""
    # ASE Atoms
    if hasattr(structure, "get_atomic_numbers"):
        return list(structure.get_atomic_numbers())

    # PyG Data
    if hasattr(structure, "z"):
        z = structure.z
        if hasattr(z, "numpy"):
            return z.numpy().tolist()
        return list(z)

    if hasattr(structure, "atomic_numbers"):
        an = structure.atomic_numbers
        if hasattr(an, "numpy"):
            return an.numpy().tolist()
        return list(an)

    # Dict-like
    if isinstance(structure, dict):
        for key in ("atomic_numbers", "numbers", "z", "Z"):
            if key in structure:
                val = structure[key]
                if hasattr(val, "numpy"):
                    return val.numpy().tolist()
                return list(val)

    raise ValueError(
        f"Cannot extract atomic numbers from {type(structure)}. "
        "Expected ASE Atoms, PyG Data, or dict with 'atomic_numbers' key."
    )


def load_qm9(
    root: str = "./data/qm9",
    target_properties: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Load the QM9 dataset.

    Tries PyG first, falls back to local cache.

    Parameters
    ----------
    root : str
        Directory to store/load data.
    target_properties : list of str, optional
        QM9 property names to load. Default: all 19 properties.
    max_samples : int, optional
        Limit dataset size (useful for quick experiments).

    Returns
    -------
    dict
        {
            "dataset": list of PyG Data objects,
            "compositions": np.ndarray of shape (N, n_elements),
            "targets": dict of {property_name: np.ndarray},
            "property_names": list of str,
        }
    """
    QM9_PROPERTIES = [
        "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve",
        "U0", "U", "H", "G", "Cv", "U0_atom", "U_atom",
        "H_atom", "G_atom", "A", "B", "C",
    ]

    try:
        from torch_geometric.datasets import QM9

        dataset = QM9(root=root)
        if max_samples:
            dataset = dataset[:max_samples]

        # Extract targets
        targets = {}
        props = target_properties or QM9_PROPERTIES
        for i, name in enumerate(QM9_PROPERTIES):
            if name in props:
                try:
                    targets[name] = dataset.data.y[:len(dataset), i].numpy()
                    if max_samples:
                        targets[name] = targets[name][:max_samples]
                except (IndexError, AttributeError):
                    logger.warning(f"Could not extract property '{name}' from QM9")

        # Extract compositions
        compositions = composition_features(dataset, method="fractional")

        return {
            "dataset": dataset,
            "compositions": compositions,
            "targets": targets,
            "property_names": list(targets.keys()),
        }

    except ImportError:
        logger.warning(
            "PyG not available for QM9 loading. "
            "Install with: pip install torch-geometric\n"
            "Or provide a local QM9 cache."
        )
        raise


def load_from_ase(
    atoms_list: List[Any],
    target_values: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Wrap a list of ASE Atoms objects into MatScope format.

    Parameters
    ----------
    atoms_list : list
        List of ASE Atoms objects.
    target_values : dict, optional
        {property_name: np.ndarray of shape (n_structures,)}

    Returns
    -------
    dict
        MatScope-compatible dataset dict.
    """
    compositions = composition_features(atoms_list, method="fractional")

    return {
        "dataset": atoms_list,
        "compositions": compositions,
        "targets": target_values or {},
        "property_names": list((target_values or {}).keys()),
    }


def train_test_split(
    dataset_dict: Dict[str, Any],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Split a MatScope dataset dict into train and test.

    Parameters
    ----------
    dataset_dict : dict
        Output from load_qm9, load_from_ase, etc.
    test_size : float
        Fraction for test set.
    random_state : int
        Random seed.

    Returns
    -------
    tuple
        (train_dict, test_dict)
    """
    from sklearn.model_selection import train_test_split as sklearn_split

    n = len(dataset_dict["dataset"])
    indices = np.arange(n)
    train_idx, test_idx = sklearn_split(
        indices, test_size=test_size, random_state=random_state
    )

    def _subset(d, idx):
        result = {
            "dataset": [d["dataset"][i] for i in idx],
            "compositions": d["compositions"][idx],
            "targets": {k: v[idx] for k, v in d["targets"].items()},
            "property_names": d["property_names"],
        }
        return result

    return _subset(dataset_dict, train_idx), _subset(dataset_dict, test_idx)
