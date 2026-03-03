"""
Representation extraction engine.

Handles the mechanics of running data through a model,
capturing intermediate representations via forward hooks,
and aggregating per-atom representations to per-structure
representations for atomistic models.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def extract_representations(
    model: Any,
    hooks: Dict[str, Callable],
    dataset: Any,
    layers: List[str],
    batch_size: int = 32,
    max_samples: Optional[int] = None,
    device: str = "cpu",
    aggregation: str = "mean",
) -> Dict[str, np.ndarray]:
    """
    Extract representations from a model using forward hooks.

    This function handles three data formats:
    1. Lists of ASE Atoms objects (atomistic models)
    2. PyTorch Geometric Data/Batch objects
    3. Raw tensors

    Parameters
    ----------
    model : Any
        The model to extract from.
    hooks : dict
        {layer_name: hook_fn}
    dataset : Any
        Input data.
    layers : list
        Which layers to extract.
    batch_size : int
        Batch size.
    max_samples : int, optional
        Maximum samples to process.
    device : str
        Device.
    aggregation : str
        How to aggregate per-atom representations to per-structure.
        "mean" (default), "sum", "max", or "none" (keep per-atom).

    Returns
    -------
    dict
        {layer_name: np.ndarray of shape (n_structures, hidden_dim)}
    """
    import torch

    model.eval()
    collected: Dict[str, List[np.ndarray]] = {layer: [] for layer in layers}

    # Register hooks
    handle_list = []
    hook_outputs: Dict[str, Any] = {}

    def make_capture_hook(layer_name):
        def capture(module, input, output):
            if hasattr(output, "detach"):
                hook_outputs[layer_name] = output.detach().cpu()
            elif isinstance(output, tuple) and hasattr(output[0], "detach"):
                hook_outputs[layer_name] = output[0].detach().cpu()
            elif isinstance(output, dict):
                for key in ["node_feats", "x", "features", "hidden"]:
                    if key in output and hasattr(output[key], "detach"):
                        hook_outputs[layer_name] = output[key].detach().cpu()
                        break
        return capture

    # Find and hook modules
    for name, module in model.named_modules():
        for layer_name in layers:
            if layer_name in name or name == layer_name:
                h = module.register_forward_hook(make_capture_hook(layer_name))
                handle_list.append(h)

    # Process data
    n_processed = 0
    try:
        with torch.no_grad():
            for batch in _iter_batches(dataset, batch_size):
                if max_samples and n_processed >= max_samples:
                    break

                hook_outputs.clear()

                # Forward pass
                if isinstance(batch, dict):
                    batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
                    _ = model(**batch)
                elif hasattr(batch, "to"):
                    batch = batch.to(device)
                    _ = model(batch)
                else:
                    _ = model(batch)

                # Collect outputs
                for layer_name in layers:
                    if layer_name in hook_outputs:
                        rep = hook_outputs[layer_name].numpy()
                        if aggregation != "none" and rep.ndim == 3:
                            rep = _aggregate(rep, method=aggregation)
                        collected[layer_name].append(rep)

                n_processed += batch_size

    finally:
        # Clean up hooks
        for h in handle_list:
            h.remove()

    # Concatenate
    result = {}
    for layer_name, arrays in collected.items():
        if arrays:
            result[layer_name] = np.concatenate(arrays, axis=0)
            if max_samples:
                result[layer_name] = result[layer_name][:max_samples]
        else:
            logger.warning(f"No representations captured for layer '{layer_name}'")

    return result


def _iter_batches(dataset, batch_size: int):
    """Yield batches from various dataset formats."""
    if hasattr(dataset, "__getitem__") and hasattr(dataset, "__len__"):
        # List-like
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size]
    elif hasattr(dataset, "__iter__"):
        # Iterable (DataLoader, generator)
        yield from dataset
    else:
        yield dataset


def _aggregate(X: np.ndarray, method: str = "mean") -> np.ndarray:
    """
    Aggregate per-atom representations to per-structure.

    Parameters
    ----------
    X : np.ndarray
        Shape (batch, n_atoms, hidden_dim) or (n_atoms, hidden_dim).
    method : str
        "mean", "sum", or "max".
    """
    axis = -2  # aggregate over atoms dimension
    if method == "mean":
        return X.mean(axis=axis)
    elif method == "sum":
        return X.sum(axis=axis)
    elif method == "max":
        return X.max(axis=axis)
    else:
        raise ValueError(f"Unknown aggregation: {method}")
