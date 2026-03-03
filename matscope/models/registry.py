"""
Model registry for scientific foundation models.

Handles loading models and auto-discovering layer hooks for
representation extraction. Each supported model has a backend
that knows how to:
1. Load the model weights
2. Register forward hooks at each message-passing / transformer layer
3. Aggregate per-atom representations to per-structure representations

Adding a new model backend:
    1. Create a class implementing ModelBackend
    2. Register it with @register_backend("model_name")
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Global registry
_BACKENDS: Dict[str, type] = {}


def register_backend(name: str):
    """Decorator to register a model backend."""
    def decorator(cls):
        _BACKENDS[name] = cls
        return cls
    return decorator


class ModelBackend:
    """Base class for model backends."""

    def load(self, model_name_or_path: str, device: str = "cpu", **kwargs) -> Any:
        """Load the model and return it."""
        raise NotImplementedError

    def get_hooks(self, model: Any) -> Dict[str, Callable]:
        """Return a dict of layer_name -> hook_fn for representation extraction."""
        raise NotImplementedError

    def get_layer_names(self, model: Any) -> List[str]:
        """Return ordered list of layer names."""
        raise NotImplementedError


@register_backend("mace-mp-0")
@register_backend("mace-off")
@register_backend("mace")
class MACEBackend(ModelBackend):
    """
    Backend for MACE foundation models.

    MACE uses equivariant message-passing with multi-body
    interactions. We hook into the output of each interaction
    block and the readout layers.
    """

    def load(self, model_name_or_path: str, device: str = "cpu", **kwargs):
        try:
            from mace.calculators import mace_mp
        except ImportError:
            raise ImportError(
                "MACE not installed. Install with: pip install mace-torch\n"
                "See: https://github.com/ACEsuit/mace"
            )

        if model_name_or_path in ("mace-mp-0", "mace"):
            calc = mace_mp(model="medium", device=device, default_dtype="float64")
        elif model_name_or_path == "mace-off":
            from mace.calculators import mace_off
            calc = mace_off(model="medium", device=device)
        else:
            # Assume local checkpoint
            from mace.calculators import MACECalculator
            calc = MACECalculator(model_paths=model_name_or_path, device=device)

        return calc.models[0] if hasattr(calc, "models") else calc

    def get_hooks(self, model) -> Dict[str, Callable]:
        hooks = {}
        # Hook into each interaction block
        if hasattr(model, "interactions"):
            for i, block in enumerate(model.interactions):
                hooks[f"interaction_{i}"] = self._make_hook(f"interaction_{i}")
        # Hook into readout blocks
        if hasattr(model, "readouts"):
            for i, block in enumerate(model.readouts):
                hooks[f"readout_{i}"] = self._make_hook(f"readout_{i}")
        return hooks

    @staticmethod
    def _make_hook(name: str):
        """Create a forward hook that stores the output."""
        def hook_fn(module, input, output):
            # MACE outputs can be complex; we extract the node features
            if hasattr(output, "node_feats"):
                return output.node_feats.detach().cpu().numpy()
            elif isinstance(output, dict) and "node_feats" in output:
                return output["node_feats"].detach().cpu().numpy()
            elif hasattr(output, "detach"):
                return output.detach().cpu().numpy()
            return output
        return hook_fn

    def get_layer_names(self, model) -> List[str]:
        names = []
        if hasattr(model, "interactions"):
            names.extend([f"interaction_{i}" for i in range(len(model.interactions))])
        if hasattr(model, "readouts"):
            names.extend([f"readout_{i}" for i in range(len(model.readouts))])
        return names


@register_backend("chgnet")
class CHGNetBackend(ModelBackend):
    """Backend for CHGNet models."""

    def load(self, model_name_or_path: str, device: str = "cpu", **kwargs):
        try:
            from chgnet.model import CHGNet
        except ImportError:
            raise ImportError("CHGNet not installed. Install with: pip install chgnet")
        return CHGNet.load(model_name=model_name_or_path)

    def get_hooks(self, model) -> Dict[str, Callable]:
        hooks = {}
        if hasattr(model, "atom_conv_layers"):
            for i, layer in enumerate(model.atom_conv_layers):
                hooks[f"atom_conv_{i}"] = self._make_hook(f"atom_conv_{i}")
        return hooks

    @staticmethod
    def _make_hook(name: str):
        def hook_fn(module, input, output):
            if hasattr(output, "detach"):
                return output.detach().cpu().numpy()
            return output
        return hook_fn

    def get_layer_names(self, model) -> List[str]:
        if hasattr(model, "atom_conv_layers"):
            return [f"atom_conv_{i}" for i in range(len(model.atom_conv_layers))]
        return []


@register_backend("generic")
class GenericTorchBackend(ModelBackend):
    """
    Fallback backend for any PyTorch model.

    Auto-discovers all nn.Module children and registers hooks.
    """

    def load(self, model_name_or_path: str, device: str = "cpu", **kwargs):
        import torch
        model = torch.load(model_name_or_path, map_location=device)
        model.eval()
        return model

    def get_hooks(self, model) -> Dict[str, Callable]:
        import torch.nn as nn
        hooks = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.TransformerEncoderLayer)):
                hooks[name] = self._make_hook(name)
        return hooks

    @staticmethod
    def _make_hook(name):
        def hook_fn(module, input, output):
            if hasattr(output, "detach"):
                return output.detach().cpu().numpy()
            return output
        return hook_fn

    def get_layer_names(self, model) -> List[str]:
        import torch.nn as nn
        return [
            name for name, module in model.named_modules()
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.TransformerEncoderLayer))
        ]


def load_model(
    model_name_or_path: str,
    device: str = "cpu",
    **kwargs,
) -> Tuple[Any, Dict[str, Callable]]:
    """
    Load a model and return (model, hooks).

    Parameters
    ----------
    model_name_or_path : str
        Model identifier or local path.
    device : str
        Target device.

    Returns
    -------
    tuple
        (model, hooks_dict)
    """
    # Find the right backend
    backend_cls = None
    for key, cls in _BACKENDS.items():
        if key in model_name_or_path.lower():
            backend_cls = cls
            break

    if backend_cls is None:
        logger.info(f"No specific backend for '{model_name_or_path}', using generic.")
        backend_cls = GenericTorchBackend

    backend = backend_cls()
    model = backend.load(model_name_or_path, device=device, **kwargs)
    hooks = backend.get_hooks(model)

    logger.info(f"Loaded model with {len(hooks)} hookable layers: {list(hooks.keys())}")
    return model, hooks


def available_backends() -> List[str]:
    """List all registered model backends."""
    return list(_BACKENDS.keys())
