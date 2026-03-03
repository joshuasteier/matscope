"""
PyTorch Geometric model backends.

Backends for GNN-based atomistic models available through PyG or TorchANI.
Each backend knows how to load the model and register hooks at each
message-passing layer.

Supported:
    - SchNet (Schütt et al., 2018)
    - DimeNet++ (Gasteiger et al., 2020)
    - PaiNN (Schütt et al., 2021)
    - ANI-2x (Devereux et al., 2020) via TorchANI
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

from matscope.models.registry import ModelBackend, register_backend

logger = logging.getLogger(__name__)


@register_backend("schnet")
class SchNetBackend(ModelBackend):
    """
    Backend for SchNet models.

    SchNet uses continuous-filter convolutions with radial basis functions.
    We hook into each SchNet interaction block.
    """

    def load(self, model_name_or_path: str, device: str = "cpu", **kwargs):
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required. Install with: pip install torch")

        if model_name_or_path in ("schnet", "schnet-qm9"):
            try:
                from torch_geometric.nn.models import SchNet

                # Load pretrained on QM9 if available, else initialize
                if "checkpoint" in kwargs:
                    model = SchNet()
                    model.load_state_dict(torch.load(kwargs["checkpoint"], map_location=device))
                else:
                    # Default architecture matching QM9 experiments
                    model = SchNet(
                        hidden_channels=kwargs.get("hidden_channels", 128),
                        num_filters=kwargs.get("num_filters", 128),
                        num_interactions=kwargs.get("num_interactions", 6),
                        num_gaussians=kwargs.get("num_gaussians", 50),
                        cutoff=kwargs.get("cutoff", 10.0),
                    )
                model = model.to(device)
                model.eval()
                return model
            except ImportError:
                raise ImportError(
                    "PyG SchNet not found. Install with:\n"
                    "pip install torch-geometric"
                )
        else:
            import torch

            model = torch.load(model_name_or_path, map_location=device)
            model.eval()
            return model

    def get_hooks(self, model) -> Dict[str, Callable]:
        hooks = {}
        # SchNet has model.interactions as a ModuleList
        if hasattr(model, "interactions"):
            for i, block in enumerate(model.interactions):
                hooks[f"interaction_{i}"] = self._make_hook(f"interaction_{i}")
        # Also hook the output MLP if present
        if hasattr(model, "lin1"):
            hooks["readout_lin1"] = self._make_hook("readout_lin1")
        if hasattr(model, "lin2"):
            hooks["readout_lin2"] = self._make_hook("readout_lin2")
        return hooks

    @staticmethod
    def _make_hook(name):
        def hook_fn(module, input, output):
            if hasattr(output, "detach"):
                return output.detach().cpu().numpy()
            return output

        return hook_fn

    def get_layer_names(self, model) -> List[str]:
        names = []
        if hasattr(model, "interactions"):
            names.extend(
                [f"interaction_{i}" for i in range(len(model.interactions))]
            )
        for attr in ("lin1", "lin2"):
            if hasattr(model, attr):
                names.append(f"readout_{attr}")
        return names


@register_backend("dimenet++")
@register_backend("dimenet")
@register_backend("dimenetpp")
class DimeNetPPBackend(ModelBackend):
    """
    Backend for DimeNet++ models.

    DimeNet++ uses directional message passing with interaction blocks
    that process both radial and angular information.
    """

    def load(self, model_name_or_path: str, device: str = "cpu", **kwargs):
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required.")

        if model_name_or_path in ("dimenet++", "dimenet", "dimenetpp", "dimenet++-qm9"):
            try:
                from torch_geometric.nn.models import DimeNetPlusPlus

                model = DimeNetPlusPlus(
                    hidden_channels=kwargs.get("hidden_channels", 128),
                    out_channels=kwargs.get("out_channels", 1),
                    num_blocks=kwargs.get("num_blocks", 4),
                    int_emb_size=kwargs.get("int_emb_size", 64),
                    basis_emb_size=kwargs.get("basis_emb_size", 8),
                    out_emb_channels=kwargs.get("out_emb_channels", 256),
                    num_spherical=kwargs.get("num_spherical", 7),
                    num_radial=kwargs.get("num_radial", 6),
                    cutoff=kwargs.get("cutoff", 5.0),
                )
                model = model.to(device)
                model.eval()
                return model
            except ImportError:
                raise ImportError(
                    "PyG DimeNet++ not found. Install with:\n"
                    "pip install torch-geometric"
                )
        else:
            import torch

            model = torch.load(model_name_or_path, map_location=device)
            model.eval()
            return model

    def get_hooks(self, model) -> Dict[str, Callable]:
        hooks = {}
        # DimeNet++ has interaction_blocks and output_blocks
        if hasattr(model, "interaction_blocks"):
            for i, block in enumerate(model.interaction_blocks):
                hooks[f"interaction_{i}"] = self._make_hook(f"interaction_{i}")
        if hasattr(model, "output_blocks"):
            for i, block in enumerate(model.output_blocks):
                hooks[f"output_{i}"] = self._make_hook(f"output_{i}")
        return hooks

    @staticmethod
    def _make_hook(name):
        def hook_fn(module, input, output):
            if hasattr(output, "detach"):
                return output.detach().cpu().numpy()
            elif isinstance(output, tuple) and hasattr(output[0], "detach"):
                return output[0].detach().cpu().numpy()
            return output

        return hook_fn

    def get_layer_names(self, model) -> List[str]:
        names = []
        if hasattr(model, "interaction_blocks"):
            names.extend(
                [f"interaction_{i}" for i in range(len(model.interaction_blocks))]
            )
        if hasattr(model, "output_blocks"):
            names.extend(
                [f"output_{i}" for i in range(len(model.output_blocks))]
            )
        return names


@register_backend("painn")
class PaiNNBackend(ModelBackend):
    """
    Backend for PaiNN (Polarizable Atom Interaction Neural Network).

    PaiNN is an equivariant model that maintains separate scalar (s)
    and vector (V) channels. We hook into the message and update blocks.
    """

    def load(self, model_name_or_path: str, device: str = "cpu", **kwargs):
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required.")

        if model_name_or_path in ("painn", "painn-qm9"):
            try:
                from torch_geometric.nn.models import PaiNN

                model = PaiNN(
                    hidden_channels=kwargs.get("hidden_channels", 128),
                    num_interactions=kwargs.get("num_interactions", 6),
                    num_rbf=kwargs.get("num_rbf", 20),
                    cutoff=kwargs.get("cutoff", 5.0),
                )
                model = model.to(device)
                model.eval()
                return model
            except ImportError:
                raise ImportError(
                    "PyG PaiNN not found. Requires torch-geometric >= 2.4.\n"
                    "pip install torch-geometric"
                )
        else:
            import torch

            model = torch.load(model_name_or_path, map_location=device)
            model.eval()
            return model

    def get_hooks(self, model) -> Dict[str, Callable]:
        hooks = {}
        # PaiNN alternates message and update layers
        if hasattr(model, "message_layers"):
            for i, layer in enumerate(model.message_layers):
                hooks[f"message_{i}"] = self._make_hook(f"message_{i}")
        if hasattr(model, "update_layers"):
            for i, layer in enumerate(model.update_layers):
                hooks[f"update_{i}"] = self._make_hook(f"update_{i}")
        return hooks

    @staticmethod
    def _make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                # PaiNN returns (scalar, vector); take scalar for probing
                if hasattr(output[0], "detach"):
                    return output[0].detach().cpu().numpy()
            if hasattr(output, "detach"):
                return output.detach().cpu().numpy()
            return output

        return hook_fn

    def get_layer_names(self, model) -> List[str]:
        names = []
        if hasattr(model, "message_layers"):
            names.extend(
                [f"message_{i}" for i in range(len(model.message_layers))]
            )
        if hasattr(model, "update_layers"):
            names.extend(
                [f"update_{i}" for i in range(len(model.update_layers))]
            )
        return names


@register_backend("ani")
@register_backend("ani-2x")
@register_backend("ani2x")
class ANI2xBackend(ModelBackend):
    """
    Backend for ANI-2x models (via TorchANI).

    ANI uses handcrafted symmetry functions (AEVs) as descriptors,
    making it a key baseline: handcrafted invariant features vs
    learned equivariant features.
    """

    def load(self, model_name_or_path: str, device: str = "cpu", **kwargs):
        try:
            import torchani

            model = torchani.models.ANI2x(periodic_table_index=True).to(device)
            model.eval()
            return model
        except ImportError:
            raise ImportError(
                "TorchANI not installed. Install with: pip install torchani\n"
                "See: https://github.com/aiqm/torchani"
            )

    def get_hooks(self, model) -> Dict[str, Callable]:
        hooks = {}
        # ANI has: aev_computer (symmetry functions) + nn (ensemble of MLPs)
        if hasattr(model, "aev_computer"):
            hooks["aev"] = self._make_hook("aev")

        # Each species network in the ensemble
        if hasattr(model, "neural_networks"):
            for i, nn in enumerate(model.neural_networks):
                hooks[f"ensemble_{i}"] = self._make_hook(f"ensemble_{i}")
                # Hook individual layers within each network
                if hasattr(nn, "0"):  # Sequential modules
                    for j in range(len(nn)):
                        if hasattr(nn[j], "weight"):  # Linear layers only
                            hooks[f"ensemble_{i}_layer_{j}"] = self._make_hook(
                                f"ensemble_{i}_layer_{j}"
                            )
        return hooks

    @staticmethod
    def _make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                # ANI aev_computer returns (species, aevs)
                for item in output:
                    if hasattr(item, "detach") and item.dim() >= 2:
                        return item.detach().cpu().numpy()
            if hasattr(output, "detach"):
                return output.detach().cpu().numpy()
            return output

        return hook_fn

    def get_layer_names(self, model) -> List[str]:
        names = ["aev"]
        if hasattr(model, "neural_networks"):
            for i in range(len(model.neural_networks)):
                names.append(f"ensemble_{i}")
        return names
