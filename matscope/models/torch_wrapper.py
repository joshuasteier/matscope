"""
Wrapper for arbitrary PyTorch nn.Module models.

Allows MatScope to work with any model, not just supported
scientific FMs. Useful for comparing scientific FMs against
standard architectures or for custom models.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple


def wrap_torch_model(
    model: Any,
    layer_names: Optional[List[str]] = None,
) -> Tuple[Any, Dict[str, Callable]]:
    """
    Wrap a PyTorch model for MatScope.

    Automatically discovers hookable layers if layer_names is None.
    """
    import torch.nn as nn

    model.eval()
    hooks = {}

    if layer_names is None:
        # Auto-discover: hook all modules that transform features
        hookable_types = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.MultiheadAttention,
            nn.LayerNorm,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
        )
        for name, module in model.named_modules():
            if isinstance(module, hookable_types):
                hooks[name] = _make_hook(name)
    else:
        for name in layer_names:
            hooks[name] = _make_hook(name)

    return model, hooks


def _make_hook(name: str):
    """Create a forward hook closure."""
    def hook_fn(module, input, output):
        if hasattr(output, "detach"):
            return output.detach().cpu().numpy()
        elif isinstance(output, tuple):
            # Some layers return (output, attention_weights)
            return output[0].detach().cpu().numpy()
        return output
    return hook_fn
