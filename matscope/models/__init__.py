from matscope.models.registry import load_model, available_backends, register_backend
from matscope.models.torch_wrapper import wrap_torch_model

# Import backends to trigger @register_backend decorators
try:
    from matscope.models import pyg_backends as _pyg  # noqa: F401
except ImportError:
    pass  # PyG/TorchANI optional
