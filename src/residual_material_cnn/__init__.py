"""Three-kernel CNN with ideal three-bank differential material encoding."""

from .config import MaterialConfig, NetworkConfig, TrainingConfig
from .network import CompiledCNN, ReferenceCNN, compile_network

__version__ = "0.2.0"

__all__ = [
    "MaterialConfig",
    "NetworkConfig",
    "TrainingConfig",
    "CompiledCNN",
    "ReferenceCNN",
    "compile_network",
]
