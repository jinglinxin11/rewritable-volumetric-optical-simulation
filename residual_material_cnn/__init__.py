"""Three-kernel CNN with ideal three-bank differential material encoding."""

from .config import MaterialConfig, NetworkConfig
from .network import CompiledCNN, ReferenceCNN, compile_network

__all__ = ['MaterialConfig', 'NetworkConfig', 'CompiledCNN', 'ReferenceCNN', 'compile_network']
