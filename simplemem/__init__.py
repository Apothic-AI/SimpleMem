"""SimpleMem package wrapper for local editable development."""

from simplemem.__version__ import __version__
from simplemem.system import SimpleMemSystem, create_system
from simplemem.models.memory_entry import MemoryEntry, Dialogue
from simplemem.config import SimpleMemConfig, get_config, set_config

__all__ = [
    "__version__",
    "SimpleMemSystem",
    "create_system",
    "MemoryEntry",
    "Dialogue",
    "SimpleMemConfig",
    "get_config",
    "set_config",
]
