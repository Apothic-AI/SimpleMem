"""Compatibility exports for legacy SimpleMem runtime."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType


def _load_legacy_main_module() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    main_path = root / "main.py"
    spec = importlib.util.spec_from_file_location("simplemem._legacy_main", main_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load SimpleMem legacy module from {main_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_legacy_main = _load_legacy_main_module()
SimpleMemSystem = _legacy_main.SimpleMemSystem
create_system = _legacy_main.create_system

__all__ = ["SimpleMemSystem", "create_system"]
