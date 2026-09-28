"""Load an optional project-local CUDA runtime on Windows.

The NVIDIA display driver is not enough for Faster-Whisper GPU decoding: CTranslate2
also needs CUDA 12 and cuDNN 9 DLLs. Keeping those DLLs under ``models/runtime``
avoids copying third-party files into Windows system folders and keeps the setup
scoped to CabInspector.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CUDA_RUNTIME_DIR = ROOT / "models" / "runtime" / "cuda12"
_dll_directory_handles: list[object] = []


def configure_project_cuda_runtime() -> Path | None:
    """Make an optional local CUDA runtime discoverable by this Python process."""

    configured = os.environ.get("CABINSPECTOR_CUDA_RUNTIME_DIR", "").strip()
    runtime_dir = Path(configured) if configured else DEFAULT_CUDA_RUNTIME_DIR
    if sys.platform != "win32" or not runtime_dir.is_dir():
        return None

    resolved = runtime_dir.resolve()
    runtime_text = str(resolved)
    current_path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if runtime_text not in current_path_parts:
        os.environ["PATH"] = runtime_text + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        # Keep the handle alive for the lifetime of the Python process.
        _dll_directory_handles.append(os.add_dll_directory(runtime_text))
    return resolved
