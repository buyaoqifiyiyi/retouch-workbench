"""Shared idempotent NVIDIA CUDA 12/13 & cuDNN DLL setup and ONNX Runtime session factory."""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any

_CUDA_DLL_INFO: dict[str, Any] | None = None
_DLL_HANDLES: list[Any] = []


def ensure_cuda_dlls() -> dict[str, Any]:
    """Register bundled NVIDIA CUDA 12/13 & cuDNN DLL directories for ONNX Runtime on Windows."""
    global _CUDA_DLL_INFO
    if _CUDA_DLL_INFO is not None:
        return dict(_CUDA_DLL_INFO)

    info: dict[str, Any] = {"preloaded": False, "dll_dirs": []}
    try:
        import onnxruntime as ort  # type: ignore

        sp_root = Path(ort.__file__).resolve().parent.parent
        nv_bins = sorted(
            set(
                glob.glob(str(sp_root / "nvidia" / "*" / "bin"))
                + glob.glob(str(sp_root / "nvidia" / "*" / "bin" / "x86_64"))
            )
        )
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        for bin_dir in nv_bins:
            info["dll_dirs"].append(bin_dir)
            if hasattr(os, "add_dll_directory"):
                _DLL_HANDLES.append(os.add_dll_directory(bin_dir))
            if bin_dir not in path_parts:
                path_parts.insert(0, bin_dir)
        os.environ["PATH"] = os.pathsep.join(path_parts)

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
            info["preloaded"] = True
    except Exception as exc:
        info["error"] = str(exc)

    _CUDA_DLL_INFO = info
    return dict(info)


def create_onnx_session(
    model_path: str | Path | bytes,
    *,
    sess_options: Any = None,
    prefer_cuda: bool = True,
) -> Any:
    """Create an ONNX Runtime InferenceSession preferring CUDA after DLL setup, with CPU fallback."""
    ensure_cuda_dlls()
    import onnxruntime as ort  # type: ignore

    target: str | bytes = model_path if isinstance(model_path, (bytes, bytearray)) else str(model_path)
    kwargs: dict[str, Any] = {}
    if sess_options is not None:
        kwargs["sess_options"] = sess_options

    avail = ort.get_available_providers()
    if prefer_cuda and "CUDAExecutionProvider" in avail:
        try:
            return ort.InferenceSession(
                target,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                **kwargs,
            )
        except Exception:
            pass
    return ort.InferenceSession(target, providers=["CPUExecutionProvider"], **kwargs)
