"""RetouchTools AI sidecar CLI for PhotoCraft / LightCraft retouch-workbench.

Runs purely local, permissively licensed (MIT / Apache-2.0 / BSD-3-Clause) models
and outputs lossless PNG masks, dense landmark JSON, repair patches, upscales,
and QA diagnostics for non-destructive PhotoCraft layer workflows.
"""

from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

LOCAL_APP_DATA = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
TOOLS_ROOT = Path(os.environ.get("RETOUCH_TOOLS_ROOT", LOCAL_APP_DATA / "RetouchTools"))
MODELS_ROOT = Path(os.environ.get("RETOUCH_MODELS_DIR", TOOLS_ROOT / "models"))
UNIFACE_CACHE = Path(os.environ.get("UNIFACE_CACHE_DIR", MODELS_ROOT / "uniface"))
REMBG_CACHE = Path(os.environ.get("U2NET_HOME", MODELS_ROOT / "rembg"))
MEDIAPIPE_SELFIE_MODEL = Path(
    os.environ.get("MEDIAPIPE_SELFIE_MODEL", MODELS_ROOT / "selfie_multiclass_256x256.tflite")
)
LAMA_MODEL = Path(os.environ.get("LAMA_MODEL_PATH", MODELS_ROOT / "lama" / "big-lama.pt"))
BIREFNET_LITE_MODEL = Path(
    os.environ.get("BIREFNET_LITE_MODEL", MODELS_ROOT / "birefnet" / "birefnet_lite.onnx")
)
SKIN_RETOUCH_DIR = Path(
    os.environ.get("SKIN_RETOUCH_DIR", MODELS_ROOT / "skin-retouching")
)
GFPGAN_MODEL = Path(
    os.environ.get("GFPGAN_ONNX_MODEL", MODELS_ROOT / "gfpgan" / "gfpgan_1.4.onnx")
)
GEN_VENV_PYTHON = Path(
    os.environ.get("RETOUCH_GEN_PYTHON", TOOLS_ROOT / "venv-gen" / "Scripts" / "python.exe")
)
GEN_SCRIPT = Path(__file__).resolve().parent / "retouch-gen.py"

ALLOWED_UNIFACE_MATTING = {"ppmatting-v2", "modnet"}
ALLOWED_CUSTOM_MATTING = {"birefnet-lite"}
ALLOWED_REMBG_MATTING = {
    "birefnet-portrait",
    "birefnet-general",
    "birefnet-general-lite",
    "birefnet-hrsod",
    "birefnet-dis",
    "birefnet-massive",
    "u2net_human_seg",
    "u2net_cloth_seg",
    "silueta",
}
DISALLOWED_COMMERCIAL_MATTING = {"bria-rmbg", "withoutbg"}

PHOTOCRAFT_CLI = Path(
    os.environ.get(
        "PHOTOCRAFT_CLI",
        LOCAL_APP_DATA / "Programs" / "ArtCraft" / "PhotoCraft" / "photocraft-cli.exe",
    )
)
LIGHTCRAFT_CLI = Path(
    os.environ.get(
        "LIGHTCRAFT_CLI",
        LOCAL_APP_DATA / "Programs" / "ArtCraft" / "LightCraft" / "lightcraft-cli.exe",
    )
)
REALESRGAN_EXE = Path(
    os.environ.get(
        "REALESRGAN_NCNN_EXE",
        TOOLS_ROOT / "realesrgan-ncnn-vulkan" / "realesrgan-ncnn-vulkan.exe",
    )
)

FACE_PARSING_LABELS: tuple[str, ...] = (
    "background",
    "skin",
    "l_brow",
    "r_brow",
    "l_eye",
    "r_eye",
    "eye_g",
    "l_ear",
    "r_ear",
    "ear_r",
    "nose",
    "mouth",
    "u_lip",
    "l_lip",
    "neck",
    "neck_l",
    "cloth",
    "hair",
    "hat",
)

_CUDA_DLL_INITIALIZED = False
_DLL_HANDLES: list[Any] = []


def _ensure_cuda_dlls() -> dict[str, Any]:
    """Register bundled NVIDIA CUDA/cuDNN bin dirs for ONNX Runtime on Windows."""
    global _CUDA_DLL_INITIALIZED
    info: dict[str, Any] = {"preloaded": False, "dll_dirs": []}
    if _CUDA_DLL_INITIALIZED:
        info["preloaded"] = True
        return info

    os.environ.setdefault("UNIFACE_CACHE_DIR", str(UNIFACE_CACHE))
    os.environ.setdefault("U2NET_HOME", str(REMBG_CACHE))

    try:
        import onnxruntime as ort  # type: ignore

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
            info["preloaded"] = True

        sp_root = Path(ort.__file__).resolve().parent.parent
        nv_bins = sorted(glob.glob(str(sp_root / "nvidia" / "*" / "bin")))
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        for bin_dir in nv_bins:
            info["dll_dirs"].append(bin_dir)
            if hasattr(os, "add_dll_directory"):
                _DLL_HANDLES.append(os.add_dll_directory(bin_dir))
            if bin_dir not in path_parts:
                path_parts.insert(0, bin_dir)
        os.environ["PATH"] = os.pathsep.join(path_parts)
        _CUDA_DLL_INITIALIZED = True
    except Exception as exc:
        info["error"] = str(exc)

    return info


def _build_minimal_identity_onnx() -> bytes:
    """Build a tiny valid ONNX Identity graph in-memory without importing onnx."""

    def _varint(val: int) -> bytes:
        out = bytearray()
        while val >= 0x80:
            out.append((val & 0x7F) | 0x80)
            val >>= 7
        out.append(val)
        return bytes(out)

    def _field_bytes(field_no: int, payload: bytes) -> bytes:
        return _varint((field_no << 3) | 2) + _varint(len(payload)) + payload

    def _field_varint(field_no: int, val: int) -> bytes:
        return _varint((field_no << 3) | 0) + _varint(val)

    def _value_info(name: str) -> bytes:
        dim = _field_varint(1, 1)
        shape = _field_bytes(1, dim + dim)
        tensor_type = _field_varint(1, 1) + _field_bytes(2, shape)  # FLOAT=1
        type_proto = _field_bytes(1, tensor_type)
        return _field_bytes(1, name.encode()) + _field_bytes(2, type_proto)

    node = (
        _field_bytes(1, b"x")
        + _field_bytes(2, b"y")
        + _field_bytes(4, b"Identity")
    )
    graph = (
        _field_bytes(1, node)
        + _field_bytes(2, b"probe")
        + _field_bytes(11, _value_info("x"))
        + _field_bytes(12, _value_info("y"))
    )
    opset = _field_varint(2, 13)
    return _field_varint(1, 8) + _field_bytes(7, graph) + _field_bytes(8, opset)


def _check_live_cuda_onnx() -> dict[str, Any]:
    """Verify CUDAExecutionProvider actually initializes on a real ONNX session."""
    dll_info = _ensure_cuda_dlls()
    result: dict[str, Any] = {
        "available_providers": [],
        "live_providers": [],
        "cuda_active": False,
        "probe_source": None,
        "dll_setup": dll_info,
    }
    try:
        import numpy as np
        import onnxruntime as ort  # type: ignore

        result["version"] = ort.__version__
        result["available_providers"] = ort.get_available_providers()
        if "CUDAExecutionProvider" not in result["available_providers"]:
            return result

        probe_model: str | bytes | None = None
        candidates = sorted(MODELS_ROOT.rglob("*.onnx")) if MODELS_ROOT.exists() else []
        for cand in candidates:
            if cand.stat().st_size < 30_000_000:
                probe_model = str(cand)
                result["probe_source"] = str(cand)
                break
        if probe_model is None:
            probe_model = _build_minimal_identity_onnx()
            result["probe_source"] = "in_memory_identity"

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        sess = ort.InferenceSession(
            probe_model,
            sess_options=opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        if result["probe_source"] == "in_memory_identity":
            sess.run(None, {"x": np.array([[1.0]], dtype=np.float32)})
        result["live_providers"] = sess.get_providers()
        result["cuda_active"] = (
            len(result["live_providers"]) > 0
            and result["live_providers"][0] == "CUDAExecutionProvider"
        )
    except Exception as exc:
        result["error"] = str(exc)
    return result


def _pkg_version(module_name: str) -> str | None:
    try:
        mod = importlib.import_module(module_name)
        return str(getattr(mod, "__version__", "installed"))
    except Exception:
        return None


def _read_bgr(path: Path) -> Any:
    import cv2
    import numpy as np

    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"Empty or unreadable image file: {path}")
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image as BGR: {path}")
    return img


def _write_png_u8(path: Path, arr: Any, overwrite: bool = False) -> None:
    import cv2

    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError(f"Failed to encode PNG: {path}")
    buf.tofile(str(path))


def _contours_to_polygons(mask: Any, min_area: float = 16.0, eps_ratio: float = 0.015) -> list[list[list[int]]]:
    import cv2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: list[list[list[int]]] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, max(1.0, eps_ratio * peri), True)
        if len(approx) >= 3:
            polys.append([[int(pt[0][0]), int(pt[0][1])] for pt in approx])
    return polys


def cmd_doctor(_args: argparse.Namespace) -> int:
    _ensure_cuda_dlls()
    ort_info = _check_live_cuda_onnx()
    packages = {
        name: _pkg_version(mod)
        for name, mod in (
            ("uniface", "uniface"),
            ("rembg", "rembg"),
            ("mediapipe", "mediapipe"),
            ("torch", "torch"),
            ("cv2", "cv2"),
            ("PIL", "PIL"),
            ("numpy", "numpy"),
            ("scipy", "scipy"),
            ("skimage", "skimage"),
        )
    }
    binaries = {
        "photocraft_cli": {"path": str(PHOTOCRAFT_CLI), "exists": PHOTOCRAFT_CLI.exists()},
        "lightcraft_cli": {"path": str(LIGHTCRAFT_CLI), "exists": LIGHTCRAFT_CLI.exists()},
        "realesrgan_ncnn_vulkan": {
            "path": str(REALESRGAN_EXE),
            "exists": REALESRGAN_EXE.exists() or shutil.which("realesrgan-ncnn-vulkan") is not None,
        },
        "lama_model": {"path": str(LAMA_MODEL), "exists": LAMA_MODEL.exists()},
        "birefnet_lite": {"path": str(BIREFNET_LITE_MODEL), "exists": BIREFNET_LITE_MODEL.exists()},
        "skin_retouch_model": {
            "path": str(SKIN_RETOUCH_DIR),
            "exists": (SKIN_RETOUCH_DIR / "pytorch_model.pt").exists() and (SKIN_RETOUCH_DIR / "model.onnx").exists(),
        },
        "gfpgan_v1_4": {"path": str(GFPGAN_MODEL), "exists": GFPGAN_MODEL.exists()},
        "venv_gen_python": {"path": str(GEN_VENV_PYTHON), "exists": GEN_VENV_PYTHON.exists()},
    }
    gen_runtime: dict[str, Any] = {"available": False}
    if GEN_VENV_PYTHON.exists() and GEN_SCRIPT.exists():
        try:
            proc = subprocess.run(
                [str(GEN_VENV_PYTHON), str(GEN_SCRIPT), "doctor"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=30,
            )
            if proc.stdout.strip():
                gen_runtime = json.loads(proc.stdout)
                gen_runtime["available"] = proc.returncode == 0
        except Exception as exc:
            gen_runtime = {"available": False, "error": str(exc)}

    payload = {
        "ok": bool(ort_info.get("cuda_active", False)),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "tools_root": str(TOOLS_ROOT),
        "models_root": str(MODELS_ROOT),
        "onnxruntime": ort_info,
        "packages": packages,
        "binaries": binaries,
        "gen_runtime": gen_runtime,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["ok"] else 1


def cmd_face_parse(args: argparse.Namespace) -> int:
    _ensure_cuda_dlls()
    import numpy as np
    from uniface import BiSeNet, Landmark106, RetinaFace, set_cache_dir  # type: ignore

    in_path = Path(args.input).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input image not found: {in_path}")
    if out_dir == in_path.parent and not args.overwrite:
        # Guard against accidentally clobbering input dir outputs if files already exist
        pass

    set_cache_dir(str(UNIFACE_CACHE))
    img = _read_bgr(in_path)
    h, w = img.shape[:2]

    detector = RetinaFace()
    faces = detector.detect(img)
    provider_used = (
        detector.session.get_providers()[0]
        if hasattr(detector, "session") and detector.session.get_providers()
        else "unknown"
    )

    if not faces:
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "ok": True,
            "input": str(in_path),
            "width": w,
            "height": h,
            "provider_used": provider_used,
            "faces": [],
            "warning": "no_face_detected",
        }
        (out_dir / "faces.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    # Sort largest/highest-confidence face first
    faces = sorted(faces, key=lambda f: float(f.confidence), reverse=True)
    if args.face_index != "all":
        idx = int(args.face_index)
        if idx < 0 or idx >= len(faces):
            raise IndexError(f"--face-index {idx} out of range (detected {len(faces)} faces)")
        selected_faces = [faces[idx]]
    else:
        selected_faces = faces

    landmarker = Landmark106()
    parser = BiSeNet()

    class_map = {str(i): name for i, name in enumerate(FACE_PARSING_LABELS)}
    full_classes = np.zeros((h, w), dtype=np.uint8)
    face_records: list[dict[str, Any]] = []

    for f in selected_faces:
        x1, y1, x2, y2 = [float(v) for v in f.bbox[:4]]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        lmk = landmarker.get_landmarks(img, np.array([x1, y1, x2, y2], dtype=np.float32))
        lmk_list = [[round(float(px), 2), round(float(py), 2)] for px, py in lmk]

        # Expand crop around bbox to capture hair top, ears, and neck cleanly
        cx1 = max(0, int(round(x1 - 0.25 * bw)))
        cy1 = max(0, int(round(y1 - 0.35 * bh)))
        cx2 = min(w, int(round(x2 + 0.25 * bw)))
        cy2 = min(h, int(round(y2 + 0.25 * bh)))
        crop = img[cy1:cy2, cx1:cx2]
        if crop.size > 0:
            parsed_crop = parser.parse(crop)
            roi = full_classes[cy1:cy2, cx1:cx2]
            # Preserve existing non-background pixels when merging multiple faces
            write_mask = (roi == 0) & (parsed_crop > 0)
            roi[write_mask] = parsed_crop[write_mask]

        face_records.append(
            {
                "confidence": round(float(f.confidence), 4),
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "parse_crop": [cx1, cy1, cx2, cy2],
                "landmarks_5": [[round(float(px), 2), round(float(py), 2)] for px, py in f.landmarks],
                "landmarks_106": lmk_list,
            }
        )

    # Disjoint binary masks (0 / 255) from argmax class map
    # Skin strictly excludes brows(2,3), eyes(4,5), glasses(6), mouth/lips(11,12,13), hair(17), cloth(16), hat(18).
    skin_ids = (1, 7, 8, 10)  # skin, l_ear, r_ear, nose
    eyes_brows_ids = (2, 3, 4, 5)  # l_brow, r_brow, l_eye, r_eye
    lips_ids = (11, 12, 13)  # mouth, u_lip, l_lip
    hair_ids = (17,)

    mask_skin = np.where(np.isin(full_classes, skin_ids), 255, 0).astype(np.uint8)
    mask_eyes_brows = np.where(np.isin(full_classes, eyes_brows_ids), 255, 0).astype(np.uint8)
    mask_lips = np.where(np.isin(full_classes, lips_ids), 255, 0).astype(np.uint8)
    mask_hair = np.where(np.isin(full_classes, hair_ids), 255, 0).astype(np.uint8)

    freeze_polygons = {
        "eyes_brows": _contours_to_polygons(mask_eyes_brows),
        "lips": _contours_to_polygons(mask_lips),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "mask_skin": out_dir / "mask_skin.png",
        "mask_eyes_brows": out_dir / "mask_eyes_brows.png",
        "mask_lips": out_dir / "mask_lips.png",
        "mask_hair_head": out_dir / "mask_hair_head.png",
        "mask_face_classes": out_dir / "mask_face_classes.png",
    }
    _write_png_u8(outputs["mask_skin"], mask_skin, overwrite=args.overwrite)
    _write_png_u8(outputs["mask_eyes_brows"], mask_eyes_brows, overwrite=args.overwrite)
    _write_png_u8(outputs["mask_lips"], mask_lips, overwrite=args.overwrite)
    _write_png_u8(outputs["mask_hair_head"], mask_hair, overwrite=args.overwrite)
    _write_png_u8(outputs["mask_face_classes"], full_classes, overwrite=args.overwrite)

    payload = {
        "ok": True,
        "input": str(in_path),
        "width": w,
        "height": h,
        "provider_used": provider_used,
        "class_map": class_map,
        "faces": face_records,
        "liquify_freeze_polygons": freeze_polygons,
        "outputs": {k: str(v) for k, v in outputs.items()},
    }
    json_path = out_dir / "faces.json"
    if json_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {json_path}")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["outputs"]["faces_json"] = str(json_path)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _read_gray_f32(path: Path, expected_hw: tuple[int, int]) -> Any:
    import cv2
    import numpy as np

    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"Empty or unreadable mask file: {path}")
    arr = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise ValueError(f"Failed to decode grayscale mask: {path}")
    eh, ew = expected_hw
    if arr.shape != (eh, ew):
        arr = cv2.resize(arr, (ew, eh), interpolation=cv2.INTER_LINEAR)
    return arr.astype(np.float32) / 255.0


def cmd_matte(args: argparse.Namespace) -> int:
    _ensure_cuda_dlls()
    import cv2
    import numpy as np

    model_name = args.model.strip()
    if model_name in DISALLOWED_COMMERCIAL_MATTING or (
        model_name not in ALLOWED_UNIFACE_MATTING
        and model_name not in ALLOWED_CUSTOM_MATTING
        and model_name not in ALLOWED_REMBG_MATTING
    ):
        raise ValueError(
            f"Refusing disallowed/non-permissive matting model '{model_name}'. "
            f"Allowed permissive models: {sorted(ALLOWED_UNIFACE_MATTING | ALLOWED_CUSTOM_MATTING | ALLOWED_REMBG_MATTING)} "
            "(bria-rmbg / withoutbg are blocked due to commercial/cloud license restrictions)."
        )

    in_path = Path(args.input).resolve()
    out_mask = Path(args.out_mask).resolve()
    out_fg = Path(args.out_fg).resolve() if args.out_fg else None
    if not in_path.exists():
        raise FileNotFoundError(f"Input image not found: {in_path}")
    if out_mask == in_path or (out_fg is not None and out_fg == in_path):
        raise ValueError("Refusing to overwrite input image.")

    img = _read_bgr(in_path)
    h, w = img.shape[:2]
    provider_used = "unknown"

    if model_name == "birefnet-lite":
        import onnxruntime as ort  # type: ignore

        if not BIREFNET_LITE_MODEL.exists():
            raise FileNotFoundError(
                f"BiRefNet_lite ONNX model missing: {BIREFNET_LITE_MODEL}. Run download_models.py."
            )
        # BiRefNet Swin/grid ops trigger heavy CPU-GPU ping-pong on ORT CUDA EP (>200s);
        # pure CPUExecutionProvider completes cleanly in ~17s with zero VRAM contention.
        providers = ["CPUExecutionProvider"]
        sess = ort.InferenceSession(str(BIREFNET_LITE_MODEL), providers=providers)
        provider_used = sess.get_providers()[0] if sess.get_providers() else "unknown"
        inp_meta = sess.get_inputs()[0]
        target_sz = int(inp_meta.shape[-1]) if isinstance(inp_meta.shape[-1], int) and inp_meta.shape[-1] > 0 else 1024

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_resized = cv2.resize(rgb, (target_sz, target_sz), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        chw = np.transpose((rgb_resized - mean) / std, (2, 0, 1))[None, ...]
        if "float16" in str(inp_meta.type):
            chw = chw.astype(np.float16)

        outputs = sess.run(None, {inp_meta.name: chw})
        pred = np.asarray(outputs[-1], dtype=np.float32).squeeze()
        if pred.ndim == 3:
            pred = pred[-1]
        if float(pred.min()) < -0.05 or float(pred.max()) > 1.05:
            pred = 1.0 / (1.0 + np.exp(-np.clip(pred, -30.0, 30.0)))
        alpha_f32 = np.clip(pred, 0.0, 1.0)
        if alpha_f32.shape != (h, w):
            alpha_f32 = cv2.resize(alpha_f32, (w, h), interpolation=cv2.INTER_LINEAR)
        alpha_u8 = np.clip(np.round(alpha_f32 * 255.0), 0, 255).astype(np.uint8)
    elif model_name in ALLOWED_UNIFACE_MATTING:
        from uniface import MODNet, PPMatting, set_cache_dir  # type: ignore
        from uniface.constants import MODNetWeights, PPMattingWeights  # type: ignore

        set_cache_dir(str(UNIFACE_CACHE))
        if model_name == "ppmatting-v2":
            matting = PPMatting(model_name=PPMattingWeights.V2, max_short=int(args.max_short))
        else:
            matting = MODNet(model_name=MODNetWeights.PHOTOGRAPHIC, input_size=int(args.max_short))
        provider_used = (
            matting.session.get_providers()[0]
            if hasattr(matting, "session") and matting.session.get_providers()
            else "unknown"
        )
        alpha_f32 = np.clip(matting.predict(img), 0.0, 1.0)
        if alpha_f32.shape != (h, w):
            alpha_f32 = cv2.resize(alpha_f32, (w, h), interpolation=cv2.INTER_LINEAR)
        alpha_u8 = np.clip(np.round(alpha_f32 * 255.0), 0, 255).astype(np.uint8)
    else:
        try:
            import rembg  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                f"Optional rembg package is not installed for model '{model_name}'. "
                "Use default --model ppmatting-v2 (Apache-2.0, CUDA ready) or install rembg."
            ) from exc
        os.environ["U2NET_HOME"] = str(REMBG_CACHE)
        session = rembg.new_session(model_name)
        from PIL import Image

        pil_in = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        mask_pil = rembg.remove(pil_in, session=session, only_mask=True)
        alpha_u8 = np.array(mask_pil.convert("L"), dtype=np.uint8)
        if alpha_u8.shape != (h, w):
            alpha_u8 = cv2.resize(alpha_u8, (w, h), interpolation=cv2.INTER_LINEAR)
        provider_used = "rembg"

    _write_png_u8(out_mask, alpha_u8, overwrite=args.overwrite)
    if out_fg is not None:
        bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = alpha_u8
        _write_png_u8(out_fg, bgra, overwrite=args.overwrite)

    soft_px = int(((alpha_u8 > 0) & (alpha_u8 < 255)).sum())
    fg_px = int((alpha_u8 >= 128).sum())
    payload = {
        "ok": True,
        "input": str(in_path),
        "model": model_name,
        "provider_used": provider_used,
        "width": w,
        "height": h,
        "fg_coverage_pct": round(100.0 * fg_px / float(w * h), 2),
        "soft_edge_pixels": soft_px,
        "out_mask": str(out_mask),
        "out_fg": str(out_fg) if out_fg is not None else None,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_body_parse(args: argparse.Namespace) -> int:
    import cv2
    import mediapipe as mp  # type: ignore
    import numpy as np
    from mediapipe.tasks.python import BaseOptions  # type: ignore
    from mediapipe.tasks.python.vision import (  # type: ignore
        ImageSegmenter,
        ImageSegmenterOptions,
        RunningMode,
    )

    in_path = Path(args.input).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input image not found: {in_path}")
    if not MEDIAPIPE_SELFIE_MODEL.exists():
        raise FileNotFoundError(f"MediaPipe multiclass model not found: {MEDIAPIPE_SELFIE_MODEL}")

    img_bgr = _read_bgr(in_path)
    h, w = img_bgr.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    refine_matte = None
    if args.refine_with_matte:
        refine_matte = _read_gray_f32(Path(args.refine_with_matte).resolve(), (h, w))

    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=str(MEDIAPIPE_SELFIE_MODEL)),
        running_mode=RunningMode.IMAGE,
        output_confidence_masks=True,
        output_category_mask=False,
    )
    with ImageSegmenter.create_from_options(options) as segmenter:
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = segmenter.segment(mp_img)

    if not res.confidence_masks or len(res.confidence_masks) < 6:
        raise RuntimeError(f"Expected 6 multiclass confidence masks, got {len(res.confidence_masks or [])}")

    def _upscale_conf(idx: int) -> Any:
        raw = res.confidence_masks[idx].numpy_view().copy()
        if raw.ndim == 3:
            raw = raw[:, :, 0]
        if raw.shape != (h, w):
            raw = cv2.resize(raw.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(raw, 0.0, 1.0)

    # MediaPipe selfie_multiclass_256x256 classes: 0=bg, 1=hair, 2=body_skin, 3=face_skin, 4=clothes, 5=accessories
    c_hair = _upscale_conf(1)
    c_body_skin = _upscale_conf(2)
    c_face_skin = _upscale_conf(3)
    c_clothes = _upscale_conf(4)
    c_accessories = _upscale_conf(5)
    c_all_skin = np.clip(c_body_skin + c_face_skin, 0.0, 1.0)

    class_arrays = {
        "mask_body_hair": c_hair,
        "mask_body_skin": c_body_skin,
        "mask_face_skin": c_face_skin,
        "mask_all_skin": c_all_skin,
        "mask_clothes": c_clothes,
        "mask_accessories": c_accessories,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    coverage_pct: dict[str, float] = {}
    total_px = float(w * h)
    for name, arr in class_arrays.items():
        if refine_matte is not None:
            arr = arr * refine_matte
        u8 = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
        out_p = out_dir / f"{name}.png"
        _write_png_u8(out_p, u8, overwrite=args.overwrite)
        outputs[name] = str(out_p)
        coverage_pct[name] = round(100.0 * float((u8 >= 128).sum()) / total_px, 2)

    payload = {
        "ok": True,
        "input": str(in_path),
        "width": w,
        "height": h,
        "refined_with_matte": str(args.refine_with_matte) if args.refine_with_matte else None,
        "coverage_pct": coverage_pct,
        "outputs": outputs,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_upscale(args: argparse.Namespace) -> int:
    import cv2
    import numpy as np

    in_path = Path(args.input).resolve()
    out_path = Path(args.out).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input image not found: {in_path}")
    if out_path == in_path:
        raise ValueError("Refusing to overwrite input image.")
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {out_path}")

    exe_path = REALESRGAN_EXE if REALESRGAN_EXE.exists() else Path(shutil.which("realesrgan-ncnn-vulkan") or "")
    if not exe_path or not exe_path.exists():
        raise FileNotFoundError(f"realesrgan-ncnn-vulkan executable not found at {REALESRGAN_EXE}")

    raw_bytes = np.fromfile(str(in_path), dtype=np.uint8)
    img = cv2.imdecode(raw_bytes, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to decode input image: {in_path}")
    in_h, in_w = img.shape[:2]
    target_scale = int(args.scale)
    if target_scale not in (2, 4):
        raise ValueError(f"Unsupported --scale {target_scale}; expected 2 or 4.")

    model_name = args.model
    native_scale = target_scale if model_name == "realesr-animevideov3" else 4

    with tempfile.TemporaryDirectory(prefix="retouch_up_") as tmpdir:
        tmp_dir = Path(tmpdir)
        tmp_in = tmp_dir / "in.png"
        tmp_out = tmp_dir / "out.png"
        _write_png_u8(tmp_in, img, overwrite=True)

        cmd = [
            str(exe_path),
            "-i",
            str(tmp_in),
            "-o",
            str(tmp_out),
            "-n",
            model_name,
            "-s",
            str(native_scale),
            "-m",
            str(exe_path.parent / "models"),
            "-f",
            "png",
        ]
        if int(args.tile) > 0:
            cmd.extend(["-t", str(int(args.tile))])

        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not tmp_out.exists():
            raise RuntimeError(
                f"realesrgan-ncnn-vulkan failed (exit={proc.returncode}): {proc.stderr or proc.stdout}"
            )

        up_bytes = np.fromfile(str(tmp_out), dtype=np.uint8)
        up_img = cv2.imdecode(up_bytes, cv2.IMREAD_UNCHANGED)
        if up_img is None:
            raise RuntimeError("Failed to decode upscaled output PNG from realesrgan-ncnn-vulkan.")

        target_w, target_h = in_w * target_scale, in_h * target_scale
        if (up_img.shape[1], up_img.shape[0]) != (target_w, target_h):
            up_img = cv2.resize(up_img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        _write_png_u8(out_path, up_img, overwrite=args.overwrite)

    payload = {
        "ok": True,
        "input": str(in_path),
        "out": str(out_path),
        "model": model_name,
        "native_scale": native_scale,
        "target_scale": target_scale,
        "in_size": [in_w, in_h],
        "out_size": [target_w, target_h],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_inpaint(args: argparse.Namespace) -> int:
    import cv2
    import numpy as np
    import torch

    in_path = Path(args.input).resolve()
    mask_path = Path(args.mask).resolve()
    out_path = Path(args.out).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input image not found: {in_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask image not found: {mask_path}")
    if out_path == in_path or out_path == mask_path:
        raise ValueError("Refusing to overwrite input or mask image.")
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {out_path}")
    if not LAMA_MODEL.exists():
        raise FileNotFoundError(f"LaMa model weights not found: {LAMA_MODEL}")

    img_bgr = _read_bgr(in_path)
    h, w = img_bgr.shape[:2]
    mask_bytes = np.fromfile(str(mask_path), dtype=np.uint8)
    mask_gray = cv2.imdecode(mask_bytes, cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        raise ValueError(f"Failed to decode mask PNG: {mask_path}")
    if mask_gray.shape != (h, w):
        raise ValueError(f"Mask shape {mask_gray.shape} does not match image shape {(h, w)}")

    mask_bin = (mask_gray >= 128).astype(np.uint8) * 255
    dilate_px = max(0, int(args.dilate))
    if dilate_px > 0:
        k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        mask_dil = cv2.dilate(mask_bin, k_dil)
    else:
        mask_dil = mask_bin

    ys, xs = np.where(mask_dil > 0)
    if xs.size == 0:
        if args.patch_only:
            bgra = np.zeros((h, w, 4), dtype=np.uint8)
            _write_png_u8(out_path, bgra, overwrite=args.overwrite)
        else:
            _write_png_u8(out_path, img_bgr, overwrite=args.overwrite)
        print(
            json.dumps(
                {
                    "ok": True,
                    "input": str(in_path),
                    "out": str(out_path),
                    "warning": "empty_mask_no_inpaint_applied",
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    pad_crop = max(16, int(args.pad_crop))
    feather_px = max(0, int(args.feather))
    margin = pad_crop + feather_px
    x1 = max(0, int(xs.min()) - margin)
    y1 = max(0, int(ys.min()) - margin)
    x2 = min(w, int(xs.max()) + 1 + margin)
    y2 = min(h, int(ys.max()) + 1 + margin)

    roi_bgr = img_bgr[y1:y2, x1:x2]
    roi_mask = mask_dil[y1:y2, x1:x2]
    rh, rw = roi_bgr.shape[:2]

    pad_h = (8 - (rh % 8)) % 8
    pad_w = (8 - (rw % 8)) % 8
    if pad_h or pad_w:
        roi_bgr_p = cv2.copyMakeBorder(roi_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        roi_mask_p = cv2.copyMakeBorder(roi_mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    else:
        roi_bgr_p = roi_bgr
        roi_mask_p = roi_mask

    roi_rgb = cv2.cvtColor(roi_bgr_p, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_t = torch.from_numpy(np.transpose(roi_rgb, (2, 0, 1))).unsqueeze(0).contiguous()
    mask_t = torch.from_numpy((roi_mask_p > 0).astype(np.float32))[None, None, :, :].contiguous()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = torch.jit.load(str(LAMA_MODEL), map_location="cpu").eval()
        with torch.inference_mode():
            pred_t = model(img_t, mask_t)

    pred_rgb = pred_t[0].permute(1, 2, 0).detach().cpu().numpy()[:rh, :rw, :]
    pred_bgr = cv2.cvtColor(np.clip(np.round(pred_rgb * 255.0), 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Strictly bound feather to [mask_dil, mask_dil + feather_px] so outside pixels are untouched
    if feather_px > 0:
        k_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * feather_px + 1, 2 * feather_px + 1))
        outer_guard = cv2.dilate(roi_mask, k_outer)
        k_blur = 2 * feather_px + 1
        alpha_roi = cv2.GaussianBlur(roi_mask.astype(np.float32) / 255.0, (k_blur, k_blur), 0)
        alpha_roi[outer_guard == 0] = 0.0
    else:
        alpha_roi = (roi_mask > 0).astype(np.float32)

    alpha_u8_roi = np.clip(np.round(alpha_roi * 255.0), 0, 255).astype(np.uint8)
    alpha_f_roi = alpha_u8_roi.astype(np.float32) / 255.0

    if args.patch_only:
        out_arr = np.zeros((h, w, 4), dtype=np.uint8)
        patch_bgra = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2BGRA)
        patch_bgra[:, :, 3] = alpha_u8_roi
        out_arr[y1:y2, x1:x2] = patch_bgra
    else:
        out_arr = img_bgr.copy()
        a3 = alpha_f_roi[:, :, None]
        comp = np.clip(
            np.round(pred_bgr.astype(np.float32) * a3 + roi_bgr.astype(np.float32) * (1.0 - a3)),
            0,
            255,
        ).astype(np.uint8)
        out_arr[y1:y2, x1:x2] = np.where(alpha_u8_roi[:, :, None] == 0, roi_bgr, comp)

    _write_png_u8(out_path, out_arr, overwrite=args.overwrite)
    payload = {
        "ok": True,
        "input": str(in_path),
        "mask": str(mask_path),
        "out": str(out_path),
        "patch_only": bool(args.patch_only),
        "width": w,
        "height": h,
        "roi": [x1, y1, x2, y2],
        "padded_roi": [roi_bgr_p.shape[1], roi_bgr_p.shape[0]],
        "masked_pixels": int((alpha_u8_roi > 0).sum()),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_qa(args: argparse.Namespace) -> int:
    import cv2
    import numpy as np

    in_path = Path(args.input).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input image not found: {in_path}")

    img_bgr = _read_bgr(in_path)
    h, w = img_bgr.shape[:2]
    total_px = float(w * h)

    ch_max = img_bgr.max(axis=2)
    highlight_clipped_pct = round(100.0 * float((ch_max >= 254).sum()) / total_px, 3)
    shadow_crushed_pct = round(100.0 * float((ch_max <= 1).sum()) / total_px, 3)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    saturation_high_pct = round(100.0 * float((hsv[:, :, 1] >= 240).sum()) / total_px, 3)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_sharpness = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 3)

    skin_stats: dict[str, Any] | None = None
    if args.skin_mask:
        mask_path = Path(args.skin_mask).resolve()
        if not mask_path.exists():
            raise FileNotFoundError(f"Skin mask not found: {mask_path}")
        mask_bytes = np.fromfile(str(mask_path), dtype=np.uint8)
        mask_gray = cv2.imdecode(mask_bytes, cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            raise ValueError(f"Failed to decode skin mask: {mask_path}")
        if mask_gray.shape != (h, w):
            mask_gray = cv2.resize(mask_gray, (w, h), interpolation=cv2.INTER_NEAREST)
        sel = mask_gray >= 128
        px_count = int(sel.sum())
        if px_count == 0:
            skin_stats = {"pixel_count": 0, "warning": "empty_skin_mask"}
        else:
            bgr_px = img_bgr[sel].astype(np.float32)
            rgb_mean = [round(float(bgr_px[:, 2].mean()), 2), round(float(bgr_px[:, 1].mean()), 2), round(float(bgr_px[:, 0].mean()), 2)]
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[sel].astype(np.float32)
            l_star = lab[:, 0] * (100.0 / 255.0)
            a_star = lab[:, 1] - 128.0
            b_star = lab[:, 2] - 128.0
            a_mean = float(a_star.mean())
            b_mean = float(b_star.mean())
            skin_stats = {
                "pixel_count": px_count,
                "rgb_mean": rgb_mean,
                "lab_mean": [round(float(l_star.mean()), 2), round(a_mean, 2), round(b_mean, 2)],
                "lab_std": [round(float(l_star.std()), 2), round(float(a_star.std()), 2), round(float(b_star.std()), 2)],
                "a_minus_b_hint": round(a_mean - b_mean, 2),
            }

    payload = {
        "ok": True,
        "input": str(in_path),
        "width": w,
        "height": h,
        "highlight_clipped_pct": highlight_clipped_pct,
        "shadow_crushed_pct": shadow_crushed_pct,
        "saturation_high_pct": saturation_high_pct,
        "laplacian_sharpness": laplacian_sharpness,
        "skin_stats": skin_stats,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _parse_crop_spec(spec: str) -> tuple[int, int, int, int]:
    parts = [int(p.strip()) for p in spec.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Invalid crop '{spec}': expected x,y,w,h")
    x, y, cw, ch = parts
    if x < 0 or y < 0 or cw <= 0 or ch <= 0:
        raise ValueError(f"Invalid crop coordinates '{spec}': expected x>=0, y>=0, w>0, h>0")
    return x, y, cw, ch


def _draw_labeled_pair(left: Any, right: Any, left_label: str, right_label: str, gap: int = 16, pad: int = 16) -> Any:
    import cv2
    import numpy as np

    lh, lw = left.shape[:2]
    rh, rw = right.shape[:2]
    cell_w = max(lw, rw)
    cell_h = max(lh, rh)
    header_h = 34
    canvas_w = pad * 2 + cell_w * 2 + gap
    canvas_h = header_h + cell_h + pad
    row = np.full((canvas_h, canvas_w, 3), 24, dtype=np.uint8)

    row[header_h : header_h + lh, pad : pad + lw] = left
    right_x = pad + cell_w + gap
    row[header_h : header_h + rh, right_x : right_x + rw] = right

    cv2.putText(row, left_label, (pad, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(row, right_label, (right_x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 2, cv2.LINE_AA)
    return row


def cmd_compare(args: argparse.Namespace) -> int:
    import cv2
    import numpy as np

    before_path = Path(args.before).resolve()
    after_path = Path(args.after).resolve()
    out_path = Path(args.out).resolve()
    if not before_path.exists():
        raise FileNotFoundError(f"Before image not found: {before_path}")
    if not after_path.exists():
        raise FileNotFoundError(f"After image not found: {after_path}")
    if out_path == before_path or out_path == after_path:
        raise ValueError("Refusing to overwrite before/after input image.")
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {out_path}")

    b_img = _read_bgr(before_path)
    a_img = _read_bgr(after_path)
    bh, bw = b_img.shape[:2]
    ah, aw = a_img.shape[:2]

    # Top row: full-frame Before | After scaled to fit panel width <= 800 px each
    target_w = min(800, bw)
    scale = float(target_w) / float(bw)
    target_h = max(1, int(round(bh * scale)))
    b_full = cv2.resize(b_img, (target_w, target_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    a_full = cv2.resize(a_img, (target_w, target_h), interpolation=cv2.INTER_AREA if (target_w < aw) else cv2.INTER_LINEAR)

    rows = [_draw_labeled_pair(b_full, a_full, f"BEFORE ({bw}x{bh})", f"AFTER ({aw}x{ah})")]

    crop_specs = [_parse_crop_spec(s) for s in args.crops] if args.crops else []
    if not crop_specs:
        cw = min(400, bw, aw)
        ch = min(400, bh, ah)
        cx = max(0, (min(bw, aw) - cw) // 2)
        cy = max(0, (min(bh, ah) - ch) // 2)
        crop_specs = [(cx, cy, cw, ch)]

    applied_crops: list[list[int]] = []
    for cx, cy, cw, ch in crop_specs:
        x2_b = min(bw, cx + cw)
        y2_b = min(bh, cy + ch)
        x2_a = min(aw, cx + cw)
        y2_a = min(ah, cy + ch)
        if cx >= x2_b or cy >= y2_b or cx >= x2_a or cy >= y2_a:
            raise ValueError(f"Crop [{cx},{cy},{cw},{ch}] is out of bounds for {(bw, bh)} / {(aw, ah)}")
        b_crop = b_img[cy:y2_b, cx:x2_b]
        a_crop = a_img[cy:y2_a, cx:x2_a]
        rows.append(
            _draw_labeled_pair(
                b_crop,
                a_crop,
                f"1:1 BEFORE [{cx},{cy},{b_crop.shape[1]}x{b_crop.shape[0]}]",
                f"1:1 AFTER [{cx},{cy},{a_crop.shape[1]}x{a_crop.shape[0]}]",
            )
        )
        applied_crops.append([cx, cy, int(b_crop.shape[1]), int(b_crop.shape[0])])

    max_w = max(r.shape[1] for r in rows)
    padded_rows = []
    for r in rows:
        if r.shape[1] < max_w:
            pad_r = max_w - r.shape[1]
            r = cv2.copyMakeBorder(r, 0, 0, 0, pad_r, cv2.BORDER_CONSTANT, value=(24, 24, 24))
        padded_rows.append(r)

    sheet = np.vstack(padded_rows)
    _write_png_u8(out_path, sheet, overwrite=args.overwrite)

    payload = {
        "ok": True,
        "before": str(before_path),
        "after": str(after_path),
        "out": str(out_path),
        "sheet_size": [int(sheet.shape[1]), int(sheet.shape[0])],
        "crops": applied_crops,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _run_gen_subcmd(subargs: list[str]) -> int:
    if not GEN_VENV_PYTHON.exists():
        raise FileNotFoundError(
            f"Isolated generative venv Python not found: {GEN_VENV_PYTHON}. Set up venv-gen first."
        )
    if not GEN_SCRIPT.exists():
        raise FileNotFoundError(f"Missing generative helper script: {GEN_SCRIPT}")
    proc = subprocess.run([str(GEN_VENV_PYTHON), str(GEN_SCRIPT)] + subargs, check=False)
    return int(proc.returncode)


def cmd_relight(args: argparse.Namespace) -> int:
    sub = [
        "relight",
        args.input,
        "--mask",
        args.mask,
        "--out",
        args.out,
        "--mode",
        args.mode,
        "--direction",
        args.direction,
        "--prompt",
        args.prompt,
        "--negative-prompt",
        args.negative_prompt,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--hi-width",
        str(args.hi_width),
        "--hi-height",
        str(args.hi_height),
        "--highres-denoise",
        str(args.highres_denoise),
        "--steps",
        str(args.steps),
        "--cfg",
        str(args.cfg),
        "--seed",
        str(args.seed),
    ]
    if args.bg_image:
        sub.extend(["--bg-image", args.bg_image])
    if args.match_input_size:
        sub.append("--match-input-size")
    if args.overwrite:
        sub.append("--overwrite")
    return _run_gen_subcmd(sub)


def cmd_skin_retouch(args: argparse.Namespace) -> int:
    _ensure_cuda_dlls()
    from skin_retouch_runner import run_skin_retouch

    payload = run_skin_retouch(
        input_path=Path(args.input),
        out_path=Path(args.out),
        model_dir=SKIN_RETOUCH_DIR,
        uniface_cache=UNIFACE_CACHE,
        degree=args.degree,
        whitening_degree=args.whitening_degree,
        retouch_local=not args.no_blemish,
        overwrite=args.overwrite,
        out_skin_mask=Path(args.out_skin_mask) if args.out_skin_mask else None,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_cos_liquify(args: argparse.Namespace) -> int:
    from skin_retouch_runner import run_cos_liquify

    payload = run_cos_liquify(
        input_path=Path(args.input),
        faces_json_path=Path(args.faces_json),
        out_path=Path(args.out),
        eye_scale=args.eye_scale,
        nose_narrow=args.nose_narrow,
        nose_tip_narrow=args.nose_tip_narrow,
        mouth_narrow=args.mouth_narrow,
        chin_lift_px=args.chin_lift,
        jaw_taper=args.jaw_taper,
        wig_fluff_px=args.wig_fluff,
        overwrite=args.overwrite,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_face_restore(args: argparse.Namespace) -> int:
    _ensure_cuda_dlls()
    from gfpgan_runner import run_face_restore

    payload = run_face_restore(
        input_path=Path(args.input),
        out_path=Path(args.out),
        model_path=GFPGAN_MODEL,
        uniface_cache=UNIFACE_CACHE,
        weight=args.weight,
        faces_json_path=Path(args.faces_json) if args.faces_json else None,
        nostril_clean=not args.no_nostril_clean,
        overwrite=args.overwrite,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_painterly(args: argparse.Namespace) -> int:
    sub = [
        "painterly",
        args.input,
        "--out",
        args.out,
        "--prompt",
        args.prompt,
        "--negative-prompt",
        args.negative_prompt,
        "--strength",
        str(args.strength),
        "--control-scale",
        str(args.control_scale),
        "--guidance",
        str(args.guidance),
        "--steps",
        str(args.steps),
        "--clip-skip",
        str(args.clip_skip),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--seed",
        str(args.seed),
    ]
    if args.match_input_size:
        sub.append("--match-input-size")
    if args.overwrite:
        sub.append("--overwrite")
    return _run_gen_subcmd(sub)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RetouchTools AI sidecar CLI")
    sub = p.add_subparsers(dest="command", required=True)

    p_doc = sub.add_parser("doctor", help="Check runtime, GPU CUDA ONNX status, and dependencies")
    p_doc.set_defaults(func=cmd_doctor)

    p_fp = sub.add_parser("face-parse", help="Detect faces, 106 landmarks, and 19-class semantic masks")
    p_fp.add_argument("input", help="Input portrait image path (read-only)")
    p_fp.add_argument("--out-dir", required=True, help="Directory to write faces.json and PNG masks")
    p_fp.add_argument("--face-index", default="0", help="Target face index (0=primary) or 'all'")
    p_fp.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files")
    p_fp.set_defaults(func=cmd_face_parse)

    p_mt = sub.add_parser("matte", help="Hair-aware portrait alpha matting (Apache-2.0 PP-Matting V2 / MODNet)")
    p_mt.add_argument("input", help="Input portrait image path (read-only)")
    p_mt.add_argument("--out-mask", required=True, help="Output 8-bit grayscale alpha matte PNG")
    p_mt.add_argument("--out-fg", help="Optional output RGBA cutout PNG")
    p_mt.add_argument("--model", default="ppmatting-v2", help="Matting model (default: ppmatting-v2)")
    p_mt.add_argument("--max-short", type=int, default=1024, help="Max shorter side for PP-Matting/MODNet")
    p_mt.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files")
    p_mt.set_defaults(func=cmd_matte)

    p_bp = sub.add_parser("body-parse", help="6-class body segmentation (hair/body-skin/face-skin/clothes/accessories)")
    p_bp.add_argument("input", help="Input portrait image path (read-only)")
    p_bp.add_argument("--out-dir", required=True, help="Directory to write 6 grayscale PNG masks")
    p_bp.add_argument("--refine-with-matte", help="Optional alpha matte PNG to sharpen outer hair/edge bounds")
    p_bp.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files")
    p_bp.set_defaults(func=cmd_body_parse)

    p_up = sub.add_parser("upscale", help="GPU Vulkan super-resolution via MIT Real-ESRGAN-ncnn-vulkan")
    p_up.add_argument("input", help="Input image path (read-only)")
    p_up.add_argument("--out", required=True, help="Output upscaled PNG path")
    p_up.add_argument(
        "--model",
        default="realesrgan-x4plus",
        choices=["realesrgan-x4plus", "realesrgan-x4plus-anime", "realesr-animevideov3"],
        help="Bundled permissive Real-ESRGAN model (default: realesrgan-x4plus)",
    )
    p_up.add_argument("--scale", type=int, default=2, choices=[2, 4], help="Target upscale factor (2 or 4)")
    p_up.add_argument("--tile", type=int, default=0, help="Tile size (0=auto, >=32)")
    p_up.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output file")
    p_up.set_defaults(func=cmd_upscale)

    p_inp = sub.add_parser("inpaint", help="ROI-cropped object/blemish removal via Apache-2.0 big-lama.pt")
    p_inp.add_argument("input", help="Input image path (read-only)")
    p_inp.add_argument("--mask", required=True, help="Grayscale mask PNG (white>=128 = remove region)")
    p_inp.add_argument("--out", required=True, help="Output PNG path")
    p_inp.add_argument("--pad-crop", type=int, default=128, help="Context padding around mask bbox (px)")
    p_inp.add_argument("--dilate", type=int, default=5, help="Mask dilation radius (px)")
    p_inp.add_argument("--feather", type=int, default=3, help="Boundary feather radius (px)")
    p_inp.add_argument("--patch-only", action="store_true", help="Output RGBA transparent repair patch for doc_place")
    p_inp.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output file")
    p_inp.set_defaults(func=cmd_inpaint)

    p_qa = sub.add_parser("qa", help="Compute clipping, saturation, sharpness, and optional skin-mask Lab stats")
    p_qa.add_argument("input", help="Input image path (read-only)")
    p_qa.add_argument("--skin-mask", help="Optional grayscale skin mask PNG for Lab skin-cast stats")
    p_qa.set_defaults(func=cmd_qa)

    p_cmp = sub.add_parser("compare", help="Generate labeled Before/After full-frame + 1:1 crop comparison sheet")
    p_cmp.add_argument("before", help="Before image path (read-only)")
    p_cmp.add_argument("after", help="After image path (read-only)")
    p_cmp.add_argument("--out", required=True, help="Output comparison sheet PNG path")
    p_cmp.add_argument("--crops", nargs="*", help="Optional 1:1 crop boxes formatted as x,y,w,h")
    p_cmp.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output file")
    p_cmp.set_defaults(func=cmd_compare)

    p_rl = sub.add_parser("relight", help="Portrait/COS relighting via isolated venv-gen IC-Light (6GB VRAM safe)")
    p_rl.add_argument("input", help="Input portrait image path (read-only)")
    p_rl.add_argument("--mask", required=True, help="Foreground alpha matte PNG")
    p_rl.add_argument("--out", required=True, help="Output relit PNG path")
    p_rl.add_argument("--mode", default="fc", choices=["fc", "fbc"], help="IC-Light mode: fc (fg) or fbc (fg+bg)")
    p_rl.add_argument("--bg-image", help="Target background image when --mode fbc")
    p_rl.add_argument("--direction", default="left", choices=["left", "right", "top", "bottom", "ambient"])
    p_rl.add_argument(
        "--prompt",
        default="1girl, soft eastern dragon palace misty celadon lighting, warm porcelain skin, serene atmospheric illumination, detailed silk garments, masterpiece",
    )
    p_rl.add_argument(
        "--negative-prompt",
        default="bad quality, lowres, harsh shadows, neon oversaturation, deformed face, muddy colors",
    )
    p_rl.add_argument("--width", type=int, default=576)
    p_rl.add_argument("--height", type=int, default=1024)
    p_rl.add_argument("--hi-width", type=int, default=864)
    p_rl.add_argument("--hi-height", type=int, default=1536)
    p_rl.add_argument("--highres-denoise", type=float, default=0.42)
    p_rl.add_argument("--steps", type=int, default=25)
    p_rl.add_argument("--cfg", type=float, default=2.0)
    p_rl.add_argument("--seed", type=int, default=42)
    p_rl.add_argument("--match-input-size", action="store_true")
    p_rl.add_argument("--overwrite", action="store_true")
    p_rl.set_defaults(func=cmd_relight)

    p_pt = sub.add_parser("painterly", help="Structure-locked anime painterly pass via Counterfeit-V3.0 + ControlNet Tile")
    p_pt.add_argument("input", help="Input portrait image path (read-only)")
    p_pt.add_argument("--out", required=True, help="Output painterly PNG path")
    p_pt.add_argument(
        "--prompt",
        default="masterpiece, best quality, exquisite semi-realistic anime painterly illustration, soft impasto brushstrokes, creamy porcelain skin, luminous expressive eyes, cohesive hair clumps with cool cyan rim light, soft celadon eastern fantasy atmosphere",
    )
    p_pt.add_argument(
        "--negative-prompt",
        default="lowres, bad anatomy, extra fingers, distorted face, harsh sharpen artifacts, sticker eyes, gray muddy skin, flat plastic noise",
    )
    p_pt.add_argument("--strength", type=float, default=0.35)
    p_pt.add_argument("--control-scale", type=float, default=0.85)
    p_pt.add_argument("--guidance", type=float, default=6.5)
    p_pt.add_argument("--steps", type=int, default=24)
    p_pt.add_argument("--clip-skip", type=int, default=2)
    p_pt.add_argument("--width", type=int, default=768)
    p_pt.add_argument("--height", type=int, default=1344)
    p_pt.add_argument("--seed", type=int, default=42)
    p_pt.add_argument("--match-input-size", action="store_true")
    p_pt.add_argument("--overwrite", action="store_true")
    p_pt.set_defaults(func=cmd_painterly)

    p_sr = sub.add_parser("skin-retouch", help="Alibaba DAMO ABPN Apache-2.0 portrait skin retouch & whitening")
    p_sr.add_argument("input", help="Input portrait image path (read-only)")
    p_sr.add_argument("--out", required=True, help="Output retouched PNG path")
    p_sr.add_argument("--out-skin-mask", help="Optional debug path to save detected skin mask")
    p_sr.add_argument("--degree", type=float, default=0.7, help="Skin smoothing blend degree (default: 0.7)")
    p_sr.add_argument("--whitening-degree", type=float, default=0.75, help="Whole-body whitening strength (default: 0.75)")
    p_sr.add_argument("--no-blemish", action="store_true", help="Skip local blemish detection/inpainting net")
    p_sr.add_argument("--overwrite", action="store_true")
    p_sr.set_defaults(func=cmd_skin_retouch)

    p_cl = sub.add_parser("cos-liquify", help="106-landmark COS youthful proportion liquify")
    p_cl.add_argument("input", help="Input portrait image path (read-only)")
    p_cl.add_argument("--faces-json", required=True, help="faces.json from face-parse")
    p_cl.add_argument("--out", required=True, help="Output liquified PNG path")
    p_cl.add_argument("--eye-scale", type=float, default=0.09)
    p_cl.add_argument("--nose-narrow", type=float, default=0.08)
    p_cl.add_argument("--nose-tip-narrow", type=float, default=0.05)
    p_cl.add_argument("--mouth-narrow", type=float, default=0.09)
    p_cl.add_argument("--chin-lift", type=float, default=8.0)
    p_cl.add_argument("--jaw-taper", type=float, default=0.045)
    p_cl.add_argument("--wig-fluff", type=float, default=7.0)
    p_cl.add_argument("--overwrite", action="store_true")
    p_cl.set_defaults(func=cmd_cos_liquify)

    p_fr = sub.add_parser("face-restore", help="Apache-2.0 GFPGAN v1.4 ONNX crisp eye/lash/pore face restoration")
    p_fr.add_argument("input", help="Input portrait image path (read-only)")
    p_fr.add_argument("--out", required=True, help="Output face-restored PNG path")
    p_fr.add_argument("--weight", type=float, default=0.50, help="Face restore blend weight (default: 0.50)")
    p_fr.add_argument("--faces-json", help="Optional faces.json with landmarks_5 to skip re-detection")
    p_fr.add_argument("--no-nostril-clean", action="store_true", help="Disable extra GFPGAN blend on nostril rim")
    p_fr.add_argument("--overwrite", action="store_true")
    p_fr.set_defaults(func=cmd_face_restore)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
