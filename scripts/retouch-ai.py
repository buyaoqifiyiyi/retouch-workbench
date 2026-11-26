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
MEDIAPIPE_FACE_TASK = Path(
    os.environ.get("MEDIAPIPE_FACE_TASK", MODELS_ROOT / "mediapipe" / "face_landmarker.task")
)
MEDIAPIPE_POSE_TASK = Path(
    os.environ.get("MEDIAPIPE_POSE_TASK", MODELS_ROOT / "mediapipe" / "pose_landmarker_heavy.task")
)
BLENDER_SCENE_KIT_SCRIPT = Path(__file__).resolve().parent / "blender_scene_kit.py"
LAMA_MODEL = Path(os.environ.get("LAMA_MODEL_PATH", MODELS_ROOT / "lama" / "big-lama.pt"))
BIREFNET_LITE_MODEL = Path(
    os.environ.get("BIREFNET_LITE_MODEL", MODELS_ROOT / "birefnet" / "birefnet_lite.onnx")
)
SEGFORMER_CLOTHES_MODEL = Path(
    os.environ.get("SEGFORMER_CLOTHES_MODEL", MODELS_ROOT / "segformer" / "segformer_b2_clothes.onnx")
)
DEPTH_ANYTHING_V2_SMALL_MODEL = Path(
    os.environ.get("DEPTH_ANYTHING_V2_SMALL_MODEL", MODELS_ROOT / "depth" / "depth_anything_v2_small.onnx")
)
SKIN_RETOUCH_DIR = Path(
    os.environ.get("SKIN_RETOUCH_DIR", MODELS_ROOT / "skin-retouching")
)
GFPGAN_MODEL = Path(
    os.environ.get("GFPGAN_ONNX_MODEL", MODELS_ROOT / "gfpgan" / "gfpgan_1.4.onnx")
)
PIH_CURVES_MODEL = Path(
    os.environ.get("PIH_CURVES_MODEL", MODELS_ROOT / "pih" / "curves.onnx")
)
PIH_GAIN_MODEL = Path(
    os.environ.get("PIH_GAIN_MODEL", MODELS_ROOT / "pih" / "gain.onnx")
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
    try:
        blender_p: Path | None = _find_blender_exe()
    except FileNotFoundError:
        blender_p = None
    binaries = {
        "photocraft_cli": {"path": str(PHOTOCRAFT_CLI), "exists": PHOTOCRAFT_CLI.exists()},
        "lightcraft_cli": {"path": str(LIGHTCRAFT_CLI), "exists": LIGHTCRAFT_CLI.exists()},
        "realesrgan_ncnn_vulkan": {
            "path": str(REALESRGAN_EXE),
            "exists": REALESRGAN_EXE.exists() or shutil.which("realesrgan-ncnn-vulkan") is not None,
        },
        "lama_model": {"path": str(LAMA_MODEL), "exists": LAMA_MODEL.exists()},
        "birefnet_lite": {"path": str(BIREFNET_LITE_MODEL), "exists": BIREFNET_LITE_MODEL.exists()},
        "segformer_clothes": {"path": str(SEGFORMER_CLOTHES_MODEL), "exists": SEGFORMER_CLOTHES_MODEL.exists()},
        "depth_anything_v2_small": {"path": str(DEPTH_ANYTHING_V2_SMALL_MODEL), "exists": DEPTH_ANYTHING_V2_SMALL_MODEL.exists()},
        "cinema_luts": {
            "path": str(Path(__file__).resolve().parent.parent / "resources" / "luts"),
            "cube_count": len(list((Path(__file__).resolve().parent.parent / "resources" / "luts").glob("*.cube"))) if (Path(__file__).resolve().parent.parent / "resources" / "luts").exists() else 0,
        },
        "blender_3d": {
            "blender_exe": str(blender_p) if blender_p else None,
            "exists": bool(blender_p and blender_p.exists()),
            "scene_kit": BLENDER_SCENE_KIT_SCRIPT.exists(),
            "fg_vfx_kit": (Path(__file__).resolve().parent / "blender_fg_vfx_kit.py").exists(),
        },
        "skin_retouch_model": {
            "path": str(SKIN_RETOUCH_DIR),
            "exists": (SKIN_RETOUCH_DIR / "pytorch_model.pt").exists() and (SKIN_RETOUCH_DIR / "model.onnx").exists(),
        },
        "gfpgan_v1_4": {"path": str(GFPGAN_MODEL), "exists": GFPGAN_MODEL.exists()},
        "pih_curves": {"path": str(PIH_CURVES_MODEL), "exists": PIH_CURVES_MODEL.exists()},
        "pih_gain": {"path": str(PIH_GAIN_MODEL), "exists": PIH_GAIN_MODEL.exists()},
        "mediapipe_face_landmarker": {"path": str(MEDIAPIPE_FACE_TASK), "exists": MEDIAPIPE_FACE_TASK.exists()},
        "mediapipe_pose_landmarker": {"path": str(MEDIAPIPE_POSE_TASK), "exists": MEDIAPIPE_POSE_TASK.exists()},
        "cv2_ximgproc": {"exists": hasattr(__import__("cv2"), "ximgproc")},
        "hair_brushes": {
            "path": str(Path(__file__).resolve().parent.parent / "resources" / "brushes"),
            "exists": all(
                (Path(__file__).resolve().parent.parent / "resources" / "brushes" / fn).exists()
                for fn in ("hair_01_multi_clump.png", "hair_02_fine_clump.png", "hair_03_single_strand.png")
            ),
            "studio_tip_count": len(list((Path(__file__).resolve().parent.parent / "resources" / "brushes").glob("*.png"))),
        },
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


def _guided_filter_gray(guide_01: Any, src_01: Any, radius: int = 12, eps: float = 1e-3) -> Any:
    import cv2
    import numpy as np

    if radius <= 0:
        return src_01
    ksize = (2 * radius + 1, 2 * radius + 1)
    mean_i = cv2.boxFilter(guide_01, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(src_01, cv2.CV_32F, ksize)
    corr_i = cv2.boxFilter(guide_01 * guide_01, cv2.CV_32F, ksize)
    corr_ip = cv2.boxFilter(guide_01 * src_01, cv2.CV_32F, ksize)
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)
    return np.clip(mean_a * guide_01 + mean_b, 0.0, 1.0)


def _read_alpha_or_gray_f32(path: Path, expected_hw: tuple[int, int]) -> Any:
    import cv2
    import numpy as np

    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"Empty or unreadable mask/alpha file: {path}")
    arr = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(f"Failed to decode mask/alpha file: {path}")
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, 3]
    elif arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    eh, ew = expected_hw
    if arr.shape[:2] != (eh, ew):
        arr = cv2.resize(arr, (ew, eh), interpolation=cv2.INTER_LINEAR)
    return arr.astype(np.float32) / 255.0


def _save_preview_rgb(path: Path, rgb_u8: Any, max_side: int = 1400, overwrite: bool = False) -> None:
    import cv2

    h, w = rgb_u8.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    out_arr = cv2.resize(rgb_u8, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA) if scale < 1.0 else rgb_u8
    _write_png_u8(path, cv2.cvtColor(out_arr, cv2.COLOR_RGB2BGR), overwrite=overwrite)


def cmd_body_parse(args: argparse.Namespace) -> int:
    _ensure_cuda_dlls()
    import cv2
    import numpy as np

    in_path = Path(args.input).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input image not found: {in_path}")

    img_bgr = _read_bgr(in_path)
    h, w = img_bgr.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    refine_matte = None
    if args.refine_with_matte:
        refine_matte = _read_gray_f32(Path(args.refine_with_matte).resolve(), (h, w))

    backend_req = getattr(args, "backend", "auto")
    if backend_req == "segformer" and not SEGFORMER_CLOTHES_MODEL.exists():
        raise FileNotFoundError(f"SegFormer-B2 Clothes ONNX model missing: {SEGFORMER_CLOTHES_MODEL}. Run download_models.py core.")

    use_segformer = backend_req == "segformer" or (backend_req == "auto" and SEGFORMER_CLOTHES_MODEL.exists())
    backend_used = "segformer_b2_clothes" if use_segformer else "mediapipe_selfie_256"
    provider_used = "unknown"

    if use_segformer:
        import onnxruntime as ort  # type: ignore

        avail = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail] or ["CPUExecutionProvider"]
        sess = ort.InferenceSession(str(SEGFORMER_CLOTHES_MODEL), providers=providers)
        provider_used = sess.get_providers()[0] if sess.get_providers() else "unknown"
        inp_meta = sess.get_inputs()[0]
        out_name = sess.get_outputs()[0].name

        rgb_512 = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        chw = np.transpose((rgb_512 - mean) / std, (2, 0, 1))[None, ...]
        if "float16" in str(inp_meta.type):
            chw = chw.astype(np.float16)

        logits = np.asarray(sess.run([out_name], {inp_meta.name: chw})[0], dtype=np.float32)[0]
        logits = logits - logits.max(axis=0, keepdims=True)
        exp_l = np.exp(logits)
        probs = exp_l / np.maximum(exp_l.sum(axis=0, keepdims=True), 1e-8)

        guide_01 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        g_radius = int(getattr(args, "guided_radius", 12))

        def _up(cls_indices: tuple[int, ...]) -> Any:
            p_sum = probs[cls_indices[0]].copy()
            for idx in cls_indices[1:]:
                p_sum = p_sum + probs[idx]
            p_sum = np.clip(p_sum, 0.0, 1.0)
            up = cv2.resize(p_sum, (w, h), interpolation=cv2.INTER_LINEAR)
            if g_radius > 0:
                up = _guided_filter_gray(guide_01, up, radius=g_radius, eps=1e-3)
            return np.clip(up, 0.0, 1.0)

        m_hair = _up((2,))
        m_upper = _up((4,))
        m_skirt = _up((5,))
        m_pants = _up((6,))
        m_dress = _up((7,))
        m_skirt_dress = _up((5, 7))
        m_shoes = _up((9, 10))
        m_face = _up((11,))
        m_legs = _up((12, 13))
        m_arms = _up((14, 15))
        m_body_skin = _up((12, 13, 14, 15))
        m_all_skin = _up((11, 12, 13, 14, 15))
        m_clothes_all = _up((4, 5, 6, 7, 8))
        m_accessories = _up((1, 3, 8, 16, 17))
        m_protected_body = np.clip(np.maximum.reduce([m_clothes_all, m_shoes, m_all_skin]), 0.0, 1.0)

        class_arrays = {
            "mask_body_hair": m_hair,
            "mask_body_skin": m_body_skin,
            "mask_face_skin": m_face,
            "mask_all_skin": m_all_skin,
            "mask_clothes": m_clothes_all,
            "mask_accessories": m_accessories,
            "mask_hair": m_hair,
            "mask_face": m_face,
            "mask_upper_clothes": m_upper,
            "mask_skirt": m_skirt,
            "mask_pants": m_pants,
            "mask_dress": m_dress,
            "mask_skirt_dress": m_skirt_dress,
            "mask_shoes": m_shoes,
            "mask_legs": m_legs,
            "mask_arms": m_arms,
            "mask_clothes_all": m_clothes_all,
            "mask_protected_body": m_protected_body,
        }
    else:
        import mediapipe as mp  # type: ignore
        from mediapipe.tasks.python import BaseOptions  # type: ignore
        from mediapipe.tasks.python.vision import (  # type: ignore
            ImageSegmenter,
            ImageSegmenterOptions,
            RunningMode,
        )

        if not MEDIAPIPE_SELFIE_MODEL.exists():
            raise FileNotFoundError(f"MediaPipe multiclass model not found: {MEDIAPIPE_SELFIE_MODEL}")
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
        provider_used = "MediaPipeCPU"

        def _upscale_conf(idx: int) -> Any:
            raw = res.confidence_masks[idx].numpy_view().copy()
            if raw.ndim == 3:
                raw = raw[:, :, 0]
            if raw.shape != (h, w):
                raw = cv2.resize(raw.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            return np.clip(raw, 0.0, 1.0)

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
            "mask_protected_body": np.clip(np.maximum(c_clothes, c_all_skin), 0.0, 1.0),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    coverage_pct: dict[str, float] = {}
    total_px = float(w * h)
    refined_arrays: dict[str, Any] = {}
    for name, arr in class_arrays.items():
        if refine_matte is not None:
            arr = arr * refine_matte
        refined_arrays[name] = arr
        u8 = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
        out_p = out_dir / f"{name}.png"
        _write_png_u8(out_p, u8, overwrite=args.overwrite)
        outputs[name] = str(out_p)
        coverage_pct[name] = round(100.0 * float((u8 >= 128).sum()) / total_px, 2)

    # Color-coded diagnostic overlay so semantic boundaries can be visually verified immediately
    overlay = rgb.astype(np.float32).copy()
    palette = [
        ("mask_body_hair", (255, 196, 40)),
        ("mask_face_skin", (255, 120, 180)),
        ("mask_body_skin", (255, 175, 130)),
        ("mask_upper_clothes" if "mask_upper_clothes" in refined_arrays else "mask_clothes", (80, 210, 255)),
        ("mask_skirt_dress" if "mask_skirt_dress" in refined_arrays else "mask_clothes", (165, 95, 255)),
        ("mask_shoes" if "mask_shoes" in refined_arrays else "mask_accessories", (60, 235, 110)),
    ]
    blend_rgb = np.zeros_like(overlay)
    blend_w = np.zeros((h, w), dtype=np.float32)
    for key, col in palette:
        if key not in refined_arrays:
            continue
        m = (refined_arrays[key] >= 0.45).astype(np.float32)
        blend_rgb[m > 0] = np.array(col, dtype=np.float32)
        blend_w = np.maximum(blend_w, m * 0.48)
    overlay = np.clip(overlay * (1.0 - blend_w[:, :, None]) + blend_rgb * blend_w[:, :, None], 0.0, 255.0).astype(np.uint8)
    overlay_p = out_dir / "preview_parse_overlay.png"
    _save_preview_rgb(overlay_p, overlay, max_side=1400, overwrite=args.overwrite)
    outputs["preview_parse_overlay"] = str(overlay_p)

    payload = {
        "ok": True,
        "input": str(in_path),
        "backend": backend_used,
        "provider": provider_used,
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

    target_w, target_h = in_w * target_scale, in_h * target_scale
    style = getattr(args, "style", "photo")

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

        if (up_img.shape[1], up_img.shape[0]) != (target_w, target_h):
            up_img = cv2.resize(up_img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        if style == "photo" and up_img.ndim == 3 and up_img.shape[2] >= 3:
            # Tame GAN over-sharpened waxy/oil-paint ridges while keeping micro-contrast
            rgb_f = up_img[:, :, :3].astype(np.float32)
            blur = cv2.GaussianBlur(rgb_f, (0, 0), sigmaX=1.4)
            detail = rgb_f - blur
            clamped = blur + np.tanh(detail / 22.0) * 22.0
            up_img[:, :, :3] = np.clip(np.round(rgb_f * 0.25 + clamped * 0.75), 0, 255).astype(np.uint8)

        _write_png_u8(out_path, up_img, overwrite=args.overwrite)

    payload: dict[str, Any] = {
        "ok": True,
        "input": str(in_path),
        "out": str(out_path),
        "model": model_name,
        "style": style,
        "native_scale": native_scale,
        "target_scale": target_scale,
        "in_size": [in_w, in_h],
        "out_size": [target_w, target_h],
    }
    if min(in_w, in_h) < 1024 and target_scale > 2:
        payload["native_resolution_warning"] = (
            "Input shorter side < 1024px with >2x GAN upscale can produce oil-paint texture; "
            "prefer tile-split + high-res tile refinement + tile-stitch."
        )
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


def _top_peaks(score: Any, n: int, sigma: float, min_dist: int) -> list[tuple[float, int, int]]:
    """Return up to n (value, x, y) peaks of a score map, suppressing neighbours within min_dist."""
    import cv2

    s = cv2.GaussianBlur(score.astype("float32"), (0, 0), sigma) if sigma > 0 else score.astype("float32").copy()
    peaks: list[tuple[float, int, int]] = []
    for _ in range(n):
        _, v, _, loc = cv2.minMaxLoc(s)
        if v <= 1e-6:
            break
        peaks.append((float(v), int(loc[0]), int(loc[1])))
        cv2.circle(s, loc, int(min_dist), 0.0, -1)
    return peaks


def cmd_mask_audit(args: argparse.Namespace) -> int:
    import cv2
    import numpy as np

    in_path = Path(args.input).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input image not found: {in_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    img_bgr = _read_bgr(in_path)
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    failures: list[str] = []
    audit_warnings: list[str] = []
    checks: dict[str, Any] = {}
    outputs: dict[str, str] = {}

    # 1. Zero Interior Hole Check (protected_body vs trim_mask)
    if args.trim_mask:
        trim = _read_gray_f32(Path(args.trim_mask).resolve(), (h, w))
        prot = _read_gray_f32(Path(args.protected_mask).resolve(), (h, w)) if args.protected_mask else None
        ov = rgb.copy()
        trimmed = trim < 0.95
        ov[trimmed] = ov[trimmed] * 0.45 + np.array([255.0, 40.0, 40.0]) * 0.55
        hole_info: dict[str, Any] = {"trimmed_px": int(trimmed.sum())}
        if prot is not None:
            holes = (prot >= 0.75) & (trim < 0.95)
            n_holes = int(holes.sum())
            hole_info["protected_interior_holes_px"] = n_holes
            if n_holes > 0:
                ys, xs = np.where(holes)
                bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                hole_info["hole_bbox_xyxy"] = bbox
                ov[holes] = np.array([255.0, 255.0, 0.0])
                failures.append(f"interior_holes_in_protected_body: {n_holes} px at {bbox}")

            # 1b. Translucent ghost strips left in trim_mask outside protected body
            near_solid = cv2.dilate((trim >= 0.88).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
            near_prot = cv2.dilate((prot >= 0.35).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
            raw_ghost = (prot < 0.35) & (trim > 0.08) & (trim < 0.88) & (~near_solid) & (~near_prot)
            ghost_open = cv2.morphologyEx(raw_ghost.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            nc_g, lbl_g, st_g, cen_g = cv2.connectedComponentsWithStats(ghost_open, connectivity=8)
            ghost_patches = [i for i in range(1, nc_g) if int(st_g[i, cv2.CC_STAT_AREA]) >= 250]
            ghost_px = int(sum(int(st_g[i, cv2.CC_STAT_AREA]) for i in ghost_patches))
            hole_info["trim_ghost_patches"] = len(ghost_patches)
            hole_info["trim_ghost_px"] = ghost_px
            if ghost_patches:
                ghost_mask = np.isin(lbl_g, ghost_patches)
                ov[ghost_mask] = np.array([255.0, 140.0, 0.0])
                biggest = max(ghost_patches, key=lambda i: int(st_g[i, cv2.CC_STAT_AREA]))
                gx, gy = int(cen_g[biggest][0]), int(cen_g[biggest][1])
                failures.append(
                    f"trim_translucent_ghost_strip: {len(ghost_patches)} patches ({ghost_px} px, largest at ({gx},{gy}))"
                )
        out_trim = out_dir / "audit_trim_overlay.png"
        _save_preview_rgb(out_trim, np.clip(ov, 0, 255).astype(np.uint8), max_side=1400, overwrite=args.overwrite)
        outputs["audit_trim_overlay"] = str(out_trim)
        checks["interior_hole_check"] = hole_info

    # 2. Bangs/Eye Translucent Ghosting Check
    if args.hair_alpha and args.eyes_brows_mask:
        hair_a = _read_alpha_or_gray_f32(Path(args.hair_alpha).resolve(), (h, w))
        eyes_m = _read_gray_f32(Path(args.eyes_brows_mask).resolve(), (h, w))
        ys_e, xs_e = np.where(eyes_m >= 0.35)
        if len(ys_e) > 0:
            # Check eyes/brows + narrow 14px band right above brow line
            k_up = np.zeros((29, 9), dtype=np.uint8)
            k_up[:16, :] = 1
            check_zone = cv2.dilate((eyes_m >= 0.35).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
            mid_alpha = check_zone & (hair_a > 0.12) & (hair_a < 0.88)
            n_ghost = int(mid_alpha.sum())
            pct_ghost = round(100.0 * float(n_ghost) / max(1.0, float(check_zone.sum())), 2)
            checks["bangs_ghosting_check"] = {"zone_px": int(check_zone.sum()), "translucent_px": n_ghost, "translucent_pct": pct_ghost}
            if n_ghost > 250 or pct_ghost > 4.0:
                failures.append(f"bangs_eye_translucent_overlap: {n_ghost} px ({pct_ghost}%)")

            pad = 200
            y0, y1 = max(0, int(ys_e.min()) - pad), min(h, int(ys_e.max()) + pad)
            x0, x1 = max(0, int(xs_e.min()) - pad), min(w, int(xs_e.max()) + pad)
            crop_ov = rgb[y0:y1, x0:x1].copy()
            seam_m = (hair_a[y0:y1, x0:x1] > 0.08) & (hair_a[y0:y1, x0:x1] < 0.92)
            crop_ov[seam_m] = crop_ov[seam_m] * 0.35 + np.array([255.0, 0.0, 220.0]) * 0.65
            out_seam = out_dir / "audit_hair_seam_overlay.png"
            _save_preview_rgb(out_seam, np.clip(crop_ov, 0, 255).astype(np.uint8), max_side=1024, overwrite=args.overwrite)
            outputs["audit_hair_seam_overlay"] = str(out_seam)

    # 3. Garment Internal Tone-Jump Check
    if args.garment_mask:
        gar = _read_gray_f32(Path(args.garment_mask).resolve(), (h, w))
        core = cv2.erode((gar >= 0.65).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
        ov_g = rgb.copy()
        gar_hi = gar >= 0.50
        ov_g[gar_hi] = ov_g[gar_hi] * 0.65 + np.array([0.0, 235.0, 255.0]) * 0.35
        out_gar = out_dir / "audit_garment_overlay.png"
        _save_preview_rgb(out_gar, np.clip(ov_g, 0, 255).astype(np.uint8), max_side=1400, overwrite=args.overwrite)
        outputs["audit_garment_overlay"] = str(out_gar)

        if int(core.sum()) >= 1000:
            luma = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
            luma_sm = cv2.GaussianBlur(luma, (0, 0), sigmaX=12.0)
            ys_g, _ = np.where(core)
            y_min, y_max = int(ys_g.min()), int(ys_g.max())
            band_means = []
            for by in range(y_min, max(y_min + 1, y_max - 39), 20):
                bm = core[by:by + 40, :]
                if bm.sum() >= 300:
                    band_means.append((by, float(luma_sm[by:by + 40, :][bm].mean())))
            max_jump = 0.0
            jump_y = None
            for i in range(len(band_means)):
                for j in range(i + 1, len(band_means)):
                    if 20 <= band_means[j][0] - band_means[i][0] <= 65:
                        d = abs(band_means[j][1] - band_means[i][1])
                        if d > max_jump:
                            max_jump = d
                            jump_y = [band_means[i][0], band_means[j][0]]
            p10 = float(np.percentile(luma_sm[core], 10))
            p50 = float(np.percentile(luma_sm[core], 50))
            p90 = float(np.percentile(luma_sm[core], 90))
            jump_limit = 58.0 if p50 >= 120.0 else 38.0
            checks["garment_tone_check"] = {
                "interior_px": int(core.sum()),
                "p10_luma": round(p10, 1),
                "p50_luma": round(p50, 1),
                "p90_luma": round(p90, 1),
                "max_band_jump": round(max_jump, 1),
                "jump_limit": jump_limit,
                "max_jump_y_pair": jump_y,
            }
            if max_jump > jump_limit:
                failures.append(f"garment_internal_tone_jump: {max_jump:.1f} luma across y={jump_y} (limit {jump_limit:.1f})")
            elif max_jump > 38.0:
                audit_warnings.append(
                    f"garment_internal_tone_gradient: {max_jump:.1f} luma across y={jump_y} (high-key p50={p50:.1f}; verify natural relighting shadow in audit_garment_overlay)"
                )

            # 3b. Neutral dark-garment chroma-cast check in CIE Lab (catches RGB dehaze brown-cast)
            lab_gar = cv2.cvtColor(np.clip(rgb / 255.0, 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2LAB)
            L_gar, a_gar, b_gar = lab_gar[:, :, 0], lab_gar[:, :, 1], lab_gar[:, :, 2]
            dark_core = core & (L_gar < 45.0)
            if int(dark_core.sum()) >= 500:
                mean_a = float(a_gar[dark_core].mean())
                mean_b = float(b_gar[dark_core].mean())
                mean_c = float(np.hypot(mean_a, mean_b))
                cast_rec: dict[str, Any] = {
                    "dark_core_px": int(dark_core.sum()),
                    "mean_a": round(mean_a, 2),
                    "mean_b": round(mean_b, 2),
                    "mean_chroma": round(mean_c, 2),
                }
                cast_failed = False
                if getattr(args, "neutral_garment", False) and (abs(mean_a) > 5.0 or abs(mean_b) > 7.5 or mean_c > 8.5):
                    cast_failed = True
                    failures.append(
                        f"garment_color_cast: dark core mean Lab (a*={mean_a:+.2f}, b*={mean_b:+.2f}, C*={mean_c:.2f}) exceeds neutral limits"
                    )
                if args.reference:
                    ref_rgb_f = cv2.cvtColor(_read_bgr(Path(args.reference).resolve()), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    lab_ref_g = cv2.cvtColor(ref_rgb_f, cv2.COLOR_RGB2LAB)
                    ref_dark = core & (lab_ref_g[:, :, 0] < 55.0)
                    if int(ref_dark.sum()) >= 500:
                        ref_a = float(lab_ref_g[:, :, 1][ref_dark].mean())
                        ref_b = float(lab_ref_g[:, :, 2][ref_dark].mean())
                        ref_c = float(np.hypot(ref_a, ref_b))
                        delta_c = float(np.hypot(mean_a - ref_a, mean_b - ref_b))
                        cast_rec.update({"ref_a": round(ref_a, 2), "ref_b": round(ref_b, 2), "ref_chroma": round(ref_c, 2), "delta_chroma_vs_ref": round(delta_c, 2)})
                        if not cast_failed and ref_c < 8.0 and delta_c > 5.5:
                            failures.append(
                                f"garment_color_cast: shifted from neutral reference (C*_ref={ref_c:.2f}) by dC*={delta_c:.2f} (> 5.5)"
                            )
                checks["garment_color_cast_check"] = cast_rec

    # 5. Extended invariants: asset resolution, hair continuity/texture, subject-background blend, garment texture
    def _k(r: int) -> Any:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

    def _mean(arr: Any, m: Any) -> float:
        return float(arr[m].mean()) if m.any() else 0.0

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    fine = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 0.7))
    ref_gray = ref_fine = ref_L = None
    if args.reference:
        ref_bgr = _read_bgr(Path(args.reference).resolve())
        if ref_bgr.shape[:2] != (h, w):
            raise ValueError(f"--reference must match input size {w}x{h}, got {ref_bgr.shape[1]}x{ref_bgr.shape[0]}")
        ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ref_fine = np.abs(ref_gray - cv2.GaussianBlur(ref_gray, (0, 0), 0.7))
        ref_L = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32) * (100.0 / 255.0)
    hotspots: list[tuple[str, float, int, int]] = []

    # 5a. Native-resolution gate for generated/placed sources (upscaled AI plates invent oil-paint texture)
    if args.asset:
        from PIL import Image

        res_rows = []
        for spec in args.asset:
            if "=" not in spec:
                raise ValueError(f"Invalid --asset '{spec}'; expected NAME=PATH[|x,y,w,h]")
            aname, rest = spec.split("=", 1)
            apath, _, abox = rest.partition("|")
            with Image.open(Path(apath).resolve()) as im_a:
                aw, ah = im_a.size
            bw, bh = (_parse_crop_spec(abox)[2:] if abox else (w, h))
            lin = round(min(aw / max(1, bw), ah / max(1, bh)), 3)
            res_rows.append({"name": aname, "native": [aw, ah], "mapped_to": [bw, bh], "linear_ratio": lin})
            if lin < 0.6:
                failures.append(f"asset_low_native_resolution: {aname} {aw}x{ah} mapped onto {bw}x{bh} (ratio {lin} < 0.6; detail is upscaler-invented)")
        checks["asset_resolution_check"] = res_rows

    def _check_hair_texture(target_hm: Any) -> float | None:
        if ref_fine is None or not args.reference_hair_mask:
            return None
        rcore = cv2.erode((_read_gray_f32(Path(args.reference_hair_mask).resolve(), (h, w)) > 0.5).astype(np.uint8), _k(3)) > 0
        core = cv2.erode(target_hm, _k(3)) > 0
        if core.sum() < 2000 or rcore.sum() < 2000:
            return None
        tex = _mean(fine, core) / max(1e-3, _mean(ref_fine, rcore))
        style_mode = getattr(args, "style", "photo")
        hi_tex = 1.20 if style_mode == "photo" else 1.40
        lo_tex = 0.65 if style_mode == "photo" else 0.55
        if tex > hi_tex:
            failures.append(f"hair_over_textured: fine detail {tex:.2f}x reference (> {hi_tex}, upscaler oil-paint look)")
        elif tex < lo_tex:
            failures.append(f"hair_plastic_smooth: fine detail {tex:.2f}x reference (< {lo_tex})")
        return round(tex, 2)

    # 5b. Hair continuity over replaced/composited hair cutouts + texture vs reference
    # In real-first mode (--native-hair-mask without --hair-alpha), inspect native 4K hair texture
    # and treat any --hair-layer overlays as decorative hair-paint wisps (safety-checked for eye overlap).
    is_real_first_audit = bool(getattr(args, "native_hair_mask", None)) and not bool(args.hair_alpha)
    hair_paths = ([args.hair_alpha] if args.hair_alpha else []) + ([] if is_real_first_audit else list(args.hair_layer or []))
    hair_union = None
    if hair_paths:
        hair_union = np.maximum.reduce([_read_alpha_or_gray_f32(Path(p).resolve(), (h, w)) for p in hair_paths])
        hm = (hair_union > 0.5).astype(np.uint8)
        hair_px = int(hm.sum())
        if hair_px >= 2000:
            mpx = hair_px / 1e6
            n_c, _, st_c, cen_c = cv2.connectedComponentsWithStats(hm, connectivity=8)
            isl = [i for i in range(1, n_c) if 8 <= st_c[i, cv2.CC_STAT_AREA] < 400]
            gap = ((cv2.morphologyEx(hm, cv2.MORPH_CLOSE, _k(4)) > 0) & (hm == 0)).astype(np.uint8)
            n_g, _, st_g, cen_g = cv2.connectedComponentsWithStats(gap, connectivity=8)
            gaps = [i for i in range(1, n_g) if st_g[i, cv2.CC_STAT_AREA] >= 6]
            hair_info: dict[str, Any] = {
                "hair_px": hair_px,
                "layers": len(hair_paths),
                "components": n_c - 1,
                "islands_per_mpx": round(len(isl) / mpx, 1),
                "gaps_per_mpx": round(len(gaps) / mpx, 1),
            }
            if len(gaps) / mpx > 30.0:
                failures.append(f"hair_broken_gaps: {len(gaps)} sub-9px gaps ({len(gaps) / mpx:.0f}/Mpx > 30)")
            if len(isl) / mpx > 15.0:
                failures.append(f"hair_detached_islands: {len(isl)} fragments ({len(isl) / mpx:.0f}/Mpx > 15)")
            heat = np.zeros((h, w), np.float32)
            for i in gaps:
                heat[int(cen_g[i][1]), int(cen_g[i][0])] += 1.0
            for i in isl:
                heat[int(cen_c[i][1]), int(cen_c[i][0])] += 1.0
            hotspots += [("hair_break", v, x, y) for v, x, y in _top_peaks(heat, 3, 40.0, 260)]
            tex_val = _check_hair_texture(hm)
            if tex_val is not None:
                hair_info["fine_texture_vs_reference"] = tex_val

            # 5b-2. Occlusion-boundary hair chop / wide mid-alpha halo / gap-shelf check
            if args.occluder_mask or args.protected_mask:
                occ_path = Path(args.occluder_mask or args.protected_mask).resolve()
                occ = (_read_gray_f32(occ_path, (h, w)) >= 0.5).astype(np.uint8)
                dist_occ = cv2.distanceTransform(1 - occ, cv2.DIST_L2, 5)
                halo = (occ == 0) & (dist_occ <= 18.0) & (hair_union > 0.08) & (hair_union < 0.85)
                halo_open = cv2.morphologyEx(halo.astype(np.uint8), cv2.MORPH_OPEN, _k(2))
                nh_o, lbl_o, st_o, cen_o = cv2.connectedComponentsWithStats(halo_open, connectivity=8)
                halo_ids = [
                    i for i in range(1, nh_o)
                    if int(st_o[i, cv2.CC_STAT_AREA]) >= 250 and float(dist_occ[lbl_o == i].min()) <= 12.0
                ]
                halo_px = int(sum(int(st_o[i, cv2.CC_STAT_AREA]) for i in halo_ids))

                gy_d, gx_d = np.gradient(dist_occ)
                mag_d = np.hypot(gx_d, gy_d) + 1e-6
                ux, uy = gx_d / mag_d, gy_d / mag_d
                yy_idx, xx_idx = np.mgrid[0:h, 0:w]
                ys_o14 = np.clip(np.round(yy_idx + 14.0 * uy).astype(np.int32), 0, h - 1)
                xs_o14 = np.clip(np.round(xx_idx + 14.0 * ux).astype(np.int32), 0, w - 1)
                ys_i18 = np.clip(np.round(yy_idx - 18.0 * uy).astype(np.int32), 0, h - 1)
                xs_i18 = np.clip(np.round(xx_idx - 18.0 * ux).astype(np.int32), 0, w - 1)
                shelf_gap = (
                    (occ == 0) & (dist_occ >= 2.0) & (dist_occ <= 10.0) & (hair_union < 0.22)
                    & (hair_union[ys_o14, xs_o14] > 0.60) & (hair_union[ys_i18, xs_i18] > 0.35)
                )
                shelf_open = cv2.morphologyEx(shelf_gap.astype(np.uint8), cv2.MORPH_OPEN, _k(1))
                ns_o, lbl_s, st_s, cen_s = cv2.connectedComponentsWithStats(shelf_open, connectivity=8)
                shelf_ids = [i for i in range(1, ns_o) if int(st_s[i, cv2.CC_STAT_AREA]) >= 200]
                shelf_px = int(sum(int(st_s[i, cv2.CC_STAT_AREA]) for i in shelf_ids))

                hair_info["occlusion_halo_patches"] = len(halo_ids)
                hair_info["occlusion_halo_px"] = halo_px
                hair_info["occlusion_gap_shelf_px"] = shelf_px
                if halo_ids or shelf_ids:
                    occ_score = np.where(np.isin(lbl_o, halo_ids) | np.isin(lbl_s, shelf_ids), 1.0, 0.0).astype(np.float32)
                    hotspots += [("hair_occ", v, x, y) for v, x, y in _top_peaks(occ_score, 3, 24.0, 220)]
                    failures.append(
                        f"hair_occlusion_chop_or_ghost: {len(halo_ids)} halo bands ({halo_px} px), {len(shelf_ids)} gap shelves ({shelf_px} px)"
                    )
            checks["hair_continuity_check"] = hair_info
    elif getattr(args, "native_hair_mask", None):
        nh_m = (_read_gray_f32(Path(args.native_hair_mask).resolve(), (h, w)) > 0.5).astype(np.uint8)
        tex_val = _check_hair_texture(nh_m)
        ht_info: dict[str, Any] = {"mode": "real-first", "hair_px": int(nh_m.sum())}
        if tex_val is not None:
            ht_info["fine_texture_vs_reference"] = tex_val
        checks["hair_texture_check"] = ht_info
        if args.hair_layer:
            hp_union = np.maximum.reduce([_read_alpha_or_gray_f32(Path(p).resolve(), (h, w)) for p in args.hair_layer])
            hp_info: dict[str, Any] = {
                "layers": len(args.hair_layer),
                "painted_px": int((hp_union > 0.05).sum()),
            }
            if args.eyes_brows_mask:
                eyes_m_hp = _read_gray_f32(Path(args.eyes_brows_mask).resolve(), (h, w))
                eye_ov_px = int(((hp_union > 0.05) & (eyes_m_hp >= 0.25)).sum())
                hp_info["eye_overlap_px"] = eye_ov_px
                if eye_ov_px > 20:
                    failures.append(f"hair_paint_eye_overlap: {eye_ov_px} px of painted hair overlap eyes/brows (> 20)")
            checks["hair_paint_safety_check"] = hp_info

    # 5c. Subject-background blend (edge warmth, light direction, grain)
    if args.subject_alpha:
        sa = _read_alpha_or_gray_f32(Path(args.subject_alpha).resolve(), (h, w))
        sub, bgm = sa > 0.95, sa < 0.02
        if sub.sum() >= 5000 and bgm.sum() >= 5000:
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            L = lab[:, :, 0] * (100.0 / 255.0)
            Bc = lab[:, :, 2] - 128.0
            soft = (sa > 0.02).astype(np.uint8)
            inner = sub & ~(cv2.erode(sub.astype(np.uint8), _k(12)) > 0)
            outer = bgm & (cv2.dilate(soft, _k(12)) > 0)
            near = bgm & (cv2.dilate(soft, _k(120)) > 0)
            left = np.arange(w)[None, :] < int(np.median(np.where(sub)[1]))
            edge_db = _mean(Bc, inner) - _mean(Bc, outer)
            subj_lr = _mean(L, sub & left) - _mean(L, sub & ~left)
            bg_lr = _mean(L, near & left) - _mean(L, near & ~left)
            blend_info: dict[str, Any] = {
                "edge_warmth_db": round(edge_db, 1),
                "subject_left_minus_right_L": round(subj_lr, 1),
                "background_left_minus_right_L": round(bg_lr, 1),
                "light_direction_mismatch": round(abs(subj_lr - bg_lr), 1),
            }
            if ref_L is not None:
                ref_subj_lr = _mean(ref_L, sub & left) - _mean(ref_L, sub & ~left)
                blend_info["reference_subject_left_minus_right_L"] = round(ref_subj_lr, 1)
            else:
                ref_subj_lr = None
            blend_warnings: list[str] = []
            # Thin-edge b* mixes skin/fabric/hair with background; record only (control image false-positived here).
            if abs(edge_db) > 5.0:
                blend_warnings.append(f"blend_edge_color_cast: subject edge b* differs from adjacent background by {edge_db:+.1f} (> 5, info)")
            if abs(subj_lr - bg_lr) > 8.0:
                msg = f"blend_light_direction_mismatch: subject L/R {subj_lr:+.1f} vs background {bg_lr:+.1f} (diff > 8)"
                sun_side = getattr(args, "sun_side", "auto")
                if getattr(args, "scene_meta", None):
                    meta_p = Path(args.scene_meta).resolve()
                    if meta_p.exists():
                        sun_side = str(json.loads(meta_p.read_text(encoding="utf-8")).get("sun_side", sun_side))
                phys_match = (sun_side == "right" and subj_lr < -2.0) or (sun_side == "left" and subj_lr > 2.0)
                if phys_match:
                    blend_info["physical_sun_check"] = f"MATCH ({sun_side}, subj L/R={subj_lr:+.1f})"
                    blend_warnings.append(f"{msg}; matches physical scene sun_side='{sun_side}', info")
                elif ref_subj_lr is not None and abs(ref_subj_lr) > 8.0:
                    blend_warnings.append(f"{msg}; albedo-driven in reference ({ref_subj_lr:+.1f}), info")
                else:
                    failures.append(msg)
            # Segmented neutral-shadow cast: 6 vertical bands, subject rim (0-30px) vs background ring (0-60px),
            # only neutral darks (L<40, chroma<14) so skin/garment colour does not count as cast.
            Cc = np.hypot(lab[:, :, 1] - 128.0, Bc)
            in_s = (sa > 0.9) & ~(cv2.erode((sa > 0.9).astype(np.uint8), _k(30)) > 0)
            out_s = bgm & (cv2.dilate(soft, _k(60)) > 0)
            ndark = (L < 40) & (Cc < 14)
            sys_ = np.where(sa > 0.5)[0]
            sy0, sy1 = int(sys_.min()), int(sys_.max())
            step = (sy1 - sy0) / 6.0
            YYi = np.arange(h)[:, None]
            segs: list[dict[str, Any]] = []
            for i in range(6):
                band = (YYi >= sy0 + i * step) & (YYi < sy0 + (i + 1) * step)
                mi, mo = in_s & band & ndark, out_s & band & ndark
                rec: dict[str, Any] = {"segment": i, "y": [int(sy0 + i * step), int(sy0 + (i + 1) * step)], "n_subject": int(mi.sum()), "n_background": int(mo.sum())}
                if mi.sum() < 2000 or mo.sum() < 2000:
                    rec["skipped"] = True
                else:
                    gap = _mean(Bc, mi) - _mean(Bc, mo)
                    rec["gap_b"] = round(gap, 1)
                    if abs(gap) > 9.0:
                        failures.append(f"blend_shadow_cast_segment: band {i} y{rec['y'][0]}-{rec['y'][1]} neutral-shadow b* gap {gap:+.1f} (> 9)")
                    elif abs(gap) > 5.0:
                        blend_warnings.append(f"blend_shadow_cast_segment: band {i} y{rec['y'][0]}-{rec['y'][1]} neutral-shadow b* gap {gap:+.1f} (5-9, style-dependent)")
                segs.append(rec)
            blend_info["shadow_cast_segments"] = segs
            core6 = cv2.erode(sub.astype(np.uint8), _k(6)) > 0
            ring = bgm & (cv2.dilate(soft, _k(250)) > 0)
            for nm_, lo_, hi_ in (("shadow", 3.0, 35.0), ("highlight", 75.0, 97.0)):
                ms_ = core6 & (L >= lo_) & (L <= hi_) & (Cc < 14)
                mb_ = ring & (L >= lo_) & (L <= hi_) & (Cc < 14)
                if ms_.sum() >= 5000 and mb_.sum() >= 5000:
                    blend_info[f"{nm_}_cast_global_gap_b"] = round(_mean(Bc, ms_) - _mean(Bc, mb_), 1)
            blend_info["warnings"] = blend_warnings
            lap = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), (0, 0), 4.0)
            hf = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 1.0))
            core = cv2.erode(sub.astype(np.uint8), _k(20)) > 0
            bgc = cv2.erode(bgm.astype(np.uint8), _k(20)) > 0
            if core.sum() >= 1000 and bgc.sum() >= 1000:
                f_core = core & (lap < np.percentile(lap[core], 30))
                f_bg = bgc & (lap < np.percentile(lap[bgc], 30))
                grain = _mean(hf, f_core) / max(1e-3, _mean(hf, f_bg))
                blend_info["grain_ratio_subject_over_bg"] = round(grain, 2)
                if not 0.7 <= grain <= 1.5:
                    failures.append(f"blend_grain_mismatch: subject/background grain ratio {grain:.2f} (outside 0.7-1.5)")
            cnts, _ = cv2.findContours((sa > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if cnts:
                seg = np.zeros((h, w), np.float32)
                for x, y in max(cnts, key=len)[:, 0, :][::60]:
                    s = np.s_[max(0, y - 40):y + 40, max(0, x - 40):x + 40]
                    if inner[s].sum() >= 50 and outer[s].sum() >= 50:
                        seg[y, x] = abs(_mean(Bc[s], inner[s]) - _mean(Bc[s], outer[s])) + 0.25 * abs(_mean(L[s], inner[s]) - _mean(L[s], outer[s]))
                hotspots += [("blend_edge", v, x, y) for v, x, y in _top_peaks(seg, 3, 0.0, 260)]
            checks["blend_check"] = blend_info

            # Edge integrity metrics: detached micro-debris & interior hairline pinholes
            fg_b = (sa >= 0.45).astype(np.uint8)
            n_e, lbl_e, st_e, _ = cv2.connectedComponentsWithStats(fg_b, connectivity=8)
            detached_px = 0
            if n_e > 2:
                main_id_e = int(np.argmax(st_e[1:, cv2.CC_STAT_AREA])) + 1
                dist_m_e = cv2.distanceTransform((lbl_e != main_id_e).astype(np.uint8), cv2.DIST_L2, 5)
                for cid_e in range(1, n_e):
                    if cid_e == main_id_e:
                        continue
                    a_e = int(st_e[cid_e, cv2.CC_STAT_AREA])
                    if a_e < 72 and float(dist_m_e[lbl_e == cid_e].min()) > 10.0:
                        detached_px += a_e
            edge_ring = (sa > 0.03) & (sa < 0.85)
            edge_luma_gap = round(abs(_mean(L, edge_ring) - _mean(L, inner)), 2) if edge_ring.any() and inner.any() else 0.0
            checks["edge_integrity_check"] = {
                "detached_micro_specks_px": detached_px,
                "edge_fringe_luma_gap": edge_luma_gap,
                "soft_fringe_px": int(edge_ring.sum()),
            }

    # 5c-bis. Cinema tone dynamic range check (catches washed-out low-contrast haze)
    L_full = lab[:, :, 0] * (100.0 / 255.0)
    hsv_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    p02_l, p50_l, p98_l = [round(float(v), 2) for v in np.percentile(L_full, [2, 50, 98])]
    dyn_range = round(p98_l - p02_l, 2)
    bright_pct_l85 = round(float((L_full > 85.0).mean() * 100.0), 2)
    sat_mean_c = round(float((hsv_full[:, :, 1] / 255.0).mean()), 3)
    checks["cinema_tone_check"] = {
        "p02_L": p02_l,
        "p50_L": p50_l,
        "p98_L": p98_l,
        "dyn_range": dyn_range,
        "bright_pct_L85": bright_pct_l85,
        "sat_mean": sat_mean_c,
    }
    if p02_l > 38.0 or dyn_range < 50.0:
        audit_warnings.append(f"washed_out_low_contrast: p02_L={p02_l} (>38) or dyn_range={dyn_range} (<50); run cinema-grade/relight-sculpt")

    # 5d. Garment fabric texture retention vs reference
    if args.garment_mask and ref_fine is not None:
        gmask = _read_gray_f32(Path(args.garment_mask).resolve(), (h, w)) >= 0.5
        if hair_union is not None:
            gmask &= hair_union < 0.3
        gmask = cv2.erode(gmask.astype(np.uint8), _k(6)) > 0
        if gmask.sum() >= 5000:
            mid = np.abs(cv2.GaussianBlur(gray, (0, 0), 1.5) - cv2.GaussianBlur(gray, (0, 0), 6.0))
            rmid = np.abs(cv2.GaussianBlur(ref_gray, (0, 0), 1.5) - cv2.GaussianBlur(ref_gray, (0, 0), 6.0))
            fine_ret = _mean(fine, gmask) / max(1e-3, _mean(ref_fine, gmask))
            retention = (_mean(fine, gmask) / max(1e-3, _mean(mid, gmask))) / max(1e-3, _mean(ref_fine, gmask) / max(1e-3, _mean(rmid, gmask)))
            checks["garment_texture_check"] = {"fine_vs_reference": round(fine_ret, 2), "texture_to_structure_retention": round(retention, 2)}
            if retention < 0.8:
                failures.append(f"garment_texture_lost: fabric micro-texture/structure retention {retention:.2f} (< 0.8 vs reference)")
            if fine_ret > 1.35:
                failures.append(f"garment_texture_noisy: fabric fine detail {fine_ret:.2f}x reference (> 1.35)")
            loss = np.clip(1.0 - cv2.GaussianBlur(fine, (0, 0), 12.0) / np.maximum(cv2.GaussianBlur(ref_fine, (0, 0), 12.0), 1e-3), 0.0, 1.0)
            hotspots += [("garment_texture", v, x, y) for v, x, y in _top_peaks(np.where(gmask, loss, 0.0), 2, 0.0, 260)]

    # 5e. Auto hotspot 1:1 review crops at the worst places the checks found (not fixed boxes)
    review_crops: dict[str, str] = {}
    if hotspots:
        overview = img_bgr.copy()
        for idx, (label, _, x, y) in enumerate(hotspots, 1):
            x0 = int(np.clip(x - 256, 0, max(0, w - 512)))
            y0 = int(np.clip(y - 256, 0, max(0, h - 512)))
            x1, y1 = min(w, x0 + 512), min(h, y0 + 512)
            hp = out_dir / f"hot_{idx:02d}_{label}_{x}_{y}_1to1.png"
            _write_png_u8(hp, img_bgr[y0:y1, x0:x1], overwrite=args.overwrite)
            review_crops[f"{idx:02d}_{label}"] = str(hp)
            cv2.rectangle(overview, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 255), 8)
            cv2.putText(overview, str(idx), (x0 + 12, y0 + 70), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 0, 255), 6)
        out_hot = out_dir / "audit_hotspots_overview.png"
        _save_preview_rgb(out_hot, cv2.cvtColor(overview, cv2.COLOR_BGR2RGB), max_side=1400, overwrite=args.overwrite)
        outputs["audit_hotspots_overview"] = str(out_hot)

    # 4. Standalone True 1:1 Single-Region Crops (NO downscaled multi-panel collapse)
    crops_saved: dict[str, str] = {}
    for spec in (args.crop or []):
        if ":" not in spec:
            raise ValueError(f"Invalid --crop '{spec}'; expected name:x,y,w,h")
        cname, box = spec.split(":", 1)
        cx, cy, cw, ch = _parse_crop_spec(box)
        x0 = max(0, min(w - 1, cx))
        y0 = max(0, min(h - 1, cy))
        x1 = max(x0 + 1, min(w, x0 + cw))
        y1 = max(y0 + 1, min(h, y0 + ch))
        crop_p = out_dir / f"crop_{cname}_1to1.png"
        _write_png_u8(crop_p, img_bgr[y0:y1, x0:x1], overwrite=args.overwrite)
        crops_saved[cname] = str(crop_p)
        outputs[f"crop_{cname}"] = str(crop_p)

    numeric_pass = len(failures) == 0
    payload = {
        "ok": True,
        "pass": numeric_pass,
        "visual_review": "done" if args.visual_reviewed else "unverified",
        "verdict": "FAIL" if not numeric_pass else ("PASS" if args.visual_reviewed else "NUMERIC_PASS_VISUAL_UNVERIFIED"),
        "failures": failures,
        "warnings": audit_warnings,
        "input": str(in_path),
        "checks": checks,
        "crops_1to1": crops_saved,
        "review_crops_1to1": review_crops,
        "outputs": outputs,
    }
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


def cmd_hair_comp(args: argparse.Namespace) -> int:
    from workbench_ops import run_hair_comp

    payload = run_hair_comp(
        subject_rgba_path=Path(args.subject_rgba).resolve(),
        subject_hair_mask_path=Path(args.subject_hair_mask).resolve(),
        protected_mask_path=Path(args.protected_mask).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        occluder_mask_path=Path(args.occluder_mask).resolve() if args.occluder_mask else None,
        accessories_mask_path=Path(args.accessories_mask).resolve() if args.accessories_mask else None,
        mode=args.mode,
        gen_hair_rgb_path=Path(args.gen_hair_rgb).resolve() if args.gen_hair_rgb else None,
        gen_hair_matte_path=Path(args.gen_hair_matte).resolve() if args.gen_hair_matte else None,
        gen_hair_mask_path=Path(args.gen_hair_mask).resolve() if args.gen_hair_mask else None,
        clean_trapped_gaps=bool(args.clean_trapped_gaps),
        haze_recover=float(args.haze_recover),
        decontam_strength=float(args.decontam_strength),
        groom=bool(args.groom),
        groom_strength=float(args.groom_strength),
        overwrite=args.overwrite,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_hair_paint(args: argparse.Namespace) -> int:
    from workbench_ops import run_hair_paint

    payload = run_hair_paint(
        subject_rgba_path=Path(args.subject_rgba).resolve(),
        hair_mask_path=Path(args.hair_mask).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        occluder_mask_path=Path(args.occluder_mask).resolve() if args.occluder_mask else None,
        face_mask_path=Path(args.face_mask).resolve() if args.face_mask else None,
        eyes_brows_mask_path=Path(args.eyes_brows_mask).resolve() if args.eyes_brows_mask else None,
        clothes_mask_path=Path(args.clothes_mask).resolve() if args.clothes_mask else None,
        wind_dir=args.wind_dir,
        intensity=args.intensity,
        seed=int(args.seed),
        overwrite=args.overwrite,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_garment_retouch(args: argparse.Namespace) -> int:
    from workbench_ops import run_garment_retouch

    payload = run_garment_retouch(
        input_path=Path(args.input).resolve(),
        out_path=Path(args.out).resolve(),
        garment_mask_path=Path(args.garment_mask).resolve() if args.garment_mask else None,
        body_parse_dir=Path(args.body_parse_dir).resolve() if args.body_parse_dir else None,
        alpha_mask_path=Path(args.alpha_mask).resolve() if args.alpha_mask else None,
        out_patch_path=Path(args.out_patch).resolve() if args.out_patch else None,
        out_dark_mask_path=Path(args.out_dark_mask).resolve() if args.out_dark_mask else None,
        material=args.material,
        spatial_dehaze=float(args.spatial_dehaze),
        target_black_l=str(args.target_black_l),
        neutral_black=float(args.neutral_black),
        drape_contrast=float(args.drape_contrast),
        weave_preserve=float(args.weave_preserve),
        overwrite=args.overwrite,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_skin_unify(args: argparse.Namespace) -> int:
    from workbench_ops import run_skin_unify

    payload = run_skin_unify(
        input_path=Path(args.input).resolve(),
        body_parse_dir=Path(args.body_parse_dir).resolve(),
        out_path=Path(args.out).resolve(),
        alpha_mask_path=Path(args.alpha_mask).resolve() if args.alpha_mask else None,
        out_patch_path=Path(args.out_patch).resolve() if args.out_patch else None,
        warmth_pull=float(args.warmth_pull),
        glare_compress=float(args.glare_compress),
        skirt_shadow=float(args.skirt_shadow),
        defringe=bool(args.defringe),
        overwrite=args.overwrite,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_harmonize(args: argparse.Namespace) -> int:
    _ensure_cuda_dlls()
    from workbench_ops import run_harmonize

    payload = run_harmonize(
        fg_rgba_path=Path(args.fg_rgba).resolve(),
        bg_path=Path(args.bg).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        pih_curves_path=PIH_CURVES_MODEL,
        pih_gain_path=PIH_GAIN_MODEL,
        skin_mask_path=Path(args.skin_mask).resolve() if args.skin_mask else None,
        garment_dark_mask_path=Path(args.garment_dark_mask).resolve() if args.garment_dark_mask else None,
        backend=args.backend,
        strength=float(args.strength),
        gain_strength=float(args.gain_strength),
        skin_preserve=float(args.skin_preserve),
        light_wrap=float(args.light_wrap),
        grain_match=bool(args.grain_match),
        overwrite=args.overwrite,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_tile_split(args: argparse.Namespace) -> int:
    from workbench_ops import run_tile_split

    payload = run_tile_split(
        input_path=Path(args.input).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        grid=args.grid,
        overlap=float(args.overlap),
        max_tile_side=int(args.max_tile_side),
        overwrite=args.overwrite,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_tile_stitch(args: argparse.Namespace) -> int:
    from workbench_ops import run_tile_stitch

    payload = run_tile_stitch(
        manifest_path=Path(args.manifest).resolve(),
        tiles_dir=Path(args.tiles_dir).resolve(),
        base_path=Path(args.base).resolve(),
        out_path=Path(args.out).resolve(),
        lock_lowfreq_sigma=float(args.lock_lowfreq_sigma),
        ecc_align=bool(args.ecc_align),
        detail_strength=float(args.detail_strength),
        overwrite=args.overwrite,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_tile_refine(args: argparse.Namespace) -> int:
    sub = [
        "tile-refine",
        args.input,
        "--out",
        args.out,
        "--prompt",
        args.prompt,
        "--negative-prompt",
        args.negative_prompt,
        "--denoise",
        str(args.denoise),
        "--control-scale",
        str(args.control_scale),
        "--guidance",
        str(args.guidance),
        "--steps",
        str(args.steps),
        "--tile-size",
        str(args.tile_size),
        "--overlap",
        str(args.overlap),
        "--seed",
        str(args.seed),
    ]
    if args.overwrite:
        sub.append("--overwrite")
    return _run_gen_subcmd(sub)


def cmd_cutout_refine(args: argparse.Namespace) -> int:
    from cutout_refine_ops import run_cutout_refine

    payload = run_cutout_refine(
        input_path=Path(args.input).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        birefnet_model_path=BIREFNET_LITE_MODEL,
        coarse_mask_path=Path(args.coarse_mask).resolve() if args.coarse_mask else None,
        body_parse_dir=Path(args.body_parse_dir).resolve() if args.body_parse_dir else None,
        hi_res_tiles=bool(args.hi_res_tiles),
        clean_pockets=bool(args.clean_pockets),
        overwrite=args.overwrite,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_skin_fs(args: argparse.Namespace) -> int:
    from pure_pc_retouch_ops import run_skin_fs

    payload = run_skin_fs(
        input_path=Path(args.input).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        pc_exe=str(PHOTOCRAFT_CLI),
        face_parse_dir=Path(args.face_parse_dir).resolve(),
        body_parse_dir=Path(args.body_parse_dir).resolve() if args.body_parse_dir else None,
        alpha_mask_path=Path(args.alpha_mask).resolve() if args.alpha_mask else None,
        face_task_path=MEDIAPIPE_FACE_TASK if MEDIAPIPE_FACE_TASK.exists() else None,
        smooth_tone=float(args.smooth_tone),
        porcelain_lift=float(args.porcelain_lift),
        warmth_balance=float(args.warmth_balance),
        db_strength=float(args.db_strength),
        overwrite=args.overwrite,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_makeup_paint(args: argparse.Namespace) -> int:
    from pure_pc_retouch_ops import run_makeup_paint

    payload = run_makeup_paint(
        input_path=Path(args.input).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        pc_exe=str(PHOTOCRAFT_CLI),
        face_task_path=MEDIAPIPE_FACE_TASK,
        faces_json_path=Path(args.faces_json).resolve() if args.faces_json else None,
        intensity=float(args.intensity),
        overwrite=args.overwrite,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_body_liquify(args: argparse.Namespace) -> int:
    from pure_pc_retouch_ops import run_body_liquify

    payload = run_body_liquify(
        input_path=Path(args.input).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        pc_exe=str(PHOTOCRAFT_CLI),
        pose_task_path=MEDIAPIPE_POSE_TASK,
        body_parse_dir=Path(args.body_parse_dir).resolve(),
        faces_json_path=Path(args.faces_json).resolve() if args.faces_json else None,
        shoulder_slim=float(args.shoulder_slim),
        arm_slim=float(args.arm_slim),
        waist_cinch=float(args.waist_cinch),
        cuff_smooth=float(args.cuff_smooth),
        overwrite=args.overwrite,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_vfx_overlay(args: argparse.Namespace) -> int:
    from vfx_overlay_ops import run_vfx_overlay

    tbox = tuple(_parse_crop_spec(args.title_box)) if args.title_box else None
    payload = run_vfx_overlay(
        vfx_path=Path(args.vfx).resolve(),
        width=int(args.width),
        height=int(args.height),
        out_dir=Path(args.out_dir).resolve(),
        subject_alpha_path=Path(args.subject_alpha).resolve() if args.subject_alpha else None,
        face_parse_dir=Path(args.face_parse_dir).resolve() if args.face_parse_dir else None,
        title_box_xywh=tbox,
        placement=args.placement,
        black_floor=str(args.black_floor),
        tint_hex=args.tint_hex,
        tint_strength=float(args.tint_strength),
        spill_strength=float(args.spill_strength),
        subject_front_keep=float(args.subject_front_keep),
        density_gamma=float(args.density_gamma),
        haze_suppress=float(args.haze_suppress),
        max_alpha=float(args.max_alpha),
        offset_x=float(args.offset_x),
        offset_y=float(args.offset_y),
        overwrite=args.overwrite,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_relight_sculpt(args: argparse.Namespace) -> int:
    from relight_cinema_ops import run_relight_sculpt

    if not DEPTH_ANYTHING_V2_SMALL_MODEL.exists():
        raise FileNotFoundError(f"Depth-Anything-V2-Small model missing: {DEPTH_ANYTHING_V2_SMALL_MODEL}. Run download_models.py core.")
    payload = run_relight_sculpt(
        fg_rgba_path=Path(args.fg_rgba).resolve(),
        depth_model_path=DEPTH_ANYTHING_V2_SMALL_MODEL,
        out_dir=Path(args.out_dir).resolve(),
        bg_path=Path(args.bg).resolve() if args.bg else None,
        sun_side=args.sun_side,
        scene_meta_path=Path(args.scene_meta).resolve() if args.scene_meta else None,
        face_parse_dir=Path(args.face_parse_dir).resolve() if args.face_parse_dir else None,
        body_parse_dir=Path(args.body_parse_dir).resolve() if args.body_parse_dir else None,
        shadow_strength=float(args.shadow_strength),
        key_strength=float(args.key_strength),
        rim_strength=float(args.rim_strength),
        bounce_strength=float(args.bounce_strength),
        face_protect=float(args.face_protect),
        overwrite=args.overwrite,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_cinema_grade(args: argparse.Namespace) -> int:
    from relight_cinema_ops import run_cinema_grade

    payload = run_cinema_grade(
        input_path=Path(args.input).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        mode=args.mode,
        preset=args.preset,
        sun_side=args.sun_side,
        subject_x_frac=float(args.subject_x_frac),
        photocraft_cli=PHOTOCRAFT_CLI,
        overwrite=args.overwrite,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_fg_vfx_3d(args: argparse.Namespace) -> int:
    from relight_cinema_ops import run_fg_vfx_3d

    kit = Path(__file__).resolve().parent / "blender_fg_vfx_kit.py"
    payload = run_fg_vfx_3d(
        blender_exe=_find_blender_exe(),
        kit_script=kit,
        preset=args.preset,
        width=int(args.width),
        height=int(args.height),
        render_width=int(args.render_width),
        render_height=int(args.render_height),
        samples=int(args.samples),
        out_dir=Path(args.out_dir).resolve(),
        subject_alpha_path=Path(args.subject_alpha).resolve() if args.subject_alpha else None,
        face_parse_dir=Path(args.face_parse_dir).resolve() if args.face_parse_dir else None,
        normal_map_path=Path(args.normal_map).resolve() if args.normal_map else None,
        subject_x_frac=float(args.subject_x_frac),
        sun_side=args.sun_side,
        overwrite=args.overwrite,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _find_blender_exe() -> Path:
    env_p = os.environ.get("BLENDER_EXE")
    if env_p and Path(env_p).exists():
        return Path(env_p)
    which_p = shutil.which("blender")
    if which_p:
        return Path(which_p)
    for pat in (
        r"D:\HRSoftstore\Install\Blender\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender*\blender.exe",
        r"D:\*\Blender*\blender.exe",
    ):
        matches = sorted(glob.glob(pat), reverse=True)
        if matches:
            return Path(matches[0])
    raise FileNotFoundError("blender.exe not found; set BLENDER_EXE environment variable.")


def cmd_scene_whitemodel(args: argparse.Namespace) -> int:
    blender = _find_blender_exe()
    out_d = Path(args.out_dir).resolve()
    out_d.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(blender),
        "--background",
        "--python",
        str(BLENDER_SCENE_KIT_SCRIPT),
        "--",
        "--preset", args.preset,
        "--aspect", args.aspect,
        "--width", str(args.width),
        "--height", str(args.height),
        "--lens", str(args.lens),
        "--sun-azimuth", str(args.sun_azimuth),
        "--sun-elevation", str(args.sun_elevation),
        "--subject-x-frac", str(args.subject_x_frac),
        "--samples", str(args.samples),
        "--out-dir", str(out_d),
    ]
    if args.subject_rgba:
        cmd += ["--subject-rgba", str(Path(args.subject_rgba).resolve())]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"Blender scene-whitemodel failed (rc={proc.returncode}): {proc.stderr[-800:] or proc.stdout[-800:]}")
    meta_p = out_d / "scene_meta.json"
    print(meta_p.read_text(encoding="utf-8"))
    return 0


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

    p_bp = sub.add_parser("body-parse", help="18-class SegFormer-B2 Clothes (default) or 6-class MediaPipe body/garment segmentation")
    p_bp.add_argument("input", help="Input portrait image path (read-only)")
    p_bp.add_argument("--out-dir", required=True, help="Directory to write grayscale PNG masks and preview_parse_overlay.png")
    p_bp.add_argument("--refine-with-matte", help="Optional alpha matte PNG to sharpen outer hair/edge bounds")
    p_bp.add_argument("--backend", default="auto", choices=["auto", "segformer", "mediapipe"], help="Segmentation backend (default: auto -> SegFormer-B2)")
    p_bp.add_argument("--guided-radius", type=int, default=12, help="Guided filter radius for edge snapping (0=off, default: 12)")
    p_bp.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files")
    p_bp.set_defaults(func=cmd_body_parse)

    p_ma = sub.add_parser("mask-audit", help="Automated hole/ghosting/tone-jump audit, neon overlays, and standalone true 1:1 crops")
    p_ma.add_argument("input", help="Input composite or subject image path (read-only)")
    p_ma.add_argument("--out-dir", required=True, help="Directory to write audit overlays and standalone 1:1 crops")
    p_ma.add_argument("--protected-mask", help="Canonical non-hair body/garment mask PNG (e.g. mask_protected_body.png)")
    p_ma.add_argument("--trim-mask", help="Subject trim mask PNG (e.g. 04a_subject_4k_hair_trim_mask.png)")
    p_ma.add_argument("--garment-mask", help="Single garment mask PNG for tone-jump check (e.g. mask_skirt_dress.png)")
    p_ma.add_argument("--hair-alpha", help="Resculpted hair RGBA or grayscale alpha PNG")
    p_ma.add_argument("--native-hair-mask", help="SegFormer hair mask for real-first runs without external hair cutout")
    p_ma.add_argument("--eyes-brows-mask", help="Transformed eyes & brows mask PNG")
    p_ma.add_argument("--crop", action="append", help="Standalone 1:1 crop formatted as name:x,y,w,h (repeatable)")
    p_ma.add_argument("--hair-layer", action="append", help="Extra visible hair layer RGBA/alpha PNG, unioned with --hair-alpha for continuity (repeatable)")
    p_ma.add_argument("--reference", help="Pixel-aligned reference image (e.g. 01_beauty_base) for hair/garment texture comparison")
    p_ma.add_argument("--reference-hair-mask", help="Hair mask of --reference for hair texture comparison")
    p_ma.add_argument("--subject-alpha", help="Subject alpha or RGBA cutout for the subject-background blend check")
    p_ma.add_argument("--asset", action="append", help="Native-resolution gate for a generated/placed source: NAME=PATH[|x,y,w,h it was mapped onto] (repeatable)")
    p_ma.add_argument("--occluder-mask", help="Mask of body parts in front of hair (arms/hands/face) for occlusion break audit")
    p_ma.add_argument("--neutral-garment", action="store_true", help="Fail if neutral dark garment interior shows brown/warm Lab color cast")
    p_ma.add_argument("--style", default="photo", choices=["photo", "painterly"], help="Texture gate strictness (default: photo)")
    p_ma.add_argument("--scene-meta", help="Optional scene_meta.json from scene-whitemodel for physical sun direction check")
    p_ma.add_argument("--sun-side", default="auto", choices=["auto", "left", "right", "back"], help="Override physical sun side check")
    p_ma.add_argument("--visual-reviewed", action="store_true", help="Set ONLY after actually viewing every review crop; otherwise verdict stays NUMERIC_PASS_VISUAL_UNVERIFIED")
    p_ma.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files")
    p_ma.set_defaults(func=cmd_mask_audit)

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
    p_up.add_argument("--style", default="photo", choices=["photo", "anime"], help="photo dampens waxy GAN ridges (default: photo)")
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

    p_hc = sub.add_parser("hair-comp", help="Style-adaptive, occlusion-clean hair compositing")
    p_hc.add_argument("--subject-rgba", required=True, help="RGBA subject cutout PNG")
    p_hc.add_argument("--subject-hair-mask", required=True, help="SegFormer hair mask PNG")
    p_hc.add_argument("--protected-mask", required=True, help="SegFormer protected body mask PNG")
    p_hc.add_argument("--occluder-mask", help="Body-in-front-of-hair mask PNG (arms/hands/face)")
    p_hc.add_argument("--accessories-mask", help="Optional accessories/ribbon mask PNG to protect from gap cleanup")
    p_hc.add_argument("--mode", default="real-first", choices=["real-first", "hybrid-flow", "full-replace"])
    p_hc.add_argument("--gen-hair-rgb", help="Generated hair RGB PNG (for hybrid-flow/full-replace)")
    p_hc.add_argument("--gen-hair-matte", help="Generated foreground matte PNG")
    p_hc.add_argument("--gen-hair-mask", help="Generated semantic hair mask PNG")
    p_hc.add_argument("--clean-trapped-gaps", action=argparse.BooleanOptionalAction, default=True)
    p_hc.add_argument("--haze-recover", type=float, default=0.35)
    p_hc.add_argument("--decontam-strength", type=float, default=0.75)
    p_hc.add_argument("--groom", action=argparse.BooleanOptionalAction, default=True)
    p_hc.add_argument("--groom-strength", type=float, default=0.38)
    p_hc.add_argument("--out-dir", required=True)
    p_hc.add_argument("--overwrite", action="store_true")
    p_hc.set_defaults(func=cmd_hair_comp)

    p_hp = sub.add_parser("hair-paint", help="Native PhotoCraft 3-brush layered COS hair painting")
    p_hp.add_argument("--subject-rgba", required=True, help="RGBA subject PNG (preserves real hair)")
    p_hp.add_argument("--hair-mask", required=True, help="SegFormer hair mask PNG")
    p_hp.add_argument("--occluder-mask", help="Arms/hands/face occluder mask PNG for 0px flush erase")
    p_hp.add_argument("--face-mask", help="Face skin mask PNG for bangs wisps & eye protection")
    p_hp.add_argument("--eyes-brows-mask", help="Eyes & brows mask PNG for strict eye protection")
    p_hp.add_argument("--clothes-mask", help="Clothes mask PNG for flush shoulder/costume erase")
    p_hp.add_argument("--wind-dir", default="auto", choices=["down-left", "down-right", "left", "right", "auto"])
    p_hp.add_argument("--intensity", default="medium", choices=["subtle", "medium", "dramatic"])
    p_hp.add_argument("--seed", type=int, default=42)
    p_hp.add_argument("--out-dir", required=True)
    p_hp.add_argument("--overwrite", action="store_true")
    p_hp.set_defaults(func=cmd_hair_paint)

    p_gr = sub.add_parser("garment-retouch", help="Material-aware CIE Lab frequency separation & luminance dehaze for garments")
    p_gr.add_argument("input", help="Input RGB or RGBA PNG")
    p_gr.add_argument("--garment-mask", help="Single garment grayscale mask PNG")
    p_gr.add_argument("--body-parse-dir", help="SegFormer body-parse dir for multi-garment auto retouch")
    p_gr.add_argument("--alpha-mask", help="Optional subject alpha matte PNG")
    p_gr.add_argument("--material", default="auto", choices=["auto", "dark-matte", "dark-twill", "light-cotton", "glossy-leather", "pastel-lace", "translucent-organza", "satin-ribbon", "knit-stocking"])
    p_gr.add_argument("--spatial-dehaze", type=float, default=-1.0)
    p_gr.add_argument("--target-black-l", default="14.0", help="Target CIE L* black level or 'upper'")
    p_gr.add_argument("--neutral-black", type=float, default=-1.0)
    p_gr.add_argument("--drape-contrast", type=float, default=0.35)
    p_gr.add_argument("--weave-preserve", type=float, default=1.0)
    p_gr.add_argument("--out", required=True, help="Retouched output PNG")
    p_gr.add_argument("--out-patch", help="Optional transparent garment delta/patch RGBA PNG for PhotoCraft")
    p_gr.add_argument("--out-dark-mask", help="Optional full dark-garment union grayscale PNG (skirt+socks+shoes+residual) for harmonize protection")
    p_gr.add_argument("--overwrite", action="store_true")
    p_gr.set_defaults(func=cmd_garment_retouch)

    p_su = sub.add_parser("skin-unify", help="Unify washed lower-limb skin with arms + defringe + skirt/leg cast & AO shadow")
    p_su.add_argument("input", help="Input RGB or RGBA PNG")
    p_su.add_argument("--body-parse-dir", required=True, help="SegFormer body-parse directory")
    p_su.add_argument("--alpha-mask", help="Optional subject alpha matte PNG")
    p_su.add_argument("--warmth-pull", type=float, default=0.72)
    p_su.add_argument("--glare-compress", type=float, default=0.62)
    p_su.add_argument("--skirt-shadow", type=float, default=0.42)
    p_su.add_argument("--defringe", action=argparse.BooleanOptionalAction, default=True)
    p_su.add_argument("--out", required=True, help="Skin-unified output PNG")
    p_su.add_argument("--out-patch", help="Optional transparent leg skin patch RGBA PNG for PhotoCraft")
    p_su.add_argument("--overwrite", action="store_true")
    p_su.set_defaults(func=cmd_skin_unify)

    p_hm = sub.add_parser("harmonize", help="Subject-background harmonization via Apache-2.0 PhotoBlaze-PIH ONNX")
    p_hm.add_argument("--fg-rgba", required=True, help="RGBA subject cutout PNG")
    p_hm.add_argument("--bg", required=True, help="Aligned background RGB PNG")
    p_hm.add_argument("--skin-mask", help="Optional grayscale skin mask PNG")
    p_hm.add_argument("--garment-dark-mask", help="Optional neutral dark garment mask PNG")
    p_hm.add_argument("--backend", default="auto", choices=["auto", "pih", "analytical"])
    p_hm.add_argument("--strength", type=float, default=0.70)
    p_hm.add_argument("--gain-strength", type=float, default=0.50)
    p_hm.add_argument("--skin-preserve", type=float, default=0.60)
    p_hm.add_argument("--light-wrap", type=float, default=0.22)
    p_hm.add_argument("--grain-match", action="store_true")
    p_hm.add_argument("--out-dir", required=True)
    p_hm.add_argument("--overwrite", action="store_true")
    p_hm.set_defaults(func=cmd_harmonize)

    p_ts = sub.add_parser("tile-split", help="Split high-res canvas into overlapping tiles with manifest")
    p_ts.add_argument("input", help="Input image PNG")
    p_ts.add_argument("--out-dir", required=True)
    p_ts.add_argument("--grid", default="2x2")
    p_ts.add_argument("--overlap", type=float, default=0.22)
    p_ts.add_argument("--max-tile-side", type=int, default=1400)
    p_ts.add_argument("--overwrite", action="store_true")
    p_ts.set_defaults(func=cmd_tile_split)

    p_tx = sub.add_parser("tile-stitch", help="Frequency-locked subpixel-aligned stitch of refined tiles")
    p_tx.add_argument("--manifest", required=True)
    p_tx.add_argument("--tiles-dir", required=True)
    p_tx.add_argument("--base", required=True)
    p_tx.add_argument("--out", required=True)
    p_tx.add_argument("--lock-lowfreq-sigma", type=float, default=48.0)
    p_tx.add_argument("--ecc-align", action=argparse.BooleanOptionalAction, default=True)
    p_tx.add_argument("--detail-strength", type=float, default=0.85)
    p_tx.add_argument("--overwrite", action="store_true")
    p_tx.set_defaults(func=cmd_tile_stitch)

    p_tr = sub.add_parser("tile-refine", help="Optional venv-gen SD1.5 ControlNet-Tile photographic detail refinement")
    p_tr.add_argument("input")
    p_tr.add_argument("--out", required=True)
    p_tr.add_argument("--prompt", default="masterpiece architectural photograph, crisp weathered natural limestone pores, delicate natural leaves and rose petals")
    p_tr.add_argument("--negative-prompt", default="oil painting, anime, illustration, CGI grid, blurry, plastic")
    p_tr.add_argument("--denoise", type=float, default=0.30)
    p_tr.add_argument("--control-scale", type=float, default=0.90)
    p_tr.add_argument("--guidance", type=float, default=5.5)
    p_tr.add_argument("--steps", type=int, default=20)
    p_tr.add_argument("--tile-size", type=int, default=768)
    p_tr.add_argument("--overlap", type=int, default=160)
    p_tr.add_argument("--seed", type=int, default=42)
    p_tr.add_argument("--overwrite", action="store_true")
    p_tr.set_defaults(func=cmd_tile_refine)

    p_cr = sub.add_parser("cutout-refine", help="High-res perimeter BiRefNet tiles + multi-level F/B edge decontamination")
    p_cr.add_argument("input", help="Input portrait image PNG")
    p_cr.add_argument("--coarse-mask", help="Optional coarse alpha matte PNG to accelerate tile detection")
    p_cr.add_argument("--body-parse-dir", help="Optional body-parse dir (enables strand pinhole bridging & safe pocket clear)")
    p_cr.add_argument("--hi-res-tiles", action=argparse.BooleanOptionalAction, default=True)
    p_cr.add_argument("--clean-pockets", action=argparse.BooleanOptionalAction, default=True)
    p_cr.add_argument("--out-dir", required=True)
    p_cr.add_argument("--overwrite", action="store_true")
    p_cr.set_defaults(func=cmd_cutout_refine)

    p_sfs = sub.add_parser("skin-fs", help="Pure non-generative Rolling-Guidance Lab frequency separation + PhotoCraft D&B")
    p_sfs.add_argument("input")
    p_sfs.add_argument("--face-parse-dir", required=True)
    p_sfs.add_argument("--body-parse-dir")
    p_sfs.add_argument("--alpha-mask")
    p_sfs.add_argument("--smooth-tone", type=float, default=0.62)
    p_sfs.add_argument("--porcelain-lift", type=float, default=0.38)
    p_sfs.add_argument("--warmth-balance", type=float, default=0.45)
    p_sfs.add_argument("--db-strength", type=float, default=0.42)
    p_sfs.add_argument("--out-dir", required=True)
    p_sfs.add_argument("--overwrite", action="store_true")
    p_sfs.set_defaults(func=cmd_skin_fs)

    p_mp = sub.add_parser("makeup-paint", help="Pure PC PhotoCraft hand-painted makeup layers via 478 MediaPipe landmarks")
    p_mp.add_argument("input")
    p_mp.add_argument("--faces-json", help="Optional faces.json fallback")
    p_mp.add_argument("--intensity", type=float, default=0.72)
    p_mp.add_argument("--out-dir", required=True)
    p_mp.add_argument("--overwrite", action="store_true")
    p_mp.set_defaults(func=cmd_makeup_paint)

    p_bl = sub.add_parser("body-liquify", help="Pose-driven PhotoCraft filter.liquify with freeze masks & mirror sheet")
    p_bl.add_argument("input")
    p_bl.add_argument("--body-parse-dir", required=True)
    p_bl.add_argument("--faces-json")
    p_bl.add_argument("--shoulder-slim", type=float, default=0.35)
    p_bl.add_argument("--arm-slim", type=float, default=0.35)
    p_bl.add_argument("--waist-cinch", type=float, default=0.25)
    p_bl.add_argument("--cuff-smooth", type=float, default=0.65)
    p_bl.add_argument("--out-dir", required=True)
    p_bl.add_argument("--overwrite", action="store_true")
    p_bl.set_defaults(func=cmd_body_liquify)

    p_vx = sub.add_parser("vfx-overlay", help="Optical Unmult + face/title avoidance for black-bg AI VFX overlays")
    p_vx.add_argument("--vfx", required=True, help="Black-background VFX image path")
    p_vx.add_argument("--width", type=int, default=3840)
    p_vx.add_argument("--height", type=int, default=2160)
    p_vx.add_argument("--subject-alpha", help="Optional 4K subject matte PNG for edge light spill")
    p_vx.add_argument("--face-parse-dir", help="Optional warped 4K face-parse dir for zero eye/face occlusion")
    p_vx.add_argument("--title-box", help="Optional x,y,w,h poster title safe box")
    p_vx.add_argument("--placement", default="auto-split", choices=["auto-split", "front", "behind"])
    p_vx.add_argument("--black-floor", default="auto")
    p_vx.add_argument("--tint-hex")
    p_vx.add_argument("--tint-strength", type=float, default=0.20)
    p_vx.add_argument("--spill-strength", type=float, default=0.25)
    p_vx.add_argument("--subject-front-keep", type=float, default=0.10)
    p_vx.add_argument("--density-gamma", type=float, default=1.40)
    p_vx.add_argument("--haze-suppress", type=float, default=0.45)
    p_vx.add_argument("--max-alpha", type=float, default=0.85)
    p_vx.add_argument("--offset-x", type=float, default=0.0)
    p_vx.add_argument("--offset-y", type=float, default=0.0)
    p_vx.add_argument("--out-dir", required=True)
    p_vx.add_argument("--overwrite", action="store_true")
    p_vx.set_defaults(func=cmd_vfx_overlay)

    p_sw = sub.add_parser("scene-whitemodel", help="Parameterized Blender GPU white-model scene kit + Z-depth & prompts")
    p_sw.add_argument("--preset", default="crystal-conservatory", choices=["crystal-conservatory", "european-arcade", "seaside-stage", "rooftop-twilight", "eastern-pavilion", "studio-cyclorama"])
    p_sw.add_argument("--aspect", default="16:9", choices=["16:9", "9:16", "2.39:1"])
    p_sw.add_argument("--width", type=int, default=1920)
    p_sw.add_argument("--height", type=int, default=1080)
    p_sw.add_argument("--lens", type=float, default=38.0)
    p_sw.add_argument("--sun-azimuth", type=float, default=42.0)
    p_sw.add_argument("--sun-elevation", type=float, default=18.0)
    p_sw.add_argument("--subject-x-frac", type=float, default=0.70)
    p_sw.add_argument("--subject-rgba", default="")
    p_sw.add_argument("--samples", type=int, default=28)
    p_sw.add_argument("--out-dir", required=True)
    p_sw.set_defaults(func=cmd_scene_whitemodel)

    p_rs = sub.add_parser("relight-sculpt", help="2.5D Depth-Anything-V2 volumetric relighting (4 PhotoCraft layers, zero pixel regen)")
    p_rs.add_argument("--fg-rgba", required=True, help="RGBA subject cutout PNG")
    p_rs.add_argument("--bg", help="Aligned background RGB PNG for key/shadow/floor color sampling")
    p_rs.add_argument("--sun-side", default="auto", choices=["auto", "right", "left", "top-right", "top-left", "back"])
    p_rs.add_argument("--scene-meta", help="Optional scene_meta.json")
    p_rs.add_argument("--face-parse-dir", help="Optional face-parse dir for eye/inner-face protection")
    p_rs.add_argument("--body-parse-dir", help="Optional body-parse dir for hair rim enhancement")
    p_rs.add_argument("--shadow-strength", type=float, default=0.58)
    p_rs.add_argument("--key-strength", type=float, default=0.45)
    p_rs.add_argument("--rim-strength", type=float, default=0.52)
    p_rs.add_argument("--bounce-strength", type=float, default=0.25)
    p_rs.add_argument("--face-protect", type=float, default=0.55)
    p_rs.add_argument("--out-dir", required=True)
    p_rs.add_argument("--overwrite", action="store_true")
    p_rs.set_defaults(func=cmd_relight_sculpt)

    p_cg = sub.add_parser("cinema-grade", help="BG shadow/spotlight anchor + fg framing + PhotoCraft CameraRaw & tetrahedral .cube LUT")
    p_cg.add_argument("input", help="Input background (bg-anchor mode) or composite PNG (grade mode)")
    p_cg.add_argument("--mode", default="grade", choices=["bg-anchor", "grade"])
    p_cg.add_argument("--preset", default="crystal-sunset-cinema", choices=["crystal-sunset-cinema", "sunset-teal-gold", "kodak2383", "moonlight-crimson", "epic-slate"])
    p_cg.add_argument("--sun-side", default="right")
    p_cg.add_argument("--subject-x-frac", type=float, default=0.70)
    p_cg.add_argument("--out-dir", required=True)
    p_cg.add_argument("--overwrite", action="store_true")
    p_cg.set_defaults(func=cmd_cinema_grade)

    p_fv = sub.add_parser("fg-vfx-3d", help="Blender GPU 3D optical-bokeh foreground props + pose-wrapped melody filaments + causal spill")
    p_fv.add_argument("--preset", default="crystal-conservatory", choices=["crystal-conservatory", "european-arcade", "seaside-stage", "rooftop-twilight", "eastern-pavilion"])
    p_fv.add_argument("--width", type=int, default=3840)
    p_fv.add_argument("--height", type=int, default=2160)
    p_fv.add_argument("--render-width", type=int, default=2560)
    p_fv.add_argument("--render-height", type=int, default=1440)
    p_fv.add_argument("--samples", type=int, default=32)
    p_fv.add_argument("--subject-alpha", help="4K subject cutout RGBA/alpha PNG")
    p_fv.add_argument("--face-parse-dir", help="Optional warped 4K face-parse dir for zero eye occlusion")
    p_fv.add_argument("--normal-map", help="Optional Depth-Anything-V2 normal_map_rgb.png for causal bounce modulation")
    p_fv.add_argument("--subject-x-frac", type=float, default=0.70)
    p_fv.add_argument("--sun-side", default="right")
    p_fv.add_argument("--out-dir", required=True)
    p_fv.add_argument("--overwrite", action="store_true")
    p_fv.set_defaults(func=cmd_fg_vfx_3d)

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
