"""Standalone ONNX Runtime runner for Apache-2.0 GFPGAN v1.4 portrait face restoration."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from skin_retouch_runner import _read_bgr_unicode, _write_png_unicode  # noqa: E402

# Standard FFHQ 512x512 5-point canonical landmark template
FFHQ_TEMPLATE_5 = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.19360],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=np.float32,
)


def _build_512_blend_mask(base_weight: float, nostril_clean: bool = True) -> np.ndarray:
    yy, xx = np.mgrid[0:512, 0:512].astype(np.float32)
    # Smooth elliptical mask strictly inside 512x512 border (zero square seams)
    d = ((xx - 256.0) / 188.0) ** 2 + ((yy - 270.0) / 198.0) ** 2
    env = np.clip(1.65 * (1.0 - d), 0.0, 1.0)
    env = cv2.GaussianBlur(env, (0, 0), 24.0)
    # Zero outer 12px margin strictly so inverse warp has clean zero boundary
    edge_guard = np.ones((512, 512), dtype=np.float32)
    edge_guard[:16, :] = 0.0
    edge_guard[-16:, :] = 0.0
    edge_guard[:, :16] = 0.0
    edge_guard[:, -16:] = 0.0
    edge_guard = cv2.GaussianBlur(edge_guard, (0, 0), 6.0)
    w_map = np.clip(env * edge_guard * float(base_weight), 0.0, 1.0)

    if nostril_clean:
        # Boost GFPGAN weight right over left/right nostril rims below nose tip (256.6, 314.0)
        d_nos = ((xx - 256.5) / 38.0) ** 2 + ((yy - 327.0) / 16.0) ** 2
        m_nos = np.clip(1.0 - d_nos, 0.0, 1.0)
        m_nos = cv2.GaussianBlur(m_nos, (0, 0), 5.0)
        w_map = np.maximum(w_map, np.clip(m_nos * 0.88 * edge_guard, 0.0, 1.0))

    return w_map.astype(np.float32)


def run_face_restore(
    input_path: Path,
    out_path: Path,
    model_path: Path,
    uniface_cache: Path,
    weight: float = 0.50,
    faces_json_path: Path | None = None,
    nostril_clean: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    import onnxruntime as ort
    from uniface import RetinaFace, set_cache_dir  # type: ignore

    in_path = input_path.resolve()
    out_resolved = out_path.resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input image not found: {in_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing GFPGAN ONNX model: {model_path}")

    bgr = _read_bgr_unicode(in_path)
    h, w = bgr.shape[:2]

    lm5_list: list[np.ndarray] = []
    if faces_json_path is not None and faces_json_path.exists():
        fdat = json.loads(faces_json_path.read_text(encoding="utf-8"))
        for rec in fdat.get("faces", []):
            if rec.get("landmarks_5"):
                lm5_list.append(np.array(rec["landmarks_5"], dtype=np.float32))
    else:
        set_cache_dir(str(uniface_cache))
        detector = RetinaFace()
        faces = sorted(detector.detect(bgr), key=lambda f: float(f.confidence), reverse=True)
        for f in faces:
            lm5_list.append(np.array(f.landmarks, dtype=np.float32))

    if not lm5_list:
        _write_png_unicode(out_resolved, bgr, overwrite=overwrite)
        return {
            "ok": True,
            "input": str(in_path),
            "out": str(out_resolved),
            "faces_restored": 0,
            "warning": "no_face_detected",
        }

    avail = ort.get_available_providers()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in avail else ["CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(str(model_path), providers=providers)
    except Exception:
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    used_provider = sess.get_providers()[0] if sess.get_providers() else "unknown"
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    mask_512 = _build_512_blend_mask(base_weight=weight, nostril_clean=nostril_clean)
    canvas = bgr.astype(np.float32)

    restored_count = 0
    for lm5 in lm5_list:
        M, _ = cv2.estimateAffinePartial2D(lm5, FFHQ_TEMPLATE_5, method=cv2.LMEDS)
        if M is None:
            continue
        crop_bgr = cv2.warpAffine(bgr, M, (512, 512), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        blob = np.transpose(crop_rgb, (2, 0, 1))[None, ...].astype(np.float32)

        pred = sess.run([out_name], {in_name: blob})[0][0]
        pred_rgb = np.clip((np.transpose(pred, (1, 2, 0)) + 1.0) * 127.5, 0.0, 255.0).astype(np.float32)
        pred_bgr = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR)

        inv_pred = cv2.warpAffine(
            pred_bgr,
            M,
            (w, h),
            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        inv_mask = cv2.warpAffine(
            mask_512,
            M,
            (w, h),
            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )[:, :, None]

        canvas = canvas * (1.0 - inv_mask) + inv_pred * inv_mask
        restored_count += 1

    out_u8 = np.clip(canvas, 0, 255).astype(np.uint8)
    _write_png_unicode(out_resolved, out_u8, overwrite=overwrite)
    return {
        "ok": True,
        "input": str(in_path),
        "out": str(out_resolved),
        "width": int(w),
        "height": int(h),
        "faces_restored": restored_count,
        "weight": float(weight),
        "nostril_clean": bool(nostril_clean),
        "provider_used": used_provider,
    }
