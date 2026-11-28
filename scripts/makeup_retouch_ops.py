"""Pure-PC Optical Digital Makeup Engine v2 (PiXimperfect Multi-Blend + Blend-If + CIE Lab Texture Lock).

Zero AI face generation. Uses MediaPipe 478 FaceLandmarker geometry, optional BiSeNet face_parse masks,
PhotoCraft native `paint.stroke` brushes, and multi-layer optical blending (`Multiply`, `Soft Light`,
`Color`, `Screen`) + `layer.layerStyle.blendingOptions` Blend-If protection.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from pure_pc_retouch_ops import (
    _J,
    _detect_mediapipe_face478,
    _render_pc_strokes_multi,
    _write_png_u8,
    ensure_studio_brush_tips,
)
from workbench_ops import lab_to_rgb255, rgb255_to_lab

# Curated preset palettes (R, G, B 0..255) — restrained, photographically believable pigments
IRIS_PRESETS: dict[str, tuple[int, int, int] | None] = {
    "keep": None,
    "garnet": (139, 29, 53),
    "ruby-dark": (146, 26, 48),
    "violet": (110, 62, 148),
    "amber": (176, 118, 48),
    "emerald": (54, 128, 92),
    "sapphire": (52, 92, 158),
}

LIP_PRESETS: dict[str, tuple[int, int, int]] = {
    "mlbb": (176, 102, 110),
    "dusty-rose": (166, 86, 102),
    "berry-wine": (132, 46, 68),
    "peach-nude": (184, 114, 100),
    "crimson-velvet": (152, 40, 56),
}

EYESHADOW_PRESETS: dict[str, tuple[int, int, int]] = {
    "taupe": (118, 92, 86),
    "rosewood": (126, 76, 84),
    "smoky-plum": (108, 60, 82),
    "peach-brown": (140, 96, 78),
}

BLUSH_PRESETS: dict[str, tuple[int, int, int]] = {
    "soft-rose": (198, 118, 134),
    "muted-peach": (204, 134, 116),
    "cool-berry": (182, 104, 130),
}

STYLE_PRESETS: dict[str, dict[str, float | str]] = {
    "natural": {
        "lash_mode": "enhance",
        "sclera_lift": 0.55,
        "eyeshadow_depth": 0.28,
        "eyeshadow_tint": 0.28,
        "aegyo_sal": 0.30,
        "iris_depth": 0.42,
        "iris_tint": 0.48,
        "iris_caustic": 0.26,
        "tightline": 0.50,
        "lash_enhance": 0.55,
        "lash_wisp": 0.0,
        "blush": 0.28,
        "nose_db": 0.35,
        "conceal_db": 0.55,
        "lip_glaze": 0.42,
        "lip_depth": 0.42,
        "lip_spec": 0.36,
        "default_lip": "mlbb",
        "default_shadow": "taupe",
        "default_blush": "soft-rose",
        "psf_sigma": 0.90,
        "group_opacity": 0.72,
    },
    "editorial": {
        "lash_mode": "enhance",
        "sclera_lift": 0.65,
        "eyeshadow_depth": 0.42,
        "eyeshadow_tint": 0.40,
        "aegyo_sal": 0.36,
        "iris_depth": 0.54,
        "iris_tint": 0.58,
        "iris_caustic": 0.36,
        "tightline": 0.62,
        "lash_enhance": 0.65,
        "lash_wisp": 0.0,
        "blush": 0.36,
        "nose_db": 0.52,
        "conceal_db": 0.65,
        "lip_glaze": 0.54,
        "lip_depth": 0.55,
        "lip_spec": 0.45,
        "default_lip": "dusty-rose",
        "default_shadow": "rosewood",
        "default_blush": "soft-rose",
        "psf_sigma": 0.85,
        "group_opacity": 0.80,
    },
    "cos-anime": {
        "lash_mode": "subtle-wisp",
        "sclera_lift": 0.68,
        "eyeshadow_depth": 0.48,
        "eyeshadow_tint": 0.46,
        "aegyo_sal": 0.50,
        "iris_depth": 0.62,
        "iris_tint": 0.66,
        "iris_caustic": 0.44,
        "tightline": 0.72,
        "lash_enhance": 0.68,
        "lash_wisp": 0.52,
        "blush": 0.38,
        "nose_db": 0.48,
        "conceal_db": 0.65,
        "lip_glaze": 0.56,
        "lip_depth": 0.58,
        "lip_spec": 0.50,
        "default_lip": "dusty-rose",
        "default_shadow": "smoky-plum",
        "default_blush": "cool-berry",
        "psf_sigma": 0.80,
        "group_opacity": 0.82,
    },
    "painterly": {
        "lash_mode": "subtle-wisp",
        "sclera_lift": 0.70,
        "eyeshadow_depth": 0.52,
        "eyeshadow_tint": 0.50,
        "aegyo_sal": 0.54,
        "iris_depth": 0.68,
        "iris_tint": 0.72,
        "iris_caustic": 0.52,
        "tightline": 0.74,
        "lash_enhance": 0.68,
        "lash_wisp": 0.56,
        "blush": 0.40,
        "nose_db": 0.56,
        "conceal_db": 0.68,
        "lip_glaze": 0.60,
        "lip_depth": 0.62,
        "lip_spec": 0.56,
        "default_lip": "berry-wine",
        "default_shadow": "smoky-plum",
        "default_blush": "cool-berry",
        "psf_sigma": 0.75,
        "group_opacity": 0.86,
    },
}


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0 + 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _read_mask_roi(
    path: Path | None,
    size_wh: tuple[int, int],
    rx0: int,
    ry0: int,
    rx1: int,
    ry1: int,
) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    im = Image.open(path).convert("L")
    if im.size != size_wh:
        im = im.resize(size_wh, Image.Resampling.BILINEAR)
    arr = np.array(im, dtype=np.float32) / 255.0
    return arr[ry0:ry1, rx0:rx1]


def _parse_hex_rgb(spec: str) -> tuple[int, int, int] | None:
    s = spec.strip().lstrip("#")
    if len(s) == 6:
        try:
            return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
        except ValueError:
            return None
    return None


def _resolve_tone(
    spec: str | None,
    presets: dict[str, tuple[int, int, int] | None],
    fallback_rgb: tuple[int, int, int],
) -> tuple[int, int, int] | None:
    if spec is None:
        return fallback_rgb
    k = spec.strip().lower()
    if k in ("auto", ""):
        return fallback_rgb
    if k in presets:
        return presets[k]
    hx = _parse_hex_rgb(spec)
    if hx is not None:
        return hx
    raise ValueError(f"Unsupported color/preset '{spec}'. Expected #rrggbb or one of {sorted(presets.keys())}")


def _harmonize_rgb(rgb: tuple[int, int, int], max_sat_scale: float = 0.88) -> np.ndarray:
    """Pull overly neon input hues softly into realistic cosmetic pigment range."""
    arr = np.array(rgb, dtype=np.float32).reshape(1, 1, 3)
    lab = rgb255_to_lab(arr)
    lab[0, 0, 1] *= max_sat_scale
    lab[0, 0, 2] *= max_sat_scale
    out = np.clip(lab_to_rgb255(lab), 0.0, 255.0).reshape(3)
    return out


def _draw_poly_band(canvas: np.ndarray, pts_xy: np.ndarray, radius: float, value: float = 1.0) -> None:
    tmp = np.zeros_like(canvas)
    thick = max(1, int(round(radius * 2.0)))
    i_pts = np.round(pts_xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(tmp, [i_pts], isClosed=False, color=float(value), thickness=thick, lineType=cv2.LINE_AA)
    canvas[:] = np.maximum(canvas, tmp)


def _save_roi_rgba(
    out_path: Path,
    w: int,
    h: int,
    rx0: int,
    ry0: int,
    rx1: int,
    ry1: int,
    rgb_roi: np.ndarray,
    a_roi: np.ndarray,
    neutral_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
    overwrite: bool = False,
) -> None:
    full = np.zeros((h, w, 4), dtype=np.float32)
    full[:, :, 0] = neutral_rgb[0]
    full[:, :, 1] = neutral_rgb[1]
    full[:, :, 2] = neutral_rgb[2]
    a_c = np.clip(a_roi, 0.0, 1.0)
    # Smoothly transition RGB -> neutral_rgb over low-alpha tails (0..0.02) to eliminate hard binary color rims
    mix_rgb_w = np.clip(a_c / 0.02, 0.0, 1.0)[:, :, None]
    neut = np.array(neutral_rgb, dtype=np.float32)[None, None, :]
    clean_rgb = np.clip(neut * (1.0 - mix_rgb_w) + np.clip(rgb_roi, 0.0, 255.0) * mix_rgb_w, 0.0, 255.0)
    full[ry0:ry1, rx0:rx1, :3] = clean_rgb
    full[ry0:ry1, rx0:rx1, 3] = a_c * 255.0
    _write_png_u8(out_path, full, "RGBA", overwrite=overwrite)


def _blendif_weight(bg_01: np.ndarray, blend_if: dict[str, Any] | None) -> np.ndarray:
    """Simulate Photoshop/PhotoCraft Blend-If Underlying Layer split ramp [b0, b1, w0, w1]."""
    if not blend_if or "underlying" not in blend_if:
        return np.ones(bg_01.shape[:2], dtype=np.float32)
    u = [float(v) / 255.0 for v in blend_if["underlying"]]
    b0, b1, w0, w1 = (u[0], u[1], u[2], u[3]) if len(u) == 4 else (u[0], u[0], u[1], u[1])
    luma = 0.2126 * bg_01[:, :, 0] + 0.7152 * bg_01[:, :, 1] + 0.0722 * bg_01[:, :, 2]
    w_lo = np.ones_like(luma) if b1 <= b0 + 1e-5 else np.clip((luma - b0) / (b1 - b0), 0.0, 1.0)
    w_hi = np.ones_like(luma) if w1 <= w0 + 1e-5 else np.clip((w1 - luma) / (w1 - w0), 0.0, 1.0)
    return np.clip(np.minimum(w_lo, w_hi), 0.0, 1.0)


def _composite_optical_layer(
    bg_01: np.ndarray,
    fg_rgba_01: np.ndarray,
    blend: str,
    opacity: float,
    blend_if: dict[str, Any] | None = None,
) -> np.ndarray:
    """Exact optical blend-mode + Blend-If compositor for accurate preview PNG generation."""
    fg = fg_rgba_01[:, :, :3]
    alpha = np.clip(fg_rgba_01[:, :, 3] * float(opacity), 0.0, 1.0)
    alpha *= _blendif_weight(bg_01, blend_if)

    b_low = blend.strip().lower()
    if b_low == "multiply":
        blended = bg_01 * fg
    elif b_low == "screen":
        blended = 1.0 - (1.0 - bg_01) * (1.0 - fg)
    elif b_low == "soft light":
        blended = np.where(
            fg <= 0.5,
            bg_01 - (1.0 - 2.0 * fg) * bg_01 * (1.0 - bg_01),
            bg_01 + (2.0 * fg - 1.0) * (np.sqrt(np.maximum(bg_01, 1e-5)) - bg_01),
        )
    elif b_low == "color":
        # Luminance-preserving CIE Lab color transfer matching Photoshop Color behavior smoothly
        bg_lab = rgb255_to_lab(np.clip(bg_01 * 255.0, 0.0, 255.0))
        fg_lab = rgb255_to_lab(np.clip(fg * 255.0, 0.0, 255.0))
        out_lab = np.dstack([bg_lab[:, :, 0], fg_lab[:, :, 1], fg_lab[:, :, 2]])
        blended = np.clip(lab_to_rgb255(out_lab) / 255.0, 0.0, 1.0)
    else:
        blended = fg

    return np.clip(bg_01 * (1.0 - alpha[:, :, None]) + blended * alpha[:, :, None], 0.0, 1.0)


def build_makeup_pcraft_cmds(
    meta: dict[str, Any],
    *,
    width: int,
    height: int,
    group_name: str = "MAKEUP_OPTICAL_GROUP",
    after_layer: str | None = None,
    group_opacity: float | None = None,
) -> list[str]:
    """Build PhotoCraft CLI command args placing all generated optical layers with native Blend-If."""
    layers = [l for l in meta.get("layers", []) if Path(l["path"]).exists()]
    if not layers:
        return []

    g_op = float(group_opacity if group_opacity is not None else meta.get("group_opacity", 0.80))
    cmds: list[str] = [
        "--cmd",
        "layer.new.group",
        "--params",
        _J({"name": group_name, "mode": "passThrough", "opacity": round(g_op * 100.0, 1)}),
    ]
    if after_layer:
        cmds += ["--cmd", "layer.moveTo", "--params", _J({"target": after_layer, "position": "above"})]

    prev_layer: str | None = None
    for idx, lay in enumerate(layers):
        lname = str(lay["layer_name"])
        blend = str(lay["blend"])
        op = float(lay["opacity"])
        bif = lay.get("blend_if")
        cmds += [
            "--cmd",
            "file.placeEmbedded",
            "--params",
            _J({"path": str(lay["path"]), "fit": False, "smartObject": True, "center": [width / 2.0, height / 2.0]}),
            "--cmd",
            "layer.setProps",
            "--params",
            _J({"name": lname, "blend": blend, "opacity": round(op, 3), "visible": True}),
        ]
        if bif is not None:
            cmds += [
                "--cmd",
                "layer.layerStyle.blendingOptions",
                "--params",
                _J({"blendIf": bif}),
            ]
        if idx == 0:
            cmds += ["--cmd", "layer.moveTo", "--params", _J({"target": group_name, "position": "into"})]
        elif prev_layer is not None:
            cmds += ["--cmd", "layer.moveTo", "--params", _J({"target": prev_layer, "position": "above"})]
        prev_layer = lname
    return cmds


def run_makeup_paint_v2(
    *,
    input_path: Path,
    out_dir: Path,
    pc_exe: str,
    face_task_path: Path,
    faces_json_path: Path | None = None,
    face_parse_dir: Path | None = None,
    style: str = "natural",
    intensity: float = 0.72,
    iris_color: str | None = None,
    lip_tone: str | None = None,
    eyeshadow_tone: str | None = None,
    blush_tone: str | None = None,
    lash_mode: str = "auto",
    aegyo_sal: float | None = None,
    skip_nose_db: bool = False,
    build_pcraft: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate 9+ non-destructive optical makeup layers with PhotoCraft Blend-If and realistic anatomy."""
    st_key = style.strip().lower()
    if st_key not in STYLE_PRESETS:
        raise ValueError(f"Unsupported makeup style '{style}'. Expected one of {sorted(STYLE_PRESETS.keys())}")
    cfg = STYLE_PRESETS[st_key]

    in_p = input_path.resolve()
    out_d = out_dir.resolve()
    out_d.mkdir(parents=True, exist_ok=True)

    rgb_u8 = np.array(Image.open(in_p).convert("RGB"), dtype=np.uint8)
    rgb_f32 = rgb_u8.astype(np.float32)
    h, w = rgb_u8.shape[:2]
    tips = ensure_studio_brush_tips()

    pts = _detect_mediapipe_face478(rgb_u8, face_task_path, faces_json_path)
    if pts is None or len(pts) < 478:
        raise RuntimeError(f"FaceLandmarker failed to detect 478 landmarks in {in_p}")

    # Auto-discover face_parse_dir alongside faces_json_path if omitted
    fp_dir = face_parse_dir.resolve() if face_parse_dir is not None else None
    if fp_dir is None and faces_json_path is not None:
        cand = faces_json_path.resolve().parent
        if (cand / "mask_skin.png").exists():
            fp_dir = cand

    # Orthonormal face coordinate frame (tracks tilted poses cleanly!)
    u_right = pts[263] - pts[33]
    u_right = u_right / max(1e-5, float(np.linalg.norm(u_right)))
    u_down = np.array([-u_right[1], u_right[0]], dtype=np.float32)

    l_w = float(np.linalg.norm(pts[133] - pts[33]))
    r_w = float(np.linalg.norm(pts[263] - pts[362]))
    ew = 0.5 * (l_w + r_w)
    l_h = float(np.linalg.norm(pts[159] - pts[145]))
    r_h = float(np.linalg.norm(pts[386] - pts[374]))
    ear_avg = 0.5 * ((l_h / max(1.0, l_w)) + (r_h / max(1.0, r_w)))
    eyes_open = ear_avg >= 0.225
    eye_span = float(np.linalg.norm(pts[263] - pts[33]))

    # Head ROI bounding box for fast, sub-pixel exact vectorized optics
    face_oval_idx = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    f_pts = pts[:468]
    pad_roi = max(48, int(round(eye_span * 0.80)))
    rx0 = max(0, int(math.floor(float(f_pts[:, 0].min()))) - pad_roi)
    ry0 = max(0, int(math.floor(float(f_pts[:, 1].min()))) - pad_roi)
    rx1 = min(w, int(math.ceil(float(f_pts[:, 0].max()))) + pad_roi)
    ry1 = min(h, int(math.ceil(float(f_pts[:, 1].max()))) + pad_roi)
    rh, rw = ry1 - ry0, rx1 - rx0
    ryy, rxx = np.mgrid[ry0:ry1, rx0:rx1].astype(np.float32)

    r_rgb = rgb_f32[ry0:ry1, rx0:rx1]
    r_lab = rgb255_to_lab(r_rgb)
    L_orig = r_lab[:, :, 0]

    def _roi_gauss(p_xy: np.ndarray, sx: float, sy: float) -> np.ndarray:
        dx = rxx - float(p_xy[0])
        dy = ryy - float(p_xy[1])
        lx = dx * float(u_right[0]) + dy * float(u_right[1])
        ly = dx * float(u_down[0]) + dy * float(u_down[1])
        return np.exp(-0.5 * ((lx / max(1.0, sx)) ** 2 + (ly / max(1.0, sy)) ** 2))

    # Load BiSeNet face_parse masks inside ROI or construct clean anatomical fallbacks
    m_skin_raw = _read_mask_roi(fp_dir / "mask_skin.png" if fp_dir else None, (w, h), rx0, ry0, rx1, ry1)
    m_lips_raw = _read_mask_roi(fp_dir / "mask_lips.png" if fp_dir else None, (w, h), rx0, ry0, rx1, ry1)
    m_hair_raw = _read_mask_roi(fp_dir / "mask_hair_head.png" if fp_dir else None, (w, h), rx0, ry0, rx1, ry1)

    if m_skin_raw is None:
        m_skin_raw = np.zeros((rh, rw), dtype=np.float32)
        cv2.fillPoly(m_skin_raw, [np.round(pts[face_oval_idx] - np.array([rx0, ry0])).astype(np.int32)], 1.0)
        m_skin_raw = cv2.GaussianBlur(m_skin_raw, (0, 0), max(3.0, ew * 0.04))

    if m_hair_raw is not None:
        bangs_solid = cv2.erode((m_hair_raw > 0.44).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(np.float32)
        bangs_guard = np.clip(cv2.GaussianBlur(bangs_solid, (0, 0), 2.5), 0.0, 1.0)
        bangs_lash_solid = cv2.erode((m_hair_raw > 0.48).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))).astype(np.float32)
        bangs_lash_guard = np.clip(cv2.GaussianBlur(bangs_lash_solid, (0, 0), 2.0), 0.0, 1.0)
    else:
        bangs_guard = np.zeros((rh, rw), dtype=np.float32)
        bangs_lash_guard = np.zeros((rh, rw), dtype=np.float32)

    # Exact eye aperture polygons
    l_eye_ring = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
    r_eye_ring = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
    eye_ap_roi = np.zeros((rh, rw), dtype=np.float32)
    for ring in (l_eye_ring, r_eye_ring):
        cv2.fillPoly(eye_ap_roi, [np.round(pts[ring] - np.array([rx0, ry0])).astype(np.int32)], 1.0)
    eye_ap_soft = np.clip(cv2.GaussianBlur(eye_ap_roi, (0, 0), max(1.2, ew * 0.015)), 0.0, 1.0)
    eye_ap_iris = cv2.erode((eye_ap_roi > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).astype(np.float32)
    eye_ap_iris = np.clip(cv2.GaussianBlur(eye_ap_iris, (0, 0), max(1.0, ew * 0.011)), 0.0, 1.0)

    # Exact zero-spill lip mask: MediaPipe outer poly intersected with dilated face_parse lips
    outer_lip_idx = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
    lip_poly = np.zeros((rh, rw), dtype=np.float32)
    cv2.fillPoly(lip_poly, [np.round(pts[outer_lip_idx] - np.array([rx0, ry0])).astype(np.int32)], 1.0)
    lip_feather_sigma = max(1.6, ew * 0.019)
    lip_poly_soft = np.clip(cv2.GaussianBlur(lip_poly, (0, 0), lip_feather_sigma), 0.0, 1.0)
    if m_lips_raw is not None:
        k_dil = max(5, int(round(ew * 0.08)) | 1)
        flips_dil = cv2.dilate((m_lips_raw > 0.16).astype(np.float32), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_dil, k_dil)))
        flips_dil = np.clip(cv2.GaussianBlur(flips_dil, (0, 0), lip_feather_sigma), 0.0, 1.0)
        lip_mask_strict = lip_poly_soft * flips_dil
    else:
        lip_mask_strict = lip_poly_soft

    # Resolve palette colors harmonized gently for natural photography / COS / 2.5D
    k_i = float(np.clip(intensity, 0.10, 1.35))
    iris_rgb_t = _resolve_tone(iris_color, IRIS_PRESETS, None) if iris_color is not None else None

    lip_key = str(cfg["default_lip"])
    lip_rgb_t = _harmonize_rgb(_resolve_tone(lip_tone, LIP_PRESETS, LIP_PRESETS[lip_key]), 0.84)  # type: ignore[arg-type]

    sh_key = str(cfg["default_shadow"])
    shadow_rgb_t = _harmonize_rgb(_resolve_tone(eyeshadow_tone, EYESHADOW_PRESETS, EYESHADOW_PRESETS[sh_key]), 0.80)  # type: ignore[arg-type]

    bl_key = str(cfg["default_blush"])
    blush_rgb_t = _harmonize_rgb(_resolve_tone(blush_tone, BLUSH_PRESETS, BLUSH_PRESETS[bl_key]), 0.80)  # type: ignore[arg-type]

    l_mode = str(cfg["lash_mode"]) if lash_mode == "auto" else lash_mode.strip().lower()
    aegyo_strength = float(cfg["aegyo_sal"]) if aegyo_sal is None else float(np.clip(aegyo_sal, 0.0, 1.0))
    psf_sig = float(cfg["psf_sigma"]) * max(0.75, ew / 95.0)

    layer_specs: list[dict[str, Any]] = []

    # =========================================================================
    # LAYER 1: Data-Driven Tear-Trough/Crease Lift & Wide Soft Nose D&B (Soft Light)
    # =========================================================================
    skin_guard = m_skin_raw * (1.0 - eye_ap_soft) * (1.0 - lip_mask_strict) * (1.0 - bangs_guard)
    sk_blur_sig = max(18.0, ew * 0.36)
    sk_w = np.clip(cv2.GaussianBlur(skin_guard, (0, 0), sk_blur_sig), 1e-4, 1.0)
    L_cheek_ref = cv2.GaussianBlur(L_orig * skin_guard, (0, 0), sk_blur_sig) / sk_w
    L_lo = cv2.GaussianBlur(L_orig, (0, 0), max(4.0, ew * 0.055))
    dark_deficit = np.clip((L_cheek_ref - L_lo - 1.8) / 14.0, 0.0, 1.0)

    ue_l = _roi_gauss(pts[468] + 0.25 * ew * u_down, 0.48 * l_w, 0.14 * l_w)
    ue_r = _roi_gauss(pts[473] + 0.25 * ew * u_down, 0.48 * r_w, 0.14 * r_w)
    tear_trough = np.clip(np.maximum(ue_l, ue_r), 0.0, 1.0)
    mouth_c = np.maximum(_roi_gauss(pts[61], ew * 0.12, ew * 0.10), _roi_gauss(pts[291], ew * 0.12, ew * 0.10))
    naso_l = _roi_gauss(0.5 * (pts[102] + pts[203]), ew * 0.14, ew * 0.18)
    naso_r = _roi_gauss(0.5 * (pts[331] + pts[423]), ew * 0.14, ew * 0.18)

    # Strictly lift only genuinely darker pixels (never creates artificial circular blobs!)
    conceal_lift = (
        np.clip(tear_trough * 0.90 + mouth_c * 0.65 + np.maximum(naso_l, naso_r) * 0.50, 0.0, 1.0)
        * (0.35 + 0.65 * dark_deficit)
        * skin_guard
    )

    db_roi = np.full((rh, rw, 3), 128.0, dtype=np.float32)
    db_hi_w = conceal_lift * 22.0 * float(cfg["conceal_db"])
    db_sh_w = np.zeros((rh, rw), dtype=np.float32)

    if not skip_nose_db:
        ndb_s = float(cfg["nose_db"]) * k_i
        # Soft nose bridge shadow ONLY from brow head to mid-bridge (stops well above nostrils!)
        nb_mid = np.vstack([pts[168], pts[6], pts[197], pts[195]])
        nb_sh_l = np.zeros((rh, rw), dtype=np.float32)
        nb_sh_r = np.zeros((rh, rw), dtype=np.float32)
        _draw_poly_band(nb_sh_l, (nb_mid - 0.15 * ew * u_right) - np.array([rx0, ry0]), radius=max(3.5, ew * 0.048), value=1.0)
        _draw_poly_band(nb_sh_r, (nb_mid + 0.15 * ew * u_right) - np.array([rx0, ry0]), radius=max(3.5, ew * 0.048), value=1.0)
        # Taper downward cleanly before nostrils
        nb_sh_mask = cv2.GaussianBlur(np.maximum(nb_sh_l, nb_sh_r), (0, 0), max(5.5, ew * 0.088))
        # Wide organic bridge-to-tip gentle lift (ZERO chalk '!' bar or pinpoint clown dot!)
        nb_full = np.vstack([pts[168], pts[6], pts[197], pts[195], pts[5], pts[4]]) - np.array([rx0, ry0])
        nb_hi = np.zeros((rh, rw), dtype=np.float32)
        _draw_poly_band(nb_hi, nb_full, radius=max(2.6, ew * 0.036), value=1.0)
        nb_hi = cv2.GaussianBlur(nb_hi, (0, 0), max(4.2, ew * 0.056))
        ntip_soft = _roi_gauss(pts[1] - 0.03 * ew * u_down, ew * 0.11, ew * 0.09)
        cupid_soft = _roi_gauss(pts[0] - 0.045 * ew * u_down, ew * 0.11, ew * 0.045)
        db_hi_w = np.clip(db_hi_w + (0.45 * nb_hi + 0.55 * ntip_soft + 0.40 * cupid_soft) * 18.0 * ndb_s * skin_guard, 0.0, 34.0)
        db_sh_w = np.clip(nb_sh_mask * 16.0 * ndb_s * skin_guard, 0.0, 22.0)

    # Warm porcelain highlight vs cool-taupe shadow (neutral 128 base)
    db_roi += db_hi_w[:, :, None] * np.array([1.05, 1.00, 0.94], dtype=np.float32)[None, None, :]
    db_roi -= db_sh_w[:, :, None] * np.array([0.92, 1.02, 0.98], dtype=np.float32)[None, None, :]
    db_a = np.clip(np.maximum(conceal_lift, db_sh_w / 18.0) * 0.92 + (db_hi_w / 26.0) * 0.70, 0.0, 1.0) * skin_guard
    p_l1 = out_d / "makeup_01_conceal_db_softlight.png"
    _save_roi_rgba(p_l1, w, h, rx0, ry0, rx1, ry1, db_roi, db_a, neutral_rgb=(128.0, 128.0, 128.0), overwrite=overwrite)
    layer_specs.append({
        "key": "conceal_db",
        "path": str(p_l1),
        "layer_name": "01_Makeup_Conceal_SoftDB",
        "blend": "Soft Light",
        "opacity": round(min(0.85, 0.76 * k_i), 3),
        "blend_if": {"channel": "gray", "underlying": [18, 60, 212, 248]},
    })

    # =========================================================================
    # LAYER 2: Restrained Spherical Sclera Clean (Normal)
    # =========================================================================
    iris_excl = np.zeros((rh, rw), dtype=np.float32)
    sclera_sphere = np.zeros((rh, rw), dtype=np.float32)
    for iris_idx, w_e, c_in, c_out in ((468, l_w, pts[133], pts[33]), (473, r_w, pts[362], pts[263])):
        ix, iy = float(pts[iris_idx][0]), float(pts[iris_idx][1])
        rad = w_e * 0.258
        dist = np.hypot(rxx - ix, ryy - iy)
        iris_excl = np.maximum(iris_excl, np.clip((rad * 1.06 - dist) / max(2.2, rad * 0.14), 0.0, 1.0))
        # Spherical falloff: protects inner/outer canthi and upper-lid corner shadow
        mid_e = 0.5 * (c_in + c_out)
        fy_e = (rxx - float(mid_e[0])) * float(u_down[0]) + (ryy - float(mid_e[1])) * float(u_down[1])
        sph = _roi_gauss(mid_e + 0.02 * w_e * u_down, 0.40 * w_e, 0.16 * w_e) * _smoothstep(-0.18 * w_e, 0.04 * w_e, fy_e)
        sclera_sphere = np.maximum(sclera_sphere, sph)

    sclera_m = np.clip(eye_ap_iris * (1.0 - iris_excl) * (0.28 + 0.72 * sclera_sphere), 0.0, 1.0)
    sc_lab = r_lab.copy()
    sc_lift = 5.5 * float(cfg["sclera_lift"]) * k_i
    sc_lab[:, :, 0] = np.clip(np.minimum(sc_lab[:, :, 0] + sc_lift, 87.5), sc_lab[:, :, 0], 92.0)
    sc_lab[:, :, 1] = sc_lab[:, :, 1] * 0.45
    sc_lab[:, :, 2] = sc_lab[:, :, 2] * 0.62 - 0.8
    sc_rgb = np.clip(lab_to_rgb255(sc_lab), 0.0, 255.0)
    p_l2 = out_d / "makeup_02_sclera_clean_normal.png"
    _save_roi_rgba(p_l2, w, h, rx0, ry0, rx1, ry1, sc_rgb, sclera_m, neutral_rgb=(215.0, 215.0, 215.0), overwrite=overwrite)
    layer_specs.append({
        "key": "sclera_clean",
        "path": str(p_l2),
        "layer_name": "02_Makeup_Sclera_Clean",
        "blend": "Normal",
        "opacity": round(min(0.70, 0.56 * k_i), 3),
        "blend_if": None,
    })

    # =========================================================================
    # LAYER 3 & 4: Anatomical Eyeshadow Depth (Multiply) + Sheer Tint / Aegyo (Soft Light)
    # =========================================================================
    sh_depth_col = np.clip(shadow_rgb_t * 0.56, 20.0, 110.0)
    sh_tint_col = np.clip(shadow_rgb_t * 1.12, 40.0, 225.0)
    # Warm champagne skin-harmonized aegyo highlight (sampled near subject cheek warmth, NEVER pure white!)
    sh_aegyo_col = np.array([242.0, 222.0, 208.0], dtype=np.float32)

    es_d_rgb = np.full((rh, rw, 3), 255.0, dtype=np.float32)
    es_d_a = np.zeros((rh, rw), dtype=np.float32)
    es_t_rgb = np.full((rh, rw, 3), 128.0, dtype=np.float32)
    es_t_a = np.zeros((rh, rw), dtype=np.float32)

    lid_guard = (1.0 - eye_ap_soft) * (1.0 - bangs_guard)
    for c_out, c_in, w_e, sgn in ((pts[33], pts[133], l_w, -1.0), (pts[263], pts[362], r_w, +1.0)):
        eye_mid = 0.5 * (c_out + c_in)
        outer_v = c_out + sgn * 0.04 * w_e * u_right - 0.02 * w_e * u_down
        # Upper orbital crease + outer V depth
        w_up_d = _roi_gauss(eye_mid - 0.13 * w_e * u_down, 0.44 * w_e, 0.13 * w_e) * 0.58
        w_ov_d = _roi_gauss(outer_v, 0.23 * w_e, 0.11 * w_e) * 0.72
        # Ultra-thin lower-outer lash root tuck (<= 0.042 * w_e, strictly tucked against outer 1/3 lashline!)
        w_lo_d = _roi_gauss(c_out - sgn * 0.16 * w_e * u_right + 0.032 * w_e * u_down, 0.22 * w_e, 0.038 * w_e) * 0.54
        d_tot = np.clip(np.maximum.reduce([w_up_d, w_ov_d, w_lo_d]), 0.0, 1.0)
        es_d_rgb = es_d_rgb * (1.0 - d_tot[:, :, None]) + sh_depth_col[None, None, :] * d_tot[:, :, None]
        es_d_a = np.maximum(es_d_a, d_tot)

        # Center-only whisper-faint aegyo-sal crease shadow (fades cleanly to 0 at both corners)
        if aegyo_strength > 0.05:
            cr_pts = np.vstack([
                eye_mid - sgn * 0.24 * w_e * u_right + 0.135 * w_e * u_down,
                eye_mid + 0.165 * w_e * u_down,
                eye_mid + sgn * 0.24 * w_e * u_right + 0.145 * w_e * u_down,
            ]) - np.array([rx0, ry0])
            cr_b = np.zeros((rh, rw), dtype=np.float32)
            _draw_poly_band(cr_b, cr_pts, radius=max(1.0, w_e * 0.011), value=1.0)
            # Central envelope tapers crease ends cleanly to 0
            cr_env = _roi_gauss(eye_mid + 0.155 * w_e * u_down, 0.22 * w_e, 0.055 * w_e)
            cr_b = cv2.GaussianBlur(cr_b, (0, 0), max(2.2, w_e * 0.025)) * cr_env * 0.26 * aegyo_strength
            taupe_cr = np.array([124.0, 102.0, 98.0], dtype=np.float32)
            es_d_rgb = es_d_rgb * (1.0 - cr_b[:, :, None]) + taupe_cr[None, None, :] * cr_b[:, :, None]
            es_d_a = np.maximum(es_d_a, cr_b)

        # Upper-lid sheer tint & warm satin central lid/aegyo lift
        w_tint = np.clip(np.maximum(_roi_gauss(eye_mid - 0.10 * w_e * u_down, 0.48 * w_e, 0.15 * w_e) * 0.72, _roi_gauss(outer_v, 0.26 * w_e, 0.12 * w_e) * 0.80), 0.0, 1.0)
        es_t_rgb = es_t_rgb * (1.0 - w_tint[:, :, None]) + sh_tint_col[None, None, :] * w_tint[:, :, None]
        es_t_a = np.maximum(es_t_a, w_tint)

        if aegyo_strength > 0.05:
            w_aeg = _roi_gauss(eye_mid + 0.10 * w_e * u_down, 0.28 * w_e, 0.048 * w_e) * 0.56 * aegyo_strength
            w_in = _roi_gauss(c_in, 0.09 * w_e, 0.07 * w_e) * 0.42 * aegyo_strength
            w_sh = np.clip(np.maximum(w_aeg, w_in), 0.0, 1.0)
            es_t_rgb = es_t_rgb * (1.0 - w_sh[:, :, None]) + sh_aegyo_col[None, None, :] * w_sh[:, :, None]
            es_t_a = np.maximum(es_t_a, w_sh)

    es_d_a *= lid_guard
    es_t_a *= lid_guard
    p_l3 = out_d / "makeup_03_eyeshadow_depth_multiply.png"
    p_l4 = out_d / "makeup_04_eyeshadow_tint_softlight.png"
    _save_roi_rgba(p_l3, w, h, rx0, ry0, rx1, ry1, es_d_rgb, es_d_a, neutral_rgb=(255.0, 255.0, 255.0), overwrite=overwrite)
    _save_roi_rgba(p_l4, w, h, rx0, ry0, rx1, ry1, es_t_rgb, es_t_a, neutral_rgb=(128.0, 128.0, 128.0), overwrite=overwrite)
    layer_specs.append({
        "key": "eyeshadow_depth",
        "path": str(p_l3),
        "layer_name": "03_Makeup_Eyeshadow_Depth",
        "blend": "Multiply",
        "opacity": round(min(0.68, float(cfg["eyeshadow_depth"]) * k_i), 3),
        "blend_if": {"channel": "gray", "underlying": [18, 65, 220, 255]},
    })
    layer_specs.append({
        "key": "eyeshadow_tint",
        "path": str(p_l4),
        "layer_name": "04_Makeup_Eyeshadow_Tint",
        "blend": "Soft Light",
        "opacity": round(min(0.68, float(cfg["eyeshadow_tint"]) * k_i), 3),
        "blend_if": {"channel": "gray", "underlying": [22, 68, 210, 248]},
    })

    # =========================================================================
    # LAYER 5 & 6a/6b: Optical Iris Depth (Multiply), Tint (Color+BlendIf), Caustic (Soft Light)
    # =========================================================================
    iris_d_rgb = np.full((rh, rw, 3), 255.0, dtype=np.float32)
    iris_d_a = np.zeros((rh, rw), dtype=np.float32)
    iris_t_rgb = np.full((rh, rw, 3), 128.0, dtype=np.float32)
    iris_t_a = np.zeros((rh, rw), dtype=np.float32)
    iris_c_rgb = np.full((rh, rw, 3), 128.0, dtype=np.float32)
    iris_c_a = np.zeros((rh, rw), dtype=np.float32)

    if eyes_open:
        iris_guard = eye_ap_iris * (1.0 - bangs_lash_guard)
        # Protect real corneal catchlights in underlying eye (L > 64)
        catch_protect = np.clip((L_orig - 58.0) / 14.0, 0.0, 1.0)
        hi_micro = np.clip(L_orig - cv2.GaussianBlur(L_orig, (0, 0), max(1.6, ew * 0.025)), -6.0, 8.0)

        target_iris_h = _harmonize_rgb(iris_rgb_t, 0.90) if iris_rgb_t is not None else None
        caustic_base = np.clip(target_iris_h * 1.38 if target_iris_h is not None else np.array([218.0, 198.0, 192.0], dtype=np.float32), 70.0, 246.0)
        limbal_tint = np.clip(target_iris_h * 0.34 if target_iris_h is not None else np.array([34.0, 26.0, 28.0], dtype=np.float32), 14.0, 72.0)

        for iris_idx, w_e in ((468, l_w), (473, r_w)):
            ix, iy = float(pts[iris_idx][0]), float(pts[iris_idx][1])
            rad = w_e * 0.252
            dx, dy = rxx - ix, ryy - iy
            fx = dx * float(u_right[0]) + dy * float(u_right[1])
            fy = dx * float(u_down[0]) + dy * float(u_down[1])
            dist = np.hypot(fx, fy)
            # Soft outer iris transition (~2.8-3.5px)
            disk = np.clip((rad * 1.02 - dist) / max(2.4, rad * 0.16), 0.0, 1.0) * iris_guard
            if disk.max() < 0.01:
                continue

            # 1) Recessed upper-lid cast shadow (top 32% of iris) + soft limbal ring + pupil center
            top_sh = _smoothstep(rad * 0.12, -rad * 0.44, fy) * disk
            limbal = _smoothstep(rad * 0.68, rad * 0.96, dist) * disk
            pupil = np.clip((rad * 0.33 - dist) / max(1.6, rad * 0.12), 0.0, 1.0) * disk * (1.0 - catch_protect)

            d_w = np.clip(np.maximum.reduce([top_sh * 0.72, limbal * 0.66, pupil * 0.74]), 0.0, 1.0) * (1.0 - 0.88 * catch_protect)
            d_col = limbal_tint * (1.0 - pupil * 0.55)[:, :, None]
            iris_d_rgb = iris_d_rgb * (1.0 - d_w[:, :, None]) + d_col * d_w[:, :, None]
            iris_d_a = np.maximum(iris_d_a, d_w)

            # 2) Pure Color shift preserving iris luminance & shadow
            if target_iris_h is not None:
                ring_zone = _smoothstep(rad * 0.22, rad * 0.42, dist) * disk * (1.0 - 0.85 * catch_protect)
                t_col = np.clip(target_iris_h, 16.0, 228.0)
                iris_t_rgb = iris_t_rgb * (1.0 - ring_zone[:, :, None]) + t_col[None, None, :] * ring_zone[:, :, None]
                iris_t_a = np.maximum(iris_t_a, ring_zone)

            # 3) Translucent lower-crescent optical refraction + real catchlight polish (Soft Light)
            lower_arc = _smoothstep(-rad * 0.05, rad * 0.48, fy) * _smoothstep(rad * 0.92, rad * 0.55, dist) * _smoothstep(rad * 0.28, rad * 0.45, dist) * disk * 0.68
            # Boost genuine catchlights detected inside raw iris
            in_disk_catch = catch_protect * np.clip((rad * 0.85 - dist) / max(1.5, rad * 0.1), 0.0, 1.0) * iris_guard
            max_orig_l = float((L_orig * disk).max()) if disk.max() > 0.2 else 80.0
            c_tot = np.clip(np.maximum(lower_arc, in_disk_catch * 0.72), 0.0, 1.0)
            c_col = caustic_base[None, None, :] * (1.0 - in_disk_catch[:, :, None]) + np.array([250.0, 248.0, 252.0], dtype=np.float32)[None, None, :] * in_disk_catch[:, :, None]

            # Fallback tiny soft catchlight ONLY if raw iris had zero bright catchlights
            if max_orig_l < 62.0:
                pin_d = np.hypot(fx + 0.26 * rad, fy + 0.14 * rad) / max(1.0, 0.13 * rad)
                pin = _smoothstep(1.15, 0.35, pin_d) * disk * 0.55
                c_tot = np.maximum(c_tot, pin)
                c_col = c_col * (1.0 - pin[:, :, None]) + np.array([250.0, 248.0, 252.0], dtype=np.float32)[None, None, :] * pin[:, :, None]

            # Modulate subtle clarity by underlying iris micro-texture
            c_col = np.clip(c_col + hi_micro[:, :, None] * 1.2, 40.0, 255.0)
            iris_c_rgb = iris_c_rgb * (1.0 - c_tot[:, :, None]) + c_col * c_tot[:, :, None]
            iris_c_a = np.maximum(iris_c_a, c_tot)

    p_l5 = out_d / "makeup_05_iris_depth_multiply.png"
    _save_roi_rgba(p_l5, w, h, rx0, ry0, rx1, ry1, iris_d_rgb, iris_d_a, neutral_rgb=(255.0, 255.0, 255.0), overwrite=overwrite)
    layer_specs.append({
        "key": "iris_depth",
        "path": str(p_l5),
        "layer_name": "05_Makeup_Iris_UpperLidShadow_Multiply",
        "blend": "Multiply",
        "opacity": round(min(0.84, float(cfg["iris_depth"]) * k_i), 3),
        "blend_if": {"channel": "gray", "underlying": [0, 0, 198, 244]},
    })

    if iris_rgb_t is not None and eyes_open:
        p_l6a = out_d / "makeup_06a_iris_tint_color.png"
        _save_roi_rgba(p_l6a, w, h, rx0, ry0, rx1, ry1, iris_t_rgb, iris_t_a, neutral_rgb=(128.0, 128.0, 128.0), overwrite=overwrite)
        layer_specs.append({
            "key": "iris_tint",
            "path": str(p_l6a),
            "layer_name": "06a_Makeup_Iris_Color",
            "blend": "Color",
            "opacity": round(min(0.86, float(cfg["iris_tint"]) * k_i), 3),
            "blend_if": {"channel": "gray", "underlying": [20, 62, 195, 242]},
        })

    p_l6b = out_d / "makeup_06b_iris_caustic_softlight.png"
    _save_roi_rgba(p_l6b, w, h, rx0, ry0, rx1, ry1, iris_c_rgb, iris_c_a, neutral_rgb=(128.0, 128.0, 128.0), overwrite=overwrite)
    layer_specs.append({
        "key": "iris_caustic",
        "path": str(p_l6b),
        "layer_name": "06b_Makeup_Iris_Caustic_SoftLight",
        "blend": "Soft Light",
        "opacity": round(min(0.75, float(cfg["iris_caustic"]) * k_i), 3),
        "blend_if": None,
    })

    # =========================================================================
    # LAYER 7: Tightline (PhotoCraft stroke + PSF) + Real Lash Contrast + Optional Outer Curves (Multiply)
    # =========================================================================
    strokes_lashes: list[dict[str, Any]] = []
    upper_l = [133, 173, 157, 158, 159, 160, 161, 246, 33]
    upper_r = [362, 398, 384, 385, 386, 387, 388, 466, 263]
    lower_l = [145, 144, 163, 7, 33]
    lower_r = [374, 373, 390, 249, 263]

    # Tightliner + optional delicate outer-45% curved micro-tapers
    for u_arc, l_arc, w_e, sgn in ((upper_l, lower_l, l_w, -1.0), (upper_r, lower_r, r_w, +1.0)):
        # Hug upper lash root cleanly
        liner_pts: list[list[float]] = []
        for k, idx in enumerate(u_arc):
            t = k / max(1, len(u_arc) - 1)
            p_xy = pts[idx] + 0.004 * w_e * u_down
            pr = 0.22 + 0.60 * math.sin(math.pi * 0.5 * t)
            liner_pts.append([round(float(p_xy[0]), 1), round(float(p_xy[1]), 1), round(pr, 3)])
        # Short subtle wing curving gracefully along lower-lid outer upward slope without cutting downward
        lo_slope = pts[l_arc[-1]] - pts[l_arc[-2]]
        lo_slope_u = lo_slope / max(1e-5, float(np.linalg.norm(lo_slope)))
        up_lift_u = (sgn * u_right - 0.30 * u_down)
        up_lift_u = up_lift_u / max(1e-5, float(np.linalg.norm(up_lift_u)))
        wing_u = 0.65 * lo_slope_u + 0.35 * up_lift_u
        wing_u = wing_u / max(1e-5, float(np.linalg.norm(wing_u)))
        wing_ext = 0.055 * w_e if l_mode == "enhance" else 0.095 * w_e
        wing_p = pts[u_arc[-1]] + wing_u * wing_ext
        liner_pts.append([round(float(wing_p[0]), 1), round(float(wing_p[1]), 1), 0.05])
        strokes_lashes.append({
            "tip": "retouch_02_micro_liner",
            "points": liner_pts,
            "size": round(max(2.4, w_e * 0.040), 1),
            "size_min": 0.10,
            "color": "#221619",
            "opacity": round(min(0.80, float(cfg["tightline"]) * k_i), 3),
            "flow": 0.78,
            "spacing": 0.035,
        })

        # Subtle outer-45% curved tapers ONLY in stylized mode (NEVER straight vertical ticks!)
        if l_mode == "subtle-wisp" and float(cfg["lash_wisp"]) > 0.05 and eyes_open:
            u_poly = pts[u_arc]
            for f_t, llen_k, ang_k in ((0.58, 0.105, 0.35), (0.70, 0.125, 0.52), (0.82, 0.140, 0.68), (0.93, 0.130, 0.84)):
                fi = f_t * (len(u_arc) - 1)
                i0 = min(len(u_arc) - 2, int(fi))
                frac = fi - i0
                bxy = u_poly[i0] * (1.0 - frac) + u_poly[i0 + 1] * frac
                dir0 = (-u_down * (0.65 - 0.25 * ang_k)) + (sgn * u_right * (0.60 + 0.45 * ang_k))
                dir0 /= max(1e-5, float(np.linalg.norm(dir0)))
                dir1 = dir0 * 0.72 + (-u_down * 0.35 + sgn * u_right * 0.25) * 0.28
                dir1 /= max(1e-5, float(np.linalg.norm(dir1)))
                ll = w_e * llen_k
                p0 = bxy
                p1 = bxy + 0.52 * ll * dir0
                p2 = p1 + 0.50 * ll * dir1
                strokes_lashes.append({
                    "tip": "hair_03_single_strand",
                    "points": [
                        [round(float(p0[0]), 1), round(float(p0[1]), 1), 0.82],
                        [round(float(p1[0]), 1), round(float(p1[1]), 1), 0.48],
                        [round(float(p2[0]), 1), round(float(p2[1]), 1), 0.05],
                    ],
                    "size": round(max(1.8, w_e * 0.024), 1),
                    "size_min": 0.05,
                    "follow_dir": True,
                    "color": "#26181c",
                    "opacity": round(min(0.62, float(cfg["lash_wisp"]) * k_i), 3),
                    "flow": 0.76,
                    "spacing": 0.03,
                })

    lash_raw_p = out_d / "_tmp_tightline_raw.png"
    _render_pc_strokes_multi(
        pc_exe=pc_exe,
        width=w,
        height=h,
        brush_tips=tips,
        strokes=strokes_lashes,
        out_png=lash_raw_p,
        tmp_pcraft=out_d / "_tmp_lashes.pcraft",
    )
    l_im = np.array(Image.open(lash_raw_p).convert("RGBA"), dtype=np.float32)[ry0:ry1, rx0:rx1]
    lash_raw_p.unlink(missing_ok=True)
    # Match optical camera PSF softness (no vector crispness)
    l_alpha = cv2.GaussianBlur(l_im[:, :, 3] / 255.0, (0, 0), psf_sig) * (1.0 - bangs_lash_guard)
    l_rgb = l_im[:, :, :3]

    # Real lash enhancement: amplify genuine dark lash strands around upper & lower lashline
    lash_band = np.zeros((rh, rw), dtype=np.float32)
    for arc in (upper_l, upper_r):
        _draw_poly_band(lash_band, pts[arc] - 0.04 * ew * u_down - np.array([rx0, ry0]), radius=max(3.0, ew * 0.06), value=1.0)
    for arc in (lower_l, lower_r):
        _draw_poly_band(lash_band, pts[arc] + 0.03 * ew * u_down - np.array([rx0, ry0]), radius=max(2.2, ew * 0.042), value=0.75)
    lash_band = cv2.GaussianBlur(lash_band, (0, 0), max(2.0, ew * 0.025)) * (1.0 - bangs_lash_guard) * (1.0 - eye_ap_iris * 0.85)
    dark_ridge = np.clip((cv2.GaussianBlur(L_orig, (0, 0), max(2.2, ew * 0.028)) - L_orig - 3.5) / 14.0, 0.0, 1.0)
    real_lash_a = np.clip(dark_ridge * lash_band * float(cfg["lash_enhance"]) * k_i, 0.0, 0.65)
    real_lash_col = np.array([34.0, 22.0, 26.0], dtype=np.float32)[None, None, :]

    tot_lash_a = 1.0 - (1.0 - l_alpha) * (1.0 - real_lash_a)
    tot_lash_rgb = np.where(
        tot_lash_a[:, :, None] > 1e-4,
        (l_rgb * l_alpha[:, :, None] + real_lash_col * real_lash_a[:, :, None] * (1.0 - l_alpha[:, :, None])) / np.maximum(tot_lash_a[:, :, None], 1e-4),
        255.0,
    )
    p_l7 = out_d / "makeup_07_tightline_lashes_multiply.png"
    _save_roi_rgba(p_l7, w, h, rx0, ry0, rx1, ry1, tot_lash_rgb, tot_lash_a, neutral_rgb=(255.0, 255.0, 255.0), overwrite=overwrite)
    layer_specs.append({
        "key": "tightline_lashes",
        "path": str(p_l7),
        "layer_name": "07_Makeup_Tightline_Lashes_Multiply",
        "blend": "Multiply",
        "opacity": round(min(0.88, 0.80 * k_i), 3),
        "blend_if": None,
    })

    # =========================================================================
    # LAYER 8: High Cheekbone Blush Strictly Guarded (Soft Light + Blend-If)
    # =========================================================================
    b_l = _roi_gauss(pts[50] - 0.14 * ew * u_right - 0.04 * ew * u_down, ew * 0.56, ew * 0.30) * 0.82
    b_r = _roi_gauss(pts[280] + 0.14 * ew * u_right - 0.04 * ew * u_down, ew * 0.56, ew * 0.30) * 0.82
    b_tot = np.maximum(b_l, b_r)
    if st_key in ("cos-anime", "painterly"):
        b_nose = _roi_gauss(pts[1] - 0.02 * ew * u_down, ew * 0.14, ew * 0.09) * 0.24
        b_tot = np.maximum(b_tot, b_nose)
    blush_a = np.clip(b_tot, 0.0, 1.0) * skin_guard * (1.0 - 0.82 * tear_trough)
    blush_roi = np.broadcast_to(blush_rgb_t[None, None, :], (rh, rw, 3)).copy()
    p_l8 = out_d / "makeup_08_blush_softlight.png"
    _save_roi_rgba(p_l8, w, h, rx0, ry0, rx1, ry1, blush_roi, blush_a, neutral_rgb=(128.0, 128.0, 128.0), overwrite=overwrite)
    layer_specs.append({
        "key": "blush",
        "path": str(p_l8),
        "layer_name": "08_Makeup_Blush_SoftLight",
        "blend": "Soft Light",
        "opacity": round(min(0.50, float(cfg["blush"]) * k_i), 3),
        "blend_if": {"channel": "gray", "underlying": [28, 78, 208, 248]},
    })

    # =========================================================================
    # LAYER 9a/b/c: PiXimperfect 3-Layer Optical Lips (Soft Light + Multiply + Screen)
    # =========================================================================
    inner_seam_pts = np.vstack([pts[78], pts[82], pts[13], pts[312], pts[308]]) - np.array([rx0, ry0])
    seam_band = np.zeros((rh, rw), dtype=np.float32)
    _draw_poly_band(seam_band, inner_seam_pts, radius=max(2.2, ew * 0.032), value=1.0)
    seam_blur = cv2.GaussianBlur(seam_band, (0, 0), max(3.5, ew * 0.055))

    # 9a) Soft Light lip glaze (preserves 100% real vertical lip creases & cupid bow)
    lip_glaze_col = np.broadcast_to(lip_rgb_t[None, None, :], (rh, rw, 3)).copy()
    p_l9a = out_d / "makeup_09a_lips_glaze_softlight.png"
    _save_roi_rgba(p_l9a, w, h, rx0, ry0, rx1, ry1, lip_glaze_col, lip_mask_strict, neutral_rgb=(128.0, 128.0, 128.0), overwrite=overwrite)
    layer_specs.append({
        "key": "lips_glaze",
        "path": str(p_l9a),
        "layer_name": "09a_Makeup_Lips_Glaze_SoftLight",
        "blend": "Soft Light",
        "opacity": round(min(0.72, float(cfg["lip_glaze"]) * k_i), 3),
        "blend_if": {"channel": "gray", "underlying": [20, 66, 200, 246]},
    })

    # 9b) Multiply inner-seam bitten gradient + tapered mouth corners + subtle under-lip cushion shadow
    lip_deep_col = np.clip(lip_rgb_t * 0.44, 24.0, 118.0)
    corner_l = _roi_gauss(pts[61], ew * 0.055, ew * 0.038)
    corner_r = _roi_gauss(pts[291], ew * 0.055, ew * 0.038)
    under_lip_sh = _roi_gauss(pts[17] + 0.08 * ew * u_down, ew * 0.24, ew * 0.055) * (1.0 - lip_mask_strict) * skin_guard * 0.28
    lip_d_a = np.clip(np.maximum.reduce([seam_blur * 0.82 * lip_mask_strict, np.maximum(corner_l, corner_r) * 0.36, under_lip_sh]), 0.0, 1.0)
    lip_d_rgb = np.broadcast_to(lip_deep_col[None, None, :], (rh, rw, 3)).copy()
    corner_col = np.array([76.0, 46.0, 52.0], dtype=np.float32)[None, None, :]
    cw = np.clip(np.maximum(np.maximum(corner_l, corner_r), under_lip_sh), 0.0, 1.0)[:, :, None]
    lip_d_rgb = lip_d_rgb * (1.0 - cw) + corner_col * cw
    p_l9b = out_d / "makeup_09b_lips_depth_multiply.png"
    _save_roi_rgba(p_l9b, w, h, rx0, ry0, rx1, ry1, lip_d_rgb, lip_d_a, neutral_rgb=(255.0, 255.0, 255.0), overwrite=overwrite)
    layer_specs.append({
        "key": "lips_depth",
        "path": str(p_l9b),
        "layer_name": "09b_Makeup_Lips_Depth_Multiply",
        "blend": "Multiply",
        "opacity": round(min(0.70, float(cfg["lip_depth"]) * k_i), 3),
        "blend_if": {"channel": "gray", "underlying": [12, 42, 224, 255]},
    })

    # 9c) Screen real-luma-gated specular lift (modulates lip micro-texture rather than stamping ovals!)
    lip_vals = L_orig[lip_mask_strict > 0.45]
    p74_lip = float(np.percentile(lip_vals, 74)) if lip_vals.size > 40 else 55.0
    p94_lip = max(p74_lip + 6.0, float(np.percentile(lip_vals, 94)) if lip_vals.size > 40 else 68.0)
    real_spec_mask = np.clip((L_orig - p74_lip) / (p94_lip - p74_lip), 0.0, 1.0)
    lo_lip_c = 0.50 * pts[14] + 0.50 * pts[17]
    cushion_env = np.clip(
        _roi_gauss(lo_lip_c, ew * 0.20, ew * 0.068) + 0.55 * _roi_gauss(pts[13] - 0.015 * ew * u_down, ew * 0.14, ew * 0.032),
        0.0,
        1.0,
    )
    # Break highlights naturally with lip vertical ridge modulation + subtle satin floor if raw lip is flat
    lip_ridges = np.clip((L_orig - cv2.GaussianBlur(L_orig, (0, 0), max(1.8, ew * 0.022))) / 5.0 + 0.65, 0.25, 1.0)
    lip_spec_a = np.clip((real_spec_mask * 0.85 + 0.15) * cushion_env * lip_ridges * lip_mask_strict, 0.0, 1.0)
    spec_rgb = np.broadcast_to(np.array([250.0, 232.0, 238.0], dtype=np.float32)[None, None, :], (rh, rw, 3)).copy()
    p_l9c = out_d / "makeup_09c_lips_specular_screen.png"
    _save_roi_rgba(p_l9c, w, h, rx0, ry0, rx1, ry1, spec_rgb, lip_spec_a, neutral_rgb=(0.0, 0.0, 0.0), overwrite=overwrite)
    layer_specs.append({
        "key": "lips_specular",
        "path": str(p_l9c),
        "layer_name": "09c_Makeup_Lips_Specular_Screen",
        "blend": "Screen",
        "opacity": round(min(0.68, float(cfg["lip_spec"]) * k_i), 3),
        "blend_if": None,
    })

    # =========================================================================
    # Composite Accurate Optical Preview & Optional Native .pcraft
    # =========================================================================
    g_op = float(cfg["group_opacity"])
    comp = r_rgb / 255.0
    for lay in layer_specs:
        ov = np.array(Image.open(lay["path"]).convert("RGBA"), dtype=np.float32)[ry0:ry1, rx0:rx1] / 255.0
        comp = _composite_optical_layer(comp, ov, lay["blend"], float(lay["opacity"]) * g_op, lay.get("blend_if"))

    full_preview = rgb_f32 / 255.0
    full_preview[ry0:ry1, rx0:rx1] = comp
    p_prev = out_d / "makeup_preview.png"
    _write_png_u8(p_prev, full_preview * 255.0, "RGB", overwrite=overwrite)

    # Also build legacy key aliases in `outputs` for smooth backward compatibility
    outputs: dict[str, str] = {lay["key"]: lay["path"] for lay in layer_specs}
    outputs["makeup_preview"] = str(p_prev)

    meta: dict[str, Any] = {
        "ok": True,
        "engine_version": "v2_optical_blendif",
        "input": str(in_p),
        "style": st_key,
        "lash_mode": l_mode,
        "eyes_open": eyes_open,
        "ear_avg": round(ear_avg, 3),
        "eye_span_px": round(eye_span, 1),
        "group_opacity": round(g_op, 3),
        "layers": layer_specs,
        "outputs": outputs,
    }

    if build_pcraft:
        p_pcraft = out_d / "makeup_layers.pcraft"
        cmds = build_makeup_pcraft_cmds(meta, width=w, height=h, group_name="MAKEUP_OPTICAL_GROUP", after_layer="BASE_INPUT", group_opacity=g_op)
        args = [
            pc_exe,
            "run",
            str(in_p),
            "--cmd",
            "layer.setProps",
            "--params",
            _J({"name": "BASE_INPUT"}),
        ] + cmds + ["--out", str(p_pcraft)]
        r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            raise RuntimeError(f"PhotoCraft build makeup_layers.pcraft failed (rc={r.returncode}): {r.stderr[-600:] or r.stdout[-600:]}")
        outputs["makeup_pcraft"] = str(p_pcraft)
        meta["makeup_pcraft"] = str(p_pcraft)

    (out_d / "makeup_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta
