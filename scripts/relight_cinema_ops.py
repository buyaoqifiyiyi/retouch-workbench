"""2.5D Depth-Anything-V2 Volumetric Relighting, Blender/AgX/Kodak2383 .cube LUT Cinema Grading & Foreground Framing."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


LUT_PRESETS = (
    "01_cinema_sunset_teal_gold",
    "02_cinema_kodak2383_filmic",
    "03_cinema_crystal_highkey_agx",
    "04_cinema_moonlight_crimson",
    "05_cinema_epic_slate_snow",
)


def _J(obj: object) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _point_light_field(
    xx: np.ndarray,
    yy: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    nz: np.ndarray,
    lx: float,
    ly: float,
    lz: float,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (ndotl [0..1], Gaussian spatial falloff [0..1]) toward a 3D scene point light."""
    span = max(1.0, float(xx.shape[1]) * 0.26)
    dx = (lx - xx) / span
    dy = (ly - yy) / span
    dz = np.full_like(dx, lz)
    inv_d = 1.0 / np.maximum(np.sqrt(dx * dx + dy * dy + dz * dz), 1e-4)
    ndotl = np.clip((nx * dx + ny * dy + nz * dz) * inv_d, 0.0, 1.0)
    dist2d = np.hypot(xx - lx, yy - ly)
    falloff = np.exp(-0.5 * (dist2d / max(32.0, radius)) ** 2)
    return ndotl, falloff


def _detect_bg_accent_lights(bg01: np.ndarray, alpha01: np.ndarray, key_side: str) -> list[dict[str, Any]]:
    """Auto-extract up to 2 vibrant chromatic accent lights (e.g. neon signs/lamps) from background pixels."""
    from workbench_ops import rgb255_to_lab

    h, w = bg01.shape[:2]
    sw = 320
    sh = max(64, int(round(h * sw / max(1, w))))
    bg_s = cv2.resize(bg01, (sw, sh), interpolation=cv2.INTER_AREA)
    a_s = cv2.resize(alpha01, (sw, sh), interpolation=cv2.INTER_AREA)
    lab_s = rgb255_to_lab(np.clip(bg_s * 255.0, 0.0, 255.0))
    luma = 0.2126 * bg_s[:, :, 0] + 0.7152 * bg_s[:, :, 1] + 0.0722 * bg_s[:, :, 2]
    chroma = np.hypot(lab_s[:, :, 1], lab_s[:, :, 2])

    score = (
        np.clip((luma - 0.30) / 0.52, 0.0, 1.0)
        * np.clip((chroma - 20.0) / 42.0, 0.0, 1.0)
        * (a_s < 0.20).astype(np.float32)
    )
    score = cv2.GaussianBlur(score, (0, 0), 7.0)

    peaks: list[dict[str, Any]] = []
    work = score.copy()
    min_sep = max(32.0, sw * 0.28)
    rad_suppress = int(max(24, min_sep))
    for _ in range(2):
        _, mv, _, ml = cv2.minMaxLoc(work)
        if mv < 0.16:
            break
        px, py = int(ml[0]), int(ml[1])
        x0, x1 = max(0, px - 10), min(sw, px + 11)
        y0, y1 = max(0, py - 10), min(sh, py + 11)
        patch = bg_s[y0:y1, x0:x1]
        p_lum = 0.2126 * patch[:, :, 0] + 0.7152 * patch[:, :, 1] + 0.0722 * patch[:, :, 2]
        top_p = patch[p_lum >= np.percentile(p_lum, 65)]
        med_c = np.median(top_p, axis=0) if top_p.size > 0 else np.median(patch.reshape(-1, 3), axis=0)
        mx = max(float(med_c.max()), 1e-4)
        col_norm = np.clip(med_c / mx, 0.05, 1.0)
        # Skip nearly-achromatic gray/moon spots unless distinctly colored (accent light)
        if float(col_norm.max() - col_norm.min()) >= 0.16:
            peaks.append({
                "x_frac": round(px / float(sw), 4),
                "y_frac": round(py / float(sh), 4),
                "z": 0.20,
                "rgb": [round(float(c), 3) for c in col_norm],
                "radius_frac": 0.36,
                "strength": round(float(np.clip(0.45 + 0.45 * mv, 0.45, 0.80)), 3),
                "source": "auto_bg",
            })
        cv2.circle(work, (px, py), rad_suppress, 0.0, -1)
    return peaks


def build_relight_pcraft_cmds(
    meta: dict[str, Any],
    *,
    width: int,
    height: int,
    base_layer_name: str,
    clip: bool = True,
) -> list[str]:
    """Build PhotoCraft CLI commands placing all 2.5D relight layers cleanly onto base_layer_name."""
    layers = [l for l in meta.get("layers", []) if Path(l["path"]).exists()]
    if not layers:
        return []
    cmds: list[str] = []
    prev = base_layer_name
    for lay in layers:
        lname = str(lay["layer_name"])
        blend = str(lay["blend"])
        op = float(lay.get("opacity", 1.0))
        fill_op = lay.get("fill_opacity")
        bif = lay.get("blend_if")
        cmds += [
            "--cmd",
            "layer.select",
            "--params",
            _J({"layer": prev}),
            "--cmd",
            "file.placeEmbedded",
            "--params",
            _J({"path": str(lay["path"]), "fit": False, "smartObject": True, "center": [width / 2.0, height / 2.0]}),
            "--cmd",
            "layer.setProps",
            "--params",
            _J({"name": lname, "blend": blend, "opacity": round(op, 3), "clipped": bool(clip), "visible": True}),
        ]
        bopts: dict[str, Any] = {}
        if fill_op is not None and int(round(float(fill_op))) < 100:
            bopts["fillOpacity"] = int(round(float(fill_op)))
        if bif is not None:
            bopts["blendIf"] = bif
        if bopts:
            cmds += ["--cmd", "layer.layerStyle.blendingOptions", "--params", _J(bopts)]
        cmds += ["--cmd", "layer.moveTo", "--params", _J({"target": prev, "position": "above"})]
        prev = lname
    return cmds


def _read_rgb_f32(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _read_rgba_f32(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.array(Image.open(path).convert("RGBA"), dtype=np.float32) / 255.0
    return arr[:, :, :3], arr[:, :, 3]


def _read_gray_f32(path: Path, size_wh: tuple[int, int]) -> np.ndarray:
    im = Image.open(path).convert("L")
    if im.size != size_wh:
        im = im.resize(size_wh, Image.Resampling.BILINEAR)
    return np.array(im, dtype=np.float32) / 255.0


def _write_png(path: Path, arr_f01: np.ndarray, mode: str, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = np.clip(np.round(arr_f01 * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(u8, mode).save(path, "PNG")


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    if abs(edge1 - edge0) < 1e-6:
        return (x >= edge1).astype(np.float32)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _filmic_grade_math(
    rgb: np.ndarray,
    *,
    toe_black: float = 0.035,
    contrast: float = 1.24,
    pivot: float = 0.44,
    shoulder_soft: float = 0.12,
    sat_boost: float = 1.08,
    shd_tint: tuple[float, float, float] = (-0.055, 0.005, 0.065),
    mid_tint: tuple[float, float, float] = (0.005, 0.0, -0.005),
    hi_tint: tuple[float, float, float] = (0.045, 0.015, -0.045),
) -> np.ndarray:
    """AgX/Kodak-2383-inspired filmic S-curve + 3-way split tone + skin-safe saturation curve."""
    x = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    # 1. Soft filmic toe & contrast around pivot (lifts micro-separation while anchoring blacks cleanly)
    lum = 0.2126 * x[:, :, 0] + 0.7152 * x[:, :, 1] + 0.0722 * x[:, :, 2]
    # Gentle exponential film curve in normalized domain
    eps = 1e-4
    shifted = np.clip((x - toe_black) / max(1e-3, 1.0 - toe_black), 0.0, 1.0)
    xp = np.clip(shifted, eps, 1.0 - eps)
    # Smooth S-curve via logit-scaled sigmoid
    p = pivot
    k = contrast
    num = np.power(xp / p, k)
    den = num + np.power((1.0 - xp) / max(1e-3, 1.0 - p), k)
    curved = num / np.maximum(den, 1e-6)
    # Soft highlight shoulder compression so bright petals/sky stay luminous without chalky clip
    curved = curved * (1.0 - shoulder_soft) + np.tanh(curved * 1.45) / math.tanh(1.45) * shoulder_soft
    curved = np.clip(curved * (1.0 - 0.015) + 0.015 * x, 0.0, 1.0)

    # Preserve hue/chroma proportions cleanly while shifting split-tone shadows & highlights
    lum_c = np.clip(0.2126 * curved[:, :, 0] + 0.7152 * curved[:, :, 1] + 0.0722 * curved[:, :, 2], 0.0, 1.0)
    w_shd = np.power(np.clip(1.0 - lum_c / 0.48, 0.0, 1.0), 1.5)[:, :, None]
    w_hi = np.power(np.clip((lum_c - 0.42) / 0.58, 0.0, 1.0), 1.4)[:, :, None]
    w_mid = np.clip(1.0 - w_shd - w_hi, 0.0, 1.0)

    # Desaturate near pure black/white naturally (AgX highlight desaturation behavior)
    chroma = curved - lum_c[:, :, None]
    desat_hi = 1.0 - 0.42 * np.power(np.clip((lum_c - 0.72) / 0.28, 0.0, 1.0), 1.5)
    desat_lo = 1.0 - 0.35 * np.power(np.clip(1.0 - lum_c / 0.14, 0.0, 1.0), 1.6)
    curved = np.clip(lum_c[:, :, None] + chroma * (sat_boost * desat_hi * desat_lo)[:, :, None], 0.0, 1.0)

    shd_v = np.array(shd_tint, dtype=np.float32)[None, None, :]
    mid_v = np.array(mid_tint, dtype=np.float32)[None, None, :]
    hi_v = np.array(hi_tint, dtype=np.float32)[None, None, :]
    out = curved + w_shd * shd_v + w_mid * mid_v + w_hi * hi_v
    return np.clip(out, 0.0, 1.0)


def _write_cube_3d(path: Path, title: str, table_bgr_rgb: np.ndarray) -> None:
    """Write standard Adobe .cube 3D LUT (R varies fastest per Adobe Cube LUT 1.0 spec)."""
    sz = table_bgr_rgb.shape[0]
    flat = table_bgr_rgb.reshape(-1, 3)
    lines = [
        f'TITLE "{title}"',
        f"LUT_3D_SIZE {sz}",
        "DOMAIN_MIN 0.0 0.0 0.0",
        "DOMAIN_MAX 1.0 1.0 1.0",
        "",
    ]
    lines.extend(f"{r:.6f} {g:.6f} {b:.6f}" for r, g, b in flat)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_cinema_lut_library(luts_dir: Path | None = None, force: bool = False) -> dict[str, Path]:
    """Ensure built-in cinema 3D .cube LUTs + Blender 5.x official OCIO AgX/Neutral LUTs exist."""
    ldir = luts_dir or (Path(__file__).resolve().parent.parent / "resources" / "luts")
    ldir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    # Copy permissive Blender 5.2 official OCIO LUTs when present
    for b_lut in (
        Path(r"D:\HRSoftstore\Install\Blender\5.2\datafiles\colormanagement\luts\AgX_Base_sRGB.cube"),
        Path(r"D:\HRSoftstore\Install\Blender\5.2\datafiles\colormanagement\luts\pbrNeutral.cube"),
    ):
        if b_lut.exists():
            dst = ldir / b_lut.name
            if force or not dst.exists():
                shutil.copy2(b_lut, dst)
            out[b_lut.stem] = dst

    sz = 33
    grid = np.linspace(0.0, 1.0, sz, dtype=np.float32)
    # Adobe .cube spec: outermost loop B, middle G, innermost (fastest) R
    bb, gg, rr = np.meshgrid(grid, grid, grid, indexing="ij")
    id_cube = np.stack([rr, gg, bb], axis=-1).reshape(sz, sz * sz, 3)

    specs: dict[str, dict[str, Any]] = {
        "01_cinema_sunset_teal_gold": {
            "toe_black": 0.032,
            "contrast": 1.24,
            "pivot": 0.44,
            "shoulder_soft": 0.14,
            "sat_boost": 1.12,
            "shd_tint": (-0.060, 0.008, 0.070),
            "mid_tint": (0.008, 0.002, -0.008),
            "hi_tint": (0.050, 0.018, -0.052),
        },
        "02_cinema_kodak2383_filmic": {
            "toe_black": 0.038,
            "contrast": 1.32,
            "pivot": 0.45,
            "shoulder_soft": 0.18,
            "sat_boost": 1.10,
            "shd_tint": (-0.042, 0.004, 0.045),
            "mid_tint": (0.012, 0.004, -0.010),
            "hi_tint": (0.042, 0.020, -0.048),
        },
        "03_cinema_crystal_highkey_agx": {
            "toe_black": 0.022,
            "contrast": 1.18,
            "pivot": 0.48,
            "shoulder_soft": 0.12,
            "sat_boost": 1.14,
            "shd_tint": (-0.052, 0.012, 0.062),
            "mid_tint": (0.004, 0.004, -0.004),
            "hi_tint": (0.038, 0.014, -0.036),
        },
        "04_cinema_moonlight_crimson": {
            "toe_black": 0.030,
            "contrast": 1.28,
            "pivot": 0.42,
            "shoulder_soft": 0.14,
            "sat_boost": 1.10,
            "shd_tint": (-0.068, 0.002, 0.088),
            "mid_tint": (0.018, -0.008, -0.006),
            "hi_tint": (0.032, -0.005, 0.010),
        },
        "05_cinema_epic_slate_snow": {
            "toe_black": 0.034,
            "contrast": 1.26,
            "pivot": 0.45,
            "shoulder_soft": 0.15,
            "sat_boost": 1.05,
            "shd_tint": (-0.048, 0.002, 0.058),
            "mid_tint": (-0.008, 0.0, 0.010),
            "hi_tint": (0.024, 0.012, -0.022),
        },
    }

    for name, params in specs.items():
        p = ldir / f"{name}.cube"
        if force or not p.exists():
            graded = _filmic_grade_math(id_cube, **params).reshape(sz, sz, sz, 3)
            _write_cube_3d(p, name, graded)
        out[name] = p

    return out


def _estimate_subject_depth_normals(
    rgb_f32: np.ndarray,
    alpha01: np.ndarray,
    depth_model_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run Depth-Anything-V2-Small ONNX tightly on subject bbox and extract polished 2.5D normals & cavity AO."""
    from cuda_env import create_onnx_session

    h, w = alpha01.shape[:2]
    ys, xs = np.nonzero(alpha01 > 0.05)
    if ys.size == 0:
        raise ValueError("Subject alpha is completely empty.")
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    pad_y = max(24, int(0.08 * (y1 - y0)))
    pad_x = max(24, int(0.08 * (x1 - x0)))
    cy0, cy1 = max(0, y0 - pad_y), min(h, y1 + pad_y)
    cx0, cx1 = max(0, x0 - pad_x), min(w, x1 + pad_x)

    crop_rgb = rgb_f32[cy0:cy1, cx0:cx1]
    crop_a = alpha01[cy0:cy1, cx0:cx1]
    # Place over quiet mid-slate background outside alpha so background clutter inside crop never distorts depth
    crop_clean = crop_rgb * crop_a[:, :, None] + 0.32 * (1.0 - crop_a[:, :, None])
    inp_sz = 518
    rsz = cv2.resize(crop_clean, (inp_sz, inp_sz), interpolation=cv2.INTER_LINEAR)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    chw = np.transpose((rsz - mean) / std, (2, 0, 1))[None, ...].astype(np.float32)

    sess = create_onnx_session(depth_model_path)
    inp_name = sess.get_inputs()[0].name
    pred = np.asarray(sess.run(None, {inp_name: chw})[0], dtype=np.float32).squeeze()
    pred_crop = cv2.resize(pred, (cx1 - cx0, cy1 - cy0), interpolation=cv2.INTER_CUBIC)

    # Normalize relative depth strictly within solid subject foreground
    fg_c = crop_a > 0.15
    d_vals = pred_crop[fg_c]
    d_lo, d_hi = float(np.percentile(d_vals, 2)), float(np.percentile(d_vals, 98))
    norm_crop = np.clip((pred_crop - d_lo) / max(1e-4, d_hi - d_lo), 0.0, 1.0)
    # Depth-Anything outputs larger values closer to camera (near=1, far=0); verify via foreground > border if inverted
    if float(norm_crop[fg_c].mean()) < float(norm_crop[~fg_c].mean()):
        norm_crop = 1.0 - norm_crop

    depth_full = np.zeros((h, w), dtype=np.float32)
    depth_full[cy0:cy1, cx0:cx1] = norm_crop

    # Edge-preserving guided filter polish using subject luminance so pleats & hair folds align crisply
    luma_guide = (0.299 * rgb_f32[:, :, 0] + 0.587 * rgb_f32[:, :, 1] + 0.114 * rgb_f32[:, :, 2]).astype(np.float32)
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
        depth_pol = np.clip(cv2.ximgproc.guidedFilter(guide=luma_guide, src=depth_full.astype(np.float32), radius=10, eps=4e-4), 0.0, 1.0)
    else:
        depth_pol = cv2.GaussianBlur(depth_full, (0, 0), 3.0)
    # Fill exterior smoothly to avoid artificial cliff normals at the alpha border
    fg_bin = (alpha01 > 0.25).astype(np.uint8)
    dist_out, lbl_out = cv2.distanceTransformWithLabels(1 - fg_bin, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys_f, xs_f = np.nonzero(fg_bin > 0)
    lut_d = np.zeros(int(lbl_out.max()) + 1, dtype=np.float32)
    lut_d[lbl_out[ys_f, xs_f]] = depth_pol[ys_f, xs_f]
    depth_ext = np.where(fg_bin > 0, depth_pol, lut_d[lbl_out] * np.exp(-dist_out / 35.0))

    # Multi-scale gradients: broad body curvature + fine pleat/hair relief
    d_broad = cv2.GaussianBlur(depth_ext, (0, 0), max(12.0, w * 0.007))
    d_mid = cv2.GaussianBlur(depth_ext, (0, 0), max(4.0, w * 0.0022))
    gy_b, gx_b = np.gradient(d_broad)
    gy_m, gx_m = np.gradient(d_mid)
    scale_g = max(w, h) * 0.14
    gx = (0.68 * gx_b + 0.32 * gx_m) * scale_g
    gy = (0.68 * gy_b + 0.32 * gy_m) * scale_g

    # Blend silhouette distance-field slope in outer 28px so outer curls & arms have true grazing edge curvature
    dist_in = cv2.distanceTransform(fg_bin, cv2.DIST_L2, 5)
    rim_w = np.clip(1.0 - dist_in / max(18.0, w * 0.009), 0.0, 1.0) * (alpha01 > 0.05)
    dist_in_sm = cv2.GaussianBlur(dist_in, (0, 0), 4.0)
    ry, rx = np.gradient(dist_in_sm)
    rmag = np.maximum(np.hypot(rx, ry), 1e-4)
    # Point outward normal (-rx, -ry) gently along silhouette edge
    nx_rim = (-rx / rmag) * 0.85
    ny_rim = (-ry / rmag) * 0.85

    nx = -gx * (1.0 - 0.55 * rim_w) + nx_rim * (0.55 * rim_w)
    ny = -gy * (1.0 - 0.55 * rim_w) + ny_rim * (0.55 * rim_w)
    nz = np.maximum(0.35, np.ones_like(nx) - 0.45 * rim_w)
    n_len = np.maximum(np.sqrt(nx * nx + ny * ny + nz * nz), 1e-6)
    normals = np.stack([nx / n_len, ny / n_len, nz / n_len], axis=-1).astype(np.float32)

    # Cavity AO: deep folds/creases sitting behind local broad depth envelope
    d_env = cv2.GaussianBlur(depth_ext, (0, 0), max(28.0, w * 0.018))
    cavity_ao = np.clip((d_env - depth_pol) * 5.5, 0.0, 1.0) * alpha01
    return depth_pol, normals, cavity_ao


def run_relight_sculpt(
    *,
    fg_rgba_path: Path,
    depth_model_path: Path,
    out_dir: Path,
    bg_path: Path | None = None,
    sun_side: str = "auto",
    scene_meta_path: Path | None = None,
    face_parse_dir: Path | None = None,
    body_parse_dir: Path | None = None,
    garment_mask_path: Path | None = None,
    skin_mask_path: Path | None = None,
    delight_strength: float = 0.65,
    delight_floor_l: float = 42.0,
    accent_lights: list[dict[str, Any]] | None = None,
    auto_bg_lights: bool = True,
    shadow_strength: float = 0.58,
    key_strength: float = 0.45,
    rim_strength: float = 0.52,
    bounce_strength: float = 0.25,
    face_protect: float = 0.55,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Pure non-generative 2.5D indoor de-lighting, Depth-Anything-V2 directional sculpt, and multi-source scene accent relighting."""
    from workbench_ops import lab_to_rgb255, rgb255_to_lab

    t0 = time.perf_counter()
    fg_p = fg_rgba_path.resolve()
    out_d = out_dir.resolve()
    out_d.mkdir(parents=True, exist_ok=True)

    rgb_f, alpha = _read_rgba_f32(fg_p)
    h, w = alpha.shape[:2]

    # Resolve light direction from scene_meta.json or override
    resolved_side = sun_side.lower().strip()
    if resolved_side == "auto" and scene_meta_path is not None and scene_meta_path.exists():
        meta_s = json.loads(scene_meta_path.read_text(encoding="utf-8"))
        resolved_side = str(meta_s.get("lighting", {}).get("sun_side") or meta_s.get("sun_side") or "right").lower()
    if resolved_side not in {"right", "left", "top-right", "top-left", "back"}:
        resolved_side = "right"

    light_vecs = {
        "right": np.array([0.80, -0.32, 0.50], dtype=np.float32),
        "top-right": np.array([0.64, -0.60, 0.48], dtype=np.float32),
        "left": np.array([-0.80, -0.32, 0.50], dtype=np.float32),
        "top-left": np.array([-0.64, -0.60, 0.48], dtype=np.float32),
        "back": np.array([0.15, -0.38, -0.91], dtype=np.float32),
    }
    L = light_vecs[resolved_side]
    L /= np.linalg.norm(L)

    # Estimate 2.5D depth, normals, and cavity AO (strictly geometry masks — ZERO pixel regeneration!)
    depth_pol, normals, cavity_ao = _estimate_subject_depth_normals(rgb_f, alpha, depth_model_path)

    # Sample physical environment light, shadow, and floor colors from background if provided
    key_rgb = np.array([1.0, 0.91, 0.75], dtype=np.float32)
    shadow_tint = np.array([0.31, 0.39, 0.50], dtype=np.float32)
    floor_rgb = np.array([0.82, 0.88, 0.92], dtype=np.float32)
    bg_blur_wrap = np.broadcast_to(key_rgb, (h, w, 3)).copy()
    bg01: np.ndarray | None = None

    if bg_path is not None and bg_path.exists():
        bg_im = Image.open(bg_path.resolve()).convert("RGB").resize((w, h), Image.Resampling.BILINEAR)
        bg = np.array(bg_im, dtype=np.float32) / 255.0
        bg01 = bg
        bg_l = 0.2126 * bg[:, :, 0] + 0.7152 * bg[:, :, 1] + 0.0722 * bg[:, :, 2]
        sun_mask = np.ones((h, w), dtype=bool)
        if "right" in resolved_side:
            sun_mask[:, : w // 2] = False
        elif "left" in resolved_side:
            sun_mask[:, w // 2 :] = False
        p90 = float(np.percentile(bg_l[sun_mask], 90))
        hi_px = bg[sun_mask & (bg_l >= p90)]
        if hi_px.size > 0:
            s_key = np.median(hi_px, axis=0).astype(np.float32)
            key_rgb = np.clip(0.55 * (s_key / max(float(s_key.max()), 1e-4)) + 0.45 * key_rgb, 0.65, 1.0)
        p20 = float(np.percentile(bg_l, 20))
        lo_px = bg[bg_l <= p20]
        if lo_px.size > 0:
            s_shd = np.median(lo_px, axis=0).astype(np.float32)
            s_shd = np.clip(s_shd / max(float(s_shd.mean()), 1e-3) * 0.36, 0.20, 0.52)
            shadow_tint = np.clip(0.45 * s_shd + 0.55 * np.array([0.30, 0.38, 0.49], dtype=np.float32), 0.18, 0.54)
        fl_px = bg[int(h * 0.74) :, :]
        floor_rgb = np.clip(np.median(fl_px.reshape(-1, 3), axis=0).astype(np.float32), 0.45, 1.0)
        bg_blur_wrap = cv2.GaussianBlur(bg, (0, 0), max(22.0, w * 0.010))

    # Protective masks for face/eyes/garment/skin
    face_shield = np.zeros((h, w), dtype=np.float32)
    face_skin_f = np.zeros((h, w), dtype=np.float32)
    eyes_shield = np.zeros((h, w), dtype=np.float32)
    if face_parse_dir is not None and face_parse_dir.exists():
        fs_p = face_parse_dir / "mask_skin.png"
        eb_p = face_parse_dir / "mask_eyes_brows.png"
        if fs_p.exists():
            f_m = _read_gray_f32(fs_p, (w, h))
            face_skin_f = np.clip(cv2.GaussianBlur(f_m, (0, 0), max(8.0, w * 0.004)), 0.0, 1.0)
            f_er = cv2.erode((f_m > 0.35).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(11, int(w * 0.008)) | 1, max(11, int(w * 0.008)) | 1))).astype(np.float32)
            face_shield = cv2.GaussianBlur(f_er, (0, 0), max(12.0, w * 0.006))
        if eb_p.exists():
            e_m = _read_gray_f32(eb_p, (w, h))
            eyes_shield = np.clip(cv2.GaussianBlur(e_m, (0, 0), 8.0), 0.0, 1.0)

    garment_m = _read_gray_f32(garment_mask_path.resolve(), (w, h)) if (garment_mask_path and garment_mask_path.exists()) else np.zeros((h, w), dtype=np.float32)
    skin_m = _read_gray_f32(skin_mask_path.resolve(), (w, h)) if (skin_mask_path and skin_mask_path.exists()) else face_skin_f.copy()
    skin_zone = np.clip(np.maximum(skin_m, face_skin_f) * (1.0 - garment_m), 0.0, 1.0)
    garment_soft = np.clip(cv2.GaussianBlur(garment_m, (0, 0), max(5.0, w * 0.0022)), 0.0, 1.0)
    k_g_er = max(13, int(round(w * 0.006)) | 1)
    garment_core = cv2.erode((garment_m > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_g_er, k_g_er))).astype(np.float32)
    garment_core = np.clip(cv2.GaussianBlur(garment_core, (0, 0), max(6.0, w * 0.0025)), 0.0, 1.0)

    # Smooth normals inside face region so Depth-Anything nose ridge never creates harsh stripe shadows
    n_soft = cv2.GaussianBlur(normals, (0, 0), max(24.0, w * 0.008))
    n_soft = n_soft / np.maximum(np.sqrt((n_soft * n_soft).sum(axis=2, keepdims=True)), 1e-4)
    face_norm_w = np.clip(0.75 * face_skin_f, 0.0, 0.82)[:, :, None]
    normals_eff = normals * (1.0 - face_norm_w) + n_soft * face_norm_w
    normals_eff = normals_eff / np.maximum(np.sqrt((normals_eff * normals_eff).sum(axis=2, keepdims=True)), 1e-4)

    # Side coordinate bias across subject bbox
    ys, xs = np.nonzero(alpha > 0.1)
    x_min, x_max = float(xs.min()), float(xs.max())
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x_rel = np.clip((xx - x_min) / max(1.0, x_max - x_min), 0.0, 1.0)
    if "left" in resolved_side:
        sun_side_bias = 1.0 - x_rel
    elif "back" == resolved_side:
        sun_side_bias = np.abs(x_rel - 0.5) * 2.0
    else:
        sun_side_bias = x_rel
    shade_side_bias = 1.0 - sun_side_bias

    ndotl = normals_eff[:, :, 0] * L[0] + normals_eff[:, :, 1] * L[1] + normals_eff[:, :, 2] * L[2]
    ndotl_eff = 0.72 * ndotl + 0.28 * (sun_side_bias * 2.0 - 1.0)

    # =========================================================================
    # STEP 0 (Optional De-Light): Frequency-preserving convention-hall flat-flash & yellow-bounce removal
    # =========================================================================
    work_rgb = rgb_f.copy()
    p_delight: Path | None = None
    dl_s = float(np.clip(delight_strength, 0.0, 1.0))
    if dl_s > 0.02:
        lab_fg = rgb255_to_lab(np.clip(rgb_f * 255.0, 0.0, 255.0))
        sig_dl = max(32.0, w * 0.014)
        L_lo = cv2.GaussianBlur(lab_fg[:, :, 0], (0, 0), sig_dl)
        L_hi = lab_fg[:, :, 0] - L_lo
        shd_geom_dl = _smoothstep(0.45, -0.30, ndotl_eff)
        shd_macro_dl = _smoothstep(0.28, 0.92, shade_side_bias)
        dl_w = np.clip(0.68 * shd_geom_dl + 0.32 * shd_macro_dl, 0.0, 1.0)
        dl_w *= (1.0 - 0.92 * eyes_shield) * np.clip(1.0 - 0.65 * float(face_protect) * face_shield, 0.25, 1.0)
        excess_l = np.clip((L_lo - float(delight_floor_l)) / 30.0, 0.0, 1.0)
        pull_l = 15.0 * dl_s * excess_l * dl_w * (1.0 - 0.55 * garment_core)
        L_new = np.clip((L_lo - pull_l) + L_hi, 1.5, 99.0)

        a_new = lab_fg[:, :, 1].copy()
        b_new = lab_fg[:, :, 2].copy()
        if garment_soft.max() > 0.05:
            # Strip warm yellow hall ceiling cast on dark garments while keeping neutral/cool sheen
            b_cln = np.where(b_new > 0.4, np.minimum(b_new, 0.4) - 0.4 * dl_s, b_new)
            a_cln = np.where(a_new > 1.4, 1.4 + (a_new - 1.4) * (1.0 - 0.65 * dl_s), a_new)
            a_new = a_new * (1.0 - garment_soft) + a_cln * garment_soft
            b_new = b_new * (1.0 - garment_soft) + b_cln * garment_soft
        if skin_zone.max() > 0.05:
            # Guard warm skin from losing healthy chroma smoothly with zero boolean cliff edges
            b_new = b_new + np.maximum(0.0, 2.5 - b_new) * 0.65 * skin_zone

        work_rgb = np.clip(lab_to_rgb255(np.dstack([L_new, a_new, b_new])) / 255.0, 0.0, 1.0)
        p_delight = out_d / "relight_00_delight_base_rgba.png"
        _write_png(p_delight, np.dstack([work_rgb, alpha]), "RGBA", overwrite=overwrite)

    luma_fg = 0.2126 * work_rgb[:, :, 0] + 0.7152 * work_rgb[:, :, 1] + 0.0722 * work_rgb[:, :, 2]
    luma_sm = cv2.GaussianBlur(luma_fg, (0, 0), max(5.0, w * 0.002))
    garment_spec = _smoothstep(0.08, 0.52, luma_sm)

    layer_specs: list[dict[str, Any]] = []

    # LAYER 1: relight_01_volume_shadow_multiply.png (Multiply: clean white outside alpha)
    shade_geom = _smoothstep(0.42, -0.35, ndotl_eff)
    shade_macro = _smoothstep(0.28, 0.92, shade_side_bias) * 0.52
    shade_w = np.clip(0.70 * shade_geom + 0.30 * shade_macro + 0.42 * cavity_ao, 0.0, 1.0)
    shade_w *= np.clip(1.0 - float(face_protect) * 0.75 * face_shield, 0.20, 1.0)
    shade_w *= (1.0 - 0.84 * eyes_shield) * (1.0 - 0.40 * garment_core)
    s_amt = np.clip(shade_w * float(shadow_strength), 0.0, 0.82) * alpha
    mult_rgb = 1.0 * (1.0 - s_amt[:, :, None]) + shadow_tint[None, None, :] * s_amt[:, :, None]

    # LAYER 2: relight_02_key_highlight_screen.png (Soft Light neutral-gray lift: preserves micro-contrast & black depth!)
    key_geom = _smoothstep(0.18, 0.85, ndotl_eff) * (1.0 - 0.65 * cavity_ao)
    key_bias = _smoothstep(0.15, 0.82, sun_side_bias)
    hi_guard = 1.0 - np.clip((luma_fg - 0.80) / 0.20, 0.0, 0.85)
    key_gate = (1.0 - garment_soft) + garment_soft * np.clip(garment_spec * 0.90 + 0.10, 0.0, 1.0)
    key_a = np.clip(key_geom * (0.45 + 0.55 * key_bias) * hi_guard * key_gate * float(key_strength) * alpha, 0.0, 1.0)
    key_sl_target = np.clip(0.5 + 0.42 * key_rgb[None, None, :], 0.5, 0.96)
    key_layer_rgb = 0.5 * (1.0 - key_a[:, :, None]) + key_sl_target * key_a[:, :, None]

    # LAYER 3: relight_03_fresnel_rim_screen.png (Screen: silhouette-gated Fresnel rim, ZERO internal body fog halos!)
    fresnel = np.power(np.clip(1.0 - normals_eff[:, :, 2], 0.0, 1.0), 1.65)
    rim_dir = _smoothstep(-0.20, 0.65, normals_eff[:, :, 0] * L[0] + normals_eff[:, :, 1] * L[1] + 0.25)
    fg_u8 = (alpha > 0.25).astype(np.uint8)
    edge_dist = cv2.distanceTransform(fg_u8, cv2.DIST_L2, 5)
    outer_rim = np.exp(-np.square(edge_dist / max(10.0, w * 0.0045))) * alpha
    outer_rim_med = np.exp(-np.square(edge_dist / max(18.0, w * 0.007))) * alpha
    hair_bonus = 1.0
    if body_parse_dir is not None and (body_parse_dir / "mask_hair.png").exists():
        hm = _read_gray_f32(body_parse_dir / "mask_hair.png", (w, h))
        hair_bonus = 1.0 + 0.45 * hm
    rim_garment_guard = 1.0 - 0.82 * garment_core * (1.0 - garment_spec)
    rim_w = np.clip((0.68 * outer_rim + 0.32 * fresnel * outer_rim_med) * (0.35 + 0.65 * rim_dir) * hair_bonus * rim_garment_guard * (1.0 - 0.88 * eyes_shield), 0.0, 1.0)
    rim_a = np.clip(rim_w * float(rim_strength) * alpha, 0.0, 1.0)
    rim_rgb = np.clip(0.55 * bg_blur_wrap + 0.45 * key_rgb[None, None, :], 0.0, 1.0)

    # LAYER 4: relight_04_floor_bounce_softlight.png (Soft Light: neutral gray outside alpha)
    down_facing = _smoothstep(-0.05, 0.55, -normals_eff[:, :, 1])
    low_zone = np.clip((yy - float(ys.min())) / max(1.0, float(ys.max() - ys.min())), 0.0, 1.0)
    bounce_w = np.clip((0.55 * down_facing + 0.45 * low_zone) * low_zone * float(bounce_strength) * (1.0 - 0.45 * garment_core), 0.0, 0.72) * alpha
    bounce_rgb = np.broadcast_to(floor_rgb[None, None, :], (h, w, 3)).copy()

    p_shd = out_d / "relight_01_volume_shadow_multiply.png"
    p_key = out_d / "relight_02_key_highlight_screen.png"
    p_rim = out_d / "relight_03_fresnel_rim_screen.png"
    p_bnc = out_d / "relight_04_floor_bounce_softlight.png"
    p_dmap = out_d / "depth_map_preview.png"
    p_nmap = out_d / "normal_map_rgb.png"

    _write_png(p_shd, np.dstack([mult_rgb, alpha]), "RGBA", overwrite=overwrite)
    _write_png(p_key, np.dstack([key_layer_rgb, key_a]), "RGBA", overwrite=overwrite)
    _write_png(p_rim, np.dstack([rim_rgb, rim_a]), "RGBA", overwrite=overwrite)
    _write_png(p_bnc, np.dstack([np.where(bounce_w[:, :, None] > 0.01, bounce_rgb, 0.5), bounce_w]), "RGBA", overwrite=overwrite)
    _write_png(p_dmap, depth_pol, "L", overwrite=overwrite)
    _write_png(p_nmap, normals_eff * 0.5 + 0.5, "RGB", overwrite=overwrite)

    layer_specs.extend([
        {"key": "volume_shadow", "path": str(p_shd), "layer_name": "01_Relight_Volume_Shadow", "blend": "Multiply", "opacity": 0.72, "fill_opacity": 100, "blend_if": None},
        {"key": "key_highlight", "path": str(p_key), "layer_name": "02_Relight_Key_Highlight", "blend": "Soft Light", "opacity": 0.65, "fill_opacity": 100, "blend_if": None},
        {"key": "fresnel_rim", "path": str(p_rim), "layer_name": "03_Relight_Fresnel_Rim", "blend": "Screen", "opacity": 0.42, "fill_opacity": 100, "blend_if": None},
        {"key": "floor_bounce", "path": str(p_bnc), "layer_name": "04_Relight_Floor_Bounce", "blend": "Soft Light", "opacity": 0.75, "fill_opacity": 100, "blend_if": None},
    ])

    # =========================================================================
    # LAYERS 5-7: Scene Multi-Source Accent Lights (auto-extracted from BG or custom) + Dark Garment Neutral Guard
    # =========================================================================
    active_accents = list(accent_lights) if accent_lights else (_detect_bg_accent_lights(bg01, alpha, resolved_side) if (auto_bg_lights and bg01 is not None) else [])
    diff_accum = np.zeros((h, w, 3), dtype=np.float32)
    spec_accum = np.zeros((h, w, 3), dtype=np.float32)

    if active_accents:
        diff_gate = (1.0 - garment_soft) + garment_soft * np.clip(garment_spec * 0.85 + outer_rim, 0.0, 1.0) * (1.0 - 0.70 * garment_core)
        spec_gate = (1.0 - garment_soft) * outer_rim * 0.85 + garment_soft * np.clip(garment_spec * (1.0 - 0.55 * garment_core) + outer_rim * 0.90, 0.0, 1.0)
        face_eye_d = (1.0 - 0.94 * eyes_shield) * (1.0 - 0.68 * face_shield)
        face_eye_s = (1.0 - 0.95 * eyes_shield) * (1.0 - 0.82 * face_shield)

        for al in active_accents:
            lx = float(al["x_frac"]) * float(w)
            ly = float(al["y_frac"]) * float(h)
            lz = float(al.get("z", 0.20))
            rad = float(al.get("radius_frac", 0.35)) * float(w)
            st = float(al.get("strength", 0.60))
            col = np.array(al["rgb"], dtype=np.float32)[None, None, :]
            n_pt, f_pt = _point_light_field(xx, yy, normals_eff[:, :, 0], normals_eff[:, :, 1], normals_eff[:, :, 2], lx, ly, lz, rad)

            d_w = np.clip(n_pt * f_pt * diff_gate * face_eye_d * st * alpha, 0.0, 1.0)
            s_w = np.clip((0.70 * n_pt + 0.30 * fresnel) * f_pt * spec_gate * face_eye_s * st * alpha, 0.0, 1.0)
            diff_accum += d_w[:, :, None] * col
            spec_accum += s_w[:, :, None] * col

    diff_a = np.clip(diff_accum.max(axis=2), 0.0, 1.0)
    diff_a = np.clip(diff_a / (1.0 + 0.45 * diff_a), 0.0, 1.0)
    diff_rgb = np.where(diff_a[:, :, None] > 1e-4, np.clip(diff_accum / np.maximum(diff_accum.max(axis=2, keepdims=True), 1e-4), 0.0, 1.0), 0.0)

    spec_a = np.clip(spec_accum.max(axis=2), 0.0, 1.0)
    spec_a = np.clip(spec_a / (1.0 + 0.55 * spec_a), 0.0, 1.0)
    spec_rgb = np.where(spec_a[:, :, None] > 1e-4, np.clip(spec_accum / np.maximum(spec_accum.max(axis=2, keepdims=True), 1e-4), 0.0, 1.0), 0.0)

    p_acc_d = out_d / "relight_05_scene_accent_diffuse_screen.png"
    p_acc_s = out_d / "relight_06_scene_accent_spec_add.png"
    _write_png(p_acc_d, np.dstack([diff_rgb, diff_a]), "RGBA", overwrite=overwrite)
    _write_png(p_acc_s, np.dstack([spec_rgb, spec_a]), "RGBA", overwrite=overwrite)
    if active_accents:
        layer_specs.extend([
            {"key": "accent_diffuse", "path": str(p_acc_d), "layer_name": "05_Relight_Accent_Diffuse", "blend": "Screen", "opacity": 0.72, "fill_opacity": 100, "blend_if": None},
            {"key": "accent_spec", "path": str(p_acc_s), "layer_name": "06_Relight_Accent_Specular", "blend": "Linear Dodge (Add)", "opacity": 0.85, "fill_opacity": 48, "blend_if": None},
        ])

    p_gcol: Path | None = None
    if garment_soft.max() > 0.05:
        p_gcol = out_d / "relight_07_dark_garment_neutral_color.png"
        g_guard_a = np.clip(garment_core * alpha * (1.0 - 0.60 * garment_spec * spec_a), 0.0, 1.0)
        _write_png(p_gcol, np.dstack([np.full((h, w, 3), 0.10, dtype=np.float32), g_guard_a]), "RGBA", overwrite=overwrite)
        layer_specs.append({
            "key": "garment_neutral_guard",
            "path": str(p_gcol),
            "layer_name": "07_Relight_DarkGarment_NeutralGuard",
            "blend": "Color",
            "opacity": 0.75,
            "fill_opacity": 100,
            "blend_if": None,
        })

    # Simulate physical composite of de-lit base + all relight layers for visual verification
    comp = work_rgb * mult_rgb
    comp = 1.0 - (1.0 - comp) * (1.0 - key_layer_rgb * key_a[:, :, None])
    comp = 1.0 - (1.0 - comp) * (1.0 - rim_rgb * rim_a[:, :, None])
    sl = np.where(bounce_rgb <= 0.5, comp - (1.0 - 2.0 * bounce_rgb) * comp * (1.0 - comp), comp + (2.0 * bounce_rgb - 1.0) * (np.sqrt(np.clip(comp, 0.0, 1.0)) - comp))
    comp = np.clip(comp * (1.0 - bounce_w[:, :, None]) + sl * bounce_w[:, :, None], 0.0, 1.0)
    if active_accents:
        comp = 1.0 - (1.0 - comp) * (1.0 - diff_rgb * (diff_a[:, :, None] * 0.72))
        comp = np.clip(comp + spec_rgb * (spec_a[:, :, None] * 0.42), 0.0, 1.0)

    def _lr_gap(img01: np.ndarray) -> float:
        lab_t = cv2.cvtColor(np.clip(np.round(img01 * 255.0), 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        Lt = lab_t[:, :, 0] * (100.0 / 255.0)
        m_l = (alpha > 0.6) & (x_rel < 0.38)
        m_r = (alpha > 0.6) & (x_rel > 0.62)
        if not m_l.any() or not m_r.any():
            return 0.0
        return round(float(Lt[m_l].mean() - Lt[m_r].mean()), 2)

    backdrop = np.array([0.14, 0.16, 0.20], dtype=np.float32)[None, None, :]
    vis_orig = rgb_f * alpha[:, :, None] + backdrop * (1.0 - alpha[:, :, None])
    vis_norm = (normals_eff * 0.5 + 0.5) * alpha[:, :, None] + backdrop * (1.0 - alpha[:, :, None])
    vis_relit = comp * alpha[:, :, None] + backdrop * (1.0 - alpha[:, :, None])
    bx0, by0, bx1, by1 = max(0, int(xs.min()) - 40), max(0, int(ys.min()) - 40), min(w, int(xs.max()) + 40), min(h, int(ys.max()) + 40)
    cw_p = min(560, bx1 - bx0)
    ch_p = max(240, int(round((by1 - by0) * cw_p / max(1, bx1 - bx0))))
    panels = []
    for arr_v in (vis_orig, vis_norm, vis_relit):
        sub_im = Image.fromarray(np.clip(np.round(arr_v[by0:by1, bx0:bx1] * 255.0), 0, 255).astype(np.uint8))
        panels.append(sub_im.resize((cw_p, ch_p), Image.Resampling.LANCZOS))
    cmp_im = Image.new("RGB", (cw_p * 3 + 48, ch_p + 56), (18, 20, 26))
    dr = ImageDraw.Draw(cmp_im)
    titles = ("1. Convention-Hall Input", "2. 2.5D Normals + Accents", f"3. De-Lit + Relit ({resolved_side})")
    for idx, (pim, ttl) in enumerate(zip(panels, titles)):
        px = 12 + idx * (cw_p + 12)
        dr.text((px, 12), ttl, fill=(230, 236, 245))
        cmp_im.paste(pim, (px, 40))
    # Annotate accent light directions on Panel 2
    for al in active_accents:
        ax_l = float(al["x_frac"]) * float(w)
        ay_l = float(al["y_frac"]) * float(h)
        px_m = int(12 + (cw_p + 12) + np.clip((ax_l - bx0) * cw_p / max(1, bx1 - bx0), 8, cw_p - 8))
        py_m = int(40 + np.clip((ay_l - by0) * ch_p / max(1, by1 - by0), 8, ch_p - 8))
        mc = tuple(int(c * 255) for c in al["rgb"])
        dr.ellipse([px_m - 7, py_m - 7, px_m + 7, py_m + 7], fill=mc, outline=(255, 255, 255))

    p_cmp = out_d / "relight_compare_preview.png"
    cmp_im.save(p_cmp, "PNG")

    fg_solid = alpha > 0.6
    luma_after = 0.2126 * comp[:, :, 0] + 0.7152 * comp[:, :, 1] + 0.0722 * comp[:, :, 2]
    outputs: dict[str, Any] = {
        "relight_01_volume_shadow_multiply": str(p_shd),
        "relight_02_key_highlight_screen": str(p_key),
        "relight_03_fresnel_rim_screen": str(p_rim),
        "relight_04_floor_bounce_softlight": str(p_bnc),
        "relight_05_scene_accent_diffuse_screen": str(p_acc_d),
        "relight_06_scene_accent_spec_add": str(p_acc_s),
        "depth_map_preview": str(p_dmap),
        "normal_map_rgb": str(p_nmap),
        "relight_compare_preview": str(p_cmp),
    }
    if p_delight is not None:
        outputs["relight_00_delight_base_rgba"] = str(p_delight)
    if p_gcol is not None:
        outputs["relight_07_dark_garment_neutral_color"] = str(p_gcol)

    res_payload: dict[str, Any] = {
        "ok": True,
        "input": str(fg_p),
        "sun_side": resolved_side,
        "delight_strength": dl_s,
        "detected_accent_lights": active_accents,
        "left_minus_right_L_before": _lr_gap(rgb_f),
        "left_minus_right_L_after": _lr_gap(comp),
        "shadow_side_mean_luma_before": round(float(luma_fg[fg_solid].mean()), 3),
        "shadow_side_mean_luma_after": round(float(luma_after[fg_solid].mean()), 3),
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "layers": layer_specs,
        "outputs": outputs,
    }
    p_meta = out_d / "relight_layers_meta.json"
    p_meta.write_text(json.dumps(res_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["relight_layers_meta"] = str(p_meta)
    return res_payload


def _cinema_preset_config(preset: str) -> dict[str, Any]:
    configs: dict[str, dict[str, Any]] = {
        "crystal-sunset-cinema": {
            "lut": "03_cinema_crystal_highkey_agx",
            "lut_opacity": 0.62,
            "camera_raw": {
                "contrast": 14,
                "highlights": -10,
                "shadows": -8,
                "whites": 4,
                "blacks": -22,
                "clarity": 8,
                "vibrance": 10,
                "gradeShadows": {"hue": 208, "sat": 22, "lum": -8},
                "gradeMidtones": {"hue": 35, "sat": 6, "lum": 2},
                "gradeHighlights": {"hue": 38, "sat": 16, "lum": 4},
                "vignetteAmount": -18,
                "vignetteMidpoint": 45,
                "vignetteFeather": 75,
            },
            "color_balance": {"shadows": [-8, 0, 10], "midtones": [2, 0, -2], "highlights": [8, 2, -7]},
        },
        "sunset-teal-gold": {
            "lut": "01_cinema_sunset_teal_gold",
            "lut_opacity": 0.72,
            "camera_raw": {
                "contrast": 18,
                "highlights": -12,
                "shadows": -10,
                "blacks": -26,
                "clarity": 12,
                "vibrance": 14,
                "gradeShadows": {"hue": 198, "sat": 26, "lum": -10},
                "gradeHighlights": {"hue": 36, "sat": 22, "lum": 4},
                "vignetteAmount": -24,
                "vignetteMidpoint": 42,
                "vignetteFeather": 75,
            },
            "color_balance": {"shadows": [-12, 2, 12], "midtones": [4, 0, -4], "highlights": [10, 3, -9]},
        },
        "kodak2383": {
            "lut": "02_cinema_kodak2383_filmic",
            "lut_opacity": 0.76,
            "camera_raw": {
                "contrast": 16,
                "highlights": -14,
                "shadows": -12,
                "blacks": -24,
                "clarity": 10,
                "gradeShadows": {"hue": 205, "sat": 18, "lum": -10},
                "gradeHighlights": {"hue": 40, "sat": 18, "lum": 2},
                "vignetteAmount": -22,
            },
            "color_balance": {"shadows": [-8, 0, 8], "midtones": [3, 0, -3], "highlights": [9, 3, -8]},
        },
        "moonlight-crimson": {
            "lut": "04_cinema_moonlight_crimson",
            "lut_opacity": 0.74,
            "camera_raw": {
                "contrast": 20,
                "highlights": -10,
                "shadows": -14,
                "blacks": -28,
                "clarity": 12,
                "gradeShadows": {"hue": 220, "sat": 28, "lum": -12},
                "gradeHighlights": {"hue": 350, "sat": 14, "lum": 2},
                "vignetteAmount": -28,
            },
            "color_balance": {"shadows": [-14, -2, 18], "midtones": [5, -4, 0], "highlights": [6, -2, 2]},
        },
        "epic-slate": {
            "lut": "05_cinema_epic_slate_snow",
            "lut_opacity": 0.72,
            "camera_raw": {
                "contrast": 18,
                "highlights": -8,
                "shadows": -10,
                "blacks": -24,
                "clarity": 14,
                "gradeShadows": {"hue": 212, "sat": 20, "lum": -8},
                "gradeHighlights": {"hue": 38, "sat": 10, "lum": 2},
                "vignetteAmount": -22,
            },
            "color_balance": {"shadows": [-10, 0, 12], "midtones": [-2, 0, 2], "highlights": [5, 2, -5]},
        },
    }
    return configs.get(preset, configs["crystal-sunset-cinema"])


def _build_bg_anchor_and_fg_frame(
    bg_path: Path,
    out_dir: Path,
    sun_side: str,
    subject_x_frac: float,
    overwrite: bool,
) -> dict[str, Any]:
    bg = _read_rgb_f32(bg_path)
    h, w = bg.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn, yn = xx / max(1.0, float(w)), yy / max(1.0, float(h))

    # 1. Spotlight + Architectural Shadow Anchoring (breaks flat milky p02=55.7 while keeping sunbeams luminous!)
    luma = 0.2126 * bg[:, :, 0] + 0.7152 * bg[:, :, 1] + 0.0722 * bg[:, :, 2]
    # Preserve brightest sunlight corridor (~top 15%) and subject focus pool
    sun_x = 0.72 if "right" in sun_side else (0.28 if "left" in sun_side else 0.50)
    subj_pool = np.exp(-0.5 * (((xn - subject_x_frac) / 0.28) ** 2 + ((yn - 0.52) / 0.42) ** 2))
    piano_pool = np.exp(-0.5 * (((xn - (1.0 - subject_x_frac * 0.7)) / 0.25) ** 2 + ((yn - 0.65) / 0.30) ** 2))
    sun_pool = np.exp(-0.5 * (((xn - sun_x) / 0.28) ** 2 + ((yn - 0.42) / 0.36) ** 2))
    focus_mask = np.clip(np.maximum.reduce([subj_pool, 0.80 * piano_pool, 0.90 * sun_pool]), 0.0, 1.0)

    # Pull washed non-highlight midtones/shadows downward with rich S-toe & cool slate split tone
    bg_toe = np.clip((bg - 0.08) / 0.92, 0.0, 1.0)
    bg_filmic = np.power(bg_toe, 1.28)
    # Cool cyan-slate shadow pull in darker architecture, keep blazing sunlight warm-ivory
    w_shd = np.power(np.clip(1.0 - luma / 0.72, 0.0, 1.0), 1.4)[:, :, None]
    w_hi = np.power(np.clip((luma - 0.76) / 0.24, 0.0, 1.0), 1.3)[:, :, None]
    cool_shd = np.array([0.82, 0.92, 1.06], dtype=np.float32)[None, None, :]
    bg_cool = np.clip(bg_filmic * (1.0 * (1.0 - 0.48 * w_shd) + cool_shd * (0.48 * w_shd)), 0.0, 1.0)

    # Blend anchored theatrical shadow depth stronger outside focal spotlight and toward corners
    corner_dark = np.clip(1.0 - 0.38 * (1.0 - focus_mask), 0.55, 1.0)[:, :, None]
    bg_out = bg_cool * (0.86 - 0.26 * focus_mask)[:, :, None] + bg * (0.14 + 0.26 * focus_mask)[:, :, None]
    bg_out = np.clip(bg_out * corner_dark, 0.0, 1.0)
    # Restore top highlight sparkle & sunlit window beam glow cleanly
    bg_out = bg_out * (1.0 - 0.55 * w_hi) + np.clip(bg * np.array([1.02, 0.99, 0.94], dtype=np.float32), 0.0, 1.0) * (0.55 * w_hi)

    p_bg_anch = out_dir / "bg_anchored_4k.png"
    _write_png(p_bg_anch, bg_out, "RGB", overwrite=overwrite)

    # 2. Defocused Dark Foreground Depth Frame RGBA (anchors outer perimeter & lower stage like 38 cinema refs!)
    fg_frame = np.zeros((h, w, 4), dtype=np.float32)
    # Smooth defocused architectural balustrade/vignette curve near bottom strip & extreme left/right bottom corners
    bot_band = np.clip((yn - 0.84) / 0.16, 0.0, 1.0)
    # Keep title area (x<0.35, y<0.40) and center hero clear; softly anchor top corners and lower perimeter
    r_edge = np.exp(-0.5 * (((xn - 1.04) / 0.08) ** 2 + ((yn - 0.72) / 0.48) ** 2))
    l_bot_arch = np.exp(-0.5 * (((xn + 0.02) / 0.18) ** 2 + ((yn - 0.96) / 0.14) ** 2))
    top_right_corner = np.exp(-0.5 * (((xn - 1.03) / 0.14) ** 2 + ((yn + 0.04) / 0.22) ** 2))
    # Subtle soft depth framing alpha (deep cool twilight teal-navy shadow)
    frame_a = np.clip(
        np.maximum.reduce([bot_band * 0.48 * (0.45 + 0.55 * np.abs(xn - 0.52)), 0.72 * r_edge, 0.65 * l_bot_arch, 0.50 * top_right_corner]),
        0.0,
        0.78,
    )
    frame_rgb = np.zeros((h, w, 3), dtype=np.float32)
    frame_rgb[:, :, 0] = 0.09
    frame_rgb[:, :, 1] = 0.14
    frame_rgb[:, :, 2] = 0.20
    fg_frame[:, :, :3] = frame_rgb
    fg_frame[:, :, 3] = frame_a
    p_fgf = out_dir / "fg_depth_frame_rgba.png"
    _write_png(p_fgf, fg_frame, "RGBA", overwrite=overwrite)

    lab_b = cv2.cvtColor(np.clip(np.round(bg * 255.0), 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)[:, :, 0] * (100.0 / 255.0)
    lab_a = cv2.cvtColor(np.clip(np.round(bg_out * 255.0), 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)[:, :, 0] * (100.0 / 255.0)
    return {
        "ok": True,
        "mode": "bg-anchor",
        "bg_p02_L_before": round(float(np.percentile(lab_b, 2)), 2),
        "bg_p02_L_after": round(float(np.percentile(lab_a, 2)), 2),
        "bg_p50_L_after": round(float(np.percentile(lab_a, 50)), 2),
        "outputs": {"bg_anchored_4k": str(p_bg_anch), "fg_depth_frame_rgba": str(p_fgf)},
    }


def run_cinema_grade(
    *,
    input_path: Path,
    out_dir: Path,
    mode: str = "grade",
    preset: str = "crystal-sunset-cinema",
    sun_side: str = "right",
    subject_x_frac: float = 0.70,
    photocraft_cli: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build anchored cinema background + fg depth frame, or apply PhotoCraft Camera Raw + Tetrahedral .cube LUT grade."""
    t0 = time.perf_counter()
    in_p = input_path.resolve()
    out_d = out_dir.resolve()
    out_d.mkdir(parents=True, exist_ok=True)
    luts = ensure_cinema_lut_library()

    if mode == "bg-anchor":
        res = _build_bg_anchor_and_fg_frame(in_p, out_d, sun_side, subject_x_frac, overwrite)
        res["elapsed_s"] = round(time.perf_counter() - t0, 2)
        res["lut_count"] = len(luts)
        return res

    # Targeted highlight-only film halation layer (strictly luma > 0.86 so midtones/shadows stay crisp!)
    rgb = _read_rgb_f32(in_p)
    h, w = rgb.shape[:2]
    luma = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    hi_src = np.clip((luma - 0.84) / 0.16, 0.0, 1.0)[:, :, None] * rgb
    halo_w = cv2.GaussianBlur(hi_src, (0, 0), max(16.0, w * 0.009))
    halo_r = cv2.GaussianBlur(hi_src[:, :, 0], (0, 0), max(32.0, w * 0.018))
    hal_rgb = np.clip(halo_w * np.array([1.0, 0.86, 0.68], dtype=np.float32)[None, None, :] + np.dstack([halo_r * 0.28, halo_r * 0.09, halo_r * 0.03]), 0.0, 1.0)
    hal_a = np.clip(hal_rgb.max(axis=2) * 0.65, 0.0, 1.0)
    p_hal = out_d / "halation_bloom_targeted.png"
    _write_png(p_hal, np.dstack([hal_rgb, hal_a]), "RGBA", overwrite=overwrite)

    cfg = _cinema_preset_config(preset)
    lut_path = luts[cfg["lut"]]

    def _J(o: object) -> str:
        return json.dumps(o, separators=(",", ":"), ensure_ascii=False)

    grade_cmds = [
        {"cmd": "filter.cameraRaw", "params": cfg["camera_raw"]},
        {
            "cmd": "layer.newAdjustmentLayer.colorLookup",
            "params": {
                "file": str(lut_path),
                "interpolation": "tetrahedral",
                "dither": True,
                "opacity": cfg["lut_opacity"],
                "name": f"LUT_{cfg['lut']}",
                "mode": "normal",
            },
        },
        {
            "cmd": "layer.newAdjustmentLayer.colorBalance",
            "params": {
                **cfg["color_balance"],
                "preserveLuminosity": True,
                "opacity": 0.55,
                "name": "Cinema_SplitTone_ColorBalance",
                "mode": "normal",
            },
        },
    ]
    p_cmds = out_d / "photocraft_grade_cmds.json"
    p_cmds.write_text(json.dumps(grade_cmds, indent=2, ensure_ascii=False), encoding="utf-8")

    p_graded = out_d / "graded_cinema_out.png"
    pc_exe = photocraft_cli
    if pc_exe is None or not pc_exe.exists():
        raise FileNotFoundError(f"PhotoCraft CLI required for native cinema grading: {pc_exe}")

    cli_args = [str(pc_exe), "run", str(in_p)]
    for item in grade_cmds:
        cli_args += ["--cmd", str(item["cmd"]), "--params", _J(item["params"])]
    cli_args += ["--out", str(p_graded)]
    proc = subprocess.run(cli_args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"PhotoCraft cinema-grade failed (rc={proc.returncode}): {proc.stderr[-1200:]}")

    g_arr = np.array(Image.open(p_graded).convert("RGB"))
    g_l = cv2.cvtColor(g_arr, cv2.COLOR_RGB2LAB).astype(np.float32)[:, :, 0] * (100.0 / 255.0)
    p02, p50, p98 = [round(float(x), 2) for x in np.percentile(g_l, [2, 50, 98])]
    return {
        "ok": True,
        "mode": "grade",
        "preset": preset,
        "lut_file": str(lut_path),
        "graded_p02_L": p02,
        "graded_p50_L": p50,
        "graded_p98_L": p98,
        "graded_dyn_range": round(p98 - p02, 2),
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "outputs": {
            "graded_cinema_out": str(p_graded),
            "halation_bloom_targeted": str(p_hal),
            "photocraft_grade_cmds": str(p_cmds),
        },
    }


def _add_silky_bloom_rgba(rgba_path: Path, sigma_inner: float, sigma_outer: float, bloom_gain: float, avoid_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    rgb, alpha = _read_rgba_f32(rgba_path)
    premult = rgb * alpha[:, :, None]
    glow1 = cv2.GaussianBlur(premult, (0, 0), sigma_inner)
    glow2 = cv2.GaussianBlur(premult, (0, 0), sigma_outer)
    comb = np.clip(premult + (0.60 * glow1 + 0.40 * glow2) * bloom_gain, 0.0, 1.0)
    if avoid_mask is not None:
        comb *= avoid_mask[:, :, None]
    out_a = np.clip(comb.max(axis=2), 0.0, 1.0)
    out_rgb = np.where(out_a[:, :, None] > 1e-4, np.clip(comb / np.maximum(out_a[:, :, None], 1e-4), 0.0, 1.0), 0.0)
    return out_rgb, out_a


def run_fg_vfx_3d(
    *,
    blender_exe: Path,
    kit_script: Path,
    preset: str,
    width: int,
    height: int,
    render_width: int,
    render_height: int,
    samples: int,
    out_dir: Path,
    subject_alpha_path: Path | None = None,
    face_parse_dir: Path | None = None,
    normal_map_path: Path | None = None,
    subject_x_frac: float = 0.70,
    sun_side: str = "right",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render 3D optical-bokeh foreground props + pose-wrapped melody ribbons + causal spill + arch back-rim via Blender Cycles."""
    t0 = time.perf_counter()
    out_d = out_dir.resolve()
    out_d.mkdir(parents=True, exist_ok=True)
    raw_dir = out_d / "raw_render"
    raw_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(blender_exe),
        "--background",
        "--python-exit-code", "1",
        "--python", str(kit_script),
        "--",
        "--preset", preset,
        "--width", str(render_width),
        "--height", str(render_height),
        "--samples", str(samples),
        "--subject-x-frac", str(subject_x_frac),
        "--out-dir", str(raw_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"Blender fg-vfx-3d failed (rc={proc.returncode}): {proc.stderr[-1000:] or proc.stdout[-1000:]}")

    # Load face/eyes shield at 4K so front ribbon or spill NEVER blurs over eyes or inner face
    face_avoid = np.ones((height, width), dtype=np.float32)
    eyes_shield = np.zeros((height, width), dtype=np.float32)
    if face_parse_dir is not None and face_parse_dir.exists():
        for fn, strength in (("mask_eyes_brows.png", 1.0), ("mask_skin.png", 0.78)):
            fp = face_parse_dir / fn
            if fp.exists():
                m = _read_gray_f32(fp, (width, height))
                m_dil = cv2.dilate((m > 0.25).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))).astype(np.float32)
                m_blur = cv2.GaussianBlur(m_dil, (0, 0), 18.0)
                face_avoid *= np.clip(1.0 - strength * m_blur, 0.0, 1.0)
                if fn == "mask_eyes_brows.png":
                    eyes_shield = np.clip(m_blur, 0.0, 1.0)

    def _upscale_rgba(src_p: Path, dst_p: Path) -> tuple[np.ndarray, np.ndarray]:
        im = Image.open(src_p).convert("RGBA")
        if im.size != (width, height):
            im = im.resize((width, height), Image.Resampling.LANCZOS)
        arr = np.array(im, dtype=np.float32) / 255.0
        _write_png(dst_p, arr, "RGBA", overwrite=overwrite)
        return arr[:, :, :3], arr[:, :, 3]

    p_fg_4k = out_d / "fg_3d_props_rgba.png"
    fg_rgb, fg_a = _upscale_rgba(raw_dir / "fg_3d_props_raw.png", p_fg_4k)

    # Upscale & add clean optical silk bloom to Behind and Front pose-wrapped filaments
    p_beh_tmp = out_d / "vfx_wrap_behind_raw_4k.png"
    p_frt_tmp = out_d / "vfx_wrap_front_raw_4k.png"
    _upscale_rgba(raw_dir / "vfx_wrap_behind_raw.png", p_beh_tmp)
    _upscale_rgba(raw_dir / "vfx_wrap_front_raw.png", p_frt_tmp)

    beh_rgb, beh_a = _add_silky_bloom_rgba(p_beh_tmp, sigma_inner=12.0, sigma_outer=42.0, bloom_gain=0.85, avoid_mask=None)
    frt_rgb, frt_a = _add_silky_bloom_rgba(p_frt_tmp, sigma_inner=14.0, sigma_outer=46.0, bloom_gain=0.95, avoid_mask=face_avoid)

    p_beh = out_d / "vfx_wrap_behind_rgba.png"
    p_frt = out_d / "vfx_wrap_front_rgba.png"
    _write_png(p_beh, np.dstack([beh_rgb, beh_a]), "RGBA", overwrite=overwrite)
    _write_png(p_frt, np.dstack([frt_rgb, frt_a]), "RGBA", overwrite=overwrite)

    outputs: dict[str, str] = {
        "fg_3d_props_rgba": str(p_fg_4k),
        "vfx_wrap_behind_rgba": str(p_beh),
        "vfx_wrap_front_rgba": str(p_frt),
    }

    # Build Causal Subject Illumination (local warm-gold/mint bounce onto Miku right beside the wrapping filaments!)
    if subject_alpha_path is not None and subject_alpha_path.exists():
        sa_im = Image.open(subject_alpha_path.resolve())
        sa_ch = sa_im.getchannel("A") if sa_im.mode in ("RGBA", "LA") else sa_im.convert("L")
        if sa_ch.size != (width, height):
            sa_ch = sa_ch.resize((width, height), Image.Resampling.BILINEAR)
        sa = np.array(sa_ch, dtype=np.float32) / 255.0

        wrap_energy = np.clip(frt_rgb * frt_a[:, :, None] + beh_rgb * beh_a[:, :, None] * 0.65, 0.0, 1.0)
        glow_tight = cv2.GaussianBlur(wrap_energy, (0, 0), 28.0)
        glow_wide = cv2.GaussianBlur(wrap_energy, (0, 0), 96.0)
        spill_rgb = np.clip(0.55 * glow_tight + 0.45 * glow_wide, 0.0, 1.0)
        # Boost edge + front-facing surface catch using normal map if provided
        geom_w = np.ones((height, width), dtype=np.float32)
        if normal_map_path is not None and normal_map_path.exists():
            nmap = _read_rgb_f32(normal_map_path.resolve())
            if nmap.shape[:2] != (height, width):
                nmap = cv2.resize(nmap, (width, height), interpolation=cv2.INTER_LINEAR)
            nz = np.clip((nmap[:, :, 2] - 0.5) * 2.0, 0.15, 1.0)
            geom_w = 0.55 + 0.45 * nz
        spill_a = np.clip(spill_rgb.max(axis=2) * 1.45 * sa * geom_w * face_avoid, 0.0, 1.0)
        spill_col = np.where(spill_a[:, :, None] > 1e-4, np.clip(spill_rgb / np.maximum(spill_rgb.max(axis=2, keepdims=True), 1e-4), 0.0, 1.0), 0.0)
        p_spill = out_d / "vfx_wrap_causal_spill_screen.png"
        _write_png(p_spill, np.dstack([spill_col, spill_a]), "RGBA", overwrite=overwrite)
        outputs["vfx_wrap_causal_spill_screen"] = str(p_spill)

        # Build Central-Arch Sunset Back-Rim along viewer-left silhouette where the blazing arch pours behind Miku
        ys, xs = np.nonzero(sa > 0.15)
        x_rel = np.clip((np.arange(width, dtype=np.float32)[None, :] - float(xs.min())) / max(1.0, float(xs.max() - xs.min())), 0.0, 1.0)
        in_fg = (sa > 0.20).astype(np.uint8)
        dist_edge = cv2.distanceTransform(in_fg, cv2.DIST_L2, 5)
        arch_rim_band = np.exp(-np.square(dist_edge / max(12.0, width * 0.005))) * sa
        left_arch_facing = _smoothstep(0.55, 0.05, x_rel) if sun_side in ("right", "top-right") else _smoothstep(0.45, 0.95, x_rel)
        arch_a = np.clip(arch_rim_band * (0.25 + 0.75 * left_arch_facing) * (1.0 - 0.85 * eyes_shield), 0.0, 1.0)
        arch_rgb = np.broadcast_to(np.array([1.0, 0.84, 0.62], dtype=np.float32)[None, None, :], (height, width, 3)).copy()
        p_arch = out_d / "subject_arch_backrim_screen.png"
        _write_png(p_arch, np.dstack([arch_rgb, arch_a]), "RGBA", overwrite=overwrite)
        outputs["subject_arch_backrim_screen"] = str(p_arch)

    # Save dark studio preview so we can visually inspect props + ribbon cleanly
    studio = np.full((height, width, 3), 0.10, dtype=np.float32)
    studio = np.clip(studio + beh_rgb * beh_a[:, :, None] + frt_rgb * frt_a[:, :, None], 0.0, 1.0)
    studio = np.clip(studio * (1.0 - fg_a[:, :, None]) + fg_rgb * fg_a[:, :, None], 0.0, 1.0)
    p_prev = out_d / "fg_vfx_3d_preview.png"
    Image.fromarray(np.clip(np.round(studio * 255.0), 0, 255).astype(np.uint8), "RGB").resize((1280, 720), Image.Resampling.LANCZOS).save(p_prev, "PNG")
    outputs["fg_vfx_3d_preview"] = str(p_prev)

    return {
        "ok": True,
        "preset": preset,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "fg_props_coverage_pct": round(100.0 * float((fg_a > 0.08).mean()), 2),
        "wrap_front_eye_overlap_px": int(((frt_a > 0.05) & (eyes_shield > 0.3)).sum()),
        "outputs": outputs,
    }
