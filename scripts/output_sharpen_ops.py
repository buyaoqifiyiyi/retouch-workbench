"""Halo-free selective focus-plane output sharpening engine for PhotoCraft workflows.

Combines linear-light luminosity Richardson-Lucy (RL) deconvolution with
asymmetric bright-halo clamping, high-contrast edge-stopping, cutout silhouette
ringing suppression, and a 4-tier semantic focus-plane weight map (eyes/lips >
hair/garment > foreground structural background, while shielding smooth skin and
bokeh/sky from gritty noise or ringing).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def _J(obj: object) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    if abs(edge1 - edge0) < 1e-6:
        return (x >= edge1).astype(np.float32)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _read_rgb_f32(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _read_optional_mask(path: Path | None, size_wh: tuple[int, int], *, channel: str = "L") -> np.ndarray | None:
    if path is None or not Path(path).exists():
        return None
    im = Image.open(Path(path).resolve())
    if channel == "A" and im.mode in ("RGBA", "LA"):
        ch = im.getchannel("A")
    else:
        ch = im.convert("L")
    if ch.size != size_wh:
        ch = ch.resize(size_wh, Image.Resampling.BILINEAR)
    return np.array(ch, dtype=np.float32) / 255.0


def _write_png(path: Path, arr01: np.ndarray, mode: str, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = np.clip(np.round(arr01 * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(u8, mode).save(path, "PNG")


def _luma_stats(img01: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    if int(mask.sum()) < 16:
        return {"lap_var": 0.0, "hp1_std": 0.0}
    gray255 = (0.2126 * img01[:, :, 0] + 0.7152 * img01[:, :, 1] + 0.0722 * img01[:, :, 2]) * 255.0
    lap = cv2.Laplacian(gray255, cv2.CV_32F)
    hp1 = gray255 - cv2.GaussianBlur(gray255, (0, 0), 1.0)
    return {
        "lap_var": round(float(np.var(lap[mask])), 2),
        "hp1_std": round(float(np.std(hp1[mask])), 3),
    }


def run_output_sharpen(
    input_path: Path,
    out_dir: Path,
    *,
    style: str = "photo",
    strength: float | None = None,
    radius: float | None = None,
    iterations: int = 8,
    bg_strength: float | None = None,
    subject_alpha: Path | None = None,
    eyes_mask: Path | None = None,
    lips_mask: Path | None = None,
    hair_mask: Path | None = None,
    garment_mask: Path | None = None,
    skin_mask: Path | None = None,
    face_skin_mask: Path | None = None,
    blend_mode: str = "Linear Light",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Compute a halo-free non-destructive luminance output-sharpening layer & focus map."""
    in_p = Path(input_path).resolve()
    out_d = Path(out_dir).resolve()
    out_d.mkdir(parents=True, exist_ok=True)

    if style not in ("photo", "painterly"):
        raise ValueError(f"Unsupported style: {style!r}")
    if blend_mode not in ("Linear Light", "Luminosity"):
        raise ValueError(f"Unsupported blend_mode: {blend_mode!r}")

    eff_strength = float(strength if strength is not None else (0.92 if style == "photo" else 0.72))
    eff_radius = float(radius if radius is not None else (0.95 if style == "photo" else 0.90))
    eff_bg = float(bg_strength if bg_strength is not None else (0.28 if style == "photo" else 0.20))
    iters = max(1, min(16, int(iterations)))

    img01 = _read_rgb_f32(in_p)
    h, w = img01.shape[:2]
    size_wh = (w, h)

    # 1. Exact sRGB -> Linear luminance for optical PSF deconvolution
    lin = np.where(img01 <= 0.04045, img01 / 12.92, np.power((img01 + 0.055) / 1.055, 2.4)).astype(np.float32)
    y_lin = np.clip(0.2126 * lin[:, :, 0] + 0.7152 * lin[:, :, 1] + 0.0722 * lin[:, :, 2], 1e-5, 1.0)
    y_srgb = np.clip(0.2126 * img01[:, :, 0] + 0.7152 * img01[:, :, 1] + 0.0722 * img01[:, :, 2], 0.0, 1.0)

    # 2. Damped Richardson-Lucy (RL) Deconvolution in Linear Light + fine micro-acutance capture
    est = y_lin.copy()
    for _ in range(iters):
        conv = np.maximum(cv2.GaussianBlur(est, (0, 0), eff_radius), 1e-5)
        ratio = np.clip(y_lin / conv, 0.55, 1.65)
        corr = cv2.GaussianBlur(ratio, (0, 0), eff_radius)
        est = np.clip(est * corr, 1e-5, 1.0)

    y_rl_srgb = np.where(est <= 0.0031308, est * 12.92, 1.055 * np.power(est, 1.0 / 2.4) - 0.055).astype(np.float32)
    dY_rl = y_rl_srgb - y_srgb
    dY_fine = (y_srgb - cv2.GaussianBlur(y_srgb, (0, 0), 0.65)) * 0.45
    dY_raw = dY_rl * 0.85 + dY_fine * 0.55

    # 3. Anti-halo, noise-floor dead-zone & cutout silhouette guards
    thresh = 0.0045 if style == "photo" else 0.0060
    dead_w = _smoothstep(thresh * 0.5, thresh * 2.2, np.abs(dY_raw))
    tone_w = _smoothstep(0.025, 0.09, y_srgb) * _smoothstep(0.96, 0.84, y_srgb)
    dY = dY_raw * dead_w * tone_w

    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    local_span = cv2.dilate(y_srgb, k5) - cv2.erode(y_srgb, k5)
    pos_edge_damp = 1.0 - 0.65 * _smoothstep(0.32, 0.68, local_span)

    alpha = _read_optional_mask(subject_alpha, size_wh, channel="A")
    if alpha is not None:
        cutout_band = ((alpha > 0.02) & (alpha < 0.98)).astype(np.float32)
        k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cutout_band_blur = cv2.GaussianBlur(cv2.dilate(cutout_band, k7), (0, 0), 2.5)
        pos_cut_guard = np.clip(1.0 - 0.85 * cutout_band_blur, 0.15, 1.0)
        neg_cut_guard = np.clip(1.0 - 0.50 * cutout_band_blur, 0.50, 1.0)
    else:
        pos_cut_guard = 1.0
        neg_cut_guard = 1.0

    max_pos = 0.095 if style == "photo" else 0.072
    max_neg = 0.135 if style == "photo" else 0.105
    dY_pos = max_pos * np.tanh((np.maximum(dY, 0.0) * pos_edge_damp * pos_cut_guard) / max_pos)
    dY_neg = -max_neg * np.tanh((np.maximum(-dY, 0.0) * neg_cut_guard) / max_neg)
    dY_clean = (dY_pos + dY_neg).astype(np.float32)

    # 4. 4-Tier Semantic Focus-Plane Weight Map
    eyes_m = _read_optional_mask(eyes_mask, size_wh)
    lips_m = _read_optional_mask(lips_mask, size_wh)
    hair_m = _read_optional_mask(hair_mask, size_wh)
    garm_m = _read_optional_mask(garment_mask, size_wh)
    skin_m = _read_optional_mask(skin_mask, size_wh)
    fskin_m = _read_optional_mask(face_skin_mask, size_wh)

    gx = cv2.Sobel(y_srgb, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y_srgb, cv2.CV_32F, 0, 1, ksize=3)
    gmag_blur = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 3.0)
    bg_struct = _smoothstep(0.035, 0.14, gmag_blur)

    has_semantic = any(m is not None for m in (alpha, eyes_m, lips_m, hair_m, garm_m, skin_m, fskin_m))
    if has_semantic:
        zero = np.zeros((h, w), dtype=np.float32)
        a_fg = alpha if alpha is not None else np.ones((h, w), dtype=np.float32)

        if eyes_m is not None:
            k13 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
            eyes_zone = np.clip(cv2.GaussianBlur(cv2.dilate((eyes_m > 0.12).astype(np.float32), k13), (0, 0), 3.0), 0.0, 1.0)
        else:
            eyes_zone = zero

        if lips_m is not None:
            k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            lips_zone = np.clip(cv2.GaussianBlur(cv2.dilate((lips_m > 0.15).astype(np.float32), k7), (0, 0), 2.5), 0.0, 1.0)
        else:
            lips_zone = zero

        hair_zone = np.clip(cv2.GaussianBlur(hair_m, (0, 0), 2.0), 0.0, 1.0) if hair_m is not None else zero
        garm_zone = np.clip(cv2.GaussianBlur(garm_m, (0, 0), 2.0), 0.0, 1.0) if garm_m is not None else zero

        skin_raw = np.maximum(skin_m if skin_m is not None else zero, fskin_m if fskin_m is not None else zero)
        skin_all = np.clip(cv2.GaussianBlur(skin_raw, (0, 0), 2.0) * (1.0 - np.maximum(eyes_zone, lips_zone)), 0.0, 1.0)

        if style == "photo":
            w_eyes, w_lips, w_hair, w_garm, w_other, w_skin = 1.00, 0.68, 0.78, 0.62, 0.42, 0.02
        else:
            w_eyes, w_lips, w_hair, w_garm, w_other, w_skin = 0.95, 0.62, 0.58, 0.65, 0.30, 0.00

        other_subj = np.clip(a_fg * (1.0 - skin_all) * (1.0 - np.maximum.reduce([eyes_zone, lips_zone, hair_zone, garm_zone])), 0.0, 1.0)
        subj_w = np.maximum.reduce([
            eyes_zone * w_eyes,
            lips_zone * w_lips,
            hair_zone * w_hair,
            garm_zone * w_garm,
            other_subj * w_other,
        ])
        skin_core = np.clip(skin_all * (1.0 - hair_zone) * (1.0 - garm_zone), 0.0, 1.0)
        body_only_skin = np.clip(skin_core * (1.0 - (cv2.GaussianBlur(fskin_m, (0, 0), 4.0) if fskin_m is not None else zero)), 0.0, 1.0)
        anat_edges = _smoothstep(0.040, 0.135, gmag_blur)
        w_anat = 0.34 if style == "photo" else 0.22
        body_anat_w = body_only_skin * anat_edges * w_anat
        bg_w = (1.0 - a_fg) * bg_struct * eff_bg
        focus_w = np.clip(subj_w * a_fg + np.maximum(skin_core * w_skin, body_anat_w) * a_fg + bg_w, 0.0, 1.0)
    else:
        focus_w = np.clip(bg_struct * 0.75, 0.0, 1.0)
        eyes_zone = lips_zone = hair_zone = garm_zone = skin_all = None

    # 5. Apply & export layers
    dY_final = (dY_clean * focus_w * eff_strength).astype(np.float32)

    # Preserve hue/chroma richness on dark lips/hair while matching clean luminance delta
    y_target = np.clip(y_srgb + dY_final, 0.0, 1.0)
    scale_ratio = np.clip(y_target / np.maximum(y_srgb, 0.02), 0.72, 1.32)
    rgb_ratio = np.clip(img01 * scale_ratio[:, :, None], 0.0, 1.0)
    rgb_add = np.clip(img01 + dY_final[:, :, None], 0.0, 1.0)
    sat_blend = _smoothstep(0.06, 0.28, y_srgb)[:, :, None]
    rgb_sharp = np.clip(rgb_add * (1.0 - 0.50 * sat_blend) + rgb_ratio * (0.50 * sat_blend), 0.0, 1.0)

    if blend_mode == "Linear Light":
        ll_val = np.clip(0.5 + dY_final * 0.5, 0.0, 1.0)
        ll_alpha = _smoothstep(0.001, 0.005, np.abs(dY_final)) * np.clip(focus_w * 2.5, 0.0, 1.0)
        layer_rgba = np.dstack([ll_val, ll_val, ll_val, ll_alpha]).astype(np.float32)
    else:
        lum_alpha = np.clip(focus_w * _smoothstep(0.0008, 0.0035, np.abs(dY_final) + 0.002 * focus_w), 0.0, 1.0)
        layer_rgba = np.dstack([rgb_sharp[:, :, 0], rgb_sharp[:, :, 1], rgb_sharp[:, :, 2], lum_alpha]).astype(np.float32)

    p_layer = out_d / "output_sharpen_layer.png"
    p_prev = out_d / "output_sharpen_preview.png"
    p_focus = out_d / "output_sharpen_focus_map.png"
    p_meta = out_d / "output_sharpen_meta.json"

    _write_png(p_layer, layer_rgba, "RGBA", overwrite=overwrite)
    _write_png(p_prev, rgb_sharp, "RGB", overwrite=overwrite)
    _write_png(p_focus, focus_w, "L", overwrite=overwrite)

    # Metrics before vs after
    opaque = (alpha > 0.95) if alpha is not None else np.ones((h, w), dtype=bool)
    m_eyes = ((eyes_zone > 0.4) & opaque) if eyes_zone is not None else np.zeros((h, w), dtype=bool)
    m_skin = ((skin_all > 0.5) & opaque) if skin_all is not None else np.zeros((h, w), dtype=bool)
    m_garm = ((garm_zone > 0.6) & opaque) if garm_zone is not None else np.zeros((h, w), dtype=bool)
    m_bg = ((alpha < 0.05) & (bg_struct > 0.3)) if alpha is not None else (bg_struct > 0.3)

    metrics = {
        "eyes": {"before": _luma_stats(img01, m_eyes), "after": _luma_stats(rgb_sharp, m_eyes)},
        "skin": {"before": _luma_stats(img01, m_skin), "after": _luma_stats(rgb_sharp, m_skin)},
        "garment": {"before": _luma_stats(img01, m_garm), "after": _luma_stats(rgb_sharp, m_garm)},
        "bg_struct": {"before": _luma_stats(img01, m_bg), "after": _luma_stats(rgb_sharp, m_bg)},
        "halo_overshoot_p99": round(float(np.percentile(np.maximum(dY_final, 0.0), 99.0)), 4),
        "neg_undershoot_p99": round(float(np.percentile(np.maximum(-dY_final, 0.0), 99.0)), 4),
        "active_pixel_pct": round(100.0 * float((np.abs(dY_final) > 0.002).mean()), 2),
    }

    meta = {
        "ok": True,
        "style": style,
        "blend_mode": blend_mode,
        "width": w,
        "height": h,
        "params": {
            "strength": eff_strength,
            "radius": eff_radius,
            "iterations": iters,
            "bg_strength": eff_bg,
            "max_pos": max_pos,
            "max_neg": max_neg,
        },
        "metrics": metrics,
        "outputs": {
            "layer_path": str(p_layer),
            "preview_path": str(p_prev),
            "focus_map_path": str(p_focus),
            "meta_path": str(p_meta),
        },
    }
    p_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def build_output_sharpen_pcraft_cmds(
    meta_or_dir: dict[str, Any] | Path | str,
    above_layer: str,
    *,
    layer_name: str = "05c3_Focus_Output_Sharpen",
    opacity: float = 1.0,
    width: int | None = None,
    height: int | None = None,
) -> list[str]:
    """Build PhotoCraft CLI `--cmd` arguments inserting output_sharpen_layer.png above `above_layer`."""
    if isinstance(meta_or_dir, (Path, str)):
        p = Path(meta_or_dir)
        meta_path = p if p.is_file() else (p / "output_sharpen_meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = meta_or_dir

    w = int(width or meta["width"])
    h = int(height or meta["height"])
    layer_path = meta["outputs"]["layer_path"]
    blend = str(meta.get("blend_mode", "Linear Light"))

    return [
        "--cmd", "layer.select", "--params", _J({"layer": above_layer}),
        "--cmd", "file.placeEmbedded",
        "--params", _J({"path": layer_path, "fit": False, "smartObject": True, "center": [w / 2, h / 2]}),
        "--cmd", "layer.setProps",
        "--params", _J({"name": layer_name, "blend": blend, "opacity": float(opacity), "visible": True}),
        "--cmd", "layer.moveTo", "--params", _J({"target": above_layer, "position": "above"}),
    ]
