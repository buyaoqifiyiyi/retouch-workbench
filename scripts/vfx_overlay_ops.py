"""Optical Unmult compositing & face/title avoidance for user-rolled black-background AI VFX overlays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def _hex_to_rgb01(hex_str: str) -> np.ndarray:
    s = hex_str.strip().lstrip("#")
    if len(s) != 6:
        raise ValueError(f"Expected #RRGGBB hex color, got {hex_str!r}")
    return np.array([int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)], dtype=np.float32) / 255.0


def _read_gray_f32(path: Path, size_wh: tuple[int, int]) -> np.ndarray:
    im = Image.open(path).convert("L")
    if im.size != size_wh:
        im = im.resize(size_wh, Image.Resampling.BILINEAR)
    return np.array(im, dtype=np.float32) / 255.0


def _read_alpha_or_gray_f32(path: Path, size_wh: tuple[int, int]) -> np.ndarray:
    im = Image.open(path)
    if im.mode in ("RGBA", "LA"):
        ch = im.getchannel("A")
    else:
        ch = im.convert("L")
    if ch.size != size_wh:
        ch = ch.resize(size_wh, Image.Resampling.BILINEAR)
    return np.array(ch, dtype=np.float32) / 255.0


def _save_rgba(path: Path, rgb01: np.ndarray, alpha01: np.ndarray, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = np.dstack([
        np.clip(np.round(rgb01 * 255.0), 0, 255),
        np.clip(np.round(alpha01 * 255.0), 0, 255),
    ]).astype(np.uint8)
    Image.fromarray(u8, "RGBA").save(path, "PNG")


def run_vfx_overlay(
    *,
    vfx_path: Path,
    width: int,
    height: int,
    out_dir: Path,
    subject_alpha_path: Path | None = None,
    face_parse_dir: Path | None = None,
    title_box_xywh: tuple[int, int, int, int] | None = None,
    placement: str = "auto-split",
    black_floor: str = "auto",
    tint_hex: str | None = None,
    tint_strength: float = 0.20,
    spill_strength: float = 0.25,
    subject_front_keep: float = 0.10,
    density_gamma: float = 1.40,
    haze_suppress: float = 0.45,
    max_alpha: float = 0.85,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert black-background AI VFX plates into straight RGBA depth-split PhotoCraft layers + edge spill."""
    in_p = vfx_path.resolve()
    out_d = out_dir.resolve()
    out_d.mkdir(parents=True, exist_ok=True)

    im_vfx = Image.open(in_p).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    raw = np.array(im_vfx, dtype=np.float32) / 255.0

    if abs(offset_x) > 1e-4 or abs(offset_y) > 1e-4:
        dx_px = float(offset_x) * float(width)
        dy_px = float(offset_y) * float(height)
        M_shift = np.array([[1.0, 0.0, dx_px], [0.0, 1.0, dy_px]], dtype=np.float32)
        raw = cv2.warpAffine(raw, M_shift, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0.0, 0.0, 0.0))
        fade_px = 120.0
        border_w = np.ones((height, width), dtype=np.float32)
        yy_b, xx_b = np.mgrid[0:height, 0:width].astype(np.float32)
        if dx_px < -1.0:
            border_w *= np.clip(((width + dx_px) - xx_b) / fade_px, 0.0, 1.0)
        elif dx_px > 1.0:
            border_w *= np.clip((xx_b - dx_px) / fade_px, 0.0, 1.0)
        if dy_px < -1.0:
            border_w *= np.clip(((height + dy_px) - yy_b) / fade_px, 0.0, 1.0)
        elif dy_px > 1.0:
            border_w *= np.clip((yy_b - dy_px) / fade_px, 0.0, 1.0)
        raw *= border_w[:, :, None]

    # 1. Adaptive black-floor subtraction (removes murky #06080C..#181A20 AI black backgrounds cleanly)
    luma = 0.2126 * raw[:, :, 0] + 0.7152 * raw[:, :, 1] + 0.0722 * raw[:, :, 2]
    if str(black_floor).strip().lower() == "auto":
        # Estimate border + low percentile dark floor so genuine soft glow stays intact
        p12 = float(np.percentile(luma, 12))
        floor_rgb = np.array([float(np.percentile(raw[:, :, c], 12)) for c in range(3)], dtype=np.float32)
        floor_rgb = np.clip(floor_rgb * 1.05, 0.0, 0.14)
        floor_scalar = float(np.clip(p12 * 1.05, 0.0, 0.14))
    else:
        floor_scalar = float(np.clip(float(black_floor), 0.0, 0.25))
        floor_rgb = np.array([floor_scalar, floor_scalar, floor_scalar], dtype=np.float32)

    cleaned = np.clip((raw - floor_rgb[None, None, :]) / np.maximum(1.0 - floor_rgb[None, None, :], 1e-4), 0.0, 1.0)
    # Soft dark-toe noise gate so tiny compression specks (< 0.012) collapse cleanly to 0
    lum_c = cleaned.max(axis=2)
    toe = np.clip((lum_c - 0.010) / 0.035, 0.0, 1.0)
    cleaned *= (toe * toe * (3.0 - 2.0 * toe))[:, :, None]

    # 2. Optical Unmult: derive straight color + density/cloud-restrained optical alpha
    raw_alpha = np.clip(cleaned.max(axis=2), 0.0, 1.0)
    gamma_v = float(np.clip(density_gamma, 0.5, 4.0))
    alpha_unmult = np.where(raw_alpha > 1e-3, np.power(raw_alpha, gamma_v), 0.0)
    if haze_suppress > 1e-3:
        broad_cloud = cv2.GaussianBlur(raw_alpha, (0, 0), sigmaX=max(24.0, width * 0.018))
        cloud_damp = np.clip(
            1.0 - float(np.clip(haze_suppress, 0.0, 1.0)) * np.clip((broad_cloud - 0.18) / 0.50, 0.0, 1.0),
            0.22,
            1.0,
        )
        alpha_unmult *= cloud_damp
    alpha_unmult = np.clip(alpha_unmult * float(np.clip(max_alpha, 0.05, 1.0)), 0.0, 1.0)
    rgb_straight = np.where(
        raw_alpha[:, :, None] > 1e-3,
        np.clip(cleaned / np.maximum(raw_alpha[:, :, None], 1e-4), 0.0, 1.0),
        0.0,
    )

    if tint_hex:
        tint = _hex_to_rgb01(tint_hex)[None, None, :]
        ts = float(np.clip(tint_strength, 0.0, 1.0))
        rgb_straight = np.clip(rgb_straight * (1.0 - ts) + (0.65 * tint + 0.35 * rgb_straight) * ts, 0.0, 1.0)

    # 3. Build editable face / eyes / pearls / title avoidance mask
    avoid = np.ones((height, width), dtype=np.float32)
    eye_mask_bin = None
    if face_parse_dir is not None and face_parse_dir.exists():
        f_skin_p = face_parse_dir / "mask_skin.png"
        f_eyes_p = face_parse_dir / "mask_eyes_brows.png"
        f_lips_p = face_parse_dir / "mask_lips.png"
        parts = []
        for p in (f_skin_p, f_eyes_p, f_lips_p):
            if p.exists():
                parts.append(_read_gray_f32(p, (width, height)))
        if f_eyes_p.exists():
            eye_mask_bin = _read_gray_f32(f_eyes_p, (width, height)) > 0.20
        if parts:
            f_union = (np.maximum.reduce(parts) > 0.25).astype(np.uint8)
            k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(max(35, width * 0.028)) | 1, int(max(35, width * 0.028)) | 1))
            f_guard = cv2.GaussianBlur(cv2.dilate(f_union, k_dil).astype(np.float32), (0, 0), max(18.0, width * 0.012))
            avoid *= np.clip(1.0 - 1.15 * f_guard, 0.0, 1.0)

    if title_box_xywh is not None:
        tx, ty, tw, th = title_box_xywh
        tb = np.zeros((height, width), dtype=np.float32)
        x0, y0 = max(0, tx), max(0, ty)
        x1, y1 = min(width, tx + tw), min(height, ty + th)
        if x1 > x0 and y1 > y0:
            tb[y0:y1, x0:x1] = 0.85
            tb = cv2.GaussianBlur(tb, (0, 0), 36.0)
            avoid *= np.clip(1.0 - tb, 0.0, 1.0)

    # Build Subject Core Protection Mask so person stays the clean hero while VFX wraps around & behind them
    subj_front_mask = np.ones((height, width), dtype=np.float32)
    subj_core_bin = None
    if subject_alpha_path is not None and subject_alpha_path.exists():
        sa_m = _read_alpha_or_gray_f32(subject_alpha_path.resolve(), (width, height))
        core_dist = cv2.distanceTransform((sa_m > 0.35).astype(np.uint8), cv2.DIST_L2, 5)
        core_bin = (core_dist > max(12.0, width * 0.004)).astype(np.float32)
        subj_core_bin = core_bin > 0.5
        ds = 4
        core_sm = cv2.resize(core_bin, (max(8, width // ds), max(8, height // ds)), interpolation=cv2.INTER_AREA)
        core_sm = cv2.GaussianBlur(core_sm, (0, 0), max(6.0, (width * 0.011) / ds))
        subj_core = np.clip(cv2.resize(core_sm, (width, height), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)
        lum_pk = cleaned.max(axis=2)
        hero_glint = np.power(np.clip((lum_pk - 0.84) / 0.16, 0.0, 1.0), 2.0)
        keep_interior = float(np.clip(subject_front_keep, 0.0, 1.0)) * hero_glint
        subj_front_mask = np.clip((1.0 - subj_core) + subj_core * keep_interior, 0.0, 1.0)

    p_avoid = out_d / "vfx_avoid_mask.png"
    if p_avoid.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite without --overwrite: {p_avoid}")
    Image.fromarray(np.clip(np.round((avoid * subj_front_mask) * 255.0), 0, 255).astype(np.uint8), "L").save(p_avoid, "PNG")

    # 4. Placement mode: auto-split separates broad soft bokeh/mist behind subject & crisp particles in front
    outputs: dict[str, str] = {"vfx_avoid_mask": str(p_avoid)}
    if placement == "auto-split":
        lum_s = cleaned.max(axis=2)
        broad_blur = cv2.GaussianBlur(lum_s, (0, 0), sigmaX=max(10.0, width * 0.0055))
        sharp_hi = np.clip(lum_s - broad_blur * 0.85, 0.0, 1.0)
        # Sharp ratio high -> belongs in foreground sparkles; broad smooth glow -> sits cleanly behind subject
        fg_share = np.clip(sharp_hi / np.maximum(lum_s, 1e-4) * 1.45, 0.0, 1.0)
        fg_share = cv2.GaussianBlur(fg_share, (0, 0), 3.0)
        a_behind = alpha_unmult * (1.0 - fg_share)
        a_front = alpha_unmult * fg_share * avoid * subj_front_mask

        p_beh = out_d / "vfx_behind_subject_rgba.png"
        p_frt = out_d / "vfx_front_subject_rgba.png"
        _save_rgba(p_beh, rgb_straight, a_behind, overwrite=overwrite)
        _save_rgba(p_frt, rgb_straight, a_front, overwrite=overwrite)
        outputs["vfx_behind_subject_rgba"] = str(p_beh)
        outputs["vfx_front_subject_rgba"] = str(p_frt)
    elif placement == "behind":
        p_beh = out_d / "vfx_behind_subject_rgba.png"
        _save_rgba(p_beh, rgb_straight, alpha_unmult, overwrite=overwrite)
        outputs["vfx_behind_subject_rgba"] = str(p_beh)
        a_front = np.zeros((height, width), dtype=np.float32)
    else:  # front
        a_front = alpha_unmult * avoid * subj_front_mask
        p_frt = out_d / "vfx_front_subject_rgba.png"
        _save_rgba(p_frt, rgb_straight, a_front, overwrite=overwrite)
        outputs["vfx_front_subject_rgba"] = str(p_frt)

    # 5. Edge light-spill wrap on subject rim when luminous particles sit close to subject
    eye_ov_px = int(((a_front > 0.05) & eye_mask_bin).sum()) if eye_mask_bin is not None else 0
    body_cov_pct = (
        round(100.0 * float(((a_front > 0.08) & subj_core_bin).sum()) / max(1, int(subj_core_bin.sum())), 3)
        if subj_core_bin is not None
        else 0.0
    )
    if subject_alpha_path is not None and subject_alpha_path.exists() and spill_strength > 0.0:
        sa = _read_alpha_or_gray_f32(subject_alpha_path.resolve(), (width, height))
        in_fg = (sa > 0.25).astype(np.uint8)
        eroded = cv2.erode(in_fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45)))
        rim = cv2.GaussianBlur((in_fg - eroded).astype(np.float32), (0, 0), 10.0) * sa
        glow_rgb = cv2.GaussianBlur(cleaned, (0, 0), max(18.0, width * 0.008))
        spill_a = np.clip(rim * np.clip(glow_rgb.max(axis=2) * 2.2, 0.0, 1.0) * float(spill_strength) * avoid, 0.0, 1.0)
        spill_rgb = np.clip(glow_rgb / np.maximum(glow_rgb.max(axis=2, keepdims=True), 1e-4), 0.0, 1.0)
        p_spill = out_d / "vfx_subject_spill_screen.png"
        _save_rgba(p_spill, spill_rgb, spill_a, overwrite=overwrite)
        outputs["vfx_subject_spill_screen"] = str(p_spill)

    return {
        "ok": True,
        "input": str(in_p),
        "size": [width, height],
        "placement": placement,
        "black_floor_subtracted": round(floor_scalar, 4),
        "front_eye_overlap_px": eye_ov_px,
        "front_subject_body_coverage_pct": body_cov_pct,
        "outputs": outputs,
    }
