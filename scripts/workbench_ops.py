"""Modular workbench operations for retouch-ai.py.

Implements style-adaptive hair compositing with tangent grooming (hair-comp),
PhotoCraft native 3-brush behind+front layered hair painting (hair-paint),
CIE Lab garment frequency separation & alpha-only defringe (garment-retouch),
lower-limb skin unification & skirt/leg occlusion shadow (skin-unify),
Apache-2.0 PhotoBlaze-PIH ONNX CUDA harmonization (harmonize), and
frequency-locked tiled image splitting/stitching (tile-split / tile-stitch).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def write_png_u8(path: Path, arr: Any, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError(f"Failed to encode PNG: {path}")
    buf.tofile(str(path))


def read_gray_f32(path: Path, expected_hw: tuple[int, int] | None = None) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"Empty or unreadable mask file: {path}")
    arr = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise ValueError(f"Failed to decode grayscale mask: {path}")
    if expected_hw is not None and arr.shape != expected_hw:
        arr = cv2.resize(arr, (expected_hw[1], expected_hw[0]), interpolation=cv2.INTER_LINEAR)
    return arr.astype(np.float32) / 255.0


def read_rgba_f32(path: Path, expected_hw: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"Empty or unreadable image file: {path}")
    arr = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(f"Failed to decode image: {path}")
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        rgb = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float32)
        alpha = arr[:, :, 3].astype(np.float32) / 255.0
    else:
        rgb = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float32)
        alpha = np.ones(arr.shape[:2], dtype=np.float32)
    if expected_hw is not None and alpha.shape != expected_hw:
        eh, ew = expected_hw
        rgb = cv2.resize(rgb, (ew, eh), interpolation=cv2.INTER_LANCZOS4)
        alpha = np.clip(cv2.resize(alpha, (ew, eh), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)
    return rgb, alpha


def write_rgba_u8(path: Path, rgb_f32: np.ndarray, alpha_f32: np.ndarray, overwrite: bool = False) -> None:
    bgr = cv2.cvtColor(np.clip(np.round(rgb_f32), 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    a_u8 = np.clip(np.round(alpha_f32 * 255.0), 0, 255).astype(np.uint8)
    bgra = np.dstack([bgr, a_u8])
    write_png_u8(path, bgra, overwrite=overwrite)


def write_rgb_u8(path: Path, rgb_f32: np.ndarray, overwrite: bool = False) -> None:
    bgr = cv2.cvtColor(np.clip(np.round(rgb_f32), 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    write_png_u8(path, bgr, overwrite=overwrite)


def guided_filter_gray(guide_01: np.ndarray, src_01: np.ndarray, radius: int = 12, eps: float = 1e-3) -> np.ndarray:
    if radius <= 0:
        return src_01
    ksize = (2 * radius + 1, 2 * radius + 1)
    g = guide_01.astype(np.float32)
    p = src_01.astype(np.float32)
    mean_i = cv2.boxFilter(g, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(p, cv2.CV_32F, ksize)
    corr_i = cv2.boxFilter(g * g, cv2.CV_32F, ksize)
    corr_ip = cv2.boxFilter(g * p, cv2.CV_32F, ksize)
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)
    return np.clip(mean_a * g + mean_b, 0.0, 1.0)


def rgb255_to_lab(rgb_f32: np.ndarray) -> np.ndarray:
    """Convert float32 RGB [0..255] to float32 CIE Lab (L 0..100, a/b -127..127)."""
    return cv2.cvtColor(np.clip(rgb_f32 / 255.0, 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2LAB)


def lab_to_rgb255(lab_f32: np.ndarray) -> np.ndarray:
    """Convert float32 CIE Lab (L 0..100, a/b -127..127) to float32 RGB [0..255]."""
    rgb01 = cv2.cvtColor(lab_f32.astype(np.float32), cv2.COLOR_LAB2RGB)
    return np.clip(rgb01 * 255.0, 0.0, 255.0)


def luma255(rgb_f32: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb_f32[:, :, 0] + 0.7152 * rgb_f32[:, :, 1] + 0.0722 * rgb_f32[:, :, 2]


def ell_kernel(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def decontaminate_fringe(
    rgb_f32: np.ndarray,
    core_mask: np.ndarray,
    fringe_weight: np.ndarray,
    sigmas: tuple[float, ...] = (4.0, 10.0, 22.0),
) -> np.ndarray:
    """Extrapolate opaque interior foreground color across semi-transparent outer fringes."""
    h, w = core_mask.shape
    eroded = cv2.erode((core_mask > 0.5).astype(np.float32), ell_kernel(4)) > 0.5
    if not eroded.any():
        return rgb_f32.copy()

    filled = eroded.copy()
    num = np.zeros_like(rgb_f32, dtype=np.float32)
    den = np.zeros((h, w), dtype=np.float32)
    num[eroded] = rgb_f32[eroded]
    den[eroded] = 1.0
    extrap = rgb_f32.copy()

    for sig in sigmas:
        b_num = cv2.GaussianBlur(num, (0, 0), sigmaX=sig)
        b_den = cv2.GaussianBlur(den, (0, 0), sigmaX=sig)
        valid = (~filled) & (b_den > 1e-3)
        if valid.any():
            fill = b_num[valid] / b_den[valid, None]
            extrap[valid] = fill
            filled[valid] = True
            den[valid] = np.maximum(den[valid], 0.05)
            num[valid] = fill * den[valid, None]

    fw = np.clip(fringe_weight, 0.0, 1.0)[:, :, None]
    return np.clip(rgb_f32 * (1.0 - fw) + extrap * fw, 0.0, 255.0)


def _clamp_over_textured_hf(src_rgb: np.ndarray, ref_rgb: np.ndarray, hair_mask: np.ndarray) -> np.ndarray:
    """Tame GAN upscaler oil-paint micro-ridges when generated hair exceeds real hair HF energy."""
    sg = cv2.cvtColor(np.clip(src_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    rg = cv2.cvtColor(np.clip(ref_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    hm = hair_mask > 0.5
    if hm.sum() < 2000:
        return src_rgb
    s_hf = np.abs(sg - cv2.GaussianBlur(sg, (0, 0), 0.8))
    r_hf = np.abs(rg - cv2.GaussianBlur(rg, (0, 0), 0.8))
    ratio = float(s_hf[hm].mean()) / max(1e-3, float(r_hf[hm].mean()))
    if ratio <= 1.15:
        return src_rgb
    # Dampen excess mid-high band softly without plastic blur
    keep = np.clip(1.05 / ratio, 0.55, 0.92)
    blur = cv2.GaussianBlur(src_rgb, (0, 0), sigmaX=1.1)
    detail = src_rgb - blur
    soft_src = blur + np.tanh(detail / 18.0) * 18.0 * keep
    w = (np.clip((hair_mask - 0.15) / 0.5, 0.0, 1.0))[:, :, None]
    return np.clip(src_rgb * (1.0 - w) + soft_src * w, 0.0, 255.0)


def run_hair_comp(
    *,
    subject_rgba_path: Path,
    subject_hair_mask_path: Path,
    protected_mask_path: Path,
    out_dir: Path,
    occluder_mask_path: Path | None = None,
    accessories_mask_path: Path | None = None,
    mode: str = "real-first",
    gen_hair_rgb_path: Path | None = None,
    gen_hair_matte_path: Path | None = None,
    gen_hair_mask_path: Path | None = None,
    clean_trapped_gaps: bool = True,
    haze_recover: float = 0.35,
    decontam_strength: float = 0.75,
    groom: bool = True,
    groom_strength: float = 0.38,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Style-adaptive, occlusion-clean hair compositing without hardcoded coordinate boxes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    subj_rgb, subj_alpha = read_rgba_f32(subject_rgba_path)
    h, w = subj_alpha.shape

    hair_m = read_gray_f32(subject_hair_mask_path, (h, w))
    prot_raw = read_gray_f32(protected_mask_path, (h, w))
    occ_raw = read_gray_f32(occluder_mask_path, (h, w)) if occluder_mask_path else prot_raw
    if accessories_mask_path is not None:
        acc_m = read_gray_f32(accessories_mask_path, (h, w))
    else:
        auto_acc = protected_mask_path.parent / "mask_accessories.png"
        acc_m = read_gray_f32(auto_acc, (h, w)) if auto_acc.exists() else np.zeros((h, w), dtype=np.float32)

    guide01 = np.clip(luma255(subj_rgb) / 255.0, 0.0, 1.0)
    prot_snap = guided_filter_gray(guide01, prot_raw, radius=8, eps=1e-3)
    occ_snap = guided_filter_gray(guide01, occ_raw, radius=8, eps=1e-3)
    # 0px dilation, 0.8px AA feather: hair terminates flush behind real occluders without a halo gap
    occ_tight = cv2.GaussianBlur((occ_snap >= 0.5).astype(np.float32), (0, 0), sigmaX=0.8)

    out_rgb = subj_rgb.copy()
    out_alpha = subj_alpha.copy()
    trim_mask = np.ones((h, w), dtype=np.float32)
    gen_layer_out = np.zeros((h, w, 4), dtype=np.float32)
    trapped_gap_px = 0

    if clean_trapped_gaps:
        lab = rgb255_to_lab(subj_rgb)
        L_lab = lab[:, :, 0]
        C_lab = np.hypot(lab[:, :, 1], lab[:, :, 2])
        hair_bin = (hair_m >= 0.45).astype(np.uint8)
        prot_bin = (prot_snap >= 0.60).astype(np.uint8)
        if hair_bin.any() and prot_bin.any():
            d_hair = cv2.distanceTransform(1 - hair_bin, cv2.DIST_L2, 5)
            d_prot = cv2.distanceTransform(1 - prot_bin, cv2.DIST_L2, 5)
            in_pocket = (d_hair <= 85.0) & (d_prot <= 85.0)
            # Trapped studio-window pockets sit outside garments/hair/accessories with bright neutral sky/wall values
            gap_raw = (
                in_pocket
                & (subj_alpha > 0.12)
                & (subj_alpha < 0.94)
                & (prot_snap < 0.42)
                & (hair_m < 0.45)
                & (acc_m < 0.40)
                & (L_lab > 56.0)
                & (C_lab < 13.0)
            )
            gap_clean = cv2.morphologyEx(gap_raw.astype(np.uint8), cv2.MORPH_OPEN, ell_kernel(2))
            gap_clean = cv2.morphologyEx(gap_clean, cv2.MORPH_CLOSE, ell_kernel(3))
            n_g, lbl_g, st_g, _ = cv2.connectedComponentsWithStats(gap_clean, connectivity=8)
            for gid in range(1, n_g):
                comp_m = lbl_g == gid
                # Reject <200px specks or solid opaque satin ribbons/collars where core is predominantly matte>=0.92
                if st_g[gid, cv2.CC_STAT_AREA] < 200 or float(np.median(subj_alpha[comp_m])) >= 0.88:
                    gap_clean[comp_m] = 0
            # Dark/mid-grey cutout sliver trapped between bright bow/accessory and adjacent hair
            bright_acc = (
                ((acc_m >= 0.40) | ((subj_alpha > 0.85) & (L_lab > 76.0) & (C_lab < 12.0)))
                & (hair_m < 0.35)
                & (prot_snap < 0.45)
            )
            d_acc = (
                cv2.distanceTransform(1 - bright_acc.astype(np.uint8), cv2.DIST_L2, 5)
                if bright_acc.any()
                else np.full((h, w), 9999.0, dtype=np.float32)
            )
            if bright_acc.any():
                bow_sliver = (
                    (d_acc <= 55.0)
                    & (d_hair <= 55.0)
                    & (d_prot <= 120.0)
                    & (subj_alpha > 0.12)
                    & (prot_snap < 0.42)
                    & (hair_m < 0.25)
                    & (acc_m < 0.40)
                    & (L_lab < 64.0)
                    & (C_lab < 10.5)
                )
                bow_open = cv2.morphologyEx(bow_sliver.astype(np.uint8), cv2.MORPH_OPEN, ell_kernel(1))
                n_b, lbl_b, st_b, _ = cv2.connectedComponentsWithStats(bow_open, connectivity=8)
                for bid in range(1, n_b):
                    if int(st_b[bid, cv2.CC_STAT_AREA]) >= 150:
                        gap_clean[lbl_b == bid] = 1
            # Opaque pure-white studio/hallway pockets trapped strictly between hair and protected body (away from white head-bow)
            white_pocket_raw = (
                (d_hair <= 85.0)
                & (d_prot <= 65.0)
                & (d_acc > 55.0)
                & (subj_alpha > 0.12)
                & (prot_snap < 0.35)
                & (hair_m < 0.25)
                & (acc_m < 0.35)
                & (L_lab > 84.0)
                & (C_lab < 6.0)
            )
            white_pocket_open = cv2.morphologyEx(white_pocket_raw.astype(np.uint8), cv2.MORPH_OPEN, ell_kernel(1))
            n_wp, lbl_wp, st_wp, _ = cv2.connectedComponentsWithStats(white_pocket_open, connectivity=8)
            white_pocket_bin = np.zeros((h, w), dtype=bool)
            for wid in range(1, n_wp):
                if int(st_wp[wid, cv2.CC_STAT_AREA]) >= 120:
                    comp_wp = lbl_wp == wid
                    gap_clean[comp_wp] = 1
                    white_pocket_bin |= comp_wp
            if gap_clean.any():
                # Strictly lock protected garment/body core (>= 0.72), dark dangling dress straps, and white bow accessories against any hole
                white_bow_guard = ((subj_alpha > 0.90) & (L_lab > 78.0) & (C_lab < 12.0)) & (~white_pocket_bin)
                dark_strap_guard = (subj_alpha > 0.35) & (L_lab < 54.0) & (C_lab < 12.0) & (hair_m < 0.35)
                prot_core = (np.maximum(prot_raw, prot_snap) >= 0.72) | (acc_m >= 0.45) | white_bow_guard | dark_strap_guard
                gap_soft = cv2.GaussianBlur(gap_clean.astype(np.float32), (0, 0), sigmaX=1.2)
                gap_soft = np.where(prot_core, 0.0, gap_soft)
                trapped_gap_px = int((gap_soft > 0.5).sum())
                trim_mask = np.clip(1.0 - gap_soft, 0.0, 1.0)
                out_alpha = np.clip(out_alpha * trim_mask, 0.0, 1.0)

    if mode == "real-first":
        hair_core = (out_alpha > 0.85) & (hair_m > 0.45) & (prot_snap < 0.35)
        fringe = np.clip((0.88 - out_alpha) / 0.85, 0.0, 1.0) * (out_alpha > 0.02) * cv2.GaussianBlur(hair_m, (0, 0), 4.0)
        if decontam_strength > 0.0 and hair_core.any():
            out_rgb = decontaminate_fringe(out_rgb, hair_core, fringe * float(decontam_strength))

        if haze_recover > 0.0:
            lab = rgb255_to_lab(out_rgb)
            L_lab, a_lab, b_lab = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
            C_lab = np.hypot(a_lab, b_lab)
            white_bow = (out_alpha > 0.85) & (L_lab > 76.0) & (C_lab < 12.0) & (hair_m < 0.50)
            opaque_ribbon = (out_alpha > 0.80) & (L_lab > 66.0) & (C_lab < 11.5) & ((hair_m < 0.65) | (b_lab < 3.6))
            hair_solid = (out_alpha > 0.55) & (hair_m > 0.50) & (prot_snap < 0.35) & (~white_bow) & (~opaque_ribbon) & (acc_m < 0.40)
            hair_fog_zone = (out_alpha > 0.35) & (hair_m > 0.50) & (prot_snap < 0.30) & (~white_bow) & (~opaque_ribbon) & (acc_m < 0.40)
            ref_h = hair_solid & (L_lab >= 36.0) & (L_lab <= 76.0) & (C_lab >= 12.0)
            if ref_h.sum() >= 500:
                a_med = float(np.median(a_lab[ref_h]))
                b_med = float(np.median(b_lab[ref_h]))
                # Blend low-frequency fog with mid/fine fiber signal so bleached-white plastic glare strands recover golden tone
                L_sm = cv2.GaussianBlur(L_lab, (0, 0), sigmaX=16.0)
                a_sm = cv2.GaussianBlur(a_lab, (0, 0), sigmaX=16.0)
                b_sm = cv2.GaussianBlur(b_lab, (0, 0), sigmaX=16.0)
                L_mix = 0.55 * L_sm + 0.45 * L_lab
                a_mix = 0.55 * a_sm + 0.45 * a_lab
                b_mix = 0.55 * b_sm + 0.45 * b_lab
                C_mix = np.hypot(a_mix, b_mix)
                w_fog = (
                    np.clip((L_mix - 66.0) / 14.0, 0.0, 1.0)
                    * np.clip((18.5 - C_mix) / 9.5, 0.0, 1.0)
                    * hair_fog_zone.astype(np.float32)
                )
                k = float(np.clip(haze_recover, 0.0, 1.0))
                lab_rec = lab.copy()
                lab_rec[:, :, 0] = np.clip(L_lab - w_fog * k * 0.62 * np.maximum(0.0, L_mix - 60.0), 8.0, 99.0)
                lab_rec[:, :, 1] = a_lab + w_fog * k * 0.78 * (a_med - a_mix)
                lab_rec[:, :, 2] = b_lab + w_fog * k * 0.78 * (b_med - b_mix)
                rec_rgb = lab_to_rgb255(lab_rec)
                bw = (hair_fog_zone.astype(np.float32) * np.clip(hair_m / 0.65, 0.0, 1.0))[:, :, None]
                out_rgb = np.clip(out_rgb * (1.0 - bw) + rec_rgb * bw, 0.0, 255.0)
                if decontam_strength > 0.0:
                    lab_post = rgb255_to_lab(out_rgb)
                    Lp, ap, bp_c = lab_post[:, :, 0], lab_post[:, :, 1], lab_post[:, :, 2]
                    Cp = np.hypot(ap, bp_c)
                    clean_gold = hair_fog_zone & (out_alpha > 0.45) & (Lp < 76.0) & (Cp > 11.0)
                    foggy_rim = hair_fog_zone & (out_alpha < 0.82) & (Lp > 74.0) & (Cp < 13.0)
                    if clean_gold.any() and foggy_rim.any():
                        rim_w = cv2.GaussianBlur(foggy_rim.astype(np.float32), (0, 0), sigmaX=1.2) * hair_fog_zone.astype(np.float32)
                        out_rgb = decontaminate_fringe(out_rgb, clean_gold, np.clip(rim_w * float(decontam_strength) * 0.45, 0.0, 1.0))

        # Structure-tensor tangent-aligned hair grooming (tames tangled frizz while keeping macro shape)
        if groom and groom_strength > 0.0:
            lab_g = rgb255_to_lab(out_rgb)
            Cg = np.hypot(lab_g[:, :, 1], lab_g[:, :, 2])
            white_bow_g = (out_alpha > 0.85) & (lab_g[:, :, 0] > 76.0) & (Cg < 12.0) & (hair_m < 0.50)
            opaque_rib_g = (out_alpha > 0.80) & (lab_g[:, :, 0] > 66.0) & (Cg < 11.5) & ((hair_m < 0.65) | (lab_g[:, :, 2] < 3.6))
            groom_mask = (out_alpha > 0.45) & (hair_m > 0.45) & (prot_snap < 0.35) & (~white_bow_g) & (~opaque_rib_g) & (acc_m < 0.40)
            if groom_mask.sum() >= 300:
                Y_g = np.clip(luma255(out_rgb) / 255.0, 0.0, 1.0)
                gx_g = cv2.Sobel(Y_g, cv2.CV_32F, 1, 0, ksize=3)
                gy_g = cv2.Sobel(Y_g, cv2.CV_32F, 0, 1, ksize=3)
                jxx = cv2.GaussianBlur(gx_g * gx_g, (0, 0), sigmaX=4.5)
                jyy = cv2.GaussianBlur(gy_g * gy_g, (0, 0), sigmaX=4.5)
                jxy = cv2.GaussianBlur(gx_g * gy_g, (0, 0), sigmaX=4.5)
                theta = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy) + (math.pi * 0.5)
                tx = np.cos(theta).astype(np.float32)
                ty = np.sin(theta).astype(np.float32)
                h_im, w_im = out_alpha.shape
                yy_idx, xx_idx = np.mgrid[0:h_im, 0:w_im].astype(np.float32)
                acc_rgb = np.zeros_like(out_rgb, dtype=np.float32)
                acc_w = np.zeros((h_im, w_im), dtype=np.float32)
                valid_hair_f = groom_mask.astype(np.float32)
                for step_d in (-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
                    wt = math.exp(-0.5 * (step_d / 2.4) ** 2)
                    if step_d == 0.0:
                        acc_rgb += out_rgb * wt
                        acc_w += wt
                    else:
                        mx = np.clip(xx_idx + step_d * tx, 0.0, float(w_im - 1))
                        my = np.clip(yy_idx + step_d * ty, 0.0, float(h_im - 1))
                        s_rgb = cv2.remap(out_rgb, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                        s_ok = cv2.remap(valid_hair_f, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
                        w_pix = wt * s_ok
                        acc_rgb += s_rgb * w_pix[:, :, None]
                        acc_w += w_pix
                groomed = acc_rgb / np.maximum(acc_w[:, :, None], 1e-4)
                # Preserve >=82% high-frequency micro-strands and protect braided cross-hatch zones from putty-smear
                coherence = np.hypot(jxx - jyy, 2.0 * jxy) / np.maximum(jxx + jyy, 1e-6)
                coh_gate = np.clip((coherence - 0.18) / 0.42, 0.15, 1.0)
                y_orig = luma255(out_rgb)
                y_groom = luma255(groomed)
                hf_keep = (y_orig - cv2.GaussianBlur(y_orig, (0, 0), 1.5)) - (y_groom - cv2.GaussianBlur(y_groom, (0, 0), 1.5))
                groomed_hf = np.clip(groomed + 0.82 * hf_keep[:, :, None], 0.0, 255.0)
                d_prot = cv2.distanceTransform((prot_snap < 0.35).astype(np.uint8), cv2.DIST_L2, 5)
                gs = float(np.clip(groom_strength, 0.0, 0.85))
                gw = (
                    gs
                    * coh_gate
                    * (0.15 + 0.85 * np.clip((d_prot - 6.0) / 28.0, 0.0, 1.0))
                    * cv2.GaussianBlur(valid_hair_f, (0, 0), sigmaX=1.2)
                    * groom_mask.astype(np.float32)
                )[:, :, None]
                out_rgb = np.clip(out_rgb * (1.0 - gw) + groomed_hf * gw, 0.0, 255.0)

        hair_union = np.clip(hair_m * out_alpha * (1.0 - occ_tight), 0.0, 1.0)
    else:
        if not (gen_hair_rgb_path and gen_hair_matte_path and gen_hair_mask_path):
            raise ValueError(f"--mode {mode} requires --gen-hair-rgb, --gen-hair-matte, and --gen-hair-mask")
        gen_rgb, _ = read_rgba_f32(gen_hair_rgb_path, (h, w))
        gen_matte = read_gray_f32(gen_hair_matte_path, (h, w))
        gen_hair = read_gray_f32(gen_hair_mask_path, (h, w))

        gen_rgb = _clamp_over_textured_hf(gen_rgb, subj_rgb, np.maximum(hair_m, gen_hair))
        gen_hair_a = np.clip(gen_matte * np.clip((gen_hair - 0.12) / 0.45, 0.0, 1.0) * (1.0 - occ_tight), 0.0, 1.0)

        if mode == "hybrid-flow":
            # Keep real hair everywhere on-body and within 24px of occluder/protected body so crossings never break
            prot_solid = (prot_snap >= 0.45).astype(np.uint8)
            d_prot = cv2.distanceTransform(1 - prot_solid, cv2.DIST_L2, 5)
            real_zone = np.clip(1.0 - np.clip((d_prot - 24.0) / 48.0, 0.0, 1.0), 0.0, 1.0)
            gen_hair_a = np.clip(gen_hair_a * (1.0 - real_zone), 0.0, 1.0)
            # Cleanly zero old air hair strictly outside real_zone + protected body (zero ghost strips)
            prot_hard = np.maximum(prot_raw, prot_snap) >= 0.45
            old_air = np.clip((hair_m >= 0.30).astype(np.float32) * (1.0 - real_zone) * (1.0 - occ_tight), 0.0, 1.0)
            trim_ext = np.where(prot_hard, 1.0, np.where(old_air > 0.5, 0.0, 1.0 - cv2.GaussianBlur(old_air, (0, 0), 0.8)))
            trim_mask = np.minimum(trim_mask, np.where(trim_ext < 0.5, 0.0, trim_ext))
            out_alpha = np.clip(out_alpha * trim_mask, 0.0, 1.0)
        else:
            prot_hard = np.maximum(prot_raw, prot_snap) >= 0.45
            old_hair_cut = (hair_m >= 0.25) & (~prot_hard)
            trim_mask = np.minimum(trim_mask, np.where(old_hair_cut, 0.0, 1.0))
            out_alpha = np.clip(out_alpha * trim_mask, 0.0, 1.0)

        # Composite gen extension seamlessly into output subject RGBA & keep standalone layer
        gen_core = gen_hair_a > 0.75
        if decontam_strength > 0.0 and gen_core.any():
            gen_fringe = np.clip((0.85 - gen_hair_a) / 0.82, 0.0, 1.0) * (gen_hair_a > 0.02)
            gen_rgb = decontaminate_fringe(gen_rgb, gen_core, gen_fringe * float(decontam_strength))
        gen_layer_out = np.dstack([gen_rgb, gen_hair_a])
        tot_a = gen_hair_a + out_alpha * (1.0 - gen_hair_a)
        pm = gen_rgb * gen_hair_a[:, :, None] + out_rgb * (out_alpha * (1.0 - gen_hair_a))[:, :, None]
        out_rgb = np.where(tot_a[:, :, None] > 1e-4, pm / np.maximum(tot_a[:, :, None], 1e-4), 0.0)
        out_alpha = np.clip(tot_a, 0.0, 1.0)
        hair_union = np.clip(np.maximum(hair_m * (subj_alpha * trim_mask), gen_hair_a) * (1.0 - occ_tight), 0.0, 1.0)

    # Non-destructive delta patch covering ONLY pixels changed from input RGBA
    diff_rgb = np.abs(out_rgb - subj_rgb).max(axis=2)
    diff_a = np.abs(out_alpha - subj_alpha) * 255.0
    changed = ((diff_rgb > 1.5) | (diff_a > 2.0)).astype(np.float32)
    patch_alpha = np.clip(cv2.GaussianBlur(changed, (0, 0), sigmaX=1.0) * np.maximum(out_alpha, subj_alpha), 0.0, 1.0)

    p_subj = out_dir / "hair_comp_subject_rgba.png"
    p_patch = out_dir / "hair_comp_patch_rgba.png"
    p_trim = out_dir / "subject_trim_mask.png"
    p_union = out_dir / "hair_alpha_union.png"
    p_occ = out_dir / "occluder_tight_mask.png"
    write_rgba_u8(p_subj, out_rgb, out_alpha, overwrite=overwrite)
    write_rgba_u8(p_patch, out_rgb, patch_alpha, overwrite=overwrite)
    write_png_u8(p_trim, np.clip(np.round(trim_mask * 255.0), 0, 255).astype(np.uint8), overwrite=overwrite)
    write_png_u8(p_union, np.clip(np.round(hair_union * 255.0), 0, 255).astype(np.uint8), overwrite=overwrite)
    write_png_u8(p_occ, np.clip(np.round(occ_tight * 255.0), 0, 255).astype(np.uint8), overwrite=overwrite)

    outputs: dict[str, str] = {
        "subject_rgba": str(p_subj),
        "patch_rgba": str(p_patch),
        "trim_mask": str(p_trim),
        "hair_alpha_union": str(p_union),
        "occluder_tight_mask": str(p_occ),
    }
    if mode != "real-first":
        p_gen = out_dir / "hair_extension_rgba.png"
        write_rgba_u8(p_gen, gen_layer_out[:, :, :3], gen_layer_out[:, :, 3], overwrite=overwrite)
        outputs["hair_extension_rgba"] = str(p_gen)

    meta = {
        "ok": True,
        "mode": mode,
        "composited_external_hair": (mode != "real-first"),
        "size": [w, h],
        "trapped_gaps_cleaned_px": trapped_gap_px,
        "occluder_dilation_px": 0,
        "haze_recover": haze_recover,
        "groom": groom,
        "groom_strength": groom_strength,
        "outputs": outputs,
    }
    (out_dir / "hair_comp_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def _retouch_garment_region(
    rgb_f32: np.ndarray,
    mask_f32: np.ndarray,
    *,
    material: str,
    spatial_dehaze: float,
    target_black_l: str | float,
    neutral_black: float,
    drape_contrast: float,
    weave_preserve: float,
    alpha_mask: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = mask_f32.shape
    lab = rgb255_to_lab(rgb_f32)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    mask_snap = guided_filter_gray(np.clip(L / 100.0, 0.0, 1.0), mask_f32, radius=6, eps=1e-3)
    if alpha_mask is not None:
        mask_snap = np.clip(mask_snap * alpha_mask, 0.0, 1.0)
    sel = mask_snap >= 0.55
    if int(sel.sum()) < 200:
        return rgb_f32.copy(), {"skipped": True, "pixels": int(sel.sum())}

    mat = material
    if mat == "auto":
        l_med = float(np.median(L[sel]))
        c_med = float(np.median(np.hypot(a[sel], b[sel])))
        if l_med >= 58.0:
            mat = "pastel-lace"
        elif l_med < 50.0 or (l_med < 58.0 and c_med < 16.0):
            mat = "dark-twill"
        else:
            mat = "light-cotton"
    is_dark = mat in ("dark-matte", "dark-twill", "glossy-leather")
    s_dehaze = float(0.78 if is_dark else 0.0) if spatial_dehaze < 0.0 else float(np.clip(spatial_dehaze, 0.0, 1.0))
    s_neut = float(0.78 if is_dark else 0.0) if neutral_black < 0.0 else float(np.clip(neutral_black, 0.0, 1.0))

    stats_before = {
        "L_p10": round(float(np.percentile(L[sel], 10)), 2),
        "L_med": round(float(np.median(L[sel])), 2),
        "L_p90": round(float(np.percentile(L[sel], 90)), 2),
        "mean_a": round(float(a[sel].mean()), 2),
        "mean_b": round(float(b[sel].mean()), 2),
    }

    # 1. Luminance-only spatial dehaze in CIE Lab (never divides RGB by warm airlight!)
    L_work = L.copy()
    L_target_used = None
    if is_dark and s_dehaze > 0.0:
        tile = 48
        hs, ws = max(1, h // tile), max(1, w // tile)
        val = np.zeros((hs, ws), dtype=np.float32)
        wgt = np.zeros((hs, ws), dtype=np.float32)
        cand = (mask_snap > 0.55) & ((L < 86.0) if mat == "glossy-leather" else (L < 68.0))
        for ty in range(hs):
            y0, y1 = ty * tile, min(h, (ty + 1) * tile)
            for tx in range(ws):
                x0, x1 = tx * tile, min(w, (tx + 1) * tile)
                cm = cand[y0:y1, x0:x1]
                if cm.sum() >= 60:
                    val[ty, tx] = float(np.percentile(L[y0:y1, x0:x1][cm], 8))
                    wgt[ty, tx] = 1.0
        if wgt.sum() > 0.0:
            num1 = cv2.GaussianBlur(val * wgt, (0, 0), sigmaX=2.0)
            den1 = cv2.GaussianBlur(wgt, (0, 0), sigmaX=2.0)
            num2 = cv2.GaussianBlur(val * wgt, (0, 0), sigmaX=6.0)
            den2 = cv2.GaussianBlur(wgt, (0, 0), sigmaX=6.0)
            dark_t = np.where(den1 > 0.08, num1 / np.maximum(den1, 1e-6), num2 / np.maximum(den2, 1e-6))
            L_floor = cv2.resize(dark_t, (w, h), interpolation=cv2.INTER_CUBIC)
            L_floor = cv2.GaussianBlur(L_floor, (0, 0), sigmaX=32.0)

            ys_s = np.where(sel)[0]
            y_split = int(ys_s.min() + 0.45 * (ys_s.max() - ys_s.min()))
            upper_tiles = (wgt > 0) & ((np.arange(hs)[:, None] * tile) < y_split)
            if str(target_black_l).strip().lower() == "upper" and upper_tiles.any():
                L_tgt = float(np.clip(np.percentile(val[upper_tiles], 25), 10.0, 18.0))
            else:
                req_l = 14.0 if str(target_black_l).strip().lower() == "upper" else float(target_black_l)
                p20_all = float(np.percentile(val[wgt > 0], 20))
                L_tgt = float(np.clip(min(req_l, max(10.0, p20_all)), 9.0, 22.0))
            L_target_used = round(L_tgt, 2)
            L_air = 94.0
            t_min = 0.26 if mat == "glossy-leather" else 0.42
            t_L = np.where(
                L_floor > L_tgt + 1.0,
                np.clip((L_air - L_floor) / max(1.0, L_air - L_tgt), t_min, 1.0),
                1.0,
            )
            L_deh = np.clip((L - L_air) / t_L + L_air, 4.5, 99.0)
            L_work = L * (1.0 - s_dehaze) + L_deh * s_dehaze

    # 2. Neutral-black chroma guard + Lab chroma speckle smoothing (preserves colored trims C* >= 20)
    a_work, b_work = a.copy(), b.copy()
    if is_dark and s_neut > 0.0:
        C_orig = np.hypot(a, b)
        neut_hi = 76.0 if mat == "glossy-leather" else 52.0
        neut_span = 34.0 if mat == "glossy-leather" else 28.0
        w_neut = np.clip((neut_hi - L_work) / neut_span, 0.0, 1.0) * np.clip((20.0 - C_orig) / 8.0, 0.0, 1.0)
        mix = s_neut * w_neut
        a_work = a * (1.0 - mix)
        b_work = b * (1.0 - mix) + (-0.8) * mix
        a_work = cv2.bilateralFilter(a_work.astype(np.float32), 5, 6.0, 5.0)
        b_work = cv2.bilateralFilter(b_work.astype(np.float32), 5, 6.0, 5.0)

    # 3. 3-Band Edge-Preserving Frequency Separation on Luminance L*
    L01 = np.clip(L_work / 100.0, 0.0, 1.0)
    L_low = guided_filter_gray(L01, L01, radius=24, eps=0.015) * 100.0
    L_mid = guided_filter_gray(L01, L01, radius=5, eps=0.002) * 100.0 - L_low
    L_hi = L_work - (L_low + L_mid)

    dc = float(np.clip(drape_contrast, 0.0, 1.0))
    wp = float(np.clip(weave_preserve, 0.0, 1.5))

    if mat in ("dark-matte", "dark-twill"):
        p50_low = float(np.median(L_low[sel]))
        p75_low = float(np.percentile(L_low[sel], 75))
        L_low_out = np.where(L_low > p75_low, L_low - 0.28 * (L_low - p50_low) * 0.55, L_low)
        L_mid_out = L_mid * (1.0 + 0.35 * dc)
        L_hi_out = L_hi * (wp + 0.05)
    elif mat == "glossy-leather":
        L_low_c = np.where(L_low > 28.0, 28.0 + (L_low - 28.0) * 0.16, L_low)
        L_low_out = np.where(L_low_c < 36.0, np.maximum(5.0, L_low_c * 0.85), L_low_c)
        L_mid_out = L_mid * (1.0 + 0.40 * dc)
        L_hi_out = L_hi * (wp + 0.06)
    elif mat == "pastel-lace":
        # Recover sunlit lace embroidery crispness & soft pastel fold roll-off without darkening
        L_low_out = np.where(L_low > 82.0, 82.0 + (L_low - 82.0) * 0.48, L_low)
        L_mid_out = L_mid * (1.0 + 0.38 * dc)
        L_hi_out = L_hi * min(1.45, wp + 0.18)
    elif mat == "translucent-organza":
        L_low_out = np.where(L_low > 85.0, 85.0 + (L_low - 85.0) * 0.55, L_low)
        L_mid_out = L_mid * (1.0 + 0.28 * dc)
        L_hi_out = L_hi * min(1.35, wp + 0.12)
    elif mat == "satin-ribbon":
        # Smooth noisy mottling while crisping satin highlight crests
        L_low_sm = cv2.bilateralFilter(L_low.astype(np.float32), 7, 5.0, 10.0)
        L_low_out = 0.65 * L_low + 0.35 * L_low_sm
        L_mid_out = L_mid * (1.0 + 0.24 * dc)
        L_hi_out = L_hi * wp
    elif mat == "knit-stocking":
        L_low_out = np.where(L_low > 86.0, 86.0 + (L_low - 86.0) * 0.58, L_low)
        L_mid_out = L_mid * (1.0 + 0.34 * dc)
        L_hi_out = L_hi * min(1.42, wp + 0.16)
    else:
        # light-cotton
        L_low_out = np.where(L_low > 86.0, 86.0 + (L_low - 86.0) * 0.65, L_low)
        L_mid_out = L_mid * (1.0 + 0.22 * dc)
        L_hi_out = L_hi * wp

    L_out = np.clip(L_low_out + L_mid_out + L_hi_out, 4.5, 99.2)
    lab_out = np.dstack([L_out, a_work, b_work]).astype(np.float32)
    retouched = lab_to_rgb255(lab_out)

    blend_w = np.clip(cv2.GaussianBlur(mask_snap, (0, 0), sigmaX=1.2), 0.0, 1.0)
    if alpha_mask is not None:
        blend_w = np.where(alpha_mask < 0.02, 0.0, blend_w)
    out_rgb = np.clip(rgb_f32 * (1.0 - blend_w[:, :, None]) + retouched * blend_w[:, :, None], 0.0, 255.0)

    lab_after = rgb255_to_lab(out_rgb)
    stats_after = {
        "L_p10": round(float(np.percentile(lab_after[:, :, 0][sel], 10)), 2),
        "L_med": round(float(np.median(lab_after[:, :, 0][sel])), 2),
        "L_p90": round(float(np.percentile(lab_after[:, :, 0][sel], 90)), 2),
        "mean_a": round(float(lab_after[:, :, 1][sel].mean()), 2),
        "mean_b": round(float(lab_after[:, :, 2][sel].mean()), 2),
    }
    return out_rgb, {
        "material": mat,
        "pixels": int(sel.sum()),
        "target_black_L": L_target_used,
        "before": stats_before,
        "after": stats_after,
    }


def run_garment_retouch(
    *,
    input_path: Path,
    out_path: Path,
    garment_mask_path: Path | None = None,
    body_parse_dir: Path | None = None,
    alpha_mask_path: Path | None = None,
    out_patch_path: Path | None = None,
    out_dark_mask_path: Path | None = None,
    material: str = "auto",
    spatial_dehaze: float = -1.0,
    target_black_l: str = "14.0",
    neutral_black: float = -1.0,
    drape_contrast: float = 0.35,
    weave_preserve: float = 1.0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Frequency-separation & CIE Lab luminance-only spatial dehaze garment retouching."""
    if not garment_mask_path and not body_parse_dir:
        raise ValueError("Provide --garment-mask or --body-parse-dir.")

    in_rgb, in_alpha = read_rgba_f32(input_path)
    h, w = in_alpha.shape
    alpha_mask = read_gray_f32(alpha_mask_path, (h, w)) if alpha_mask_path else (in_alpha if in_alpha.min() < 0.99 else None)

    cur_rgb = in_rgb.copy()
    region_reports: dict[str, Any] = {}
    combined_mask = np.zeros((h, w), dtype=np.float32)
    dark_union = np.zeros((h, w), dtype=np.float32)

    if garment_mask_path is not None:
        gmask = read_gray_f32(garment_mask_path, (h, w))
        combined_mask = np.maximum(combined_mask, gmask)
        cur_rgb, region_reports["custom_garment"] = _retouch_garment_region(
            cur_rgb,
            gmask,
            material=material,
            spatial_dehaze=spatial_dehaze,
            target_black_l=target_black_l,
            neutral_black=neutral_black,
            drape_contrast=drape_contrast,
            weave_preserve=weave_preserve,
            alpha_mask=alpha_mask,
        )
    else:
        assert body_parse_dir is not None
        bp = body_parse_dir.resolve()
        lab0 = rgb255_to_lab(cur_rgb)
        L0, a0, b0 = lab0[:, :, 0], lab0[:, :, 1], lab0[:, :, 2]

        # 1. Skirt / dress / pants (protect warm knee skin mislabeled inside SegFormer mask_pants)
        for cname in ("mask_skirt_dress", "mask_pants"):
            cp = bp / f"{cname}.png"
            if cp.exists():
                m = read_gray_f32(cp, (h, w))
                if cname == "mask_pants":
                    m = m * np.where((L0 > 50.0) & (a0 > 4.2), 0.0, 1.0)
                if (m >= 0.5).sum() >= 300:
                    combined_mask = np.maximum(combined_mask, m)
                    cur_rgb, region_reports[cname] = _retouch_garment_region(
                        cur_rgb, m,
                        material=material if material != "auto" else "auto",
                        spatial_dehaze=spatial_dehaze,
                        target_black_l=target_black_l,
                        neutral_black=neutral_black,
                        drape_contrast=drape_contrast,
                        weave_preserve=weave_preserve,
                        alpha_mask=alpha_mask,
                    )
                    if region_reports[cname].get("material") in ("dark-twill", "dark-matte", "glossy-leather"):
                        dark_union = np.maximum(dark_union, m)

        # 2. Upper clothes (automatically separates light blouse vs dark bodice/straps by Lab luminance)
        up_p = bp / "mask_upper_clothes.png"
        if up_p.exists():
            um = read_gray_f32(up_p, (h, w))
            dark_part = um * np.clip((52.0 - L0) / 14.0, 0.0, 1.0)
            light_part = um * np.clip((L0 - 52.0) / 14.0, 0.0, 1.0)
            if (dark_part >= 0.5).sum() >= 300:
                combined_mask = np.maximum(combined_mask, dark_part)
                dark_union = np.maximum(dark_union, dark_part)
                cur_rgb, region_reports["upper_dark_trim"] = _retouch_garment_region(
                    cur_rgb, dark_part,
                    material="dark-twill" if material == "auto" else material,
                    spatial_dehaze=spatial_dehaze,
                    target_black_l=target_black_l,
                    neutral_black=neutral_black,
                    drape_contrast=drape_contrast,
                    weave_preserve=weave_preserve,
                    alpha_mask=alpha_mask,
                )
            if (light_part >= 0.5).sum() >= 300:
                combined_mask = np.maximum(combined_mask, light_part)
                cur_rgb, region_reports["upper_light_blouse"] = _retouch_garment_region(
                    cur_rgb, light_part,
                    material="light-cotton",
                    spatial_dehaze=0.0,
                    target_black_l=target_black_l,
                    neutral_black=0.0,
                    drape_contrast=drape_contrast * 0.75,
                    weave_preserve=weave_preserve,
                    alpha_mask=alpha_mask,
                )

        # 3. Shoes + dark socks detected strictly inside SegFormer mask_legs
        legs_p = bp / "mask_legs.png"
        shoes_p = bp / "mask_shoes.png"
        if legs_p.exists():
            lm = read_gray_f32(legs_p, (h, w))
            # Dark knee-socks vs bright white knit stockings (never confuse white stockings with black socks)
            raw_socks = (lm > 0.40) & (L0 < 52.0) & (np.hypot(a0, b0) < 13.0) & (b0 < 9.5) & (a0 < 4.0)
            sock_clean = cv2.morphologyEx(raw_socks.astype(np.uint8), cv2.MORPH_OPEN, ell_kernel(2))
            sock_clean = cv2.morphologyEx(sock_clean, cv2.MORPH_CLOSE, ell_kernel(3))
            sock_m = guided_filter_gray(np.clip(L0 / 100.0, 0.0, 1.0), sock_clean.astype(np.float32), radius=8, eps=5e-4) * lm
            if (sock_m >= 0.5).sum() >= 300:
                combined_mask = np.maximum(combined_mask, sock_m)
                dark_union = np.maximum(dark_union, sock_m)
                cur_rgb, region_reports["dark_socks_in_legs"] = _retouch_garment_region(
                    cur_rgb, sock_m,
                    material="dark-matte",
                    spatial_dehaze=spatial_dehaze if spatial_dehaze >= 0 else 0.85,
                    target_black_l=target_black_l,
                    neutral_black=neutral_black if neutral_black >= 0 else 0.82,
                    drape_contrast=drape_contrast,
                    weave_preserve=weave_preserve,
                    alpha_mask=alpha_mask,
                )
        if shoes_p.exists():
            sm = read_gray_f32(shoes_p, (h, w))
            if (sm >= 0.5).sum() >= 300:
                combined_mask = np.maximum(combined_mask, sm)
                dark_union = np.maximum(dark_union, sm)
                cur_rgb, region_reports["shoes"] = _retouch_garment_region(
                    cur_rgb, sm,
                    material="glossy-leather" if material == "auto" else material,
                    spatial_dehaze=spatial_dehaze if spatial_dehaze >= 0 else 0.85,
                    target_black_l=target_black_l,
                    neutral_black=neutral_black if neutral_black >= 0 else 0.82,
                    drape_contrast=drape_contrast,
                    weave_preserve=weave_preserve,
                    alpha_mask=alpha_mask,
                )

        # 4. Catch residual fogged dark footwear/socks missed by SegFormer (e.g., kicked shoe toe outside mask_skirt_dress)
        skin_p = bp / "mask_all_skin.png"
        hair_p = bp / "mask_hair.png"
        up_p2 = bp / "mask_upper_clothes.png"
        pants_p2 = bp / "mask_pants.png"
        skirt_p2 = bp / "mask_skirt_dress.png"
        skin_m = read_gray_f32(skin_p, (h, w)) if skin_p.exists() else np.zeros((h, w), dtype=np.float32)
        hair_m_bp = read_gray_f32(hair_p, (h, w)) if hair_p.exists() else np.zeros((h, w), dtype=np.float32)
        up_m_bp = read_gray_f32(up_p2, (h, w)) if up_p2.exists() else np.zeros((h, w), dtype=np.float32)
        shoes_m = read_gray_f32(shoes_p, (h, w)) if shoes_p.exists() else np.zeros((h, w), dtype=np.float32)
        pants_m = read_gray_f32(pants_p2, (h, w)) if pants_p2.exists() else np.zeros((h, w), dtype=np.float32)
        legs_m = read_gray_f32(legs_p, (h, w)) if legs_p.exists() else np.zeros((h, w), dtype=np.float32)
        skirt_m = read_gray_f32(skirt_p2, (h, w)) if skirt_p2.exists() else np.zeros((h, w), dtype=np.float32)
        lower_anchor = (shoes_m >= 0.5) | (pants_m >= 0.5) | (legs_m >= 0.5)
        if (skirt_m >= 0.5).sum() >= 300:
            y_sk_mid = int(np.median(np.where(skirt_m >= 0.5)[0]))
            lower_anchor = lower_anchor | ((skirt_m >= 0.5) & (np.arange(h)[:, None] >= y_sk_mid))
        d_lower = cv2.distanceTransform((~lower_anchor).astype(np.uint8), cv2.DIST_L2, 5) if lower_anchor.any() else np.full((h, w), 9999.0, np.float32)
        d_hair = cv2.distanceTransform((hair_m_bp < 0.5).astype(np.uint8), cv2.DIST_L2, 5) if (hair_m_bp >= 0.5).any() else np.full((h, w), 9999.0, np.float32)
        a_eff = alpha_mask if alpha_mask is not None else np.ones((h, w), dtype=np.float32)
        res_dark = (
            (a_eff > 0.50) & (skin_m < 0.35) & (hair_m_bp < 0.35) & (up_m_bp < 0.30)
            & (d_lower <= 95.0) & (d_hair > 50.0)
            & (combined_mask < 0.40) & (L0 < 48.0) & (np.hypot(a0, b0) < 14.0)
            & (~((L0 > 54.0) & (a0 > 4.2)))
        )
        res_close = cv2.morphologyEx(res_dark.astype(np.uint8), cv2.MORPH_CLOSE, ell_kernel(7))
        res_open = cv2.morphologyEx(res_close, cv2.MORPH_OPEN, ell_kernel(2))
        n_rd, lbl_rd, st_rd, _ = cv2.connectedComponentsWithStats(res_open, connectivity=8)
        res_clean = np.zeros((h, w), dtype=np.float32)
        for rid in range(1, n_rd):
            comp = lbl_rd == rid
            if int(st_rd[rid, cv2.CC_STAT_AREA]) >= 400 and float(a0[comp].mean()) < 4.2:
                res_clean[comp] = 1.0
        if (res_clean >= 0.5).sum() >= 400:
            res_soft = cv2.GaussianBlur(res_clean, (0, 0), sigmaX=1.5) * a_eff
            combined_mask = np.maximum(combined_mask, res_soft)
            dark_union = np.maximum(dark_union, res_soft)
            cur_rgb, region_reports["residual_dark_garment"] = _retouch_garment_region(
                cur_rgb, res_soft,
                material="glossy-leather",
                spatial_dehaze=spatial_dehaze if spatial_dehaze >= 0 else 0.90,
                target_black_l="14.0",
                neutral_black=neutral_black if neutral_black >= 0 else 0.88,
                drape_contrast=drape_contrast,
                weave_preserve=weave_preserve,
                alpha_mask=alpha_mask,
            )

        # 5. Defringe semi-transparent alpha halo along outer dark-garment border (never draws heavy outline inside solid silhouette)
        if (dark_union >= 0.40).any():
            lab_d = rgb255_to_lab(cur_rgb)
            Ld, ad, bd = lab_d[:, :, 0], lab_d[:, :, 1], lab_d[:, :, 2]
            dark_core = (dark_union >= 0.45) & (a_eff >= 0.90) & (skin_m < 0.25)
            d_dark_edge = cv2.distanceTransform((dark_union < 0.40).astype(np.uint8), cv2.DIST_L2, 5)
            cutout_rim = ((a_eff > 0.02) & (a_eff < 0.88)) | (
                (a_eff > 0.02) & (cv2.dilate((a_eff < 0.50).astype(np.uint8), ell_kernel(1)) > 0)
            )
            pale_halo = (
                cutout_rim
                & (d_dark_edge <= 3.0)
                & (skin_m < 0.25)
                & (up_m_bp < 0.30)
                & (~((Ld > 54.0) & (ad > 4.2)))
                & (Ld > 30.0)
                & (np.hypot(ad, bd) < 13.0)
            )
            if dark_core.any() and pale_halo.any():
                rim_w = cv2.GaussianBlur(pale_halo.astype(np.float32), (0, 0), sigmaX=0.6)
                cur_rgb = decontaminate_fringe(cur_rgb, dark_core, np.clip(rim_w * 0.85, 0.0, 1.0), sigmas=(2.5, 6.0, 12.0))
                combined_mask = np.maximum(combined_mask, rim_w * a_eff)
                region_reports["dark_garment_defringe_px"] = int(pale_halo.sum())

    if alpha_mask is not None or in_alpha.min() < 0.99:
        write_rgba_u8(out_path, cur_rgb, in_alpha, overwrite=overwrite)
    else:
        write_rgb_u8(out_path, cur_rgb, overwrite=overwrite)

    if out_patch_path is not None:
        patch_a = np.clip(combined_mask * (alpha_mask if alpha_mask is not None else 1.0), 0.0, 1.0)
        write_rgba_u8(out_patch_path, cur_rgb, patch_a, overwrite=overwrite)

    if out_dark_mask_path is not None:
        a_out = alpha_mask if alpha_mask is not None else in_alpha
        dark_u8 = np.clip(np.round(dark_union * np.clip(a_out, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
        write_png_u8(out_dark_mask_path, dark_u8, overwrite=overwrite)

    return {
        "ok": True,
        "input": str(input_path),
        "out": str(out_path),
        "out_patch": str(out_patch_path) if out_patch_path else None,
        "out_dark_mask": str(out_dark_mask_path) if out_dark_mask_path else None,
        "regions": region_reports,
    }


def run_skin_unify(
    *,
    input_path: Path,
    body_parse_dir: Path,
    out_path: Path,
    alpha_mask_path: Path | None = None,
    out_patch_path: Path | None = None,
    warmth_pull: float = 0.72,
    glare_compress: float = 0.62,
    skirt_shadow: float = 0.42,
    defringe: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Unify washed/glared lower-limb skin with sunlit arms, defringe outer leg halo, and add skirt/leg cast & AO shadows."""
    in_rgb, in_alpha = read_rgba_f32(input_path)
    h, w = in_alpha.shape
    a_eff = read_gray_f32(alpha_mask_path, (h, w)) if alpha_mask_path else (in_alpha if in_alpha.min() < 0.99 else np.ones((h, w), dtype=np.float32))
    bp = body_parse_dir.resolve()

    def _r(name: str) -> np.ndarray:
        p = bp / f"{name}.png"
        return read_gray_f32(p, (h, w)) if p.exists() else np.zeros((h, w), dtype=np.float32)

    arms_m = _r("mask_arms")
    legs_m = _r("mask_legs")
    face_m = _r("mask_face_skin")
    skirt_m = _r("mask_skirt_dress")
    pants_m = _r("mask_pants")
    up_m = _r("mask_upper_clothes")
    shoes_m = _r("mask_shoes")
    hair_m = _r("mask_hair")

    lab0 = rgb255_to_lab(in_rgb)
    L0, a0, b0 = lab0[:, :, 0], lab0[:, :, 1], lab0[:, :, 2]
    C0 = np.hypot(a0, b0)

    # Identify dark socks inside or near legs + all dark garments
    dark_socks_raw = (legs_m >= 0.40) & (L0 < 52.0) & (C0 < 12.5)
    dark_socks = cv2.morphologyEx(dark_socks_raw.astype(np.uint8), cv2.MORPH_CLOSE, ell_kernel(3)) > 0
    dark_socks = dark_socks | ((a_eff > 0.50) & (L0 < 40.0) & (C0 < 11.0) & (arms_m < 0.35) & (face_m < 0.35))
    dark_pants = (pants_m >= 0.45) & ((L0 < 52.0) | (a0 < 2.2) | (b0 < 2.5))
    non_leg = (skirt_m >= 0.45) | dark_pants | (shoes_m >= 0.45) | (hair_m >= 0.40) | (up_m >= 0.45) | (arms_m >= 0.45) | (face_m >= 0.45) | dark_socks

    # 1. Expand leg skin up to 165px around SegFormer mask_legs to catch unmasked/pants-mislabeled upper thigh/knee seamlessly
    leg_seed = (legs_m >= 0.50) & (~non_leg) & (a_eff > 0.50) & (a0 >= 1.4) & (b0 >= 1.8)
    if leg_seed.sum() < 200:
        write_rgba_u8(out_path, in_rgb, in_alpha, overwrite=overwrite) if (alpha_mask_path or in_alpha.min() < 0.99) else write_rgb_u8(out_path, in_rgb, overwrite=overwrite)
        return {"ok": True, "skipped": True, "reason": "insufficient leg_seed pixels"}

    d_leg = cv2.distanceTransform((legs_m < 0.40).astype(np.uint8), cv2.DIST_L2, 5)
    leg_cand = (
        (a_eff > 0.35)
        & (~non_leg)
        & (d_leg <= 165.0)
        & (L0 >= 38.0)
        & (a0 >= 1.4)
        & (b0 >= 1.8)
    )
    leg_closed = cv2.morphologyEx(leg_cand.astype(np.uint8), cv2.MORPH_CLOSE, ell_kernel(3))
    n_lc, lbl_lc, st_lc, _ = cv2.connectedComponentsWithStats(leg_closed, connectivity=8)
    leg_keep = np.zeros((h, w), dtype=np.float32)
    for cid in range(1, n_lc):
        comp = lbl_lc == cid
        if int(st_lc[cid, cv2.CC_STAT_AREA]) >= 300 and (comp & leg_seed).any():
            leg_keep[comp] = 1.0

    leg_snap = guided_filter_gray(np.clip(L0 / 100.0, 0.0, 1.0), leg_keep, radius=6, eps=5e-4)
    leg_snap = np.clip(leg_snap * (1.0 - np.clip(dark_socks.astype(np.float32), 0.0, 1.0)) * (1.0 - np.clip(skirt_m, 0.0, 1.0)) * (1.0 - np.clip(shoes_m, 0.0, 1.0)) * a_eff, 0.0, 1.0)
    leg_sel = leg_snap >= 0.45

    # Reference sunlit arm skin
    arm_ref = (arms_m >= 0.60) & (a_eff > 0.85) & (L0 >= 45.0) & (L0 <= 86.0) & (C0 >= 12.0)
    if arm_ref.sum() < 200:
        arm_ref = (face_m >= 0.60) & (a_eff > 0.85) & (L0 >= 42.0) & (L0 <= 84.0) & (C0 >= 11.0)
    L_arm = float(np.median(L0[arm_ref]))
    a_arm = float(np.median(a0[arm_ref]))
    b_arm = float(np.median(b0[arm_ref]))

    legs_before = {
        "L_med": round(float(np.median(L0[leg_sel])), 2),
        "a_med": round(float(np.median(a0[leg_sel])), 2),
        "b_med": round(float(np.median(b0[leg_sel])), 2),
    }

    # 2. Normalized-convolution low-frequency skin tone & glare recovery (keeps L_fine texture crisp)
    mask_f = leg_sel.astype(np.float32)
    den = np.maximum(cv2.GaussianBlur(mask_f, (0, 0), sigmaX=18.0), 1e-4)
    L_lf = cv2.GaussianBlur(np.where(leg_sel, L0, 0.0), (0, 0), sigmaX=18.0) / den
    a_lf = cv2.GaussianBlur(np.where(leg_sel, a0, 0.0), (0, 0), sigmaX=18.0) / den
    b_lf = cv2.GaussianBlur(np.where(leg_sel, b0, 0.0), (0, 0), sigmaX=18.0) / den

    L_fine = L0 - L_lf
    w_wash = np.clip((L_lf - (L_arm - 2.0)) / 10.0, 0.0, 1.0)
    w_desat = np.clip((a_arm - a_lf) / 8.0, 0.0, 1.0)
    w_glare = np.clip(0.55 * w_desat + 0.45 * w_wash, 0.0, 1.0)

    gc = float(np.clip(glare_compress, 0.0, 1.0))
    wp = float(np.clip(warmth_pull, 0.0, 1.0))

    L_target_floor = L_arm * 0.94
    L_lf_new = L_lf - gc * w_glare * np.maximum(0.0, L_lf - L_target_floor)
    L_new = np.clip(L_lf_new + L_fine * 0.95, 10.0, 96.0)

    chroma_pull = wp * np.clip(0.32 + 0.68 * w_desat, 0.0, 1.0)
    a_new = a0 + chroma_pull * (a_arm - a_lf)
    b_new = b0 + chroma_pull * (b_arm - b_lf)

    # 3. Skirt & overlapping crossed-leg + dark sock-cuff cast / AO shadows
    ss = float(np.clip(skirt_shadow, 0.0, 0.85))
    if ss > 0.0:
        sk_bin = (skirt_m >= 0.45).astype(np.uint8)
        d_sk = cv2.distanceTransform(1 - sk_bin, cv2.DIST_L2, 5)
        M_sk = np.array([[1.0, 0.0, 34.0], [0.0, 1.0, 52.0]], dtype=np.float32)
        sk_sh = cv2.warpAffine(sk_bin.astype(np.float32), M_sk, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        d_sk_sh = cv2.distanceTransform((sk_sh < 0.40).astype(np.uint8), cv2.DIST_L2, 5)
        skirt_term = np.maximum(0.72 * np.exp(-d_sk / 82.0), 0.95 * np.exp(-d_sk_sh / 52.0))

        # Overlapping leg / dark sock cuff contact + directional shadow (sun from upper-left -> cast down-right)
        sock_bin = dark_socks.astype(np.uint8)
        if sock_bin.any():
            d_so = cv2.distanceTransform(1 - sock_bin, cv2.DIST_L2, 5)
            M_so = np.array([[1.0, 0.0, 22.0], [0.0, 1.0, 34.0]], dtype=np.float32)
            so_sh = cv2.warpAffine(sock_bin.astype(np.float32), M_so, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            d_so_sh = cv2.distanceTransform((so_sh < 0.40).astype(np.uint8), cv2.DIST_L2, 5)
            sock_term = np.maximum(0.65 * np.exp(-d_so / 24.0), 0.82 * np.exp(-d_so_sh / 30.0))
        else:
            sock_term = np.zeros((h, w), dtype=np.float32)

        shad = np.clip(ss * (0.90 * skirt_term + 0.65 * sock_term) * leg_snap, 0.0, 0.56)
        L_new = np.clip(L_new * (1.0 - 0.38 * shad), 8.0, 96.0)
        a_new = a_new * (1.0 - 0.06 * shad) + 0.6 * shad
        b_new = b_new * (1.0 - 0.10 * shad)

    unified_rgb = lab_to_rgb255(np.dstack([L_new, a_new, b_new]).astype(np.float32))
    bw = np.clip(cv2.GaussianBlur(leg_snap, (0, 0), sigmaX=1.2), 0.0, 1.0)[:, :, None]
    out_rgb = np.clip(in_rgb * (1.0 - bw) + unified_rgb * bw, 0.0, 255.0)

    # 4. Outer leg-skin white-rim defringe (eliminates bright left-edge cutout border along knees/thighs)
    defringe_px = 0
    if defringe:
        lab_u = rgb255_to_lab(out_rgb)
        Lu, au, bu = lab_u[:, :, 0], lab_u[:, :, 1], lab_u[:, :, 2]
        leg_core = (leg_snap >= 0.75) & (a_eff > 0.88) & (au >= 7.0) & (bu >= 9.0)
        cutout_edge = ((a_eff > 0.03) & (a_eff < 0.94)) | (
            (a_eff > 0.03) & (cv2.dilate((a_eff < 0.35).astype(np.uint8), ell_kernel(5)) > 0)
        )
        outer_halo = (leg_snap >= 0.15) & cutout_edge & ((au < 9.5) | (Lu > 74.0) | (a_eff < 0.90))
        if leg_core.any() and outer_halo.any():
            rim_w = cv2.GaussianBlur(outer_halo.astype(np.float32), (0, 0), sigmaX=0.8) * np.clip(leg_snap * 2.0, 0.0, 1.0)
            out_rgb = decontaminate_fringe(out_rgb, leg_core, np.clip(rim_w * 0.90, 0.0, 1.0))
            defringe_px = int(outer_halo.sum())

    lab_after = rgb255_to_lab(out_rgb)
    legs_after = {
        "L_med": round(float(np.median(lab_after[:, :, 0][leg_sel])), 2),
        "a_med": round(float(np.median(lab_after[:, :, 1][leg_sel])), 2),
        "b_med": round(float(np.median(lab_after[:, :, 2][leg_sel])), 2),
    }

    if alpha_mask_path is not None or in_alpha.min() < 0.99:
        write_rgba_u8(out_path, out_rgb, in_alpha, overwrite=overwrite)
    else:
        write_rgb_u8(out_path, out_rgb, overwrite=overwrite)

    if out_patch_path is not None:
        patch_a = np.clip(cv2.GaussianBlur(leg_snap, (0, 0), sigmaX=1.2) * (a_eff if alpha_mask_path else in_alpha), 0.0, 1.0)
        write_rgba_u8(out_patch_path, out_rgb, patch_a, overwrite=overwrite)

    return {
        "ok": True,
        "input": str(input_path),
        "out": str(out_path),
        "out_patch": str(out_patch_path) if out_patch_path else None,
        "leg_pixels": int(leg_sel.sum()),
        "defringe_px": defringe_px,
        "arms_ref_lab": {"L_med": round(L_arm, 2), "a_med": round(a_arm, 2), "b_med": round(b_arm, 2)},
        "legs_before": legs_before,
        "legs_after": legs_after,
    }


def _apply_pih_curves_np(curves_3x32: np.ndarray, img01: np.ndarray) -> np.ndarray:
    """Exact separable 32-entry LUT linear interpolation matching upstream adobe/PIH."""
    pos = np.clip(img01 * 32.0 - 0.5, 0.0, 31.0)
    lo = np.floor(pos).astype(np.int32)
    hi = np.minimum(lo + 1, 31)
    frac = pos - lo
    out = np.empty_like(img01, dtype=np.float32)
    for c in range(3):
        lut = curves_3x32[c]
        p_lo = lut[lo[:, :, c]]
        p_hi = lut[hi[:, :, c]]
        out[:, :, c] = p_lo + (p_hi - p_lo) * frac[:, :, c]
    return np.clip(out, 0.0, 1.0)


def run_harmonize(
    *,
    fg_rgba_path: Path,
    bg_path: Path,
    out_dir: Path,
    pih_curves_path: Path,
    pih_gain_path: Path,
    skin_mask_path: Path | None = None,
    garment_dark_mask_path: Path | None = None,
    backend: str = "auto",
    strength: float = 0.70,
    gain_strength: float = 0.50,
    skin_preserve: float = 0.60,
    light_wrap: float = 0.22,
    grain_match: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Harmonize subject foreground to scene background via Apache-2.0 PhotoBlaze-PIH ONNX."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fg_rgb, fg_a = read_rgba_f32(fg_rgba_path)
    h, w = fg_a.shape
    bg_rgb, _ = read_rgba_f32(bg_path, (h, w))

    use_pih = (
        backend == "pih"
        or (backend == "auto" and pih_curves_path.exists() and pih_gain_path.exists())
    )
    provider_used = "analytical"
    backend_used = "analytical"

    fg01 = np.clip(fg_rgb / 255.0, 0.0, 1.0).astype(np.float32)
    bg01 = np.clip(bg_rgb / 255.0, 0.0, 1.0).astype(np.float32)

    if use_pih:
        if not pih_curves_path.exists() or not pih_gain_path.exists():
            raise FileNotFoundError(
                f"PIH ONNX weights missing: {pih_curves_path} / {pih_gain_path}. Run download_models.py core."
            )
        from cuda_env import create_onnx_session

        fg512 = cv2.resize(fg01, (512, 512), interpolation=cv2.INTER_AREA)
        bg512 = cv2.resize(bg01, (512, 512), interpolation=cv2.INTER_AREA)
        m512 = np.clip(cv2.resize(fg_a, (512, 512), interpolation=cv2.INTER_AREA), 0.0, 1.0)
        comp512 = fg512 * m512[:, :, None] + bg512 * (1.0 - m512[:, :, None])

        t_comp = np.transpose(comp512, (2, 0, 1))[None, ...]
        t_bg = np.transpose(bg512, (2, 0, 1))[None, ...]
        t_m = m512[None, None, ...]

        sess_c = create_onnx_session(pih_curves_path)
        provider_used = sess_c.get_providers()[0] if sess_c.get_providers() else "unknown"
        curves = np.asarray(
            sess_c.run(None, {"inputs": np.concatenate([t_comp, t_bg, t_m], axis=1).astype(np.float32)})[0],
            dtype=np.float32,
        )[0]  # (3, 32)
        del sess_c

        curved512 = _apply_pih_curves_np(curves, comp512)
        inter512 = curved512 * m512[:, :, None] + bg512 * (1.0 - m512[:, :, None])
        t_inter = np.transpose(inter512, (2, 0, 1))[None, ...]

        sess_g = create_onnx_session(pih_gain_path)
        gain512 = np.asarray(
            sess_g.run(None, {"inputs": np.concatenate([t_comp, t_bg, t_m, t_inter], axis=1).astype(np.float32)})[0],
            dtype=np.float32,
        )[0, 0]
        del sess_g
        backend_used = "pih_onnx"

        fg_curved = _apply_pih_curves_np(curves, fg01)
        gain_4k = cv2.GaussianBlur(cv2.resize(gain512, (w, h), interpolation=cv2.INTER_CUBIC), (0, 0), sigmaX=8.0)
        gain_4k = np.clip(gain_4k, 0.58, 1.42)
        gain_eff = 1.0 + (gain_4k - 1.0) * float(np.clip(gain_strength, 0.0, 1.0))
        s = float(np.clip(strength, 0.0, 1.0))
        fg_harm = np.clip((fg01 * (1.0 - s) + fg_curved * s) * gain_eff[:, :, None], 0.0, 1.0) * 255.0
    else:
        # Analytical fallback: match adjacent L/R illumination slope and shadow neutrality
        sub = fg_a > 0.9
        bgm = fg_a < 0.05
        lab_bg = rgb255_to_lab(bg_rgb)
        lab_fg = rgb255_to_lab(fg_rgb)
        left = np.arange(w)[None, :] < (int(np.median(np.where(sub)[1])) if sub.any() else w // 2)
        bg_lr = float(lab_bg[:, :, 0][bgm & left].mean() - lab_bg[:, :, 0][bgm & ~left].mean()) if bgm.any() else 0.0
        fg_lr = float(lab_fg[:, :, 0][sub & left].mean() - lab_fg[:, :, 0][sub & ~left].mean()) if sub.any() else 0.0
        slope_fix = np.clip((bg_lr - fg_lr) / 80.0, -0.25, 0.25)
        x_norm = np.linspace(1.0, -1.0, w, dtype=np.float32)[None, :]
        gain_eff = np.clip(1.0 + slope_fix * x_norm * 0.5, 0.72, 1.25)
        fg_harm = np.clip(fg_rgb * gain_eff[:, :, None], 0.0, 255.0)

    # Protect skin natural hue/warmth while keeping PIH directional luminance shading
    if skin_mask_path is not None and skin_preserve > 0.0:
        skin_m = read_gray_f32(skin_mask_path, (h, w)) * fg_a
        skin_w = guided_filter_gray(np.clip(luma255(fg_rgb) / 255.0, 0.0, 1.0), skin_m, radius=6, eps=1e-3)
        lab_orig = rgb255_to_lab(fg_rgb)
        lab_h = rgb255_to_lab(fg_harm)
        sp = float(np.clip(skin_preserve, 0.0, 1.0)) * skin_w
        lab_h[:, :, 1] = lab_h[:, :, 1] * (1.0 - sp) + lab_orig[:, :, 1] * sp
        lab_h[:, :, 2] = lab_h[:, :, 2] * (1.0 - sp) + lab_orig[:, :, 2] * sp
        # Prevent excessive overall skin darkening
        lab_h[:, :, 0] = lab_h[:, :, 0] * (1.0 - 0.25 * skin_w) + np.maximum(lab_h[:, :, 0], lab_orig[:, :, 0] - 8.0) * (0.25 * skin_w)
        fg_harm = lab_to_rgb255(lab_h)

    # Prevent PIH from tinting neutral dark garments brown/cyan
    if garment_dark_mask_path is not None:
        gdark = read_gray_f32(garment_dark_mask_path, (h, w)) * fg_a
        gdark_w = cv2.GaussianBlur(gdark, (0, 0), sigmaX=2.0)
        lab_orig = rgb255_to_lab(fg_rgb)
        lab_h = rgb255_to_lab(fg_harm)
        # Keep neutral shadow chroma very close to original neutral dark garment
        c_orig = np.hypot(lab_orig[:, :, 1], lab_orig[:, :, 2])
        lock_c = gdark_w * np.clip((18.0 - c_orig) / 8.0, 0.0, 1.0) * 0.90
        lab_h[:, :, 1] = lab_h[:, :, 1] * (1.0 - lock_c) + lab_orig[:, :, 1] * lock_c
        lab_h[:, :, 2] = lab_h[:, :, 2] * (1.0 - lock_c) + lab_orig[:, :, 2] * lock_c
        fg_harm = lab_to_rgb255(lab_h)

    # Dual-radius optical Light Wrap (PTC Blend-If protected on deep blacks)
    lw = float(np.clip(light_wrap, 0.0, 1.0))
    er_near = cv2.erode(fg_a, ell_kernel(8))
    er_far = cv2.erode(fg_a, ell_kernel(21))
    ring_near = np.clip(fg_a - cv2.GaussianBlur(er_near, (0, 0), sigmaX=4.5), 0.0, 1.0)
    ring_far = np.clip(fg_a - cv2.GaussianBlur(er_far, (0, 0), sigmaX=14.0), 0.0, 1.0)
    bg_near = cv2.GaussianBlur(bg_rgb, (0, 0), sigmaX=14.0)
    bg_far = cv2.GaussianBlur(bg_rgb, (0, 0), sigmaX=48.0)
    luma_fg = luma255(fg_harm)
    dark_guard = np.clip((luma_fg - 28.0) / 36.0, 0.12, 1.0)
    bg_bright = np.clip(luma255(bg_near) / 255.0, 0.2, 1.0)
    wrap_rgb = np.clip(
        (bg_near * ring_near[:, :, None] * 0.65 + bg_far * ring_far[:, :, None] * 0.35)
        * (lw * 1.15 * dark_guard * bg_bright * fg_a)[:, :, None],
        0.0,
        255.0,
    )

    # Preview & grain match
    preview_subj = np.clip(fg_harm + wrap_rgb * 0.45, 0.0, 255.0)
    comp = bg_rgb * (1.0 - fg_a[:, :, None]) + preview_subj * fg_a[:, :, None]

    # Measure flat-region HF grain ratio using exact mask-audit formula
    gray_c = cv2.cvtColor(np.clip(comp, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray_c, cv2.CV_32F)), (0, 0), 4.0)
    hf = np.abs(gray_c - cv2.GaussianBlur(gray_c, (0, 0), 1.0))
    sub_c = cv2.erode((fg_a > 0.95).astype(np.uint8), ell_kernel(20)) > 0
    bg_c = cv2.erode((fg_a < 0.02).astype(np.uint8), ell_kernel(20)) > 0
    fg_grain = float(hf[sub_c & (lap < np.percentile(lap[sub_c], 30))].mean()) if sub_c.sum() > 1000 else 1.0
    bg_grain = float(hf[bg_c & (lap < np.percentile(lap[bg_c], 30))].mean()) if bg_c.sum() > 1000 else 1.0
    ratio_before = fg_grain / max(1e-3, bg_grain)

    grain_layer = np.full((h, w, 3), 128.0, dtype=np.float32)
    if grain_match:
        rng = np.random.default_rng(2026)
        target_hf = max(fg_grain, bg_grain, 0.85)
        # Apply calibrated soft-light noise to whichever region is smoother + slight film floor globally
        sigma_fg = max(0.0, np.sqrt(max(0.0, target_hf ** 2 - fg_grain ** 2)) * 2.1)
        sigma_bg = max(0.0, np.sqrt(max(0.0, target_hf ** 2 - bg_grain ** 2)) * 2.1)
        n_map = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
        n_map = 0.65 * n_map + 0.35 * cv2.GaussianBlur(n_map, (0, 0), 0.7)
        fg_w = cv2.GaussianBlur(fg_a, (0, 0), sigmaX=3.0)
        amp = max(sigma_fg, 0.9) * fg_w + max(sigma_bg, 0.9) * (1.0 - fg_w)
        grain_layer = np.clip(128.0 + n_map * amp[:, :] * 1.4, 96.0, 160.0)
        grain_layer = np.repeat(grain_layer[:, :, None], 3, axis=2)

    p_harm = out_dir / "harmonized_fg_rgba.png"
    p_gain = out_dir / "pih_gain_map_multiply.png"
    p_wrap = out_dir / "pih_light_wrap_screen.png"
    p_grain = out_dir / "pih_grain_overlay.png"
    p_prev = out_dir / "harmonized_preview.png"

    write_rgba_u8(p_harm, fg_harm, fg_a, overwrite=overwrite)
    gain_rgb = np.repeat(np.clip(gain_eff * 255.0, 0.0, 255.0)[:, :, None], 3, axis=2)
    write_rgb_u8(p_gain, gain_rgb, overwrite=overwrite)
    write_rgb_u8(p_wrap, wrap_rgb, overwrite=overwrite)
    write_rgb_u8(p_grain, grain_layer, overwrite=overwrite)
    write_rgb_u8(p_prev, comp, overwrite=overwrite)

    meta = {
        "ok": True,
        "backend": backend_used,
        "provider": provider_used,
        "size": [w, h],
        "grain_fg_hf": round(fg_grain, 3),
        "grain_bg_hf": round(bg_grain, 3),
        "grain_ratio_before": round(ratio_before, 3),
        "outputs": {
            "harmonized_fg_rgba": str(p_harm),
            "pih_gain_map_multiply": str(p_gain),
            "pih_light_wrap_screen": str(p_wrap),
            "pih_grain_overlay": str(p_grain),
            "harmonized_preview": str(p_prev),
        },
    }
    (out_dir / "harmonize_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def _parse_grid(grid_str: str) -> tuple[int, int]:
    parts = grid_str.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid --grid '{grid_str}', expected RxC e.g. 2x2")
    rows, cols = int(parts[0]), int(parts[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"Invalid --grid '{grid_str}', rows and cols must be >= 1")
    return rows, cols


def _compute_tile_boxes(w: int, h: int, rows: int, cols: int, overlap: float) -> list[dict[str, Any]]:
    ov = float(np.clip(overlap, 0.05, 0.45))
    tw = int(np.ceil(w / max(0.5, cols - (cols - 1) * ov)))
    th = int(np.ceil(h / max(0.5, rows - (rows - 1) * ov)))
    tw = min(w, max(64, tw))
    th = min(h, max(64, th))
    xs = [0] if cols == 1 else [int(round(c * (w - tw) / (cols - 1))) for c in range(cols)]
    ys = [0] if rows == 1 else [int(round(r * (h - th) / (rows - 1))) for r in range(rows)]
    tiles = []
    for r, y0 in enumerate(ys):
        y1 = min(h, y0 + th)
        for c, x0 in enumerate(xs):
            x1 = min(w, x0 + tw)
            tiles.append({"row": r, "col": c, "name": f"tile_r{r}_c{c}", "box_xyxy": [x0, y0, x1, y1]})
    return tiles


def run_tile_split(
    *,
    input_path: Path,
    out_dir: Path,
    grid: str = "2x2",
    overlap: float = 0.22,
    max_tile_side: int = 1400,
    overwrite: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb, _ = read_rgba_f32(input_path)
    h, w = rgb.shape[:2]
    rows, cols = _parse_grid(grid)
    tiles = _compute_tile_boxes(w, h, rows, cols, overlap)

    saved_tiles = []
    for t in tiles:
        x0, y0, x1, y1 = t["box_xyxy"]
        crop = rgb[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]
        scale = min(1.0, float(max_tile_side) / float(max(ch, cw))) if max_tile_side > 0 else 1.0
        sw, sh = (int(round(cw * scale)), int(round(ch * scale))) if scale < 1.0 else (cw, ch)
        crop_save = cv2.resize(crop, (sw, sh), interpolation=cv2.INTER_AREA) if scale < 1.0 else crop
        tp = out_dir / f"{t['name']}.png"
        write_rgb_u8(tp, crop_save, overwrite=overwrite)
        saved_tiles.append({**t, "path": str(tp), "saved_size": [sw, sh]})

    manifest = {
        "ok": True,
        "input": str(input_path),
        "orig_size": [w, h],
        "grid": [rows, cols],
        "overlap": float(overlap),
        "max_tile_side": int(max_tile_side),
        "tiles": saved_tiles,
    }
    mp = out_dir / "tiles_manifest.json"
    mp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {**manifest, "manifest": str(mp)}


def _find_refined_tile(tiles_dir: Path, name: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        cand = tiles_dir / f"{name}{ext}"
        if cand.exists():
            return cand
    matches = sorted(tiles_dir.glob(f"{name}.*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Refined tile not found for {name} in {tiles_dir}")


def _raised_cosine_weight(cw: int, ch: int, r: int, c: int, rows: int, cols: int, ov_x: int, ov_y: int) -> np.ndarray:
    wx = np.ones(cw, dtype=np.float32)
    wy = np.ones(ch, dtype=np.float32)
    if c > 0 and ov_x > 2:
        n = min(cw // 2, ov_x)
        wx[:n] = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0.0, 1.0, n, dtype=np.float32))
    if c < cols - 1 and ov_x > 2:
        n = min(cw // 2, ov_x)
        wx[-n:] = 0.5 - 0.5 * np.cos(np.pi * np.linspace(1.0, 0.0, n, dtype=np.float32))
    if r > 0 and ov_y > 2:
        n = min(ch // 2, ov_y)
        wy[:n] = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0.0, 1.0, n, dtype=np.float32))
    if r < rows - 1 and ov_y > 2:
        n = min(ch // 2, ov_y)
        wy[-n:] = 0.5 - 0.5 * np.cos(np.pi * np.linspace(1.0, 0.0, n, dtype=np.float32))
    return np.outer(wy, wx)


def run_tile_stitch(
    *,
    manifest_path: Path,
    tiles_dir: Path,
    base_path: Path,
    out_path: Path,
    lock_lowfreq_sigma: float = 48.0,
    ecc_align: bool = True,
    detail_strength: float = 0.85,
    overwrite: bool = False,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_rgb, _ = read_rgba_f32(base_path)
    h, w = base_rgb.shape[:2]
    rows, cols = int(manifest["grid"][0]), int(manifest["grid"][1])
    overlap = float(manifest.get("overlap", 0.22))

    # Compute exact overlap widths between neighboring tiles
    boxes = [t["box_xyxy"] for t in manifest["tiles"]]
    ov_x = int(max([0] + [boxes[c][2] - boxes[c + 1][0] for c in range(max(0, cols - 1))]))
    ov_y = int(max([0] + [boxes[r * cols][3] - boxes[(r + 1) * cols][1] for r in range(max(0, rows - 1))]))

    accum = np.zeros((h, w, 3), dtype=np.float32)
    w_sum = np.zeros((h, w), dtype=np.float32)
    tile_reports: list[dict[str, Any]] = []
    ratios: list[float] = []

    for t in manifest["tiles"]:
        x0, y0, x1, y1 = t["box_xyxy"]
        cw, ch = x1 - x0, y1 - y0
        r, c = int(t["row"]), int(t["col"])
        tp = _find_refined_tile(tiles_dir, t["name"])
        tile_raw, _ = read_rgba_f32(tp)
        nat_h, nat_w = tile_raw.shape[:2]
        ratios.append(min(nat_w / max(1, cw), nat_h / max(1, ch)))

        tile_fit = cv2.resize(tile_raw, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
        base_crop = base_rgb[y0:y1, x0:x1]
        corr = 1.0
        tile_w_scale = 1.0

        if ecc_align:
            scale_d = min(1.0, 480.0 / float(max(cw, ch)))
            sw_d, sh_d = max(32, int(round(cw * scale_d))), max(32, int(round(ch * scale_d)))
            bg_g = cv2.cvtColor(cv2.resize(base_crop, (sw_d, sh_d)).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            tl_g = cv2.cvtColor(cv2.resize(tile_fit, (sw_d, sh_d)).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            M = np.eye(2, 3, dtype=np.float32)
            try:
                corr, M = cv2.findTransformECC(
                    bg_g, tl_g, M, cv2.MOTION_AFFINE,
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-4),
                )
                M_full = M.copy()
                M_full[0, 2] *= float(cw) / float(sw_d)
                M_full[1, 2] *= float(ch) / float(sh_d)
                inv_M = cv2.invertAffineTransform(M_full)
                # Guard against crazy warps by keeping shift/scale bounded
                if abs(inv_M[0, 2]) < cw * 0.08 and abs(inv_M[1, 2]) < ch * 0.08:
                    tile_fit = cv2.warpAffine(
                        tile_fit, inv_M, (cw, ch), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101
                    )
            except Exception:
                # Fallback correlation score
                corr = float(np.corrcoef(bg_g.ravel(), tl_g.ravel())[0, 1])

            if corr < 0.70:
                tile_w_scale = 0.25

        w_tile = _raised_cosine_weight(cw, ch, r, c, rows, cols, ov_x, ov_y) * tile_w_scale
        # Blend safely back toward base_crop if ECC warned of drift
        eff_tile = tile_fit if tile_w_scale == 1.0 else (base_crop * 0.65 + tile_fit * 0.35)
        accum[y0:y1, x0:x1] += eff_tile * w_tile[:, :, None]
        w_sum[y0:y1, x0:x1] += w_tile
        tile_reports.append({
            "name": t["name"],
            "box_xyxy": [x0, y0, x1, y1],
            "refined_native_size": [nat_w, nat_h],
            "ecc_corr": round(float(corr), 4),
            "low_corr_warning": bool(corr < 0.70),
        })

    stitched = np.where(w_sum[:, :, None] > 1e-5, accum / np.maximum(w_sum[:, :, None], 1e-5), base_rgb)
    sig = float(lock_lowfreq_sigma)
    if sig > 0.0:
        base_low = cv2.GaussianBlur(base_rgb, (0, 0), sigmaX=sig)
        stitch_low = cv2.GaussianBlur(stitched, (0, 0), sigmaX=sig)
        detail_hf = stitched - stitch_low
        out_rgb = np.clip(base_low + detail_hf * float(detail_strength) + (stitch_low - base_low) * 0.15, 0.0, 255.0)
    else:
        out_rgb = np.clip(stitched, 0.0, 255.0)

    write_rgb_u8(out_path, out_rgb, overwrite=overwrite)
    eff_lin = float(min(ratios)) if ratios else 1.0
    eff_nat = [int(round(w * eff_lin)), int(round(h * eff_lin))]
    return {
        "ok": True,
        "out": str(out_path),
        "out_size": [w, h],
        "effective_native_size": eff_nat,
        "effective_linear_ratio": round(eff_lin, 3),
        "overlap": overlap,
        "tiles": tile_reports,
    }


def _rgb_to_hex(rgb: np.ndarray) -> str:
    r, g, b = [int(np.clip(round(float(v)), 0, 255)) for v in rgb[:3]]
    return f"#{r:02x}{g:02x}{b:02x}"


def ensure_hair_brush_tips(brushes_dir: Path | None = None, force: bool = False) -> dict[str, Path]:
    """Ensure the 3 reusable grayscale PhotoCraft hair brush tip PNGs exist (black=ink, white=transparent)."""
    bdir = brushes_dir or (Path(__file__).resolve().parent.parent / "resources" / "brushes")
    bdir.mkdir(parents=True, exist_ok=True)
    specs = {
        "hair_01_multi_clump": [
            (128, 34, 4.0, 4.0, 0.72),
            (131, 72, 4.4, 4.4, 0.92),
            (125, 110, 4.2, 4.2, 0.85),
            (130, 148, 4.5, 4.5, 1.00),
            (126, 186, 4.0, 4.0, 0.84),
            (130, 224, 3.6, 3.6, 0.68),
        ],
        "hair_02_fine_clump": [
            (128, 86, 3.8, 3.8, 0.88),
            (131, 128, 4.4, 4.4, 1.00),
            (126, 170, 3.6, 3.6, 0.82),
        ],
        "hair_03_single_strand": [
            (128, 128, 9.0, 9.0, 1.00),
        ],
    }
    yy, xx = np.mgrid[0:256, 0:256].astype(np.float32)
    out: dict[str, Path] = {}
    for name, dabs in specs.items():
        p = bdir / f"{name}.png"
        if force or (not p.exists()):
            canvas = np.zeros((256, 256), dtype=np.float32)
            for cx, cy, rx, ry, wt in dabs:
                d2 = ((xx - cx) / max(1.0, rx)) ** 2 + ((yy - cy) / max(1.0, ry)) ** 2
                g = np.exp(-2.4 * d2) * np.clip(1.25 - 0.55 * np.sqrt(d2), 0.0, 1.0)
                canvas = np.maximum(canvas, wt * g)
            arr = np.clip(np.round(255.0 * (1.0 - np.clip(canvas, 0.0, 1.0))), 0, 255).astype(np.uint8)
            write_png_u8(p, arr, overwrite=True)
        out[name] = p
    return out


def _sample_cubic_bezier_stroke(
    p0: tuple[float, float],
    wind: tuple[float, float],
    length: float,
    bend1: float,
    bend2: float,
    n_pts: int = 12,
    p_min: float = 0.04,
    p_peak: float = 0.96,
    p_pow: float = 0.78,
) -> list[list[float]]:
    wx, wy = wind
    nx, ny = -wy, wx
    x0, y0 = p0
    x1, y1 = x0 + 0.34 * length * wx + bend1 * nx, y0 + 0.34 * length * wy + bend1 * ny
    x2, y2 = x0 + 0.72 * length * wx + bend2 * nx, y0 + 0.72 * length * wy + bend2 * ny
    x3, y3 = x0 + 1.00 * length * wx + 0.45 * bend2 * nx, y0 + 1.00 * length * wy + 0.45 * bend2 * ny
    pts: list[list[float]] = []
    for i in range(n_pts):
        t = float(i) / float(max(1, n_pts - 1))
        mt = 1.0 - t
        x = mt**3 * x0 + 3 * mt**2 * t * x1 + 3 * mt * t**2 * x2 + t**3 * x3
        y = mt**3 * y0 + 3 * mt**2 * t * y1 + 3 * mt * t**2 * y2 + t**3 * y3
        pr = p_min + (p_peak - p_min) * (math.sin(math.pi * t) ** p_pow)
        pts.append([round(x, 1), round(y, 1), round(pr, 3)])
    return pts


def _render_pc_stroke_layer(
    *,
    pc_exe: str,
    width: int,
    height: int,
    brush_tips: dict[str, Path],
    strokes: list[dict[str, Any]],
    out_png: Path,
    tmp_pcraft: Path,
) -> None:
    """Render strokes via PhotoCraft native paint.stroke in safe CLI-length chunks."""
    if not strokes:
        write_rgba_u8(out_png, np.zeros((height, width, 3), np.float32), np.zeros((height, width), np.float32), overwrite=True)
        return

    def _J(obj: object) -> str:
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)

    chunk_size = 14
    chunks = [strokes[i : i + chunk_size] for i in range(0, len(strokes), chunk_size)]
    for ci, ch in enumerate(chunks):
        if ci == 0:
            base_args = [
                pc_exe,
                "run",
                "--new",
                _J({"width": width, "height": height, "mode": "rgb", "depth": 8, "background": "transparent"}),
            ]
        else:
            base_args = [pc_exe, "run", str(tmp_pcraft)]

        cmds: list[str] = []
        used_tips = sorted({str(s["tip"]) for s in ch})
        for tip_name in used_tips:
            cmds += [
                "--cmd",
                "brush.presets.importImage",
                "--params",
                _J({"path": str(brush_tips[tip_name]), "name": tip_name, "select": False}),
            ]

        active_tip: str | None = None
        for s in ch:
            tname = str(s["tip"])
            if tname != active_tip:
                cmds += ["--cmd", "brush.presets.setCurrent", "--params", _J({"name": tname})]
                cmds += [
                    "--cmd",
                    "tools.setBrush",
                    "--params",
                    _J({
                        "size": float(s["size"]),
                        "spacing": float(s.get("spacing", 0.035)),
                        "shapeDynamics": {
                            "enabled": True,
                            "size": {"control": "penPressure", "minimum": float(s.get("size_min", 0.06))},
                            "angle": {"control": "direction"},
                        },
                        "transfer": {
                            "enabled": True,
                            "opacity": {"control": "penPressure", "minimum": 0.0},
                        },
                    }),
                ]
                active_tip = tname
            cmds += [
                "--cmd",
                "paint.stroke",
                "--params",
                _J({
                    "points": s["points"],
                    "size": float(s["size"]),
                    "color": str(s["color"]),
                    "opacity": float(s["opacity"]),
                    "flow": float(s.get("flow", 0.90)),
                    "smoothing": 0.25,
                }),
            ]

        is_last = ci == len(chunks) - 1
        if is_last:
            out_args = ["--out", str(out_png)]
        else:
            out_args = ["--out", str(tmp_pcraft)] + (["--in-place"] if ci > 0 else [])

        r = subprocess.run(base_args + cmds + out_args, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            raise RuntimeError(f"PhotoCraft hair-paint stroke batch {ci} failed (rc={r.returncode}): {r.stderr[-600:] or r.stdout[-600:]}")
    if tmp_pcraft.exists():
        tmp_pcraft.unlink(missing_ok=True)


def run_hair_paint(
    *,
    subject_rgba_path: Path,
    hair_mask_path: Path,
    out_dir: Path,
    occluder_mask_path: Path | None = None,
    face_mask_path: Path | None = None,
    eyes_brows_mask_path: Path | None = None,
    clothes_mask_path: Path | None = None,
    wind_dir: str = "auto",
    intensity: str = "medium",
    photocraft_cli: str | None = None,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Native PhotoCraft behind+front layered COS hair painting.

    Never deletes original real hair; blends local root hair colors with global dark/mid/bright/glow
    tones; renders clustered silky single-strand Bezier wisps following natural gravity+breeze arcs
    across a behind-subject flow layer (hair_paint_01_soft_flow_behind.png) and two front layers
    (hair_paint_02_airy_wisps_front.png, hair_paint_03_add_backlit_flow.png) with dilated face/eye
    protection and 0px flush occluder carve-out.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pc_exe = photocraft_cli or str(
        Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        / "Programs" / "ArtCraft" / "PhotoCraft" / "photocraft-cli.exe"
    )
    if not Path(pc_exe).exists():
        raise FileNotFoundError(f"PhotoCraft CLI not found at: {pc_exe}")

    subj_rgb, subj_alpha = read_rgba_f32(subject_rgba_path)
    h, w = subj_alpha.shape
    hair_m = read_gray_f32(hair_mask_path, (h, w))
    occ_m = read_gray_f32(occluder_mask_path, (h, w)) if occluder_mask_path else np.zeros((h, w), dtype=np.float32)
    face_m = read_gray_f32(face_mask_path, (h, w)) if face_mask_path else np.zeros((h, w), dtype=np.float32)
    if eyes_brows_mask_path is None:
        auto_eyes = hair_mask_path.parent / "mask_eyes_brows.png"
        eyes_brows_mask_path = auto_eyes if auto_eyes.exists() else None
    eyes_m = read_gray_f32(eyes_brows_mask_path, (h, w)) if eyes_brows_mask_path else np.zeros((h, w), dtype=np.float32)
    clothes_m = read_gray_f32(clothes_mask_path, (h, w)) if clothes_mask_path else np.zeros((h, w), dtype=np.float32)

    guide01 = np.clip(luma255(subj_rgb) / 255.0, 0.0, 1.0)
    occ_snap = guided_filter_gray(guide01, occ_m, radius=8, eps=1e-3) if occ_m.max() > 0.05 else occ_m
    clo_snap = guided_filter_gray(guide01, clothes_m, radius=8, eps=1e-3) if clothes_m.max() > 0.05 else clothes_m
    face_snap = guided_filter_gray(guide01, face_m, radius=8, eps=1e-3) if face_m.max() > 0.05 else face_m
    eyes_guard = (
        cv2.GaussianBlur(cv2.dilate((eyes_m >= 0.25).astype(np.uint8), ell_kernel(28)).astype(np.float32), (0, 0), sigmaX=4.0)
        if eyes_m.max() > 0.05
        else np.zeros((h, w), dtype=np.float32)
    )
    face_guard = (
        cv2.GaussianBlur(
            cv2.dilate((face_snap >= 0.35).astype(np.uint8), ell_kernel(8)).astype(np.float32),
            (0, 0),
            sigmaX=2.0,
        )
        if face_m.max() > 0.05
        else np.zeros((h, w), dtype=np.float32)
    )

    # 1. Eyedropper-only color sampling strictly from real hair pixels
    lab = rgb255_to_lab(subj_rgb)
    L_lab = lab[:, :, 0]
    C_lab = np.hypot(lab[:, :, 1], lab[:, :, 2])
    hair_sample = (hair_m > 0.65) & (subj_alpha > 0.80) & (L_lab >= 18.0) & (L_lab <= 82.0) & (C_lab >= 10.0)
    if int(hair_sample.sum()) < 300:
        hair_sample = (hair_m > 0.45) & (subj_alpha > 0.55)
    if int(hair_sample.sum()) < 50:
        raise ValueError("Insufficient hair pixels found in --hair-mask for eyedropper sampling.")

    L_vals = L_lab[hair_sample]
    rgb_vals = subj_rgb[hair_sample]

    def _band_rgb(p_lo: float, p_hi: float) -> np.ndarray:
        lo_v, hi_v = float(np.percentile(L_vals, p_lo)), float(np.percentile(L_vals, p_hi))
        sel = (L_vals >= lo_v) & (L_vals <= max(lo_v + 0.5, hi_v))
        return np.median(rgb_vals[sel], axis=0) if sel.any() else np.median(rgb_vals, axis=0)

    dark_rgb = _band_rgb(14.0, 26.0)
    mid_rgb = _band_rgb(42.0, 56.0)
    bright_rgb = _band_rgb(72.0, 84.0)
    glow_raw = _band_rgb(84.0, 94.0)
    glow_rgb = np.clip(glow_raw * 0.72 + np.array([28.0, 24.0, 18.0], dtype=np.float32), 0.0, 235.0)
    tone_rgb_map = {"dark": dark_rgb, "mid": mid_rgb, "bright": bright_rgb, "glow": glow_rgb}
    palette = {k: _rgb_to_hex(v) for k, v in tone_rgb_map.items()}

    def _local_hair_hex(ax: float, ay: float, tone: str, local_mix: float = 0.65) -> str:
        ix, iy = int(round(ax)), int(round(ay))
        y0, y1 = max(0, iy - 7), min(h, iy + 8)
        x0, x1 = max(0, ix - 7), min(w, ix + 8)
        sub_m = (hair_m[y0:y1, x0:x1] > 0.45) & (subj_alpha[y0:y1, x0:x1] > 0.65)
        base_tone = tone_rgb_map[tone]
        if sub_m.sum() >= 6:
            loc = np.median(subj_rgb[y0:y1, x0:x1][sub_m], axis=0)
            blended = local_mix * loc + (1.0 - local_mix) * base_tone
            return _rgb_to_hex(blended)
        return palette[tone]

    # 2. Dynamic outer-silhouette anchor discovery following natural hair curvature (zero flat ribbons or cheek cross-hatching)
    hair_bin = ((hair_m >= 0.55) & (subj_alpha >= 0.65)).astype(np.uint8)
    ys_h, xs_h = np.where(hair_bin > 0)
    hx0, hx1 = int(xs_h.min()), int(xs_h.max())
    hy0, hy1 = int(ys_h.min()), int(ys_h.max())
    hair_w = max(1, hx1 - hx0)
    hair_h = max(1, hy1 - hy0)
    hair_span = float(max(hair_w, hair_h))
    hcx = float(np.median(xs_h))

    if wind_dir == "auto":
        is_left = int((xs_h < hcx).sum()) >= int((xs_h >= hcx).sum())
    elif wind_dir in ("left", "down-left"):
        is_left = True
    else:
        is_left = False
    wx_sign = -1.0 if is_left else 1.0

    def _flow_vec(rel_y: float, airy: bool = False) -> tuple[float, float]:
        # Follow natural gravity+breeze curvature: contour-hugging silky flow vs wind-lifted outer wisps
        t_y = float(np.clip((rel_y - 0.55) / 0.40, 0.0, 1.0))
        if airy:
            vx = wx_sign * (0.60 + 0.16 * t_y)
            vy = 0.80 - 0.15 * t_y
        else:
            vx = wx_sign * (0.42 + 0.18 * t_y)
            vy = 0.91 - 0.11 * t_y
        vn = math.hypot(vx, vy) or 1.0
        return vx / vn, vy / vn

    avoid_occ = (np.maximum(occ_snap, clo_snap) > 0.30) | (face_snap > 0.30)
    d_avoid = cv2.distanceTransform((~avoid_occ).astype(np.uint8), cv2.DIST_L2, 5) if avoid_occ.any() else np.full((h, w), 999.0, np.float32)
    white_bow = (subj_alpha > 0.82) & (L_lab > 76.0) & (C_lab < 12.0) & (hair_m < 0.50)
    d_bow = cv2.distanceTransform((~white_bow).astype(np.uint8), cv2.DIST_L2, 5) if white_bow.any() else np.full((h, w), 999.0, np.float32)

    rng = np.random.default_rng(seed)

    outer_wind_anchors: list[tuple[float, float, float]] = []
    lower_tip_anchors: list[tuple[float, float]] = []
    crown_anchors: list[tuple[float, float]] = []

    for y_row in range(hy0, hy1 + 1, 6):
        cols = np.where(hair_bin[y_row, :] > 0)[0]
        if cols.size < 18:
            continue
        xL = int(cols[0])
        xR = int(cols[-1])
        rel_y = (y_row - hy0) / max(1.0, float(hair_h))

        x_wind = xL if is_left else xR
        in_sign = 1.0 if is_left else -1.0
        for push in (9.0, 16.0):
            ax = float(np.clip(x_wind + in_sign * push, 0, w - 1))
            ay = float(y_row)
            if hair_m[int(ay), int(ax)] >= 0.45 and d_bow[int(ay), int(ax)] >= 20.0:
                if d_avoid[int(ay), int(ax)] >= 12.0 and 0.14 <= rel_y <= 0.90:
                    outer_wind_anchors.append((ax, ay, rel_y))
                    break
        ax_cr = int(np.clip(x_wind + in_sign * 10.0, 0, w - 1))
        if 0.04 <= rel_y <= 0.18 and hair_m[y_row, ax_cr] >= 0.45 and d_avoid[y_row, ax_cr] >= 14.0 and d_bow[y_row, ax_cr] >= 16.0:
            crown_anchors.append((float(ax_cr), float(y_row)))

    for x_col in range(hx0, hx1 + 1, 10):
        rows_c = np.where((hair_bin[:, x_col] > 0) & (np.arange(h) >= hy0 + 0.62 * hair_h))[0]
        if rows_c.size < 14:
            continue
        yB = int(rows_c[-1])
        ax_b = float(x_col)
        ay_b = float(max(0, yB - int(rng.integers(18, 34))))
        if hair_m[int(ay_b), int(ax_b)] >= 0.45 and d_avoid[int(ay_b), int(ax_b)] >= 14.0 and d_bow[int(ay_b), int(ax_b)] >= 24.0:
            lower_tip_anchors.append((ax_b, ay_b))

    if not outer_wind_anchors:
        outer_wind_anchors = [(ax, ay, 0.85) for ax, ay in lower_tip_anchors]

    # Whisper-thin outer temple wisps strictly outside dilated face_guard & eyes_guard (zero cheek/chest crossing)
    bangs_anchors: list[tuple[float, float, float]] = []
    face_protect_center = np.zeros((h, w), dtype=np.float32)
    if (face_snap > 0.5).sum() >= 500:
        fys, fxs = np.where(face_snap > 0.5)
        fx0, fx1 = int(fxs.min()), int(fxs.max())
        fy0, fy1 = int(fys.min()), int(fys.max())
        fcx = 0.5 * (fx0 + fx1)
        fw, fh = max(1, fx1 - fx0), max(1, fy1 - fy0)
        yy_f, xx_f = np.mgrid[0:h, 0:w].astype(np.float32)
        ell = ((xx_f - fcx) / (0.48 * fw)) ** 2 + ((yy_f - (fy0 + 0.52 * fh)) / (0.52 * fh)) ** 2
        face_protect_center = np.clip((1.15 - ell) / 0.25, 0.0, 1.0)
        band = (
            (hair_m > 0.55)
            & (np.abs(xx_f - fcx) >= 0.50 * fw)
            & (yy_f >= fy0 + 0.18 * fh)
            & (yy_f <= fy1 + 0.02 * fh)
            & (xx_f >= fx0 - 0.26 * fw)
            & (xx_f <= fx1 + 0.26 * fw)
        )
        b_cand = band & (face_guard < 0.05) & (face_protect_center < 0.05) & (eyes_guard < 0.05)
        bys, bxs = np.where(b_cand)
        if len(bxs) > 0:
            step_b = max(1, len(bxs) // 30)
            for bx, by in zip(bxs[::step_b], bys[::step_b]):
                side_dir = -1.0 if bx < fcx else 1.0
                bangs_anchors.append((float(bx), float(by), side_dir))

    scale_mult = {"subtle": 0.72, "medium": 1.0, "dramatic": 1.35}.get(intensity, 1.0)
    op_mult = {"subtle": 0.72, "medium": 0.88, "dramatic": 1.0}.get(intensity, 0.88)

    def _pick(pool: list[Any], count: int) -> list[Any]:
        if not pool or count <= 0:
            return []
        idx = np.linspace(0, len(pool) - 1, min(count, len(pool))).round().astype(int)
        picks = [pool[int(i)] for i in idx]
        while len(picks) < count:
            picks.append(pool[int(rng.integers(0, len(pool)))])
        return picks

    def _emit_bundle(
        target: list[dict[str, Any]],
        ax: float,
        ay: float,
        rel_y: float,
        n_strands: int,
        tone: str,
        size_rng: tuple[float, float],
        len_rng: tuple[float, float],
        op_rng: tuple[float, float],
        airy: bool = False,
    ) -> None:
        vx0, vy0 = _flow_vec(rel_y, airy=airy)
        col_hex = _local_hair_hex(ax, ay, tone, local_mix=0.65)
        base_len = float(rng.uniform(*len_rng) * hair_span)
        s_sign = 1.0 if wx_sign < 0 else -1.0
        if airy:
            base_bend1 = s_sign * float(rng.uniform(0.14, 0.24) * base_len)
            base_bend2 = -s_sign * float(rng.uniform(0.08, 0.18) * base_len)
        else:
            base_bend1 = float(rng.uniform(-0.14, 0.16) * base_len)
            base_bend2 = float(rng.uniform(-0.16, 0.18) * base_len)
        for _ in range(n_strands):
            jit = float(rng.uniform(-0.14, 0.14))
            c, sn = math.cos(jit), math.sin(jit)
            vwx, vwy = vx0 * c - vy0 * sn, vx0 * sn + vy0 * c
            sx = ax + float(rng.uniform(-6.0, 6.0))
            sy = ay + float(rng.uniform(-5.0, 5.0))
            slen = base_len * float(rng.uniform(0.82, 1.16))
            b1 = base_bend1 + float(rng.uniform(-0.06, 0.06) * slen)
            b2 = base_bend2 + float(rng.uniform(-0.08, 0.08) * slen)
            target.append({
                "tip": "hair_03_single_strand",
                "size": round(float(rng.uniform(*size_rng)), 1),
                "size_min": 0.02,
                "spacing": 0.035,
                "color": col_hex,
                "opacity": round(float(rng.uniform(*op_rng) * op_mult), 3),
                "flow": 0.88,
                "points": _sample_cubic_bezier_stroke((sx, sy), (vwx, vwy), slen, b1, b2, n_pts=12, p_pow=0.90),
            })

    # Layer 1 (BEHIND subject): silky tapered bundles (~55% contour-hugging, ~45% breeze-lifted airy arcs)
    s_l1: list[dict[str, Any]] = []
    for i, (ax, ay, ry) in enumerate(_pick(outer_wind_anchors, int(round(20 * scale_mult)))):
        is_airy = (i % 2 == 1)
        _emit_bundle(s_l1, ax, ay, ry, 4, "mid" if (i % 2 == 0) else "dark", (2.6, 4.5), (0.14, 0.25) if is_airy else (0.10, 0.20), (0.40, 0.66), airy=is_airy)
    for i, (ax, ay) in enumerate(_pick(lower_tip_anchors, int(round(7 * scale_mult)))):
        is_airy = (i % 2 == 1)
        _emit_bundle(s_l1, ax, ay, 0.90, 3, "mid", (2.4, 4.2), (0.12, 0.20) if is_airy else (0.09, 0.16), (0.38, 0.62), airy=is_airy)

    # Layer 2 (FRONT Normal): fine airy secondary silky wisps & lower tips + whisper-thin temple wisps
    s_l2: list[dict[str, Any]] = []
    for i, (ax, ay, ry) in enumerate(_pick(outer_wind_anchors, int(round(22 * scale_mult)))):
        is_airy = (i % 2 == 1)
        _emit_bundle(s_l2, ax, ay, ry, 4, "bright" if (i % 2 == 0) else "mid", (2.2, 3.8), (0.14, 0.26) if is_airy else (0.10, 0.20), (0.46, 0.76), airy=is_airy)
    for i, (ax, ay) in enumerate(_pick(lower_tip_anchors, int(round(8 * scale_mult)))):
        is_airy = (i % 2 == 1)
        _emit_bundle(s_l2, ax, ay, 0.92, 3, "bright" if (i % 2 == 0) else "mid", (2.0, 3.5), (0.12, 0.21) if is_airy else (0.09, 0.16), (0.46, 0.72), airy=is_airy)
    for ax, ay, side_dir in _pick(bangs_anchors, 5):
        vwx, vwy = side_dir * float(rng.uniform(0.18, 0.34)), float(rng.uniform(0.92, 0.98))
        vn = math.hypot(vwx, vwy) or 1.0
        slen = float(rng.uniform(55.0, 105.0))
        s_l2.append({
            "tip": "hair_03_single_strand",
            "size": round(float(rng.uniform(1.8, 2.8)), 1),
            "size_min": 0.02,
            "spacing": 0.035,
            "color": _local_hair_hex(ax, ay, "bright", local_mix=0.75),
            "opacity": round(float(rng.uniform(0.36, 0.56) * op_mult), 3),
            "flow": 0.84,
            "points": _sample_cubic_bezier_stroke((ax, ay), (vwx / vn, vwy / vn), slen, side_dir * 0.12 * slen, -side_dir * 0.06 * slen, n_pts=10),
        })

    # Layer 3 (FRONT Linear Dodge Add): luminous breeze-lifted backlit golden flyaways & crown wisps ("毛流感")
    s_l3: list[dict[str, Any]] = []
    for i, (ax, ay, ry) in enumerate(_pick(outer_wind_anchors, int(round(22 * scale_mult)))):
        _emit_bundle(s_l3, ax, ay, ry, 3, "glow" if (i % 3 != 0) else "bright", (1.8, 3.3), (0.14, 0.26), (0.46, 0.78), airy=True)
    for i, (ax, ay) in enumerate(_pick(lower_tip_anchors, int(round(9 * scale_mult)))):
        _emit_bundle(s_l3, ax, ay, 0.94, 2, "glow", (1.8, 3.2), (0.12, 0.22), (0.44, 0.72), airy=True)
    for ax, ay in _pick(crown_anchors, 10):
        ang_jit = float(rng.uniform(-0.18, 0.18))
        c, sn = math.cos(ang_jit), math.sin(ang_jit)
        vx0, vy0 = wx_sign * 0.62, 0.78
        vwx, vwy = vx0 * c - vy0 * sn, vx0 * sn + vy0 * c
        vn = math.hypot(vwx, vwy) or 1.0
        slen = float(rng.uniform(75.0, 150.0))
        s_sign = 1.0 if wx_sign < 0 else -1.0
        s_l3.append({
            "tip": "hair_03_single_strand",
            "size": round(float(rng.uniform(1.8, 2.8)), 1),
            "size_min": 0.02,
            "spacing": 0.035,
            "color": _local_hair_hex(ax, ay, "glow", local_mix=0.60),
            "opacity": round(float(rng.uniform(0.42, 0.68) * op_mult), 3),
            "flow": 0.86,
            "points": _sample_cubic_bezier_stroke((ax, ay), (vwx / vn, vwy / vn), slen, s_sign * 0.16 * slen, -s_sign * 0.09 * slen, n_pts=10, p_pow=0.92),
        })

    # 3. Render via native PhotoCraft brush engine
    brush_tips = ensure_hair_brush_tips(force=True)
    p_l1 = out_dir / "hair_paint_01_soft_flow_behind.png"
    p_l2 = out_dir / "hair_paint_02_airy_wisps_front.png"
    p_l3 = out_dir / "hair_paint_03_add_backlit_flow.png"
    tmp_pc = out_dir / "_hair_paint_tmp.pcraft"

    _render_pc_stroke_layer(pc_exe=pc_exe, width=w, height=h, brush_tips=brush_tips, strokes=s_l1, out_png=p_l1, tmp_pcraft=tmp_pc)
    _render_pc_stroke_layer(pc_exe=pc_exe, width=w, height=h, brush_tips=brush_tips, strokes=s_l2, out_png=p_l2, tmp_pcraft=tmp_pc)
    _render_pc_stroke_layer(pc_exe=pc_exe, width=w, height=h, brush_tips=brush_tips, strokes=s_l3, out_png=p_l3, tmp_pcraft=tmp_pc)

    # 4. Crisp 85mm optical softening + 0px-dilation flush occluder & dilated face carve-out
    dark_strap_block = cv2.GaussianBlur(((subj_alpha > 0.40) & (L_lab < 45.0) & (C_lab < 12.0)).astype(np.float32), (0, 0), sigmaX=0.8)
    bow_block = cv2.GaussianBlur(white_bow.astype(np.float32), (0, 0), sigmaX=1.0)
    occ_behind = np.clip(np.maximum.reduce([eyes_guard, face_guard, bow_block]), 0.0, 1.0)
    occ_tight_full = np.clip(np.maximum.reduce([
        cv2.GaussianBlur((np.maximum(occ_snap, clo_snap) >= 0.45).astype(np.float32), (0, 0), sigmaX=0.8),
        eyes_guard,
        face_guard,
        dark_strap_block,
        bow_block,
    ]), 0.0, 1.0)
    occ_tight_bangs = np.clip(np.maximum.reduce([
        cv2.GaussianBlur((np.maximum(np.where(face_snap > 0.45, 0.0, occ_snap), clo_snap) >= 0.45).astype(np.float32), (0, 0), sigmaX=0.8),
        face_protect_center,
        face_guard,
        eyes_guard,
        dark_strap_block,
        bow_block,
    ]), 0.0, 1.0)

    def _finish_layer(path: Path, occ_mask: np.ndarray, blur_sigma: float) -> tuple[int, int, np.ndarray]:
        rgb_l, a_l = read_rgba_f32(path, (h, w))
        if blur_sigma > 0.0:
            pm = rgb_l * a_l[:, :, None]
            pm_b = cv2.GaussianBlur(pm, (0, 0), sigmaX=blur_sigma)
            a_b = cv2.GaussianBlur(a_l, (0, 0), sigmaX=blur_sigma)
            rgb_l = np.where(a_b[:, :, None] > 1e-4, pm_b / np.maximum(a_b[:, :, None], 1e-4), rgb_l)
            a_l = a_b
        a_l = np.clip(a_l * (1.0 - occ_mask), 0.0, 1.0)
        write_rgba_u8(path, np.clip(rgb_l, 0.0, 255.0), a_l, overwrite=True)
        tot = int((a_l > 0.05).sum())
        ext = int(((a_l > 0.05) & (subj_alpha < 0.50)).sum())
        return tot, ext, a_l

    px_l1, ext_l1, a_l1 = _finish_layer(p_l1, occ_behind, 0.42)
    px_l2, ext_l2, a_l2 = _finish_layer(p_l2, occ_tight_bangs, 0.38)
    px_l3, ext_l3, a_l3 = _finish_layer(p_l3, np.maximum(occ_tight_full, face_protect_center), 0.35)

    hp_any = np.maximum.reduce([a_l1, a_l2, a_l3]) > 0.05
    safety_eyes_px = int((hp_any & (eyes_m >= 0.25)).sum()) if eyes_m.max() > 0.05 else 0
    safety_face_px = int((hp_any & (face_m >= 0.50)).sum()) if face_m.max() > 0.05 else 0

    meta = {
        "ok": True,
        "engine": "photocraft_native_brush",
        "size": [w, h],
        "wind_dir": wind_dir,
        "wind_vec": [round(v, 3) for v in _flow_vec(0.65)],
        "intensity": intensity,
        "palette": palette,
        "brush_tips": {k: str(v) for k, v in brush_tips.items()},
        "anchor_counts": {
            "outer_wind": len(outer_wind_anchors),
            "lower_tips": len(lower_tip_anchors),
            "crown": len(crown_anchors),
            "bangs": len(bangs_anchors),
        },
        "stroke_counts": {
            "layer1_soft_flow_behind": len(s_l1),
            "layer2_airy_wisps_front": len(s_l2),
            "layer3_add_backlit_flow": len(s_l3),
        },
        "painted_px": {
            "layer1_soft_flow_behind": px_l1,
            "layer2_airy_wisps_front": px_l2,
            "layer3_add_backlit_flow": px_l3,
        },
        "external_air_px": {
            "layer1_soft_flow_behind": ext_l1,
            "layer2_airy_wisps_front": ext_l2,
            "layer3_add_backlit_flow": ext_l3,
        },
        "safety": {
            "painted_in_eyes_px": safety_eyes_px,
            "painted_in_face_px": safety_face_px,
        },
        "outputs": {
            "layer1_soft_flow_behind": str(p_l1),
            "layer2_airy_wisps_front": str(p_l2),
            "layer3_add_backlit_flow": str(p_l3),
        },
    }
    (out_dir / "hair_paint_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta
