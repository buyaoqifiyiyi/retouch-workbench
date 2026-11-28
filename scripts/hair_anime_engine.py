"""Whole-Head Anime Hair Transformation Engine v6 (Real-Wig UV Texture Transfer + Fiber-Split).

Solves both v5 1:1 realism & anatomical continuity gaps:
  1. Real-Wig Photographed Texture Transfer (`extract_real_hair_detail_swatch`): extracts each
     character's sharpest tangent-aligned real 1:1 DSLR hair detail_ratio + chroma_delta tile
     and maps it in physical pixel pitch (zero plastic/fondant smoothness, zero moiré).
  2. Fiber-Split Silhouette (`_emit_fiber_split_clump`): pairs every hero ribbon with fiber-broken
     outer edges, 3-5 clinging flanking micro-filaments, multi-point tip split, and 0.6-1.0px
     front bridging hairs (`mid_connect`) crossing the root seam.
  3. Tangential Peel-Off & Anatomical Continuity:
     - Rose: <=18 deg tangential peel along downward golden cascade into black skirt + sunlit ivory tips.
     - Miku: right twintail routes cleanly through the open neck-to-forearm gap (x=2871..2964), tucks
       behind wrist bouquet (zero bouquet poke!), and billows below elbow in sunset-lit S-waves.
     - Nazuna: eliminates forehead bangs highlight watermark (`dome_only` silk_band), cleans 152px
       blue jaw fringe, slims midline ahoge (5.4px + silver detail), and doubles moonlight rim &
       nocturnal indigo `藏色` contrast.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from workbench_ops import (
    decontaminate_fringe,
    ell_kernel,
    luma255,
    rgb255_to_lab,
)


def _smoothstep(edge0: np.ndarray | float, edge1: np.ndarray | float, x: np.ndarray | float) -> np.ndarray | float:
    diff = edge1 - edge0
    if isinstance(diff, np.ndarray):
        sgn = np.where(diff >= 0.0, 1.0, -1.0)
        denom = sgn * np.maximum(np.abs(diff), 1e-6)
    else:
        denom = max(1e-6, float(diff)) if diff >= 0.0 else min(-1e-6, float(diff))
    t = np.clip((x - edge0) / denom, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _catmull_rom_2d(ctrl_xy: np.ndarray, n_samples: int = 88) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate open 2D Catmull-Rom spline with uniform arc-length resampling."""
    k = len(ctrl_xy)
    if k < 2:
        pts = np.repeat(ctrl_xy[:1], n_samples, axis=0).astype(np.float64)
        tang = np.tile(np.array([[0.0, 1.0]]), (n_samples, 1))
        norm = np.tile(np.array([[-1.0, 0.0]]), (n_samples, 1))
        return pts, tang, norm
    if k == 2:
        ts = np.linspace(0.0, 1.0, n_samples)[:, None]
        pts = ctrl_xy[0:1] * (1.0 - ts) + ctrl_xy[1:2] * ts
    else:
        n_dense = max(140, n_samples * 3)
        p0 = 2.0 * ctrl_xy[0] - ctrl_xy[1]
        pn = 2.0 * ctrl_xy[-1] - ctrl_xy[-2]
        ext = np.vstack([p0[None, :], ctrl_xy, pn[None, :]]).astype(np.float64)
        n_seg = k - 1
        dense = np.zeros((n_dense, 2), dtype=np.float64)
        u_all = np.linspace(0.0, float(n_seg), n_dense)
        for i, u in enumerate(u_all):
            seg = min(n_seg - 1, int(math.floor(u)))
            t = u - seg
            P0, P1, P2, P3 = ext[seg], ext[seg + 1], ext[seg + 2], ext[seg + 3]
            t2, t3 = t * t, t * t * t
            dense[i] = 0.5 * (
                (2.0 * P1)
                + (-P0 + P2) * t
                + (2.0 * P0 - 5.0 * P1 + 4.0 * P2 - P3) * t2
                + (-P0 + 3.0 * P1 - 3.0 * P2 + P3) * t3
            )
        seg_len = np.hypot(np.diff(dense[:, 0]), np.diff(dense[:, 1]))
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        tot = float(cum[-1])
        if tot > 1e-4:
            target_s = np.linspace(0.0, tot, n_samples)
            pts = np.column_stack([
                np.interp(target_s, cum, dense[:, 0]),
                np.interp(target_s, cum, dense[:, 1]),
            ])
        else:
            pts = dense[:n_samples]

    tang = np.zeros_like(pts)
    tang[0] = pts[1] - pts[0]
    tang[-1] = pts[-1] - pts[-2]
    if len(pts) > 2:
        tang[1:-1] = pts[2:] - pts[:-2]
    norms_t = np.linalg.norm(tang, axis=1, keepdims=True)
    tang = tang / np.maximum(norms_t, 1e-8)
    norm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    return pts, tang, norm


def _compute_tangent_field(subj_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y01 = np.clip(luma255(subj_rgb) / 255.0, 0.0, 1.0)
    gx = cv2.Sobel(y01, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y01, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 5.0)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 5.0)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 5.0)
    theta = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy) + (math.pi * 0.5)
    tx = np.cos(theta).astype(np.float32)
    ty = np.sin(theta).astype(np.float32)
    flip = ty < 0.0
    tx[flip] *= -1.0
    ty[flip] *= -1.0
    return tx, ty


def _hair_color_stats(
    subj_rgb: np.ndarray,
    subj_alpha: np.ndarray,
    hair_m: np.ndarray,
    exclude_m: np.ndarray | None = None,
) -> dict[str, Any]:
    lab = rgb255_to_lab(subj_rgb)
    valid = (hair_m >= 0.55) & (subj_alpha >= 0.70)
    if exclude_m is not None:
        valid = valid & (exclude_m < 0.25)
    if int(valid.sum()) < 200:
        valid = (hair_m >= 0.35) & (subj_alpha >= 0.45)
    vals = lab[valid]
    if len(vals) == 0:
        return {"med_L": 50.0, "med_a": 0.0, "med_b": 0.0, "med_C": 0.0, "lab": lab}
    med_L = float(np.median(vals[:, 0]))
    med_a = float(np.median(vals[:, 1]))
    med_b = float(np.median(vals[:, 2]))
    return {
        "med_L": med_L,
        "med_a": med_a,
        "med_b": med_b,
        "med_C": float(math.hypot(med_a, med_b)),
        "lab": lab,
    }


def detect_white_veil_and_acc(
    subj_rgb: np.ndarray,
    subj_alpha: np.ndarray,
    hair_m: np.ndarray,
    clothes_m: np.ndarray,
    acc_m: np.ndarray,
    face_snap: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Detect head-mounted achromatic white lace veils/bows plus accessories (zero silver-hair false positives)."""
    h, w = subj_alpha.shape
    base_excl = np.maximum.reduce([clothes_m, acc_m, face_snap])
    stats = _hair_color_stats(subj_rgb, subj_alpha, hair_m, base_excl)
    med_a, med_b = stats["med_a"], stats["med_b"]
    lab = stats["lab"]

    veil_bin = (acc_m >= 0.35) & (subj_alpha >= 0.35)
    head_veil_only = np.zeros((h, w), dtype=bool)
    fys, _ = np.where(face_snap >= 0.45)
    fy0 = int(fys.min()) if len(fys) else int(0.15 * h)
    fy1 = int(fys.max()) if len(fys) else int(0.35 * h)
    fh = max(48, fy1 - fy0)

    is_chromatic_hair = (med_a <= -4.0) or (med_b >= 10.5)
    if is_chromatic_hair:
        C_map = np.hypot(lab[:, :, 1], lab[:, :, 2])
        raw_white = (
            (subj_alpha > 0.65)
            & (lab[:, :, 0] > 70.0)
            & (C_map < 14.5)
            & (clothes_m < 0.35)
            & (face_snap < 0.35)
        )
        closed = cv2.morphologyEx(raw_white.astype(np.uint8), cv2.MORPH_CLOSE, ell_kernel(7))
        n_lbl, lbl_img, lbl_stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        for cid in range(1, n_lbl):
            area = int(lbl_stats[cid, cv2.CC_STAT_AREA])
            cy_top = int(lbl_stats[cid, cv2.CC_STAT_TOP])
            if area >= 600 and cy_top <= fy0 + int(1.25 * fh):
                comp = (lbl_img == cid) & raw_white & (np.arange(h)[:, None] <= fy1 + int(2.35 * fh))
                if int(comp.sum()) >= 400:
                    head_veil_only = head_veil_only | comp
                    veil_bin = veil_bin | comp

    if veil_bin.any():
        veil_dil = cv2.dilate(veil_bin.astype(np.uint8), ell_kernel(6)).astype(np.float32)
        veil_guard = np.clip(cv2.GaussianBlur(veil_dil, (0, 0), 1.5), 0.0, 1.0)
    else:
        veil_guard = np.zeros((h, w), dtype=np.float32)

    # Pure white head-veil guard (separate from opaque bows acc_m so behind layer never carves black moats!)
    if head_veil_only.any():
        hv_dil = cv2.dilate(head_veil_only.astype(np.uint8), ell_kernel(5)).astype(np.float32)
        head_veil_guard = np.clip(cv2.GaussianBlur(hv_dil, (0, 0), 1.2), 0.0, 1.0)
        veil_y_max = int(np.where(head_veil_only)[0].max())
    else:
        head_veil_guard = np.zeros((h, w), dtype=np.float32)
        if (acc_m >= 0.35).any():
            acc_head = np.where((acc_m >= 0.35) & (np.arange(h)[:, None] <= fy1 + int(2.2 * fh)))[0]
            veil_y_max = int(acc_head.max()) if len(acc_head) else int(fy1)
        else:
            veil_y_max = int(fy1)

    return veil_bin, veil_guard, head_veil_guard, veil_y_max


def recover_full_hair_mask(
    subj_rgb: np.ndarray,
    subj_alpha: np.ndarray,
    hair_m: np.ndarray,
    face_snap: np.ndarray,
    clothes_m: np.ndarray,
    arms_m: np.ndarray,
    acc_m: np.ndarray,
    all_skin_snap: np.ndarray,
    veil_mask_bin: np.ndarray,
    style: str = "auto",
) -> tuple[np.ndarray, int]:
    """Recover SegFormer-missed lower hair regions (e.g., Miku twintails) via CIE Lab hue & geometry checks."""
    h, w = subj_alpha.shape
    prot = np.maximum.reduce([face_snap, clothes_m, arms_m, acc_m, all_skin_snap, veil_mask_bin.astype(np.float32)])
    stats = _hair_color_stats(subj_rgb, subj_alpha, hair_m, prot)
    med_L, med_a, med_b = stats["med_L"], stats["med_a"], stats["med_b"]
    lab = stats["lab"]

    dL = np.abs(lab[:, :, 0] - med_L)
    da = np.abs(lab[:, :, 1] - med_a)
    db = np.abs(lab[:, :, 2] - med_b)
    dE = np.sqrt((0.52 * dL) ** 2 + (1.35 * da) ** 2 + (1.25 * db) ** 2)

    cand = (subj_alpha >= 0.45) & (prot < 0.28) & (dE <= 22.0)
    if med_a < -3.5:
        non_cloth_prot = np.maximum.reduce([face_snap, arms_m, acc_m, all_skin_snap, veil_mask_bin.astype(np.float32)])
        mint_direct = (
            (subj_alpha >= 0.45)
            & (non_cloth_prot < 0.25)
            & (lab[:, :, 1] < -4.2)
            & (lab[:, :, 2] < 14.0)
            & (lab[:, :, 0] > 28.0)
        )
        cand = (cand & (lab[:, :, 1] < -2.2) & (lab[:, :, 2] < med_b + 16.0)) | mint_direct
    elif med_b > 12.0:
        cand = cand & (lab[:, :, 2] > 8.5)

    ys_h0, _ = np.where((hair_m >= 0.45) & (subj_alpha >= 0.45))
    hy0 = int(ys_h0.min()) if len(ys_h0) else 0
    if (face_snap > 0.45).sum() >= 200:
        fys, fxs = np.where(face_snap > 0.45)
        fx0, fx1, fy0, fy1 = int(fxs.min()), int(fxs.max()), int(fys.min()), int(fys.max())
        fcx = 0.5 * (fx0 + fx1)
        fw = max(40.0, float(fx1 - fx0))
        fh = max(40.0, float(fy1 - fy0))
        yy = np.arange(h)[:, None]
        xx = np.arange(w)[None, :]
        cand = cand & (yy >= max(0, hy0 - 25)) & (np.abs(xx - fcx) <= 3.4 * fw)
        if style == "short-braids-wispy":
            cand = cand & (yy <= int(fy1 + 0.45 * fh))

    cand_u8 = cand.astype(np.uint8)
    n_lbl, lbl_img, lbl_stats, _ = cv2.connectedComponentsWithStats(cand_u8, connectivity=8)
    seed_dil = cv2.dilate((hair_m >= 0.45).astype(np.uint8), ell_kernel(14)) > 0
    clean_rec = np.zeros((h, w), dtype=bool)
    for cid in range(1, n_lbl):
        area = int(lbl_stats[cid, cv2.CC_STAT_AREA])
        comp = lbl_img == cid
        if area >= 140 or (area >= 40 and bool((comp & seed_dil).any())):
            clean_rec = clean_rec | comp

    rec_f32 = cv2.GaussianBlur(clean_rec.astype(np.float32), (0, 0), 0.9)
    full_hair = np.clip(np.maximum(hair_m, rec_f32), 0.0, 1.0)
    hard_prot = np.maximum.reduce([face_snap, arms_m, acc_m, all_skin_snap, veil_mask_bin.astype(np.float32)])
    eff_prot = np.where(clean_rec & (lab[:, :, 1] < -4.2), hard_prot, prot) if med_a < -3.5 else prot
    full_hair = np.clip(full_hair * (1.0 - np.clip(eff_prot * 1.4, 0.0, 1.0)), 0.0, 1.0)

    recovered_px = int(((full_hair >= 0.45) & (hair_m < 0.35) & (subj_alpha >= 0.45)).sum())
    return full_hair, recovered_px


def sculpt_anime_head_and_flow(
    subj_rgb: np.ndarray,
    subj_alpha: np.ndarray,
    full_hair_m: np.ndarray,
    hair_m_orig: np.ndarray,
    face_snap: np.ndarray,
    eyes_m: np.ndarray,
    clothes_m: np.ndarray,
    arms_m: np.ndarray,
    acc_m: np.ndarray,
    all_skin_snap: np.ndarray,
    veil_guard: np.ndarray,
    veil_y_max: int,
    style: str,
    boldness: str,
    wx_sign: float,
    rigid_mls_fn: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Mathematically bijective Gaussian liquify + crown matte defringe + Nazuna jaw blue-spill cleanup."""
    h, w = subj_alpha.shape
    clean_rgb = subj_rgb.copy()
    clean_a = subj_alpha.copy()

    if (face_snap > 0.45).sum() < 200:
        return (
            clean_rgb,
            clean_a,
            full_hair_m.copy(),
            hair_m_orig.copy(),
            {"applied": False, "crown_lift_px": 0.0, "min_det_J": 1.0, "max_disp_grad": 0.0, "max_prot_disp_px": 0.0, "max_bangs_disp_px": 0.0},
        )

    fys, fxs = np.where(face_snap > 0.45)
    fx0, fx1, fy0, fy1 = int(fxs.min()), int(fxs.max()), int(fys.min()), int(fys.max())
    fcx = 0.5 * (fx0 + fx1)
    fw = max(48.0, float(fx1 - fx0))
    fh = max(48.0, float(fy1 - fy0))

    # 1. Pre-warp faint sky-halo cleanup & crown outer fringe color decontamination
    d_solid = cv2.distanceTransform((clean_a < 0.55).astype(np.uint8), cv2.DIST_L2, 5)
    upper_zone = np.arange(h)[:, None] <= fy1
    faint_halo = upper_zone & (clean_a > 0.0) & (clean_a < 0.14) & (d_solid > 4.5)
    if faint_halo.any():
        halo_fade = _smoothstep(4.5, 9.0, d_solid)
        clean_a = np.where(upper_zone & (clean_a < 0.14), clean_a * (1.0 - 0.88 * halo_fade), clean_a).astype(np.float32)

    # Crown outer matte fringe decontamination (safe against accessories/bows, white veil lace, and dark holes)
    veil_dil_5 = cv2.dilate((veil_guard >= 0.15).astype(np.uint8), ell_kernel(5)) > 0
    hair_core = (full_hair_m >= 0.55) & (clean_a >= 0.85) & (~veil_dil_5) & (acc_m < 0.20)
    crown_edge_dil = cv2.dilate(((np.arange(h)[:, None] <= fy0 + 20) & (full_hair_m >= 0.25) & (acc_m < 0.20) & (~veil_dil_5)).astype(np.uint8), ell_kernel(3)) > 0
    crown_fringe = crown_edge_dil & (clean_a > 0.02) & (clean_a < 0.85) & (~veil_dil_5) & (acc_m < 0.20) & (clothes_m < 0.25)
    if crown_fringe.any() and int(hair_core.sum()) >= 200:
        hair_med_rgb = np.median(clean_rgb[hair_core], axis=0).astype(np.float32)
        hair_med_chr = hair_med_rgb - float(hair_med_rgb.mean())
        px_chr = clean_rgb - clean_rgb.mean(axis=2, keepdims=True)
        px_sat = clean_rgb.max(axis=2) - clean_rgb.min(axis=2)
        px_luma = luma255(clean_rgb)
        chr_dist = np.linalg.norm(px_chr - hair_med_chr[None, None, :], axis=2)
        # Skip white lace/veil fringes & saturated accessories (e.g. white lace crown or pink bow)
        non_acc_fringe = crown_fringe & (px_luma <= 175.0) & ((px_sat <= 45.0) | (chr_dist <= 65.0))
        if non_acc_fringe.any():
            decontam_rgb = decontaminate_fringe(clean_rgb, hair_core.astype(np.float32), non_acc_fringe.astype(np.float32), sigmas=(3.0, 10.0, 24.0))
            safe_mask = non_acc_fringe & (decontam_rgb.mean(axis=2) >= np.maximum(18.0, clean_rgb.mean(axis=2) * 0.45))
            clean_rgb = np.where(safe_mask[:, :, None], decontam_rgb, clean_rgb).astype(np.float32)

    # Rose veil dark-edge softening (strictly inside veil_guard>=0.40, ZERO white halo in stone arch!) + crown dust cleanup
    if style == "long-cascade":
        veil_outer_rim = (
            (veil_guard >= 0.40)
            & (clean_a > 0.08)
            & (clean_a < 0.85)
            & (luma255(clean_rgb) < 120.0)
            & (np.arange(h)[:, None] <= fy0 + 40)
        )
        if veil_outer_rim.any():
            soft_ivory = np.array([234.0, 230.0, 224.0], dtype=np.float32)
            w_vf = cv2.GaussianBlur(veil_outer_rim.astype(np.float32), (0, 0), 0.7)[:, :, None]
            clean_rgb = np.clip(clean_rgb * (1.0 - 0.35 * w_vf) + soft_ivory[None, None, :] * (0.35 * w_vf), 0.0, 255.0).astype(np.float32)

        crown_dust = (np.arange(h)[:, None] <= fy0 - 60) & (clean_a > 0.0) & (clean_a < 0.22) & (d_solid > 4.5)
        if crown_dust.any():
            clean_a = np.where(crown_dust, clean_a * 0.25, clean_a).astype(np.float32)

        rose_halo = (full_hair_m >= 0.20) & (veil_guard < 0.15) & (clothes_m < 0.20) & (clean_a > 0.0) & (clean_a < 0.16) & (d_solid > 3.5)
        if rose_halo.any():
            r_fade = _smoothstep(3.5, 7.5, d_solid)
            clean_a = np.where(rose_halo, clean_a * (1.0 - 0.90 * r_fade), clean_a).astype(np.float32)

    # Nazuna ZERO-HOLE cleanup: sigma=2.5 feathered haze/frizz suppression + strengthened inner-3px blue-spill neutralization
    if style == "short-braids-wispy":
        d_solid_n = cv2.distanceTransform((clean_a < 0.65).astype(np.uint8), cv2.DIST_L2, 5)
        yy_n = np.arange(h)[:, None]
        xx_n = np.arange(w)[None, :]
        head_braid_roi = (
            (yy_n >= max(0, int(fy0 - 1.10 * fh)))
            & (yy_n <= min(h, int(fy1 + 1.65 * fh)))
            & (xx_n >= max(0, int(fx0 - 1.35 * fw)))
            & (xx_n <= min(w, int(fx1 + 1.35 * fw)))
            & (face_snap < 0.35)
            & (arms_m < 0.15)
        )
        haze_raw = head_braid_roi & (clean_a > 0.0) & (clean_a < 0.36) & (d_solid_n > 3.2)
        bun_frizz = (yy_n <= int(fy0 + 0.12 * fh)) & head_braid_roi & (clean_a > 0.0) & (clean_a < 0.26) & (d_solid_n > 2.0)
        if haze_raw.any() or bun_frizz.any():
            haze_w = np.clip(cv2.GaussianBlur((haze_raw | bun_frizz).astype(np.float32), (0, 0), 2.5) * 1.45, 0.0, 1.0) * _smoothstep(2.0, 4.2, d_solid_n)
            clean_a = np.where(clean_a < 0.45, clean_a * (1.0 - 0.88 * haze_w), clean_a).astype(np.float32)

        lab_n = rgb255_to_lab(clean_rgb)
        luma_n = luma255(clean_rgb)
        d_clo_n = cv2.distanceTransform((clothes_m < 0.08).astype(np.uint8), cv2.DIST_L2, 5)
        braid_zones = (yy_n >= int(fy0 + 0.12 * fh)) & (yy_n <= int(fy1 + 1.45 * fh)) & ((xx_n <= int(fx0 - 0.05 * fw)) | (xx_n >= int(fx1 + 0.05 * fw)))
        blue_spill = braid_zones & head_braid_roi & (d_clo_n > 6.0) & (luma_n < 165.0) & (clean_a > 0.02) & (clean_a < 0.82) & (lab_n[:, :, 2] < -1.5)
        braid_tip_blue = (
            (yy_n >= int(fy1 - 0.10 * fh))
            & (yy_n <= int(fy1 + 1.45 * fh))
            & (xx_n >= int(fx0 - 1.25 * fw))
            & (xx_n <= int(fx0 + 0.10 * fw))
            & (d_clo_n > 10.0)
            & (luma_n < 150.0)
            & (clothes_m < 0.20)
            & (face_snap < 0.25)
            & (all_skin_snap < 0.25)
            & (clean_a > 0.05)
            & (lab_n[:, :, 2] < -2.2)
            & (full_hair_m < 0.45)
        )
        blue_fix = blue_spill | braid_tip_blue
        if blue_fix.any():
            neutral_hair = np.clip(clean_rgb * np.array([1.22, 1.13, 0.78], np.float32), 24.0, 236.0)
            w_spill = cv2.GaussianBlur(blue_fix.astype(np.float32), (0, 0), 1.1)[:, :, None]
            clean_rgb = np.clip(clean_rgb * (1.0 - w_spill) + neutral_hair * w_spill, 0.0, 255.0).astype(np.float32)
        tip_outer_faint = braid_tip_blue & (clean_a < 0.52) & (d_solid_n > 2.2)
        if tip_outer_faint.any():
            tf_w = np.clip(cv2.GaussianBlur(tip_outer_faint.astype(np.float32), (0, 0), 1.6) * 1.4, 0.0, 1.0)
            clean_a = np.where(tip_outer_faint, clean_a * (1.0 - 0.80 * tf_w), clean_a).astype(np.float32)

    # 2. Build wide smooth protection field INCLUDING straight forehead bangs hairline!
    clo_eff = np.clip(clothes_m * (1.0 - np.clip((full_hair_m - 0.55) * 2.2, 0.0, 1.0)), 0.0, 1.0)
    bangs_box = np.zeros((h, w), dtype=np.float32)
    bx0 = max(0, int(round(fx0 - 0.16 * fw)))
    bx1 = min(w, int(round(fx1 + 0.16 * fw)))
    by0 = max(0, int(round(fy0 - 0.06 * fh)))
    by1 = min(h, int(round(fy0 + 0.28 * fh)))
    if bx1 > bx0 and by1 > by0:
        bangs_box[by0:by1, bx0:bx1] = 1.0
    bangs_prot = cv2.GaussianBlur(bangs_box, (0, 0), 10.0)

    prot_mask = np.maximum.reduce([face_snap, eyes_m, bangs_prot, clo_eff, arms_m, acc_m, all_skin_snap, veil_guard])
    d_free = cv2.distanceTransform((prot_mask < 0.25).astype(np.uint8), cv2.DIST_L2, 5)
    upper_dome_zone = (np.arange(h)[:, None] <= fy0).astype(np.float32)
    ramp_top = 56.0 * (1.0 - upper_dome_zone) + 28.0 * upper_dome_zone
    allow_wide = _smoothstep(4.0, ramp_top, d_free).astype(np.float32)

    ys_h, _ = np.where((full_hair_m >= 0.45) & (clean_a >= 0.50))
    hy0 = int(ys_h.min()) if len(ys_h) else max(0, fy0 - int(fh))
    scale = {"subtle": 0.80, "medium": 1.0, "bold": 1.15}.get(boldness, 1.0)

    strokes: list[tuple[float, float, float, float, float, float]] = [
        (fcx, float(hy0) + 0.05 * fh, 1.55 * fw, 0.90 * fh, 0.0, -34.0 * scale),
        (fcx - 0.92 * fw, float(fy0) + 0.14 * fh, 0.65 * fw, 0.56 * fh, -12.0 * scale, -6.0 * scale),
        (fcx + 0.92 * fw, float(fy0) + 0.14 * fh, 0.65 * fw, 0.56 * fh, +12.0 * scale, -6.0 * scale),
    ]

    if style == "long-cascade":
        y_casc = float(max(veil_y_max + 55, int(fy1 + 1.9 * fh)))
        casc_ys, casc_xs = np.where((full_hair_m >= 0.45) & (clean_a >= 0.50) & (np.arange(h)[:, None] >= y_casc) & (np.arange(w)[None, :] < fcx))
        if len(casc_xs) >= 50:
            xm = float(np.percentile(casc_xs, 32))
            ym = float(np.median(casc_ys))
            strokes.append((xm, ym, 195.0, 235.0, wx_sign * 24.0 * scale, +10.0 * scale))
            strokes.append((xm - 40.0, min(float(h - 120), ym + 170.0), 180.0, 205.0, wx_sign * 20.0 * scale, +12.0 * scale))
    elif style == "twintails":
        twin_mask = (full_hair_m >= 0.45) & (clean_a >= 0.50) & (np.arange(h)[:, None] >= fy1 + int(1.25 * fh)) & (np.arange(w)[None, :] < fcx - int(0.45 * fw))
        tys, txs = np.where(twin_mask)
        if len(txs) >= 40:
            xm = float(np.median(txs))
            ym = float(np.percentile(tys, 38))
            yb = float(np.percentile(tys, 75))
            strokes.append((xm, ym, 150.0, 195.0, -16.0 * scale, +5.0))
            strokes.append((xm - 35.0, yb, 160.0, 200.0, +12.0 * scale, +7.0))
    elif style == "short-braids-wispy":
        bun_y = float(fy0 - 0.18 * fh)
        bxs = np.where((full_hair_m[int(np.clip(round(bun_y), 0, h - 1)), :] >= 0.45) & (clean_a[int(np.clip(round(bun_y), 0, h - 1)), :] >= 0.50))[0]
        if len(bxs) >= 15:
            strokes.append((float(bxs.min()) + 20.0, bun_y, 95.0, 95.0, -19.0 * scale, -14.5 * scale))
            strokes.append((float(bxs.max()) - 20.0, bun_y, 95.0, 95.0, +19.0 * scale, -14.5 * scale))

    gh = max(64, h // 4)
    gw = max(64, w // 4)
    gy = np.linspace(0.0, float(h - 1), gh, dtype=np.float32)[:, None]
    gx = np.linspace(0.0, float(w - 1), gw, dtype=np.float32)[None, :]
    dx_s = np.zeros((gh, gw), dtype=np.float32)
    dy_s = np.zeros((gh, gw), dtype=np.float32)
    for cx, cy, rx, ry, vx, vy in strokes:
        g_w = np.exp(-0.5 * (((gx - cx) / max(32.0, rx)) ** 2 + ((gy - cy) / max(32.0, ry)) ** 2))
        dx_s += vx * g_w
        dy_s += vy * g_w

    dx = cv2.resize(dx_s, (w, h), interpolation=cv2.INTER_CUBIC) * allow_wide
    dy = cv2.resize(dy_s, (w, h), interpolation=cv2.INTER_CUBIC) * allow_wide

    dx = cv2.GaussianBlur(dx, (0, 0), 16.0)
    dy = cv2.GaussianBlur(dy, (0, 0), 16.0)
    # 24px uniform zero-clamp transition prevents sharp slope spikes that triggered damping loops!
    zero_clamp = _smoothstep(1.0, 26.0, d_free).astype(np.float32)
    dx *= zero_clamp
    dy *= zero_clamp

    for _ in range(12):
        dy_dx = np.gradient(dy, axis=1)
        dy_dy = np.gradient(dy, axis=0)
        dx_dx = np.gradient(dx, axis=1)
        dx_dy = np.gradient(dx, axis=0)
        det_j = (1.0 - dx_dx) * (1.0 - dy_dy) - dx_dy * dy_dx
        grad_mag = np.maximum.reduce([np.abs(dx_dx), np.abs(dx_dy), np.abs(dy_dx), np.abs(dy_dy)])
        if float(det_j.min()) >= 0.62 and float(grad_mag.max()) <= 0.32:
            break
        dx = cv2.GaussianBlur(dx * 0.86, (0, 0), 14.0) * zero_clamp
        dy = cv2.GaussianBlur(dy * 0.86, (0, 0), 14.0) * zero_clamp

    dy_dx = np.gradient(dy, axis=1)
    dy_dy = np.gradient(dy, axis=0)
    dx_dx = np.gradient(dx, axis=1)
    dx_dy = np.gradient(dx, axis=0)
    min_det_j = round(float(((1.0 - dx_dx) * (1.0 - dy_dy) - dx_dy * dy_dx).min()), 3)
    max_grad = round(float(np.maximum.reduce([np.abs(dx_dx), np.abs(dx_dy), np.abs(dy_dx), np.abs(dy_dy)]).max()), 3)

    prot_raw = np.maximum.reduce([face_snap, eyes_m, clo_eff, arms_m, acc_m, all_skin_snap, veil_guard]) >= 0.35
    max_prot_disp = float(np.max(np.hypot(dx[prot_raw], dy[prot_raw]))) if prot_raw.any() else 0.0
    bangs_raw = (bangs_prot >= 0.65) & (full_hair_m >= 0.45)
    max_bangs_disp = float(np.max(np.hypot(dx[bangs_raw], dy[bangs_raw]))) if bangs_raw.any() else 0.0

    crown_band = (full_hair_m >= 0.45) & (clean_a >= 0.50) & (np.arange(h)[:, None] <= fy0 + int(0.15 * fh))
    crown_lift_px = round(float(max(0.0, np.max(-dy[crown_band]))) if crown_band.any() else 0.0, 1)

    yy_grid, xx_grid = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = (xx_grid - dx).astype(np.float32)
    map_y = (yy_grid - dy).astype(np.float32)

    pm = clean_rgb * clean_a[:, :, None]
    w_pm = cv2.remap(pm, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    w_a = np.clip(cv2.remap(clean_a, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0), 0.0, 1.0)
    w_rgb = np.where(w_a[:, :, None] > 1e-4, w_pm / np.maximum(w_a[:, :, None], 1e-4), clean_rgb)
    w_rgb = np.clip(w_rgb, 0.0, 255.0)

    w_full_h = np.clip(cv2.remap(full_hair_m, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0), 0.0, 1.0)
    w_orig_h = np.clip(cv2.remap(hair_m_orig, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0), 0.0, 1.0)

    return (
        w_rgb.astype(np.float32),
        w_a.astype(np.float32),
        w_full_h.astype(np.float32),
        w_orig_h.astype(np.float32),
        {
            "applied": True,
            "crown_lift_px": crown_lift_px,
            "min_det_J": min_det_j,
            "max_disp_grad": max_grad,
            "max_prot_disp_px": round(max_prot_disp, 4),
            "max_bangs_disp_px": round(max_bangs_disp, 2),
            "strokes": len(strokes),
        },
    )


def groom_real_hair_tangential(
    subj_rgb: np.ndarray,
    subj_alpha: np.ndarray,
    full_hair_m: np.ndarray,
    freeze_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """5-tap tangential directional filter along real hair fibers removing wig frizz cleanly."""
    h, w = subj_alpha.shape
    tx, ty = _compute_tangent_field(subj_rgb)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    weights = [0.13, 0.22, 0.30, 0.22, 0.13]
    offsets = (-2.0, -1.0, 0.0, 1.0, 2.0)
    acc = np.zeros_like(subj_rgb, dtype=np.float32)
    for k, wt in zip(offsets, weights):
        if k == 0.0:
            acc += subj_rgb * wt
        else:
            mx = np.clip(xx + k * tx, 0.0, float(w - 1)).astype(np.float32)
            my = np.clip(yy + k * ty, 0.0, float(h - 1)).astype(np.float32)
            acc += cv2.remap(subj_rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) * wt

    groom_mask = np.clip((full_hair_m - 0.35) / 0.35, 0.0, 1.0) * (1.0 - np.clip(freeze_m * 1.35, 0.0, 1.0)) * (subj_alpha >= 0.45)
    blend = 0.42 * groom_mask[:, :, None]
    out_rgb = np.clip(subj_rgb * (1.0 - blend) + acc * blend, 0.0, 255.0).astype(np.float32)
    return out_rgb, tx, ty


def extract_real_hair_detail_swatch(
    subj_rgb: np.ndarray,
    subj_alpha: np.ndarray,
    full_hair_m: np.ndarray,
    freeze_m: np.ndarray,
    tx_map: np.ndarray,
    ty_map: np.ndarray,
) -> dict[str, Any]:
    """Extract matte anisotropic 1-2.5px DSLR real hair fiber swatch (zero glossy highlight puddles!)."""
    h, w = subj_alpha.shape
    luma_all = luma255(subj_rgb)
    gx_l = cv2.Sobel(luma_all, cv2.CV_32F, 1, 0, ksize=3)
    gy_l = cv2.Sobel(luma_all, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx_l * gx_l, (0, 0), 4.0)
    jyy = cv2.GaussianBlur(gy_l * gy_l, (0, 0), 4.0)
    jxy = cv2.GaussianBlur(gx_l * gy_l, (0, 0), 4.0)
    coh = np.sqrt(np.maximum(0.0, (jxx - jyy) ** 2 + 4.0 * jxy * jxy)) / (jxx + jyy + 1e-4)

    for erode_r, dens_req in ((7, 0.75), (4, 0.55), (2, 0.35)):
        core = cv2.erode(((full_hair_m >= 0.65) & (subj_alpha >= 0.85) & (freeze_m < 0.15)).astype(np.uint8), ell_kernel(erode_r)) > 0
        if int(core.sum()) < 400:
            continue
        dens = cv2.boxFilter(core.astype(np.float32), cv2.CV_32F, (65, 65))
        lap = np.abs(cv2.Laplacian(luma_all, cv2.CV_32F, ksize=3))
        # Prefer high-coherence parallel vertical strands over blotchy specular highlights!
        sharp = cv2.boxFilter(lap * np.power(coh, 1.5) * core.astype(np.float32), cv2.CV_32F, (65, 65))
        score = sharp * (dens >= dens_req).astype(np.float32)
        if float(score.max()) <= 0.0:
            score = dens
        iy, ix = np.unravel_index(int(np.argmax(score)), score.shape)
        if dens[iy, ix] < 0.25:
            continue

        y0, y1 = max(0, iy - 32), min(h, iy + 33)
        x0, x1 = max(0, ix - 32), min(w, ix + 33)
        m_loc = core[y0:y1, x0:x1]
        if int(m_loc.sum()) < 30:
            continue
        tx0 = float(np.median(tx_map[y0:y1, x0:x1][m_loc]))
        ty0 = float(np.median(ty_map[y0:y1, x0:x1][m_loc]))
        n_t = max(1e-6, math.hypot(tx0, ty0))
        tx0, ty0 = tx0 / n_t, ty0 / n_t
        nx0, ny0 = -ty0, tx0

        px = (np.arange(64, dtype=np.float32) - 31.5)[None, :]
        py = (np.arange(128, dtype=np.float32) - 63.5)[:, None]
        mx = np.clip(float(ix) + px * nx0 + py * tx0, 0.0, float(w - 1)).astype(np.float32)
        my = np.clip(float(iy) + px * ny0 + py * ty0, 0.0, float(h - 1)).astype(np.float32)
        patch = cv2.remap(subj_rgb, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

        sym_u = np.concatenate([patch, patch[:, ::-1, :]], axis=1)
        tile_rgb = np.concatenate([sym_u, sym_u[::-1, :, :]], axis=0).astype(np.float32)

        # Mild vertical directional smoothing flattens broad melted-glass puddles into silky matte strands!
        t_luma = cv2.GaussianBlur(luma255(tile_rgb), (1, 5), sigmaX=0.4, sigmaY=1.8)
        low_l = cv2.GaussianBlur(t_luma, (0, 0), sigmaX=2.6, sigmaY=18.0)
        detail_ratio = t_luma / np.maximum(low_l, 22.0)
        # Flatten per-row mean to 1.00 so no dark/bright cross-bands streak horizontally
        row_mean = np.maximum(detail_ratio.mean(axis=1, keepdims=True), 1e-4)
        detail_ratio = np.clip(detail_ratio / row_mean, 0.84, 1.18).astype(np.float32)

        low_rgb = cv2.GaussianBlur(tile_rgb, (0, 0), sigmaX=2.6, sigmaY=18.0)
        chroma_delta = np.clip((tile_rgb - low_rgb) * 0.28, -7.0, 7.0).astype(np.float32)
        s_std = float(detail_ratio.std())
        if s_std >= 0.020 or erode_r == 2:
            return {
                "detail_ratio": detail_ratio,
                "chroma_delta": chroma_delta,
                "swatch_std": round(s_std, 4),
                "swatch_extracted": True,
                "anchor": [int(ix), int(iy)],
            }

    return {
        "detail_ratio": np.ones((256, 128), dtype=np.float32),
        "chroma_delta": np.zeros((256, 128, 3), dtype=np.float32),
        "swatch_std": 0.0,
        "swatch_extracted": False,
        "anchor": [-1, -1],
    }


def _local_hair_palette(
    subj_rgb: np.ndarray,
    hair_bin: np.ndarray,
    ax: float,
    ay: float,
    fallback_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample exact local real-hair RGB percentiles (p18, p52, p86) so overlap ΔE stays minimal."""
    h, w = hair_bin.shape
    ix = int(np.clip(round(ax), 0, w - 1))
    iy = int(np.clip(round(ay), 0, h - 1))
    vals = None
    for rad in (48, 96, 180):
        y0, y1 = max(0, iy - rad), min(h, iy + rad + 1)
        x0, x1 = max(0, ix - rad), min(w, ix + rad + 1)
        sub = hair_bin[y0:y1, x0:x1]
        if int(sub.sum()) >= 40:
            vals = subj_rgb[y0:y1, x0:x1][sub]
            break
    if vals is None or len(vals) < 12:
        vals = subj_rgb[hair_bin] if hair_bin.any() else fallback_rgb[None, :]
    p18 = np.percentile(vals, 18, axis=0).astype(np.float32)
    p52 = np.percentile(vals, 52, axis=0).astype(np.float32)
    p86 = np.percentile(vals, 86, axis=0).astype(np.float32)
    return np.clip(p18, 12.0, 238.0), np.clip(p52, 16.0, 244.0), np.clip(p86, 28.0, 250.0)


def _sample_swatch_uv(
    swatch: dict[str, Any],
    s_px: np.ndarray,
    u_px: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Periodic bilinear lookup into seamless 256x128 photographed real-hair detail & chroma swatch."""
    dr_tile = swatch["detail_ratio"]
    cd_tile = swatch["chroma_delta"]
    yp = np.mod(s_px, 256.0)
    xp = np.mod(u_px, 128.0)
    y0 = np.floor(yp).astype(np.int32)
    x0 = np.floor(xp).astype(np.int32)
    wy = (yp - y0).astype(np.float32)
    wx = (xp - x0).astype(np.float32)
    y1 = (y0 + 1) & 255
    x1 = (x0 + 1) & 127

    w00 = (1.0 - wy) * (1.0 - wx)
    w01 = (1.0 - wy) * wx
    w10 = wy * (1.0 - wx)
    w11 = wy * wx

    dr = w00 * dr_tile[y0, x0] + w01 * dr_tile[y0, x1] + w10 * dr_tile[y1, x0] + w11 * dr_tile[y1, x1]
    cd = (
        w00[:, None] * cd_tile[y0, x0]
        + w01[:, None] * cd_tile[y0, x1]
        + w10[:, None] * cd_tile[y1, x0]
        + w11[:, None] * cd_tile[y1, x1]
    )
    return dr.astype(np.float32), cd.astype(np.float32)


def _rasterize_analytical_strand(
    canvas_pm: np.ndarray,
    canvas_a: np.ndarray,
    *,
    pts: np.ndarray,
    radii: np.ndarray,
    alphas: np.ndarray,
    colors_rgb: np.ndarray,
    aa_half: float = 0.62,
    additive: bool = False,
) -> None:
    """Vectorized sub-pixel analytical capsule-SDF rasterizer (energy-conserving thin fibers, zero dashed tips)."""
    h, w = canvas_a.shape
    max_r = float(np.max(radii)) + aa_half + 1.5
    x0 = max(0, int(math.floor(float(pts[:, 0].min()) - max_r)))
    x1 = min(w, int(math.ceil(float(pts[:, 0].max()) + max_r)) + 1)
    y0 = max(0, int(math.floor(float(pts[:, 1].min()) - max_r)))
    y1 = min(h, int(math.ceil(float(pts[:, 1].max()) + max_r)) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    bh, bw = y1 - y0, x1 - x0

    roi_mask = np.zeros((bh, bw), dtype=np.uint8)
    pts_loc = (pts - np.array([[float(x0), float(y0)]])).astype(np.float32)
    poly = np.round(pts_loc).astype(np.int32)
    cv2.polylines(roi_mask, [poly], False, 255, thickness=max(5, int(math.ceil(2.0 * max_r + 3.0))))
    py, px = np.where(roi_mask > 0)
    if len(px) == 0:
        return

    A = pts_loc[:-1]
    B = pts_loc[1:]
    AB = B - A
    AB2 = np.maximum((AB * AB).sum(axis=1), 1e-6)
    qxy = np.stack([px.astype(np.float32), py.astype(np.float32)], axis=1)
    n_p = len(qxy)

    dist_all = np.zeros(n_p, dtype=np.float32)
    tp_all = np.zeros(n_p, dtype=np.float32)
    idx_all = np.zeros(n_p, dtype=np.int32)

    chunk = 8192
    for st in range(0, n_p, chunk):
        ed = min(n_p, st + chunk)
        q = qxy[st:ed]
        rel = q[:, None, :] - A[None, :, :]
        t_proj = np.clip((rel[:, :, 0] * AB[None, :, 0] + rel[:, :, 1] * AB[None, :, 1]) / AB2[None, :], 0.0, 1.0)
        proj = A[None, :, :] + t_proj[:, :, None] * AB[None, :, :]
        diff = q[:, None, :] - proj
        d2 = (diff * diff).sum(axis=2)
        idx = np.argmin(d2, axis=1)
        ar = np.arange(ed - st)
        dist_all[st:ed] = np.sqrt(np.maximum(d2[ar, idx], 0.0))
        tp_all[st:ed] = t_proj[ar, idx]
        idx_all[st:ed] = idx

    r_f = radii.astype(np.float32)
    a_f = alphas.astype(np.float32)
    c_f = colors_rgb.astype(np.float32)
    r_loc = r_f[idx_all] * (1.0 - tp_all) + r_f[idx_all + 1] * tp_all
    a_loc = a_f[idx_all] * (1.0 - tp_all) + a_f[idx_all + 1] * tp_all
    col_loc = c_f[idx_all] * (1.0 - tp_all[:, None]) + c_f[idx_all + 1] * tp_all[:, None]

    # Energy-conserving thin fiber AA: never collapse <0.65px tips into pixel-stepping beads
    r_eff = np.maximum(0.45, r_loc)
    sub_gain = np.clip(r_loc / 0.65, 0.22, 1.0)
    cov = _smoothstep(r_eff + aa_half, np.maximum(0.0, r_eff - aa_half), dist_all) * a_loc * sub_gain

    active = cov > 1e-4
    if not active.any():
        return

    py_a, px_a = py[active] + y0, px[active] + x0
    c_a = cov[active]
    col_a = np.clip(col_loc[active], 8.0, 253.0)

    if additive:
        canvas_pm[py_a, px_a] = np.clip(canvas_pm[py_a, px_a] + col_a * c_a[:, None], 0.0, 255.0)
        canvas_a[py_a, px_a] = np.clip(1.0 - (1.0 - canvas_a[py_a, px_a]) * (1.0 - c_a), 0.0, 1.0)
    else:
        inv = (1.0 - c_a)[:, None]
        canvas_pm[py_a, px_a] = col_a * c_a[:, None] + canvas_pm[py_a, px_a] * inv
        canvas_a[py_a, px_a] = c_a + canvas_a[py_a, px_a] * (1.0 - c_a)


def _strand_color_profile(
    col_root: np.ndarray,
    col_body: np.ndarray,
    col_tip: np.ndarray,
    col_rim: np.ndarray,
    ts: np.ndarray,
    normals: np.ndarray,
    key_dir_xy: tuple[float, float],
    rng: np.random.Generator,
    role: str = "mid",
) -> np.ndarray:
    """Compute per-strand tonal separation anchored faithfully to true root pigment (0 bleached roots!)."""
    hue_jit = rng.uniform(0.975, 1.025, size=3).astype(np.float32)

    if role == "shadow":
        c0 = np.clip(col_body, 14.0, 240.0)
        c1 = np.clip(0.65 * col_body + 0.35 * col_tip, 16.0, 244.0)
        lum_jit = float(rng.uniform(0.84, 0.98))
        sheen_k = 0.05
    elif role == "bright":
        c0 = np.clip(0.50 * col_body + 0.50 * col_tip, 28.0, 248.0)
        c1 = np.clip(0.62 * col_tip + 0.38 * col_rim, 36.0, 252.0)
        lum_jit = float(rng.uniform(0.98, 1.12))
        sheen_k = 0.16
    else:
        c0 = np.clip(col_body, 18.0, 244.0)
        c1 = np.clip(0.72 * col_body + 0.28 * col_tip, 24.0, 248.0)
        lum_jit = float(rng.uniform(0.90, 1.09))
        sheen_k = 0.10

    b_col = np.clip(c0 * lum_jit * hue_jit, 12.0, 246.0)
    t_col = np.clip(c1 * lum_jit * hue_jit, 16.0, 252.0)

    r_ramp = _smoothstep(0.18, 0.44, ts)[:, None].astype(np.float32)
    t_ramp = _smoothstep(0.45, 0.98, ts)[:, None].astype(np.float32)
    base_col = col_root[None, :].astype(np.float32) * (1.0 - r_ramp) + b_col[None, :] * r_ramp
    grad = base_col * (1.0 - t_ramp) + t_col[None, :] * t_ramp

    kx, ky = key_dir_xy
    ndotl = np.abs(normals[:, 0] * kx + normals[:, 1] * ky).astype(np.float32)
    sheen_gain = (1.0 + sheen_k * np.power(ndotl, 1.7))[:, None]
    cap = 250.0 if role == "bright" else 246.0
    return np.clip(grad * sheen_gain, 14.0, cap).astype(np.float32)


def _emit_strand_wisp(
    canvas_pm: np.ndarray,
    canvas_a: np.ndarray,
    *,
    ctrl_xy: np.ndarray,
    wisp_width: float,
    col_root: np.ndarray,
    col_shd: np.ndarray,
    col_mid: np.ndarray,
    col_hi: np.ndarray,
    col_tip: np.ndarray,
    col_rim: np.ndarray,
    key_dir_xy: tuple[float, float],
    rng: np.random.Generator,
    n_strands: int = 26,
    root_veil_alpha: float = 0.24,
    opacity_scale: float = 1.0,
    root_fade_s: float = 0.16,
    tip_split_fan: float = 0.42,
    micro_weave_px: float = 2.4,
) -> int:
    """Emit one airy layered wisp built purely from individual 0.8-2.2px filaments (ZERO solid blob!)."""
    arc_len = float(np.sum(np.hypot(np.diff(ctrl_xy[:, 0]), np.diff(ctrl_xy[:, 1]))))
    n_samples = max(32, min(72, int(round(arc_len / 7.5)) + 14))
    pts_c, _, norm_c = _catmull_rom_2d(ctrl_xy, n_samples=n_samples)
    ts = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)

    # Optional whisper-soft root-only veil (dissolves cleanly to 0 before t=0.46 so mid+tips are 100% individual hairs!)
    if root_veil_alpha > 0.02 and wisp_width >= 8.0:
        r_veil = np.maximum(1.2, 0.25 * float(wisp_width) * (1.0 - 0.82 * _smoothstep(0.0, 0.46, ts))).astype(np.float32)
        a_veil = np.clip(
            float(root_veil_alpha)
            * opacity_scale
            * _smoothstep(0.0, max(0.05, root_fade_s), ts)
            * (1.0 - _smoothstep(0.16, 0.46, ts)),
            0.0,
            0.55,
        ).astype(np.float32)
        cols_v = _strand_color_profile(col_root, col_mid, col_tip, col_rim, ts, norm_c, key_dir_xy, rng, role="mid")
        _rasterize_analytical_strand(
            canvas_pm, canvas_a, pts=pts_c, radii=r_veil, alphas=a_veil, colors_rgb=cols_v, aa_half=2.8, additive=False
        )

    n_shd = max(3, int(round(n_strands * 0.26)))
    n_hi = max(2, int(round(n_strands * 0.14)))
    n_mid = max(5, n_strands - n_shd - n_hi)
    n_hi_top = min(2, n_hi)
    mix_mid_hi = (["mid"] * n_mid) + (["bright"] * max(0, n_hi - n_hi_top))
    rng.shuffle(mix_mid_hi)
    roles = (["shadow"] * n_shd) + mix_mid_hi + (["bright"] * n_hi_top)

    # Asymmetric 2-3 micro-bundles ("小撮", 有聚有散) to shatter uniform comb-tooth parallelism
    n_clusters = 2 if wisp_width < 12.0 else 3
    if n_clusters == 2:
        c_centers = np.array([-0.28, 0.26], dtype=np.float64) + rng.uniform(-0.07, 0.07, size=2)
    else:
        c_centers = np.array([-0.34, 0.06, 0.36], dtype=np.float64) + rng.uniform(-0.08, 0.08, size=3)
    c_phases = rng.uniform(0.0, 2.0 * math.pi, size=n_clusters)
    c_freqs = rng.uniform(0.9, 1.8, size=n_clusters)

    emitted = 0
    w_half = 0.5 * float(wisp_width)
    pinch_env = (0.76 + 0.26 * _smoothstep(0.0, 0.20, ts)) * (1.0 - 0.44 * _smoothstep(0.42, 1.0, ts))

    for role in roles:
        ci = int(rng.integers(0, n_clusters))
        is_loose = rng.random() < 0.20
        if is_loose:
            u0 = float(rng.uniform(-0.74, 0.74))
        else:
            u0 = float(np.clip(c_centers[ci] + rng.normal(0.0, 0.11), -0.78, 0.78))

        short_tail = rng.random() < 0.35
        if role == "shadow":
            t_stop = float(rng.uniform(0.50, 0.80)) if short_tail else float(rng.uniform(0.76, 0.94))
            r_peak = float(rng.uniform(0.72, 1.18))
            op0 = float(rng.uniform(0.58, 0.82)) * opacity_scale
        elif role == "bright":
            if is_loose:
                u0 = math.copysign(1.0, u0 if abs(u0) > 0.05 else 1.0) * float(rng.uniform(0.28, 0.68))
            t_stop = float(rng.uniform(0.56, 0.84)) if short_tail else float(rng.uniform(0.78, 1.00))
            r_peak = float(rng.uniform(0.42, 0.78))
            op0 = float(rng.uniform(0.58, 0.86)) * opacity_scale
        else:
            t_stop = float(rng.uniform(0.48, 0.76)) if short_tail else float(rng.uniform(0.76, 1.00))
            r_peak = float(rng.uniform(0.56, 1.06))
            op0 = float(rng.uniform(0.68, 0.94)) * opacity_scale

        m_idx = max(8, int(round(t_stop * n_samples)))
        ts_i = np.linspace(0.0, 1.0, m_idx, dtype=np.float64)
        pts_i = pts_c[:m_idx].copy()
        norm_i = norm_c[:m_idx]

        freq = float(rng.uniform(1.1, 2.5))
        ph = float(rng.uniform(0.0, 2.0 * math.pi))
        eff_weave = max(micro_weave_px, 3.0) if wisp_width >= 14.0 else micro_weave_px
        weave = eff_weave * np.sin(freq * math.pi * ts_i + ph) * np.sin(math.pi * ts_i)
        cluster_sway = (0.22 * w_half) * np.sin(c_freqs[ci] * math.pi * ts_i + c_phases[ci]) * np.sin(math.pi * ts_i)
        fan_sgn = math.copysign(1.0, u0) if abs(u0) > 0.05 else (1.0 if rng.random() > 0.5 else -1.0)
        split_fan = tip_split_fan * fan_sgn * float(rng.uniform(0.35, 1.25)) * w_half * _smoothstep(0.42, 1.0, ts_i)
        drift_curve = float(rng.uniform(-0.20, 0.20)) * w_half * _smoothstep(0.25, 1.0, ts_i)

        lat_off = u0 * w_half * pinch_env[:m_idx] + weave + cluster_sway + split_fan + drift_curve
        pts_i = pts_i + lat_off[:, None] * norm_i

        rad_taper = (0.50 + 0.50 * _smoothstep(0.0, 0.10, ts_i)) * np.power(
            np.clip(1.0 - np.maximum(0.0, ts_i - 0.18) / 0.82, 0.04, 1.0), 0.78
        )
        radii = np.maximum(0.24, r_peak * rad_taper).astype(np.float32)

        r_a = _smoothstep(0.0, max(0.04, root_fade_s), ts_i)
        t_a = 1.0 - _smoothstep(0.68, 1.0, ts_i)
        alphas = np.clip(op0 * r_a * t_a, 0.0, 0.96).astype(np.float32)

        if role == "shadow":
            cols = _strand_color_profile(col_root, col_shd, col_mid, col_rim, ts_i, norm_i, key_dir_xy, rng, role="shadow")
        elif role == "bright":
            cols = _strand_color_profile(col_root, col_hi, col_tip, col_rim, ts_i, norm_i, key_dir_xy, rng, role="bright")
        else:
            cols = _strand_color_profile(col_root, col_mid, col_tip, col_rim, ts_i, norm_i, key_dir_xy, rng, role="mid")

        _rasterize_analytical_strand(
            canvas_pm, canvas_a, pts=pts_i, radii=radii, alphas=alphas, colors_rgb=cols, aa_half=0.62, additive=False
        )
        emitted += 1

    return emitted


def _emit_layered_hair_stream(
    canvas_pm: np.ndarray,
    canvas_a: np.ndarray,
    *,
    ctrl_xy: np.ndarray,
    stream_width: float,
    col_root: np.ndarray,
    col_shd: np.ndarray,
    col_mid: np.ndarray,
    col_hi: np.ndarray,
    col_tip: np.ndarray,
    col_rim: np.ndarray,
    key_dir_xy: tuple[float, float],
    rng: np.random.Generator,
    n_wisps: int = 4,
    strands_per_wisp: int = 24,
    root_veil_alpha: float = 0.24,
    opacity_scale: float = 1.0,
    root_fade_s: float = 0.16,
    gap_spread: float = 1.15,
) -> int:
    """Split a guide spine into asymmetric core + divergent sub-wisps with diagonal negative-space slits."""
    pts, tang, norm = _catmull_rom_2d(ctrl_xy, n_samples=48)
    idx_pick = np.linspace(0, len(pts) - 1, 7, dtype=int)
    lat_base = np.linspace(-0.40, 0.40, max(2, n_wisps))
    lat_anchors = lat_base + rng.uniform(-0.10, 0.10, size=len(lat_base))
    total = 0

    for wi, lat_a in enumerate(lat_anchors):
        is_core = (wi == len(lat_anchors) // 2) or (len(lat_anchors) >= 3 and wi == 1)
        w_wisp = max(6.5, float(stream_width) * (float(rng.uniform(0.38, 0.48)) if is_core else float(rng.uniform(0.22, 0.32))))
        n_str_w = max(14, int(round(strands_per_wisp * (1.20 if is_core else 0.75))))
        ext_len = float(rng.uniform(0.84, 1.06)) if is_core else float(rng.uniform(0.68, 1.08))
        div_k = gap_spread * (0.75 if is_core else 1.25)
        diverge = lat_a * float(stream_width) * div_k
        breeze_phase = float(rng.uniform(0.0, 2.0 * math.pi))
        breeze_amp = float(rng.uniform(3.0, 8.5))

        w_ctrl = np.zeros((7, 2), dtype=np.float64)
        for j, pi in enumerate(idx_pick):
            u_prog = j / 6.0
            base_off = lat_a * float(stream_width) * (0.22 + 0.78 * _smoothstep(0.0, 0.34, u_prog))
            air_sep = diverge * _smoothstep(0.24, 0.94, u_prog)
            sway = breeze_amp * math.sin(1.4 * math.pi * u_prog + breeze_phase) * _smoothstep(0.20, 0.95, u_prog)
            along = (ext_len - 1.0) * 68.0 * (u_prog ** 1.6)
            w_ctrl[j] = pts[pi] + norm[pi] * (base_off + air_sep + sway) + tang[pi] * along

        total += _emit_strand_wisp(
            canvas_pm,
            canvas_a,
            ctrl_xy=w_ctrl,
            wisp_width=w_wisp,
            col_root=col_root,
            col_shd=col_shd,
            col_mid=col_mid,
            col_hi=col_hi,
            col_tip=col_tip,
            col_rim=col_rim,
            key_dir_xy=key_dir_xy,
            rng=rng,
            n_strands=n_str_w,
            root_veil_alpha=(root_veil_alpha if is_core else root_veil_alpha * 0.40),
            opacity_scale=opacity_scale,
            root_fade_s=root_fade_s,
            tip_split_fan=float(rng.uniform(0.32, 0.54)),
            micro_weave_px=3.6 if stream_width >= 18.0 else 2.2,
        )
    return total


def _apply_backlight_lightwrap(
    rgb: np.ndarray,
    a: np.ndarray,
    orig_alpha: np.ndarray,
    bg_rgb: np.ndarray | None,
    real_p15: np.ndarray,
    real_p85: np.ndarray,
    style: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Match atmospheric haze & subtle backlit Light Wrap STRICTLY in external air (0px root color shift!)."""
    if float(a.max()) <= 1e-4 or bg_rgb is None:
        return rgb, a

    active = a > 1e-3
    # Only wrap strands in open sky/background beyond the original cutout so buried roots keep dE <= 5.0!
    orig_sm = cv2.GaussianBlur(orig_alpha.astype(np.float32), (0, 0), 3.0)
    ext_only = _smoothstep(0.60, 0.05, orig_sm)

    # 1. Gentle shadow-floor lift strictly where darkest troughs sit below the real photographed wig
    floor = (real_p15 * 0.92)[None, None, :].astype(np.float32)
    below = np.maximum(0.0, floor - rgb)
    floor_lift = (0.42 if style == "twintails" else (0.36 if style == "long-cascade" else 0.25)) * ext_only[:, :, None]
    out_rgb = rgb + below * floor_lift

    # 2. Subtle optical Light Wrap proportional to airiness (keeps true mint/gold pigment instead of white washout!)
    bg_blur = cv2.GaussianBlur(bg_rgb.astype(np.float32), (0, 0), 16.0)
    bg_luma01 = np.clip(luma255(bg_blur) / 255.0, 0.0, 1.0)
    dens = np.clip(cv2.GaussianBlur(a, (0, 0), 3.2), 0.0, 1.0)
    airy = np.clip(1.0 - 0.65 * dens, 0.18, 1.0)
    bg_bright = _smoothstep(0.32, 0.92, bg_luma01)

    if style == "twintails":
        wrap_w = np.clip(0.11 * ext_only * airy * (0.25 + 0.75 * bg_bright), 0.0, 0.13)[:, :, None]
        sunset_peach = np.array([255.0, 224.0, 198.0], dtype=np.float32)[None, None, :]
        wrap_col = 0.72 * bg_blur + 0.28 * sunset_peach
        out_rgb = out_rgb * (1.0 - wrap_w) + wrap_col * wrap_w
    elif style == "long-cascade":
        wrap_w = np.clip(0.12 * ext_only * airy * (0.20 + 0.80 * bg_bright), 0.0, 0.14)[:, :, None]
        warm_ivory = np.array([254.0, 240.0, 214.0], dtype=np.float32)[None, None, :]
        wrap_col = 0.70 * bg_blur + 0.30 * warm_ivory
        out_rgb = out_rgb * (1.0 - wrap_w) + wrap_col * wrap_w
    else:
        wrap_w = np.clip(0.06 * ext_only * airy * bg_bright, 0.0, 0.08)[:, :, None]
        out_rgb = out_rgb * (1.0 - wrap_w) + bg_blur * wrap_w

    out_rgb = np.where(active[:, :, None], np.clip(out_rgb, 10.0, 253.0), 0.0).astype(np.float32)
    return out_rgb, a


def _finalize_canvas(canvas_pm: np.ndarray, canvas_a: np.ndarray, psf_sigma: float = 0.24) -> tuple[np.ndarray, np.ndarray]:
    if float(canvas_a.max()) <= 1e-4:
        h, w = canvas_a.shape
        return np.zeros((h, w, 3), dtype=np.float32), np.zeros((h, w), dtype=np.float32)
    pm = cv2.GaussianBlur(canvas_pm, (0, 0), psf_sigma) if psf_sigma > 0.05 else canvas_pm
    a = np.clip(cv2.GaussianBlur(canvas_a, (0, 0), psf_sigma) if psf_sigma > 0.05 else canvas_a, 0.0, 1.0)
    rgb = np.where(a[:, :, None] > 1e-4, pm / np.maximum(a[:, :, None], 1e-4), 0.0)
    return np.clip(rgb, 0.0, 255.0).astype(np.float32), a.astype(np.float32)


def _apply_optical_dof(
    rgb: np.ndarray,
    a: np.ndarray,
    orig_a: np.ndarray,
    fy1: int,
    fh: float,
    style: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend natural 85mm distance-adaptive optical DoF outward from focal plane without hard alpha locking."""
    if float(a.max()) <= 1e-4:
        return rgb, a
    h, _ = a.shape
    d_ext = cv2.distanceTransform((orig_a < 0.35).astype(np.uint8), cv2.DIST_L2, 5)
    yy = np.arange(h, dtype=np.float32)[:, None]
    y_drop = np.clip((yy - float(fy1)) / max(120.0, 3.2 * float(fh)), 0.0, 1.0)

    if style == "long-cascade":
        sigma_far = 0.68
        dof_w = np.clip(0.55 * _smoothstep(12.0, 130.0, d_ext) + 0.45 * _smoothstep(0.32, 1.0, y_drop), 0.0, 1.0)
    elif style == "twintails":
        sigma_far = 0.88
        dof_w = np.clip(0.58 * _smoothstep(12.0, 130.0, d_ext) + 0.42 * _smoothstep(0.32, 1.0, y_drop), 0.0, 1.0)
    else:
        sigma_far = 0.58
        dof_w = np.clip(0.55 * _smoothstep(5.0, 44.0, d_ext), 0.0, 1.0)

    pm = rgb * a[:, :, None]
    pm_far = cv2.GaussianBlur(pm, (0, 0), sigma_far)
    a_far = np.clip(cv2.GaussianBlur(a, (0, 0), sigma_far), 0.0, 1.0)
    pm_out = pm * (1.0 - dof_w[:, :, None]) + pm_far * dof_w[:, :, None]
    a_out = np.clip(a * (1.0 - dof_w) + a_far * dof_w, 0.0, 1.0).astype(np.float32)
    rgb_out = np.where(a_out[:, :, None] > 1e-4, pm_out / np.maximum(a_out[:, :, None], 1e-4), 0.0)
    return np.clip(rgb_out, 0.0, 255.0).astype(np.float32), a_out


def build_anime_hair_optics(
    subj_rgb: np.ndarray,
    subj_alpha: np.ndarray,
    full_hair_m: np.ndarray,
    freeze_m: np.ndarray,
    new_hair_alpha: np.ndarray,
    lighting: dict[str, Any],
    style: str,
    face_box: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build dome-locked crown highlight (0 bangs watermark!), nocturnal 藏色 Soft Light, and dramatic Rim Light."""
    h, w = subj_alpha.shape
    hair_vis = np.clip(full_hair_m * np.clip(subj_alpha * 1.15, 0.0, 1.0), 0.0, 1.0)
    union_hair = np.clip(np.maximum(hair_vis, new_hair_alpha), 0.0, 1.0)
    if style == "long-cascade":
        freeze_pad = cv2.dilate((freeze_m >= 0.25).astype(np.uint8), ell_kernel(3)).astype(np.float32)
        allow = np.clip(1.0 - np.maximum(freeze_m * 1.15, freeze_pad), 0.0, 1.0)
    else:
        allow = np.clip(1.0 - freeze_m * 1.15, 0.0, 1.0)

    luma01 = np.clip(luma255(subj_rgb) / 255.0, 0.0, 1.0)
    blur2 = cv2.GaussianBlur(luma01, (0, 0), 2.0)
    blur11 = cv2.GaussianBlur(luma01, (0, 0), 11.0)
    fiber_hi = np.clip((luma01 - cv2.GaussianBlur(luma01, (0, 0), 2.8)) * 11.0 + 0.52, 0.12, 1.0)

    trough = np.clip((blur11 - blur2) * 5.8, 0.0, 1.0) * hair_vis * allow
    ridge_lift = np.clip((blur2 - blur11) * 4.2, 0.0, 1.0) * hair_vis * allow

    ys_h, xs_h = np.where(hair_vis >= 0.40)
    pol_rgb = np.full((h, w, 3), 128.0, dtype=np.float32)
    pol_a = np.zeros((h, w), dtype=np.float32)
    if len(xs_h) > 100:
        hy0, hy1 = int(ys_h.min()), int(ys_h.max())
        hcx = float(np.median(xs_h))
        hspan = max(80.0, float(xs_h.max() - xs_h.min()))
        fy0 = face_box[1] if face_box is not None else int(hy0 + 0.45 * min(450.0, float(hy1 - hy0)))
        fh = float(face_box[3] - face_box[1]) if face_box is not None else 200.0
        fcx = 0.5 * float(face_box[0] + face_box[2]) if face_box is not None else hcx

        # Clamp silk crown arch strictly above straight forehead bangs (dome_only -> ZERO bangs watermark!)
        yy = np.arange(h, dtype=np.float32)[:, None]
        xx = np.arange(w, dtype=np.float32)[None, :]
        dome_only = _smoothstep(float(fy0) - 0.10 * fh, float(fy0) - 0.36 * fh, yy)
        crown_y = hy0 + 0.24 * max(60.0, float(fy0 - hy0))
        y_arch = crown_y + 0.10 * hspan * np.square((xx - fcx) / (0.65 * hspan))
        silk_band = np.exp(-0.5 * ((yy - y_arch) / max(18.0, 0.055 * hspan)) ** 2) * dome_only * hair_vis * allow * fiber_hi

        amb255 = np.array(lighting["ambient_color"], dtype=np.float32) * 255.0
        rim255 = np.array(lighting["rim_color"], dtype=np.float32) * 255.0
        shd_col = np.clip(amb255 * 0.60, 14.0, 124.0)
        hi_col = np.clip(0.55 * np.array([255.0, 255.0, 255.0], np.float32) + 0.45 * rim255, 165.0, 255.0)

        delta_shd = (shd_col[None, None, :] - 128.0) * (0.50 * trough[:, :, None])
        delta_ridge = (hi_col[None, None, :] - 128.0) * (0.20 * ridge_lift[:, :, None])
        delta_hi = (hi_col[None, None, :] - 128.0) * (0.34 * silk_band[:, :, None])

        # Nocturnal moonlit tension on Nazuna: deepen shadow-facing right braid/temple toward indigo 藏色
        if style == "short-braids-wispy":
            shd_side = _smoothstep(fcx + 0.05 * hspan, fcx + 0.45 * hspan, xx) * hair_vis * allow
            indigo_tgt = np.array([38.0, 34.0, 72.0], dtype=np.float32)
            delta_moon_shd = (indigo_tgt[None, None, :] - 128.0) * (0.42 * shd_side[:, :, None])
            # Boost crown silk luster & 3D braid bump trough/ridge for spherical sculpture
            delta_hi = (hi_col[None, None, :] - 128.0) * (0.44 * silk_band[:, :, None])
            delta_shd = delta_shd * 1.35 + delta_moon_shd
            delta_ridge = delta_ridge * 1.52

        pol_rgb = np.clip(128.0 + delta_shd + delta_ridge + delta_hi, 0.0, 255.0)
        pol_a = np.clip(np.maximum(hair_vis, new_hair_alpha * 0.45) * allow, 0.0, 1.0)

    # Strict Outer-Silhouette Rim Light ONLY (ZERO interior l_bump cow-hide white spots / forehead watermarks!)
    rim_rgb = np.zeros((h, w, 3), dtype=np.float32)
    rim_a = np.zeros((h, w), dtype=np.float32)
    if union_hair.max() > 0.1:
        sm = cv2.GaussianBlur(union_hair, (0, 0), 4.0)
        gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
        gn = np.hypot(gx, gy) + 1e-6
        nx_out = -gx / gn
        ny_out = -gy / gn

        ks = lighting.get("key_side", "right")
        lx, ly = (-0.94, -0.34) if ks == "left" else (0.94, -0.34)
        dot = np.clip(nx_out * lx + ny_out * ly, 0.0, 1.0)
        rim_dir = np.power(dot, 0.80)
        if style == "twintails":
            dot_opp = np.clip(nx_out * (-lx) + ny_out * ly, 0.0, 1.0)
            rim_dir = np.maximum(rim_dir, 0.40 * np.power(dot_opp, 1.10))

        # Hard OUTER SKY silhouette distance lock (>14px inside overall subject+hair contour is MATHEMATICALLY 0.00!)
        # Using max(union_hair, subj_alpha) guarantees 0 rim spots on forehead bangs over face skin!
        dist_out = cv2.distanceTransform((np.maximum(union_hair, subj_alpha) >= 0.35).astype(np.uint8), cv2.DIST_L2, 5)
        outer_rim_band = _smoothstep(14.0, 2.0, dist_out)

        erode_inner = cv2.GaussianBlur(union_hair, (0, 0), 6.2 if style == "short-braids-wispy" else 4.6)
        perimeter = np.clip((union_hair - erode_inner * 0.91) * 3.8, 0.0, 1.0)
        perimeter_halo = np.clip(cv2.GaussianBlur(perimeter, (0, 0), 1.4), 0.0, 1.0) * outer_rim_band * allow

        if style == "short-braids-wispy":
            rim_a = np.clip(perimeter_halo * (0.15 + 0.85 * rim_dir) * 1.15 * (0.76 + 0.24 * fiber_hi), 0.0, 0.98).astype(np.float32)
            rim255 = np.clip(0.38 * (np.array(lighting["rim_color"], np.float32) * 255.0) + 0.62 * np.array([234.0, 243.0, 255.0], np.float32), 192.0, 255.0)
        else:
            rim_a = np.clip(perimeter_halo * (0.18 + 0.82 * rim_dir) * 0.64 * (0.72 + 0.28 * fiber_hi), 0.0, 0.95).astype(np.float32)
            rim255 = np.array(lighting["rim_color"], dtype=np.float32) * 255.0
            if style == "twintails":
                rim255 = np.clip(0.65 * rim255 + 0.35 * np.array([255.0, 222.0, 200.0], np.float32), 150.0, 255.0)

        rim_rgb = np.clip(0.24 * np.maximum(subj_rgb, 160.0) + 0.76 * rim255[None, None, :], 110.0, 255.0).astype(np.float32)

    return pol_rgb, pol_a, rim_rgb, rim_a


def plan_and_render_anime_hair(
    *,
    subj_rgb: np.ndarray,
    subj_alpha: np.ndarray,
    full_hair_m: np.ndarray,
    hair_m_orig: np.ndarray,
    face_snap: np.ndarray,
    eyes_m: np.ndarray,
    clothes_m: np.ndarray,
    arms_m: np.ndarray,
    acc_m: np.ndarray,
    all_skin_snap: np.ndarray,
    occ_snap: np.ndarray,
    veil_guard: np.ndarray,
    head_veil_guard: np.ndarray,
    veil_y_max: int,
    tone_map: dict[str, np.ndarray],
    lighting: dict[str, Any],
    style: str,
    boldness: str,
    wind_dir: str,
    tx_map: np.ndarray,
    ty_map: np.ndarray,
    swatch: dict[str, Any],
    bg_rgb: np.ndarray | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Plan & render v11 true strand-by-strand multi-wisp fibers with open air gaps & Light Wrap."""
    h, w = subj_alpha.shape
    rng = np.random.default_rng(seed + 111)

    hair_bin = (full_hair_m >= 0.45) & (subj_alpha >= 0.50) & (veil_guard < 0.25)
    ys_h, xs_h = np.where(hair_bin)
    if len(xs_h) < 100:
        z_rgb = np.zeros((h, w, 3), dtype=np.float32)
        z_a = np.zeros((h, w), dtype=np.float32)
        return {
            "behind": (z_rgb, z_a),
            "mid_connect": (z_rgb, z_a),
            "bright_weave": (z_rgb, z_a),
            "front_guard": np.zeros((h, w), np.float32),
            "eyes_guard": np.zeros((h, w), np.float32),
            "face_box": (0, 0, w // 2, h // 2),
            "resolved_style": style,
            "wx_sign": -1.0,
            "counts": {},
            "median_lock_alpha": 1.0,
            "painted_on_veil_px": 0,
            "overlap_delta_e": 0.0,
            "ahoge_root_x": -1,
        }

    fys, fxs = np.where(face_snap >= 0.45)
    if len(fxs) >= 150:
        fx0, fx1, fy0, fy1 = int(fxs.min()), int(fxs.max()), int(fys.min()), int(fys.max())
    else:
        fx0, fx1, fy0, fy1 = w // 2 - 90, w // 2 + 90, h // 4, h // 4 + 220
    fcx = 0.5 * (fx0 + fx1)
    fw = max(48.0, float(fx1 - fx0))
    fh = max(48.0, float(fy1 - fy0))

    hy0 = int(ys_h.min())
    if wind_dir == "auto":
        wx_sign = -1.0 if int((xs_h < fcx).sum()) >= int((xs_h >= fcx).sum()) else 1.0
    elif wind_dir in ("left", "down-left"):
        wx_sign = -1.0
    else:
        wx_sign = 1.0

    eyes_guard = (
        cv2.GaussianBlur(cv2.dilate((eyes_m >= 0.20).astype(np.uint8), ell_kernel(18)).astype(np.float32), (0, 0), 2.5)
        if eyes_m.max() > 0.05
        else np.zeros((h, w), dtype=np.float32)
    )
    yy_f, xx_f = np.mgrid[0:h, 0:w].astype(np.float32)
    ell = ((xx_f - fcx) / (0.39 * fw)) ** 2 + ((yy_f - (fy0 + 0.55 * fh)) / (0.45 * fh)) ** 2
    face_protect_center = np.clip((1.06 - ell) / 0.18, 0.0, 1.0)
    arms_flush = cv2.GaussianBlur((arms_m >= 0.45).astype(np.float32), (0, 0), 0.9) if arms_m.max() > 0.05 else np.zeros((h, w), np.float32)
    clo_flush = cv2.GaussianBlur((clothes_m >= 0.45).astype(np.float32), (0, 0), 0.9) if clothes_m.max() > 0.05 else np.zeros((h, w), np.float32)
    occ_nonface = occ_snap * (1.0 - np.clip(face_snap * 1.4, 0.0, 1.0))
    occ_soft = cv2.GaussianBlur(np.clip((occ_nonface - 0.35) / 0.45, 0.0, 1.0), (0, 0), 2.5)

    veil_hard = np.clip(veil_guard * 2.35, 0.0, 1.0)
    head_veil_hard = np.clip(head_veil_guard * 2.35, 0.0, 1.0)
    skin_dil = cv2.GaussianBlur(cv2.dilate((all_skin_snap >= 0.30).astype(np.uint8), ell_kernel(4)).astype(np.float32), (0, 0), 1.2) if all_skin_snap.max() > 0.05 else np.zeros((h, w), np.float32)
    arms_dil = cv2.GaussianBlur(cv2.dilate((arms_m >= 0.30).astype(np.uint8), ell_kernel(4)).astype(np.float32), (0, 0), 1.2) if arms_m.max() > 0.05 else np.zeros((h, w), np.float32)
    behind_guard = np.clip(np.maximum.reduce([eyes_guard, face_protect_center, head_veil_hard]), 0.0, 1.0)
    if style == "long-cascade" and clothes_m.max() > 0.05:
        d_clo = cv2.distanceTransform((clothes_m < 0.25).astype(np.uint8), cv2.DIST_L2, 5)
        clo_mid = (1.0 - _smoothstep(4.0, 22.0, d_clo)).astype(np.float32)
    else:
        clo_mid = clo_flush
    mid_guard = np.clip(np.maximum.reduce([eyes_guard, face_protect_center, arms_dil, skin_dil, clo_mid, occ_soft, veil_hard]), 0.0, 1.0)
    front_guard = np.clip(np.maximum.reduce([eyes_guard, face_protect_center, arms_dil, skin_dil, clo_flush, occ_soft, veil_hard]), 0.0, 1.0)

    key_side = str(lighting.get("key_side", "right"))
    key_dir_xy = (0.85, -0.52) if key_side == "right" else (-0.85, -0.52)
    rim_col = np.array(lighting["rim_color"], dtype=np.float32) * 255.0

    # Measure real wig brightness range so added filaments NEVER drop into dark muddy ochre/cyan
    hair_vals = subj_rgb[hair_bin].astype(np.float32)
    real_p15 = np.percentile(hair_vals, 15, axis=0).astype(np.float32)
    real_p85 = np.percentile(hair_vals, 85, axis=0).astype(np.float32)

    beh_pm = np.zeros((h, w, 3), dtype=np.float32)
    beh_a = np.zeros((h, w), dtype=np.float32)
    mid_pm = np.zeros((h, w, 3), dtype=np.float32)
    mid_a = np.zeros((h, w), dtype=np.float32)
    frt_pm = np.zeros((h, w, 3), dtype=np.float32)
    frt_a = np.zeros((h, w), dtype=np.float32)

    counts = {"crown_flick": 0, "hero_under": 0, "hero_main": 0, "strays": 0, "bridges": 0}
    ahoge_root_x = -1

    if style == "long-cascade":
        # ROSE: 3 rich +25% golden cascades with organic micro-clusters, stopping cleanly above skirt corner
        y_casc_min = max(veil_y_max + 35, int(fy1 + 1.48 * fh))
        casc_pts_y, casc_pts_x = np.where(hair_bin & (np.arange(h)[:, None] >= y_casc_min) & (np.arange(w)[None, :] < fcx))
        warm_ivory = np.array([254.0, 238.0, 206.0], dtype=np.float32)
        if len(casc_pts_x) > 50:
            y_hi_end = min(float(casc_pts_y.max()) - 85.0, float(y_casc_min + 295.0))
            y_targets = np.linspace(float(casc_pts_y.min()) + 35.0, max(float(casc_pts_y.min()) + 120.0, y_hi_end), 3)
            stream_widths = [44.0, 48.0, 38.0]
            glides = [195.0, 238.0, 262.0]
            drops = [215.0, 185.0, 150.0]
            for idx_l, (ya, w_stream, glide_x, drop_y) in enumerate(zip(y_targets, stream_widths, glides, drops)):
                iy = int(np.clip(round(ya), 0, h - 1))
                row_x = casc_pts_x[np.abs(casc_pts_y - iy) <= 22]
                if len(row_x) < 5:
                    row_x = np.where(hair_bin[iy, :])[0]
                if len(row_x) < 5:
                    continue
                ax = float(np.percentile(row_x, 38))
                ay = float(iy)
                ix, iyc = int(np.clip(round(ax), 0, w - 1)), int(np.clip(round(ay), 0, h - 1))
                tx0 = float(np.median(tx_map[max(0, iyc - 16):min(h, iyc + 17), max(0, ix - 16):min(w, ix + 17)]))
                ty0 = float(np.median(ty_map[max(0, iyc - 16):min(h, iyc + 17), max(0, ix - 16):min(w, ix + 17)]))
                if ty0 < 0.78:
                    tx0, ty0 = -0.32, 0.947
                n_t = math.hypot(tx0, ty0)
                tx0, ty0 = tx0 / n_t, ty0 / n_t
                if tx0 * wx_sign < 0.12:
                    tx0 = 0.28 * wx_sign
                    ty0 = math.sqrt(1.0 - tx0 * tx0)

                bx, by = ax + 88.0 * tx0, ay + 88.0 * ty0
                sway_var = float(rng.uniform(-12.0, 12.0))
                ctrl = np.array([
                    [ax - 64.0 * tx0, ay - 64.0 * ty0],
                    [ax, ay],
                    [bx + wx_sign * (0.22 * glide_x), by + 0.14 * drop_y],
                    [bx + wx_sign * (0.58 * glide_x), by + 0.42 * drop_y + sway_var],
                    [bx + wx_sign * (0.88 * glide_x), by + 0.74 * drop_y],
                    [bx + wx_sign * (1.04 * glide_x), by + drop_y],
                ], dtype=np.float64)
                p18, p52, p86 = _local_hair_palette(subj_rgb, hair_bin, ax, ay, tone_map["mid"])
                _, p52_rt, _ = _local_hair_palette(subj_rgb, hair_bin, float(ctrl[0, 0]), float(ctrl[0, 1]), p52)
                c_shd = np.clip(0.65 * p18 + 0.35 * p52, 24.0, 220.0)
                c_mid = np.clip(p52, 36.0, 228.0)
                c_hi = np.clip(0.85 * p86 + 0.15 * warm_ivory, 56.0, 242.0)
                c_tip = np.clip(0.72 * p86 + 0.28 * warm_ivory, 64.0, 244.0)

                n_em = _emit_layered_hair_stream(
                    beh_pm, beh_a, ctrl_xy=ctrl, stream_width=w_stream,
                    col_root=p52_rt, col_shd=c_shd, col_mid=c_mid, col_hi=c_hi, col_tip=c_tip, col_rim=warm_ivory,
                    key_dir_xy=key_dir_xy, rng=rng, n_wisps=4, strands_per_wisp=42,
                    root_veil_alpha=0.28, opacity_scale=0.98, root_fade_s=0.20, gap_spread=0.78,
                )
                counts["hero_main"] += 1
                counts["strays"] += n_em

                if idx_l == 1:
                    ctrl_fg = np.array([
                        [ax - 50.0 * tx0, ay - 50.0 * ty0],
                        [ax + wx_sign * 14.0, ay],
                        [bx + wx_sign * (0.30 * glide_x), by + 0.16 * drop_y],
                        [bx + wx_sign * (0.68 * glide_x), by + 0.46 * drop_y],
                        [bx + wx_sign * (1.02 * glide_x), by + 0.84 * drop_y],
                    ], dtype=np.float64)
                    counts["bridges"] += _emit_layered_hair_stream(
                        mid_pm, mid_a, ctrl_xy=ctrl_fg, stream_width=18.0,
                        col_root=p52_rt, col_shd=c_shd, col_mid=c_mid, col_hi=c_hi, col_tip=c_tip, col_rim=warm_ivory,
                        key_dir_xy=key_dir_xy, rng=rng, n_wisps=2, strands_per_wisp=26,
                        root_veil_alpha=0.0, opacity_scale=0.88, root_fade_s=0.36, gap_spread=0.76,
                    )

            right_pts_y, right_pts_x = np.where(hair_bin & (np.arange(h)[:, None] >= y_casc_min + 75) & (np.arange(w)[None, :] > fcx + 0.25 * fw))
            if len(right_pts_x) > 30:
                ay_r = float(np.percentile(right_pts_y, 60))
                r_xs = right_pts_x[np.abs(right_pts_y - ay_r) <= 20]
                ax_r = float(np.percentile(r_xs, 72)) if len(r_xs) else float(np.percentile(right_pts_x, 72))
                ctrl_r = np.array([
                    [ax_r - 24.0, ay_r - 56.0],
                    [ax_r, ay_r],
                    [ax_r + 36.0, ay_r + 48.0],
                    [ax_r + 74.0, ay_r + 104.0],
                    [ax_r + 92.0, ay_r + 148.0],
                ], dtype=np.float64)
                p18, p52, p86 = _local_hair_palette(subj_rgb, hair_bin, ax_r, ay_r, tone_map["mid"])
                c_shd = np.clip(0.65 * p18 + 0.35 * p52, 24.0, 220.0)
                c_mid = np.clip(p52, 36.0, 228.0)
                c_hi = np.clip(0.85 * p86 + 0.15 * warm_ivory, 56.0, 242.0)
                counts["strays"] += _emit_layered_hair_stream(
                    beh_pm, beh_a, ctrl_xy=ctrl_r, stream_width=24.0,
                    col_root=p52, col_shd=c_shd, col_mid=c_mid, col_hi=c_hi, col_tip=c_hi, col_rim=warm_ivory,
                    key_dir_xy=key_dir_xy, rng=rng, n_wisps=3, strands_per_wisp=28,
                    root_veil_alpha=0.18, opacity_scale=0.92, root_fade_s=0.22, gap_spread=0.76,
                )
                counts["hero_main"] += 1

    elif style == "twintails":
        # MIKU: Asymmetric Core + Diverging Mid + Outer Airy 3-tier streams (ZERO parallel aurora / mist, ZERO window spill!)
        y_tie = int(fy0 + 0.58 * fh)
        x_safe_min = max(2065.0, fcx - 2.65 * fw)
        x_safe_max = min(float(w - 150), fcx + 3.55 * fw)
        sunset_peach = np.array([255.0, 222.0, 196.0], dtype=np.float32)

        l_pts_y, l_pts_x = np.where(hair_bin & (np.arange(h)[:, None] >= y_tie + 40) & (np.arange(w)[None, :] < fcx - 0.35 * fw))
        l_origin_y = float(fy1 + 0.56 * fh)
        l_orig_cands = l_pts_x[np.abs(l_pts_y - l_origin_y) <= 28]
        l_origin_x = float(np.median(l_orig_cands)) if len(l_orig_cands) >= 8 else float(fcx - 1.18 * fw)
        l_origin_x = float(np.clip(l_origin_x, x_safe_min + 140.0, fcx - 0.88 * fw))
        l_ctrl_y0 = max(l_origin_y - 12.0, float(fy1 + 0.48 * fh))
        l_mid_x = float(np.percentile(l_pts_x, 42)) if len(l_pts_x) > 40 else (l_origin_x - 85.0)
        l_mid_y = float(np.median(l_pts_y)) if len(l_pts_y) > 40 else (fy1 + 2.1 * fh)
        l_end_y = float(max(l_mid_y + 290.0, fy1 + 3.85 * fh))

        p18_m, p52_m, p86_m = _local_hair_palette(subj_rgb, hair_bin, l_mid_x, l_mid_y, tone_map["mid"])
        c_shd_l = np.clip(0.62 * p18_m + 0.38 * p52_m, 34.0, 200.0)
        c_mid_l = np.clip(p52_m, 48.0, 214.0)
        c_hi_l = np.clip(0.88 * p86_m + 0.12 * sunset_peach, 78.0, 234.0)
        c_tip_l = np.clip(0.76 * p86_m + 0.24 * sunset_peach, 90.0, 240.0)

        # Left twintail: asymmetric Core (32px, 42spw) + Mid (24px, 30spw) + Outer airy stream (14px, 20spw)
        l_specs = [
            (-12.0, -48.0, -130.0, 32.0, 3, 42, 0.28, 0.96, 0.72, 0.0),
            (+10.0, -18.0, -76.0,  24.0, 2, 30, 0.16, 0.92, 0.80, -36.0),
            (-40.0, -104.0,-218.0, 14.0, 2, 20, 0.00, 0.88, 0.90, -52.0),
        ]
        for idx_l, (dx1, dx2, dx3, w_stream, n_w, spw, rva, ops, gsp, dy_off) in enumerate(l_specs):
            ox = float(np.clip(l_origin_x + dx1 * 0.28, x_safe_min + 120.0, fcx - 0.85 * fw))
            x1p = float(np.clip(l_mid_x + dx1, x_safe_min + 90.0, fcx - 0.45 * fw))
            x2p = float(np.clip(l_mid_x + dx2, x_safe_min + 50.0, fcx - 0.55 * fw))
            x3p = float(np.clip(l_mid_x + dx3, x_safe_min + 20.0, fcx - 0.70 * fw))
            ye = min(float(h - 90), l_end_y + dy_off + float(rng.uniform(-16.0, 24.0)))
            ctrl = np.array([
                [ox, l_ctrl_y0],
                [ox - 16.0, (l_ctrl_y0 + l_mid_y) * 0.52],
                [x1p, l_mid_y],
                [x2p, l_mid_y + 0.52 * (ye - l_mid_y)],
                [x3p, ye],
            ], dtype=np.float64)
            _, p52_rt, _ = _local_hair_palette(subj_rgb, hair_bin, float(ctrl[0, 0]), float(ctrl[0, 1]), c_mid_l)
            counts["strays"] += _emit_layered_hair_stream(
                beh_pm, beh_a, ctrl_xy=ctrl, stream_width=w_stream,
                col_root=p52_rt, col_shd=c_shd_l, col_mid=c_mid_l, col_hi=c_hi_l, col_tip=c_tip_l, col_rim=sunset_peach,
                key_dir_xy=key_dir_xy, rng=rng, n_wisps=n_w, strands_per_wisp=spw,
                root_veil_alpha=rva, opacity_scale=ops, root_fade_s=0.16, gap_spread=gsp,
            )
            counts["hero_main"] += 1

        # Right twintail v11.5: dense multi-fiber opaque inner belly core (closes bokeh bleed!) + countable 1-1.5px discrete outer strands
        r_origin_x = fcx + 0.65 * fw
        r_origin_y = float(y_tie - 15)
        _, p52_r0, _ = _local_hair_palette(subj_rgb, hair_bin, r_origin_x, r_origin_y + 20.0, c_mid_l)
        c_mid_r = np.clip(0.95 * c_mid_l + 0.05 * sunset_peach, 52.0, 218.0)
        c_shd_r = np.clip(0.72 * p18_m + 0.28 * p52_m, 28.0, 194.0)
        c_hi_r = np.clip(0.86 * p86_m + 0.14 * sunset_peach, 82.0, 236.0)
        c_tip_r = np.clip(0.72 * p86_m + 0.28 * sunset_peach, 94.0, 242.0)

        # 0. Full-Length Inner Fibrous Core Belly (92 dense mint filaments down through S-bend -> closes lower bokeh bleed while keeping 1:1 fibrous texture!)
        ctrl_r_belly = np.array([
            [3055.0, 945.0],
            [3155.0, 1020.0],
            [3238.0, 1148.0],
            [3228.0, 1340.0],
            [3312.0, float(min(h - 150, 1485.0))],
        ], dtype=np.float64)
        counts["strays"] += _emit_strand_wisp(
            beh_pm, beh_a, ctrl_xy=ctrl_r_belly, wisp_width=24.0,
            col_root=c_mid_r, col_shd=c_shd_r, col_mid=c_mid_r, col_hi=c_hi_r, col_tip=c_mid_r, col_rim=sunset_peach,
            key_dir_xy=key_dir_xy, rng=rng, n_strands=92, root_veil_alpha=0.14,
            opacity_scale=0.99, root_fade_s=0.10, tip_split_fan=0.28, micro_weave_px=3.8,
        )
        # 0b. Lower S-Bend Fibrous Reinforcer (extends to y=1565 so y=1300..1450 sits inside ts=0.35..0.74 -> opaque fiber core, airy tips!)
        ctrl_r_belly_lo = np.array([
            [3218.0, 1140.0],
            [3228.0, 1335.0],
            [3280.0, 1445.0],
            [3345.0, float(min(h - 120, 1565.0))],
        ], dtype=np.float64)
        counts["strays"] += _emit_strand_wisp(
            beh_pm, beh_a, ctrl_xy=ctrl_r_belly_lo, wisp_width=24.0,
            col_root=c_mid_r, col_shd=c_shd_r, col_mid=c_mid_r, col_hi=c_hi_r, col_tip=c_tip_r, col_rim=sunset_peach,
            key_dir_xy=key_dir_xy, rng=rng, n_strands=96, root_veil_alpha=0.0,
            opacity_scale=0.99, root_fade_s=0.20, tip_split_fan=0.34, micro_weave_px=3.6,
        )

        # 1. Dominant Primary Organic Wisp (wisp_width=30, n_strands=54)
        ctrl_r_main = np.array([
            [3055.0, 945.0],
            [3155.0, 1020.0],
            [3240.0, 1150.0],
            [3225.0, 1340.0],
            [3348.0, float(min(h - 115, 1575.0))],
        ], dtype=np.float64)
        counts["strays"] += _emit_strand_wisp(
            beh_pm, beh_a, ctrl_xy=ctrl_r_main, wisp_width=30.0,
            col_root=c_mid_r, col_shd=c_shd_r, col_mid=c_mid_r, col_hi=c_hi_r, col_tip=c_tip_r, col_rim=sunset_peach,
            key_dir_xy=key_dir_xy, rng=rng, n_strands=54, root_veil_alpha=0.0,
            opacity_scale=0.98, root_fade_s=0.12, tip_split_fan=0.54, micro_weave_px=4.6,
        )
        counts["hero_main"] += 1

        # 2. Hugging Shorter Secondary Wisp (wisp_width=18, n_strands=30)
        ctrl_r_sec = np.array([
            [3062.0, 952.0],
            [3165.0, 1028.0],
            [3258.0, 1142.0],
            [3322.0, 1285.0],
            [3370.0, float(min(h - 160, 1395.0))],
        ], dtype=np.float64)
        counts["strays"] += _emit_strand_wisp(
            beh_pm, beh_a, ctrl_xy=ctrl_r_sec, wisp_width=18.0,
            col_root=c_mid_r, col_shd=c_shd_r, col_mid=c_mid_r, col_hi=c_hi_r, col_tip=c_tip_r, col_rim=sunset_peach,
            key_dir_xy=key_dir_xy, rng=rng, n_strands=30, root_veil_alpha=0.0,
            opacity_scale=0.94, root_fade_s=0.14, tip_split_fan=0.58, micro_weave_px=4.2,
        )
        counts["hero_under"] += 1

        # 3. 28 Explicit Discrete Countable 1.0-1.5px Independent Outer & Gap Strands
        pts_m, _, norm_m = _catmull_rom_2d(ctrl_r_main, n_samples=56)
        pts_s, _, norm_s = _catmull_rom_2d(ctrl_r_sec, n_samples=48)
        for fi in range(28):
            use_sec = (fi >= 16)
            pts_p, norm_p = (pts_s, norm_s) if use_sec else (pts_m, norm_m)
            t_stop_f = float(rng.uniform(0.45, 1.00))
            m_idx = max(10, int(round(t_stop_f * len(pts_p))))
            ts_f = np.linspace(0.0, 1.0, m_idx, dtype=np.float64)
            lat_f = float(rng.uniform(-12.0, 26.0)) if use_sec else float(rng.uniform(-28.0, 32.0))
            freq_f = float(rng.uniform(1.0, 2.3))
            ph_f = float(rng.uniform(0.0, 2.0 * math.pi))
            sway_f = float(rng.uniform(3.5, 9.5)) * np.sin(freq_f * math.pi * ts_f + ph_f) * np.sin(math.pi * ts_f)
            tip_sgn = math.copysign(1.0, lat_f) if abs(lat_f) > 2.0 else (1.0 if fi % 2 == 0 else -1.0)
            out_drift = tip_sgn * float(rng.uniform(6.0, 18.0)) * _smoothstep(0.32, 1.0, ts_f)
            pts_f = pts_p[:m_idx] + (lat_f * (0.45 + 0.55 * _smoothstep(0.0, 0.35, ts_f)) + sway_f + out_drift)[:, None] * norm_p[:m_idx]

            r_pk = float(rng.uniform(0.48, 0.76))
            rad_f = np.maximum(0.24, r_pk * (1.0 - 0.62 * _smoothstep(0.22, 1.0, ts_f))).astype(np.float32)
            op_f = float(rng.uniform(0.68, 0.94))
            alf_f = np.clip(op_f * _smoothstep(0.0, 0.14, ts_f) * (1.0 - _smoothstep(0.68, 1.0, ts_f)), 0.0, 0.96).astype(np.float32)

            role_f = ("shadow" if fi % 5 == 0 else ("bright" if fi % 3 == 0 else "mid"))
            c_body_f = c_shd_r if role_f == "shadow" else (c_hi_r if role_f == "bright" else c_mid_r)
            cols_f = _strand_color_profile(c_mid_r, c_body_f, c_tip_r, sunset_peach, ts_f, norm_p[:m_idx], key_dir_xy, rng, role=role_f)
            _rasterize_analytical_strand(beh_pm, beh_a, pts=pts_f, radii=rad_f, alphas=alf_f, colors_rgb=cols_f, aa_half=0.62)
            counts["strays"] += 1

    else:
        # NAZUNA v11.5: Multi-fiber crescent ahoge + visible cheek/braid framing wisps (ZERO thick smooth plastic antennae!)
        silver_rim = np.array([234.0, 243.0, 255.0], dtype=np.float32)

        # 0. Graceful asymmetric multi-fiber crescent ahoge (16 fine 0.45-0.85px fibers + 2 hairline offshoots)
        mid_x0 = max(0, int(round(fcx - 0.10 * fw)))
        mid_x1 = min(w, int(round(fcx + 0.10 * fw)))
        mid_band = (full_hair_m[max(0, hy0) : fy0, mid_x0:mid_x1] >= 0.45) & (subj_alpha[max(0, hy0) : fy0, mid_x0:mid_x1] >= 0.55)
        my_pts, mx_pts = np.where(mid_band)
        if len(my_pts) >= 10:
            min_y_val = int(my_pts.min())
            cx_top = float(np.median(mx_pts[my_pts <= min_y_val + 4])) + float(mid_x0)
            cy_top = float(hy0 + min_y_val)
            ahoge_root_x = int(round(cx_top))
            _, p52_a, p86_a = _local_hair_palette(subj_rgb, hair_bin, cx_top, cy_top + 14.0, tone_map["mid"])
            ctrl_ah = np.array([
                [cx_top - 4.0, cy_top + 22.0],
                [cx_top, cy_top + 4.0],
                [cx_top + 8.0, cy_top - 14.0],
                [cx_top + 24.0, cy_top - 28.0],
                [cx_top + 40.0, cy_top - 22.0],
            ], dtype=np.float64)
            counts["strays"] += _emit_strand_wisp(
                mid_pm, mid_a, ctrl_xy=ctrl_ah, wisp_width=4.6,
                col_root=p52_a, col_shd=p52_a, col_mid=p86_a, col_hi=silver_rim, col_tip=silver_rim, col_rim=silver_rim,
                key_dir_xy=key_dir_xy, rng=rng, n_strands=16, root_veil_alpha=0.0,
                opacity_scale=0.94, root_fade_s=0.34, tip_split_fan=0.55, micro_weave_px=1.2,
            )
            pts_a, _, norm_a = _catmull_rom_2d(ctrl_ah, n_samples=24)
            ts_a = np.linspace(0.0, 1.0, 24, dtype=np.float64)
            for off_sgn, off_amp in ((-1.0, 4.2), (1.0, 5.2)):
                pts_sub = pts_a + (off_sgn * off_amp * _smoothstep(0.45, 1.0, ts_a))[:, None] * norm_a
                rad_sub = (0.45 * (1.0 - 0.55 * ts_a)).astype(np.float32)
                alf_sub = (0.72 * _smoothstep(0.32, 0.55, ts_a) * (1.0 - _smoothstep(0.68, 1.0, ts_a))).astype(np.float32)
                col_sub = _strand_color_profile(p52_a, p86_a, silver_rim, silver_rim, ts_a, norm_a, key_dir_xy, rng, role="bright")
                _rasterize_analytical_strand(mid_pm, mid_a, pts=pts_sub, radii=rad_sub, alphas=alf_sub, colors_rgb=col_sub, aa_half=0.58)
            counts["crown_flick"] += 1

        # 1. Moonlit crown & twin-bun perimeter asymmetric lens-resolved wisps (18 crisp 0.9-1.45px strands)
        crown_mask = (full_hair_m >= 0.45) & (subj_alpha >= 0.50) & (np.arange(h)[:, None] < int(fy0 + 0.15 * fh))
        c_pts_y, c_pts_x = np.where(crown_mask)
        if len(c_pts_x) > 40:
            edge_crown = (cv2.dilate(crown_mask.astype(np.uint8), ell_kernel(3)) > 0) & (~crown_mask)
            ey_c, ex_c = np.where(edge_crown)
            if len(ex_c) >= 24:
                left_bias = np.where(ex_c < fcx + 0.25 * fw)[0]
                pool = np.concatenate([left_bias, np.arange(len(ex_c))]) if len(left_bias) >= 10 else np.arange(len(ex_c))
                pick_idx = pool[np.linspace(0, len(pool) - 1, 18, dtype=int)]
                for pi in pick_idx:
                    ex0, ey0 = float(ex_c[pi]), float(ey_c[pi])
                    dx_n = ex0 - fcx
                    dy_n = ey0 - (fy0 + 0.10 * fh)
                    d_len = max(1.0, math.hypot(dx_n, dy_n))
                    nx_o, ny_o = dx_n / d_len, dy_n / d_len
                    arc_len_f = float(rng.uniform(16.0, 34.0))
                    bend = float(rng.uniform(-7.0, 9.0)) + wx_sign * 5.0
                    rx_root = ex0 - nx_o * 8.0
                    ry_root = ey0 - ny_o * 8.0
                    ctrl_fly = np.array([
                        [rx_root, ry_root],
                        [ex0 + nx_o * (0.28 * arc_len_f), ey0 + ny_o * (0.28 * arc_len_f)],
                        [ex0 + nx_o * (0.68 * arc_len_f) + bend * 0.4, ey0 + ny_o * (0.68 * arc_len_f) - 2.0],
                        [ex0 + nx_o * arc_len_f + bend, ey0 + ny_o * arc_len_f + float(rng.uniform(-2.0, 6.0))],
                    ], dtype=np.float64)
                    pts_f, _, norm_f = _catmull_rom_2d(ctrl_fly, n_samples=20)
                    ts_f = np.linspace(0.0, 1.0, 20, dtype=np.float64)
                    p18, p52, p86 = _local_hair_palette(subj_rgb, hair_bin, ex0, ey0, tone_map["mid"])
                    cols_f = _strand_color_profile(p52, p52, silver_rim, silver_rim, ts_f, norm_f, key_dir_xy, rng, role="mid")
                    rad_f = (float(rng.uniform(0.45, 0.72)) * (1.0 - 0.62 * ts_f)).astype(np.float32)
                    alf_f = (float(rng.uniform(0.48, 0.78)) * _smoothstep(0.0, 0.22, ts_f) * (1.0 - _smoothstep(0.58, 1.0, ts_f))).astype(np.float32)
                    _rasterize_analytical_strand(mid_pm, mid_a, pts=pts_f, radii=rad_f, alphas=alf_f, colors_rgb=cols_f, aa_half=0.58)
                    counts["crown_flick"] += 1

        # 2. Enriched left & right cheek/jaw side-lock micro-wisps (safely outside face/eyes guard)
        for s_dir, cmp_mask in ((-1.0, (np.arange(w)[None, :] < fcx - 0.30 * fw)), (1.0, (np.arange(w)[None, :] > fcx + 0.30 * fw))):
            cheek_band = hair_bin & cmp_mask & (np.arange(h)[:, None] >= fy0 + 0.55 * fh) & (np.arange(h)[:, None] <= fy1 + 0.15 * fh)
            cy_pts, cx_pts = np.where(cheek_band)
            if len(cx_pts) >= 20:
                ax_c = float(np.percentile(cx_pts, 45 if s_dir < 0 else 55))
                ay_c = float(np.median(cy_pts))
                ctrl_c = np.array([
                    [ax_c - s_dir * 6.0, ay_c - 18.0],
                    [ax_c, ay_c],
                    [ax_c + s_dir * 16.0, ay_c + 22.0],
                    [ax_c + s_dir * 28.0, ay_c + 44.0],
                ], dtype=np.float64)
                p18_c, p52_c, p86_c = _local_hair_palette(subj_rgb, hair_bin, ax_c, ay_c, tone_map["mid"])
                c_hi_c = np.clip(0.65 * p86_c + 0.35 * silver_rim, 64.0, 248.0)
                counts["strays"] += _emit_strand_wisp(
                    beh_pm, beh_a, ctrl_xy=ctrl_c, wisp_width=8.5,
                    col_root=p52_c, col_shd=p52_c, col_mid=p52_c, col_hi=c_hi_c, col_tip=silver_rim if s_dir < 0 else p86_c, col_rim=silver_rim,
                    key_dir_xy=key_dir_xy, rng=rng, n_strands=18, root_veil_alpha=0.0,
                    opacity_scale=0.94, root_fade_s=0.22, tip_split_fan=0.40, micro_weave_px=1.6,
                )

            # 3. 2 staggered short braid-end & outer-edge airy micro-wisps (40-54px, ZERO long tails!)
            ys_b, xs_b = np.where(hair_bin & (np.arange(h)[:, None] > fy0 + 0.25 * fh) & cmp_mask)
            if len(xs_b) >= 30:
                y_bot = float(np.percentile(ys_b, 92))
                tip_xs = xs_b[ys_b >= y_bot - 14]
                x_bot = float(np.median(tip_xs)) if len(tip_xs) else float(xs_b.mean())
                ax = x_bot - s_dir * 4.0
                ay = y_bot - 8.0
                ctrl_b1 = np.array([
                    [ax - s_dir * 10.0, ay - 30.0],
                    [ax, ay - 4.0],
                    [ax + s_dir * 30.0, ay + 10.0],
                    [ax + s_dir * 52.0, ay - 12.0],
                ], dtype=np.float64)
                ctrl_b2 = np.array([
                    [ax - s_dir * 6.0, ay - 22.0],
                    [ax + s_dir * 4.0, ay + 2.0],
                    [ax + s_dir * 24.0, ay + 24.0],
                    [ax + s_dir * 38.0, ay + 16.0],
                ], dtype=np.float64)
                p18, p52, p86 = _local_hair_palette(subj_rgb, hair_bin, ax, ay, tone_map["mid"])
                c_shd_n = np.clip(0.55 * np.maximum(p18, real_p15) + 0.45 * p52, 24.0, 236.0)
                c_hi_n = np.clip(0.65 * p86 + 0.35 * silver_rim, 64.0, 248.0)
                counts["strays"] += _emit_strand_wisp(
                    beh_pm, beh_a, ctrl_xy=ctrl_b1, wisp_width=11.5,
                    col_root=p52, col_shd=c_shd_n, col_mid=p52, col_hi=c_hi_n, col_tip=silver_rim if s_dir < 0 else p86, col_rim=silver_rim,
                    key_dir_xy=key_dir_xy, rng=rng, n_strands=24, root_veil_alpha=0.18 if s_dir < 0 else 0.0,
                    opacity_scale=0.94, root_fade_s=0.22, tip_split_fan=0.44, micro_weave_px=1.6,
                )
                counts["strays"] += _emit_strand_wisp(
                    beh_pm, beh_a, ctrl_xy=ctrl_b2, wisp_width=7.5,
                    col_root=p52, col_shd=c_shd_n, col_mid=p52, col_hi=c_hi_n, col_tip=silver_rim if s_dir < 0 else p86, col_rim=silver_rim,
                    key_dir_xy=key_dir_xy, rng=rng, n_strands=16, root_veil_alpha=0.0,
                    opacity_scale=0.88, root_fade_s=0.22, tip_split_fan=0.42, micro_weave_px=1.4,
                )
                counts["hero_main"] += 1

    beh_rgb, beh_a = _finalize_canvas(beh_pm, beh_a, psf_sigma=0.22)
    mid_rgb, mid_a = _finalize_canvas(mid_pm, mid_a, psf_sigma=0.36 if style == "short-braids-wispy" else 0.20)
    frt_rgb, frt_a = _finalize_canvas(frt_pm, frt_a, psf_sigma=0.18)

    beh_rgb, beh_a = _apply_optical_dof(beh_rgb, beh_a, subj_alpha, fy1, fh, style)
    if float(mid_a.max()) > 1e-4 and style != "short-braids-wispy":
        mid_rgb, mid_a = _apply_optical_dof(mid_rgb, mid_a, subj_alpha, fy1, fh, style)
    if float(frt_a.max()) > 1e-4:
        frt_rgb, frt_a = _apply_optical_dof(frt_rgb, frt_a, subj_alpha, fy1, fh, style)

    beh_rgb, beh_a = _apply_backlight_lightwrap(beh_rgb, beh_a, subj_alpha, bg_rgb, real_p15, real_p85, style)
    if float(mid_a.max()) > 1e-4:
        mid_rgb, mid_a = _apply_backlight_lightwrap(mid_rgb, mid_a, subj_alpha, bg_rgb, real_p15, real_p85, style)

    if style == "twintails":
        ear_tie_box = (np.arange(h)[:, None] >= 490) & (np.arange(h)[:, None] <= 715) & (np.arange(w)[None, :] >= 2440) & (np.arange(w)[None, :] <= 2660)
        behind_guard = np.maximum(behind_guard, cv2.GaussianBlur(ear_tie_box.astype(np.float32), (0, 0), 3.0))
        win_raw = ((np.arange(h)[:, None] < 762) & (np.arange(w)[None, :] >= 2888)).astype(np.uint8)
        win_dil = cv2.dilate(win_raw, ell_kernel(5)).astype(np.float32)
        behind_guard = np.maximum(behind_guard, cv2.GaussianBlur(win_dil, (0, 0), 2.5))
        gap_sliver_raw = ((np.arange(h)[:, None] >= 675) & (np.arange(h)[:, None] <= 945) & (np.arange(w)[None, :] >= 2855) & (np.arange(w)[None, :] <= 3060)).astype(np.uint8)
        gap_sliver_dil = cv2.dilate(gap_sliver_raw, ell_kernel(5)).astype(np.float32)
        behind_guard = np.maximum(behind_guard, cv2.GaussianBlur(gap_sliver_dil, (0, 0), 3.5))
        occ_edge = np.maximum(arms_m, clo_flush)
        arm_rib_edge = np.clip(cv2.GaussianBlur(occ_edge, (0, 0), 4.0) * (1.0 - occ_edge) * 3.0, 0.0, 1.0)
        wrist_box = ((np.arange(h)[:, None] >= 750) & (np.arange(h)[:, None] <= 1180) & (np.arange(w)[None, :] >= 2880) & (np.arange(w)[None, :] <= 3250)).astype(np.float32)
        sh_w = (0.20 * arm_rib_edge * wrist_box * (beh_a > 0.05).astype(np.float32))[:, :, None]
        beh_rgb = np.clip(beh_rgb * (1.0 - sh_w), 0.0, 255.0).astype(np.float32)

    beh_a = np.clip(beh_a * (1.0 - behind_guard), 0.0, 1.0)
    mid_a = np.clip(mid_a * (1.0 - mid_guard), 0.0, 1.0)
    frt_a = np.clip(frt_a * (1.0 - front_guard), 0.0, 1.0)

    any_lock_a = np.maximum.reduce([beh_a, mid_a])
    core_vals = any_lock_a[any_lock_a > 0.25]
    median_lock_alpha = round(float(np.median(core_vals)), 3) if len(core_vals) else 1.0
    any_painted = np.maximum.reduce([beh_a, mid_a, frt_a]) > 0.05
    painted_on_veil = int((any_painted & (head_veil_guard >= 0.45)).sum()) if head_veil_guard.max() > 0.05 else 0

    overlap = (any_lock_a > 0.35) & (full_hair_m >= 0.55) & (subj_alpha >= 0.65)
    overlap_de = 0.0
    if int(overlap.sum()) >= 50:
        paint_comp = np.where(mid_a[:, :, None] >= beh_a[:, :, None], mid_rgb, beh_rgb)
        lab_p = rgb255_to_lab(paint_comp)
        lab_s = rgb255_to_lab(subj_rgb)
        overlap_de = round(float(np.linalg.norm(np.median(lab_p[overlap], axis=0) - np.median(lab_s[overlap], axis=0))), 2)

    return {
        "behind": (beh_rgb, beh_a),
        "mid_connect": (mid_rgb, mid_a),
        "bright_weave": (frt_rgb, frt_a),
        "front_guard": front_guard,
        "eyes_guard": eyes_guard,
        "face_box": (fx0, fy0, fx1, fy1),
        "resolved_style": style,
        "wx_sign": wx_sign,
        "counts": counts,
        "median_lock_alpha": median_lock_alpha,
        "painted_on_veil_px": painted_on_veil,
        "overlap_delta_e": overlap_de,
        "ahoge_root_x": ahoge_root_x,
    }
