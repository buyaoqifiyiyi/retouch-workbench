"""Hair Sculpt Ops: Real-strand + Bold Anime-Flowing Hair Retouch Engine.

Combines:
  1. Zero-blur synthetic-wig cleanup + vectorized Rigid MLS crown lift & wind-sweep reshape
     (strictly 0.00px locked on face, eyes, arms, accessories, and garments).
  2. Anti-muddy real-hair Lab color sampling + scene key/rim/ambient (藏色) extraction.
  3. Topology-aware 3D anime ribbon lock planner (long-cascade, twintails, short-braids-wispy)
     with root-tangent locking and negative-space breathing windows.
  4. Blender 5.2 Cycles OptiX physical hair_curves rendering + root-overlap Lab calibration +
     0px flush occluder carve-out + camera sensor grain matching + PhotoCraft layer stack builder.
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
from PIL import Image

from workbench_ops import (
    decontaminate_fringe,
    ell_kernel,
    guided_filter_gray,
    lab_to_rgb255,
    luma255,
    read_gray_f32,
    read_rgba_f32,
    rgb255_to_lab,
    write_png_u8,
    write_rgba_u8,
)


def _J(obj: object) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray | float) -> np.ndarray | float:
    t = np.clip((x - edge0) / max(1e-6, edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _parse_hex_rgb01(hx: str | None) -> list[float] | None:
    if not hx:
        return None
    s = hx.strip().lstrip("#")
    if len(s) != 6:
        raise ValueError(f"Invalid hex color: {hx!r}")
    return [int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0, int(s[4:6], 16) / 255.0]


def _rgb255_to_hex(rgb: np.ndarray) -> str:
    r, g, b = [int(np.clip(round(float(v)), 0, 255)) for v in rgb[:3]]
    return f"#{r:02x}{g:02x}{b:02x}"


def _auto_find_mask(base_dir: Path, names: tuple[str, ...]) -> Path | None:
    search_dirs = [base_dir, base_dir.parent, base_dir.parent / "face_parse", base_dir.parent / "body_parse"]
    for d in search_dirs:
        if not d.exists():
            continue
        for nm in names:
            p = d / nm
            if p.exists():
                return p
    return None


def _resize_rgba_premul(path: Path, expected_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Read RGBA and resize premultiplied RGB+Alpha with bicubic interpolation (zero edge ringing/halos)."""
    rgb, alpha = read_rgba_f32(path)
    eh, ew = expected_hw
    if alpha.shape == (eh, ew):
        return rgb, alpha
    pm = rgb * alpha[:, :, None]
    pm_r = cv2.resize(pm, (ew, eh), interpolation=cv2.INTER_CUBIC)
    a_r = np.clip(cv2.resize(alpha, (ew, eh), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)
    rgb_r = np.where(a_r[:, :, None] > 1e-4, pm_r / np.maximum(a_r[:, :, None], 1e-4), 0.0)
    return np.clip(rgb_r, 0.0, 255.0), a_r


def _rigid_mls_displacement(
    h: int,
    w: int,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    grid_step: int = 24,
    alpha: float = 1.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute vectorized Rigid Moving Least Squares (Schaefer et al. 2006) displacement field (dx, dy)."""
    if len(src_pts) < 4:
        return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

    gy = np.linspace(0, h - 1, max(16, int(math.ceil(h / grid_step)) + 1), dtype=np.float32)
    gx = np.linspace(0, w - 1, max(16, int(math.ceil(w / grid_step)) + 1), dtype=np.float32)
    gxx, gyy = np.meshgrid(gx, gy)
    gh, gw = gxx.shape
    v = np.stack([gxx.ravel(), gyy.ravel()], axis=1).astype(np.float64)  # (N, 2)

    p = src_pts.astype(np.float64)  # (K, 2)
    q = dst_pts.astype(np.float64)  # (K, 2)

    diff = v[:, None, :] - p[None, :, :]  # (N, K, 2)
    d2 = (diff * diff).sum(axis=2) + 1.0  # (N, K)
    w_k = 1.0 / np.power(d2, alpha)  # (N, K)
    w_sum = np.maximum(w_k.sum(axis=1, keepdims=True), 1e-9)

    p_star = (w_k @ p) / w_sum  # (N, 2)
    q_star = (w_k @ q) / w_sum  # (N, 2)

    p_hat = p[None, :, :] - p_star[:, None, :]  # (N, K, 2)
    q_hat = q[None, :, :] - q_star[:, None, :]  # (N, K, 2)

    # Optimal 2D rigid rotation per query point via complex dot product sum(w * q_hat * conj(p_hat))
    a_re = (w_k * (q_hat[:, :, 0] * p_hat[:, :, 0] + q_hat[:, :, 1] * p_hat[:, :, 1])).sum(axis=1)
    b_im = (w_k * (q_hat[:, :, 1] * p_hat[:, :, 0] - q_hat[:, :, 0] * p_hat[:, :, 1])).sum(axis=1)
    norm_ab = np.maximum(np.hypot(a_re, b_im), 1e-9)
    cos_t = a_re / norm_ab
    sin_t = b_im / norm_ab

    vp = v - p_star  # (N, 2)
    rx = cos_t * vp[:, 0] - sin_t * vp[:, 1] + q_star[:, 0]
    ry = sin_t * vp[:, 0] + cos_t * vp[:, 1] + q_star[:, 1]

    dx_grid = (rx - v[:, 0]).reshape(gh, gw).astype(np.float32)
    dy_grid = (ry - v[:, 1]).reshape(gh, gw).astype(np.float32)

    dx_full = cv2.resize(dx_grid, (w, h), interpolation=cv2.INTER_CUBIC)
    dy_full = cv2.resize(dy_grid, (w, h), interpolation=cv2.INTER_CUBIC)
    return dx_full, dy_full


def _clean_wig_and_reshape(
    subj_rgb: np.ndarray,
    subj_alpha: np.ndarray,
    hair_m: np.ndarray,
    face_snap: np.ndarray,
    eyes_m: np.ndarray,
    clothes_m: np.ndarray,
    arms_m: np.ndarray,
    acc_m: np.ndarray,
    wx_sign: float,
    boldness: str,
    do_reshape: bool,
    neck_body_prot: np.ndarray | None = None,
    style: str = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Remove static wig frizz/trapped dust without blurring braids, then run safe Rigid MLS crown/tip reshape."""
    h, w = subj_alpha.shape
    out_rgb = subj_rgb.copy()
    out_alpha = subj_alpha.copy()
    out_hair = hair_m.copy()

    lab = rgb255_to_lab(out_rgb)
    L_lab, a_lab, b_lab = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    C_lab = np.hypot(a_lab, b_lab)

    body_ext = neck_body_prot if neck_body_prot is not None else np.zeros((h, w), np.float32)
    fys_w = np.where(face_snap >= 0.45)[0]
    y_bow_max = int(fys_w.max() + 0.08 * max(20, fys_w.max() - fys_w.min())) if len(fys_w) else int(0.40 * h)
    above_chin = np.arange(h)[:, None] <= y_bow_max
    white_acc = (out_alpha > 0.78) & (L_lab > 76.0) & (C_lab < 12.0) & (out_hair < 0.50) & above_chin
    prot_hard = (np.maximum.reduce([face_snap, clothes_m, arms_m, acc_m, body_ext]) >= 0.32) | white_acc

    # 1. Clean faint outer static/cobweb frizz & trapped neutral cutout haze strictly outside hair core
    hair_core = (out_hair > 0.58) & (out_alpha > 0.85) & (~prot_hard)
    if hair_core.any():
        fringe = np.clip((0.86 - out_alpha) / 0.82, 0.0, 1.0) * (out_alpha > 0.02) * cv2.GaussianBlur(out_hair, (0, 0), 3.0)
        out_rgb = decontaminate_fringe(out_rgb, hair_core, fringe * 0.82)

        # Suppress isolated low-alpha cobweb dust specks far from solid core
        d_core = cv2.distanceTransform((~hair_core).astype(np.uint8), cv2.DIST_L2, 5)
        cobweb = (
            (out_hair > 0.20)
            & (~prot_hard)
            & (out_alpha > 0.04)
            & (out_alpha < 0.42)
            & (d_core > 10.0)
            & (C_lab < 14.0)
        )
        if cobweb.any():
            cob_soft = cv2.GaussianBlur(cobweb.astype(np.float32), (0, 0), 0.9) * (1.0 - prot_hard.astype(np.float32))
            out_alpha = np.clip(out_alpha * (1.0 - 0.85 * cob_soft), 0.0, 1.0)

        # Mildly recover blown-out plastic specular hot-bands on synthetic wigs (keeps 100% micro-strands!)
        ref_h = hair_core & (L_lab >= 34.0) & (L_lab <= 76.0) & (C_lab >= 10.0)
        if int(ref_h.sum()) >= 400:
            a_med = float(np.median(a_lab[ref_h]))
            b_med = float(np.median(b_lab[ref_h]))
            L_sm = cv2.GaussianBlur(L_lab, (0, 0), 12.0)
            C_sm = np.hypot(cv2.GaussianBlur(a_lab, (0, 0), 12.0), cv2.GaussianBlur(b_lab, (0, 0), 12.0))
            w_plastic = (
                np.clip((0.5 * L_sm + 0.5 * L_lab - 72.0) / 14.0, 0.0, 1.0)
                * np.clip((15.0 - C_sm) / 8.0, 0.0, 1.0)
                * (out_hair > 0.55).astype(np.float32)
                * (1.0 - prot_hard.astype(np.float32))
            )
            if float(w_plastic.max()) > 0.05:
                lab_rec = lab.copy()
                lab_rec[:, :, 0] = np.clip(L_lab - w_plastic * 0.35 * np.maximum(0.0, L_lab - 68.0), 8.0, 98.0)
                lab_rec[:, :, 1] = a_lab + w_plastic * 0.50 * (a_med - a_lab)
                lab_rec[:, :, 2] = b_lab + w_plastic * 0.50 * (b_med - b_lab)
                rec_rgb = lab_to_rgb255(lab_rec)
                out_rgb = np.clip(out_rgb * (1.0 - w_plastic[:, :, None]) + rec_rgb * w_plastic[:, :, None], 0.0, 255.0)

    reshape_info: dict[str, Any] = {"reshaped": False, "crown_lift_px": 0.0, "tip_sweep_px": 0.0}
    if not do_reshape:
        return out_rgb, out_alpha, out_hair, reshape_info

    hair_bin = (out_hair >= 0.45) & (out_alpha >= 0.50) & (~prot_hard)
    ys_h, xs_h = np.where(hair_bin)
    if len(xs_h) < 600:
        return out_rgb, out_alpha, out_hair, reshape_info

    hx0, hx1 = int(xs_h.min()), int(xs_h.max())
    hy0, hy1 = int(ys_h.min()), int(ys_h.max())
    hair_span = float(max(hx1 - hx0, hy1 - hy0))

    # Build strict zero-displacement anchors on face, eyes, neck/chest skin, arms, accessories, and clothes
    prot_union = np.maximum.reduce([
        face_snap,
        cv2.dilate((eyes_m >= 0.20).astype(np.float32), ell_kernel(14)),
        arms_m,
        clothes_m,
        acc_m,
        body_ext,
        white_acc.astype(np.float32),
    ])
    prot_bin = (prot_union >= 0.26).astype(np.uint8)
    prot_edge = cv2.morphologyEx(prot_bin, cv2.MORPH_GRADIENT, ell_kernel(3))
    pys, pxs = np.where(prot_edge > 0)
    src_list: list[list[float]] = []
    dst_list: list[list[float]] = []

    if len(pxs) > 0:
        step_p = max(1, len(pxs) // 110)
        for px, py in zip(pxs[::step_p], pys[::step_p]):
            src_list.append([float(px), float(py)])
            dst_list.append([float(px), float(py)])

    # Also pin face center & neck/collar interior
    fys, fxs = np.where(face_snap >= 0.5)
    if len(fxs) > 0:
        fcx, fcy = float(np.mean(fxs)), float(np.mean(fys))
        fy0 = float(fys.min())
        fw = float(max(20, fxs.max() - fxs.min()))
        fh = float(max(20, fys.max() - fys.min()))
        src_list.append([fcx, fcy])
        dst_list.append([fcx, fcy])
    else:
        fcx, fy0 = 0.5 * (hx0 + hx1), float(hy0 + 0.18 * (hy1 - hy0))
        fw, fh = float(0.26 * (hx1 - hx0)), float(0.18 * (hy1 - hy0))

    d_prot = cv2.distanceTransform((prot_union < 0.22).astype(np.uint8), cv2.DIST_L2, 5)
    bold_mult = {"natural": 0.60, "medium": 1.0, "dramatic": 1.15}.get(boldness, 1.0)

    # Gentle outer crown lift only (<=18px so rigid MLS never tears background holes in inner hair)
    crown_lift = min(0.018 * hair_span * bold_mult, 18.0)
    crown_pts = 0
    for x_c in np.linspace(hx0 + 0.16 * (hx1 - hx0), hx1 - 0.16 * (hx1 - hx0), 12):
        ix = int(round(x_c))
        col_y = np.where(hair_bin[:, ix] & (np.arange(h) < fy0 + 0.20 * fh))[0]
        if len(col_y) > 0:
            iy = int(col_y[0]) + 8
            if iy < h and d_prot[iy, ix] >= 18.0:
                dx_c = (float(ix) - fcx) / max(1.0, 0.5 * (hx1 - hx0)) * 0.25 * crown_lift
                src_list.append([float(ix), float(iy)])
                dst_list.append([float(ix) + dx_c, float(iy) - crown_lift])
                crown_pts += 1

    # Gentle outer silhouette breeze-flow anchors along free real-hair blocks (softens rigid wigs safely!)
    tip_sweep = min(0.048 * hair_span * bold_mult, 48.0)
    is_twintail_mls = (style == "twintails")
    if style in ("auto", "wind-dramatic") and len(xs_h) > 600:
        yy_c, xx_c = np.mgrid[0:h, 0:w]
        low_m = hair_bin & (yy_c > fy0 + fh)
        lc = int((low_m & (xx_c < fcx - 0.55 * fw)).sum())
        rc = int((low_m & (xx_c > fcx + 0.55 * fw)).sum())
        cc = int((low_m & (np.abs(xx_c - fcx) <= 0.45 * fw)).sum())
        if lc > 1.4 * max(200, cc) and rc > 1.4 * max(200, cc):
            is_twintail_mls = True

    if is_twintail_mls:
        max_sil_sweep = min(0.042 * hair_span * bold_mult, 28.0)
        sides_scan = (-1.0, 1.0)
    elif style == "short-braids-wispy":
        max_sil_sweep = min(0.026 * hair_span * bold_mult, 14.0)
        sides_scan = (wx_sign,)
    else:
        max_sil_sweep = min(0.036 * hair_span * bold_mult, 36.0)
        sides_scan = (wx_sign,)

    tip_pts = 0
    for y_r in np.linspace(hy0 + 0.22 * (hy1 - hy0), hy1 - 0.06 * (hy1 - hy0), 12):
        iy = int(round(y_r))
        rel_y = float(np.clip((iy - (hy0 + 0.18 * (hy1 - hy0))) / max(20.0, 0.80 * (hy1 - hy0)), 0.0, 1.0))
        row_h = np.where(hair_bin[iy, :])[0]
        if len(row_h) < 12:
            continue
        for s_dir in sides_scan:
            if is_twintail_mls:
                cand = row_h[row_h < fcx] if s_dir < 0 else row_h[row_h > fcx]
                if len(cand) < 8:
                    continue
                ix_edge = int(cand[0] if s_dir < 0 else cand[-1])
            else:
                ix_edge = int(row_h[0] if s_dir < 0 else row_h[-1])
            ix = int(np.clip(ix_edge - s_dir * 8.0, 0, w - 1))
            if d_prot[iy, ix] >= 16.0:
                sw = float(s_dir * max_sil_sweep * (0.28 + 0.72 * (rel_y ** 1.35)) * (0.85 + 0.22 * math.sin(1.3 * math.pi * rel_y)))
                dy_s = -0.16 * abs(sw) * math.sin(math.pi * rel_y)
                src_list.append([float(ix), float(iy)])
                dst_list.append([float(ix) + sw, float(iy) + dy_s])
                tip_pts += 1

    if crown_pts + tip_pts >= 3 and len(src_list) >= 12:
        dx_f, dy_f = _rigid_mls_displacement(
            h, w, np.array(src_list, np.float32), np.array(dst_list, np.float32), grid_step=28, alpha=1.45
        )
        # Hard mathematical mask: ZERO displacement inside protected body/neck/face/clothes/arms/accessories!
        prot_lock = cv2.GaussianBlur(
            cv2.dilate((prot_union >= 0.20).astype(np.uint8), ell_kernel(12)).astype(np.float32),
            (0, 0),
            sigmaX=10.0,
        )
        hair_zone = cv2.GaussianBlur(
            cv2.dilate(hair_bin.astype(np.uint8), ell_kernel(22)).astype(np.float32),
            (0, 0),
            sigmaX=10.0,
        )
        allow_w = np.clip((1.0 - prot_lock) * hair_zone, 0.0, 1.0)
        dx_f *= allow_w
        dy_f *= allow_w

        yy_g, xx_g = np.mgrid[0:h, 0:w].astype(np.float32)
        map_x = np.clip(xx_g - dx_f, 0.0, float(w - 1))
        map_y = np.clip(yy_g - dy_f, 0.0, float(h - 1))
        pm_rgb = out_rgb * out_alpha[:, :, None]
        w_pm = cv2.remap(pm_rgb, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        w_a = np.clip(cv2.remap(out_alpha, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0), 0.0, 1.0)
        w_hm = np.clip(cv2.remap(out_hair, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0), 0.0, 1.0)
        w_rgb = np.where(w_a[:, :, None] > 1e-4, w_pm / np.maximum(w_a[:, :, None], 1e-4), out_rgb)

        out_rgb = np.clip(out_rgb * prot_lock[:, :, None] + w_rgb * (1.0 - prot_lock[:, :, None]), 0.0, 255.0)
        out_alpha = np.clip(out_alpha * prot_lock + w_a * (1.0 - prot_lock), 0.0, 1.0)
        out_hair = np.clip(out_hair * prot_lock + w_hm * (1.0 - prot_lock), 0.0, 1.0)
        reshape_info = {
            "reshaped": True,
            "crown_lift_px": round(float(crown_lift), 1),
            "tip_sweep_px": round(float(max_sil_sweep), 1),
            "anchors_active": crown_pts + tip_pts,
            "anchors_locked": len(src_list) - (crown_pts + tip_pts),
        }

    return out_rgb, out_alpha, out_hair, reshape_info


def _feather_blunt_wig_tips(
    subj_rgb: np.ndarray,
    subj_alpha: np.ndarray,
    hair_m: np.ndarray,
    face_snap: np.ndarray,
    prot_union: np.ndarray,
    tx_map: np.ndarray,
    ty_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Tutorial @睡颜 (03:52) directional micro-smudge feathering on blunt lower outer wig tips."""
    h, w = subj_alpha.shape
    px_scale = max(w, h) / 3840.0
    hair_bin = (hair_m >= 0.45) & (subj_alpha >= 0.35)
    if int(hair_bin.sum()) < 400:
        return subj_rgb, subj_alpha, 0

    fys_f = np.where(face_snap >= 0.45)[0]
    y_min_tip = int(fys_f.min() + 0.50 * max(20, fys_f.max() - fys_f.min())) if len(fys_f) else int(0.28 * h)

    prot_safe = cv2.dilate((prot_union >= 0.25).astype(np.uint8), ell_kernel(14)) > 0
    d_prot = cv2.distanceTransform((~prot_safe).astype(np.uint8), cv2.DIST_L2, 5)
    d_inner = cv2.distanceTransform((subj_alpha >= 0.50).astype(np.uint8), cv2.DIST_L2, 5)
    edge_grad = cv2.morphologyEx(subj_alpha.astype(np.float32), cv2.MORPH_GRADIENT, ell_kernel(3))

    # Strictly target sharp/blunt outer lower wig boundary edges clear of face, neck, clothes & arms
    blunt_band = (
        (cv2.dilate(hair_bin.astype(np.uint8), ell_kernel(8)) > 0)
        & (d_inner <= 12.0 * px_scale)
        & (d_prot >= 22.0 * px_scale)
        & (np.arange(h)[:, None] >= y_min_tip)
        & (cv2.GaussianBlur((edge_grad > 0.14).astype(np.float32), (0, 0), 3.0) > 0.18)
    )
    feathered_px = int(blunt_band.sum())
    if feathered_px < 40:
        return subj_rgb, subj_alpha, 0

    # Directional micro-pull along strand tangents modulated by subtle perpendicular streak noise
    yy_g, xx_g = np.mgrid[0:h, 0:w].astype(np.float32)
    tx_s = cv2.GaussianBlur(tx_map, (0, 0), 4.0)
    ty_s = cv2.GaussianBlur(ty_map, (0, 0), 4.0)
    tn = np.maximum(np.hypot(tx_s, ty_s), 1e-4)
    tx_s /= tn
    ty_s /= tn

    rng = np.random.default_rng(707)
    streak = rng.uniform(0.65, 1.0, size=(max(8, h // 4), max(8, w // 4))).astype(np.float32)
    streak = cv2.resize(streak, (w, h), interpolation=cv2.INTER_LINEAR)
    streak = cv2.GaussianBlur(streak, (0, 0), 1.6)

    pm_in = subj_rgb * subj_alpha[:, :, None]
    pm_acc = np.zeros_like(pm_in)
    a_acc = np.zeros_like(subj_alpha)
    w_tot = 0.0
    for step, wt in ((2.5, 0.40), (5.5, 0.30), (9.0, 0.20), (12.5, 0.10)):
        pull = step * px_scale * streak
        mx = np.clip(xx_g - tx_s * pull, 0.0, float(w - 1))
        my = np.clip(yy_g - ty_s * pull, 0.0, float(h - 1))
        pm_acc += wt * cv2.remap(pm_in, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        a_acc += wt * cv2.remap(subj_alpha, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        w_tot += wt
    pm_acc /= max(1e-4, w_tot)
    a_acc = np.clip(a_acc / max(1e-4, w_tot), 0.0, 1.0)
    rgb_smudge = np.where(a_acc[:, :, None] > 1e-4, pm_acc / np.maximum(a_acc[:, :, None], 1e-4), subj_rgb)

    feather_w = cv2.GaussianBlur(blunt_band.astype(np.float32), (0, 0), 2.4) * (1.0 - np.clip(prot_safe.astype(np.float32), 0.0, 1.0)) * 0.58
    out_rgb = np.clip(subj_rgb * (1.0 - feather_w[:, :, None]) + rgb_smudge * feather_w[:, :, None], 0.0, 255.0)
    out_a = np.clip(np.maximum(subj_alpha * (1.0 - 0.45 * feather_w), subj_alpha * (1.0 - feather_w) + a_acc * feather_w), 0.0, 1.0)
    return out_rgb.astype(np.float32), out_a.astype(np.float32), feathered_px


def _sample_hair_palette(
    subj_rgb: np.ndarray,
    subj_alpha: np.ndarray,
    hair_m: np.ndarray,
    prot_union: np.ndarray,
    rim_col01: list[float] | None = None,
    amb_col01: list[float] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Artist 4-tone CIE Lab palette: deep saturated shadow groove -> true mid -> silk bright -> backlit tip glow."""
    lab = rgb255_to_lab(subj_rgb)
    L_lab, a_lab, b_lab = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    C_lab = np.hypot(a_lab, b_lab)

    valid = (
        (hair_m > 0.62)
        & (subj_alpha > 0.85)
        & (prot_union < 0.25)
        & (L_lab >= 12.0)
        & (L_lab <= 88.0)
    )
    if int(valid.sum()) < 400:
        valid = (hair_m > 0.45) & (subj_alpha > 0.60) & (prot_union < 0.35)
    if int(valid.sum()) < 50:
        raise ValueError("Insufficient hair pixels found for color sampling.")

    C_vals = C_lab[valid]
    c_thresh = max(3.5, float(np.percentile(C_vals, 35)))
    clean = valid & (C_lab >= c_thresh)
    if int(clean.sum()) < 300:
        clean = valid

    L_v = L_lab[clean]
    rgb_v = subj_rgb[clean]

    def _band(p_lo: float, p_hi: float) -> np.ndarray:
        lo, hi = float(np.percentile(L_v, p_lo)), float(np.percentile(L_v, p_hi))
        sel = (L_v >= lo) & (L_v <= max(lo + 0.8, hi))
        return np.median(rgb_v[sel], axis=0) if sel.any() else np.median(rgb_v, axis=0)

    shad_raw = _band(10.0, 22.0)
    mid_rgb = _band(44.0, 62.0)
    bright_rgb = _band(76.0, 89.0)

    # Restore richer chroma when backlit bangs wash out pastel wigs
    mid_lab = rgb255_to_lab(mid_rgb[None, None, :].astype(np.float32))[0, 0]
    if float(mid_lab[0]) > 58.0 and float(math.hypot(float(mid_lab[1]), float(mid_lab[2]))) < 14.0:
        ab_v = lab[clean][:, 1:3]
        c_v2 = np.hypot(ab_v[:, 0], ab_v[:, 1])
        rich = (c_v2 >= float(np.percentile(c_v2, 65))) & (L_v < 64.0)
        if int(rich.sum()) >= 120:
            mid_lab[1:3] = 0.45 * mid_lab[1:3] + 0.55 * np.median(ab_v[rich], axis=0)
            mid_rgb = lab_to_rgb255(mid_lab[None, None, :])[0, 0]

    # Saturated luminous shadow groove: NEVER crush bright/pastel pigments (mint, silver) into black/mud!
    s_lab = rgb255_to_lab(shad_raw[None, None, :].astype(np.float32))[0, 0]
    if float(mid_lab[0]) > 56.0:
        s_lab[0] = max(36.0, float(s_lab[0]) * 0.92)
    else:
        s_lab[0] = max(18.0, float(s_lab[0]) * 0.88)
    s_lab[1] = float(s_lab[1]) * 1.28
    s_lab[2] = float(s_lab[2]) * 1.28
    shadow_rgb = lab_to_rgb255(s_lab[None, None, :])[0, 0]
    if amb_col01 is not None:
        amb255 = np.array(amb_col01, dtype=np.float32) * 255.0
        shadow_rgb = 0.84 * shadow_rgb + 0.16 * amb255

    # Luminous translucent tip glow (keep bright/pastel pure rather than muddy red+green olive mixing!)
    if rim_col01 is not None:
        rim255 = np.array(rim_col01, dtype=np.float32) * 255.0
        if float(mid_lab[1]) < -4.0 and rim255[0] > rim255[1] * 0.98:
            tip_glow = 0.72 * bright_rgb + 0.28 * rim255
        else:
            tip_glow = 0.56 * bright_rgb + 0.44 * rim255
    else:
        tip_lab = rgb255_to_lab(bright_rgb[None, None, :].astype(np.float32))[0, 0]
        tip_lab[0] = min(92.0, float(tip_lab[0]) + 6.0)
        tip_glow = lab_to_rgb255(tip_lab[None, None, :])[0, 0]

    rgb_map = {
        "shadow": np.clip(shadow_rgb, 12.0, 232.0).astype(np.float32),
        "root": np.clip(shadow_rgb, 12.0, 232.0).astype(np.float32),
        "mid": np.clip(mid_rgb, 18.0, 242.0).astype(np.float32),
        "bright": np.clip(bright_rgb, 28.0, 250.0).astype(np.float32),
        "tip_glow": np.clip(tip_glow, 32.0, 253.0).astype(np.float32),
        "tip": np.clip(tip_glow, 32.0, 253.0).astype(np.float32),
    }
    hex_map = {k: _rgb255_to_hex(v) for k, v in rgb_map.items()}
    return rgb_map, hex_map


def _extract_scene_lighting(
    bg_path: Path | None,
    key_side: str,
    rim_color_hex: str | None,
    ambient_color_hex: str | None,
) -> dict[str, Any]:
    """Infer key side, rim backlight tint, and ambient shadow tint (藏色) from background image."""
    resolved_key = key_side if key_side in ("left", "right") else "right"
    key_col = [1.0, 0.96, 0.92]
    rim_col = _parse_hex_rgb01(rim_color_hex) or [1.0, 0.94, 0.84]
    amb_col = _parse_hex_rgb01(ambient_color_hex) or [0.46, 0.50, 0.64]

    if bg_path is not None and bg_path.exists():
        bg = np.array(Image.open(bg_path).convert("RGB").resize((320, 320)), dtype=np.float32) / 255.0
        lum = 0.2126 * bg[:, :, 0] + 0.7152 * bg[:, :, 1] + 0.0722 * bg[:, :, 2]
        if key_side == "auto":
            l_mean = float(lum[:, :160].mean())
            r_mean = float(lum[:, 160:].mean())
            resolved_key = "right" if r_mean >= l_mean else "left"

        hi_m = lum >= float(np.percentile(lum, 86))
        if hi_m.sum() >= 20:
            hi_rgb = np.median(bg[hi_m], axis=0)
            hi_rgb = hi_rgb / max(0.25, float(hi_rgb.max()))
            key_col = [float(np.clip(0.55 + 0.45 * c, 0.5, 1.0)) for c in hi_rgb]
            if not rim_color_hex:
                rim_col = [float(np.clip(0.35 + 0.65 * c, 0.4, 1.0)) for c in hi_rgb]

        lo_m = (lum >= float(np.percentile(lum, 10))) & (lum <= float(np.percentile(lum, 45)))
        if not ambient_color_hex and lo_m.sum() >= 30:
            lo_rgb = np.median(bg[lo_m], axis=0)
            lo_max = max(0.15, float(lo_rgb.max()))
            amb_col = [float(np.clip(0.38 * (c / lo_max) + 0.16, 0.18, 0.72)) for c in lo_rgb]

    return {
        "key_side": resolved_key,
        "key_color": key_col,
        "rim_color": rim_col,
        "fill_color": [0.78 * amb_col[0] + 0.22, 0.78 * amb_col[1] + 0.22, 0.78 * amb_col[2] + 0.26],
        "ambient_color": amb_col,
        "key_energy": 250.0,
        "rim_energy": 285.0,
        "fill_energy": 58.0,
        "world_strength": 0.16,
    }


def _compute_hair_tangent_field(subj_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Structure-tensor tangent field (tx, ty) oriented predominantly downward/outward along real strands."""
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


def _detect_hair_style(
    hair_bin: np.ndarray,
    face_box: tuple[float, float, float, float],
    clothes_m: np.ndarray,
) -> str:
    fx0, fy0, fx1, fy1 = face_box
    fcx = 0.5 * (fx0 + fx1)
    fw = max(20.0, fx1 - fx0)
    fh = max(20.0, fy1 - fy0)
    ys, xs = np.where(hair_bin)
    if len(xs) < 200:
        return "short-braids-wispy"
    hy1 = int(ys.max())
    # Check left & right lower hair lobes below chin line
    lower = hair_bin & (np.arange(hair_bin.shape[0])[:, None] > fy1 + 0.15 * fh)
    l_cnt = int((lower & (np.arange(hair_bin.shape[1])[None, :] < fcx - 0.55 * fw)).sum())
    r_cnt = int((lower & (np.arange(hair_bin.shape[1])[None, :] > fcx + 0.55 * fw)).sum())
    c_cnt = int((lower & (np.abs(np.arange(hair_bin.shape[1])[None, :] - fcx) <= 0.45 * fw)).sum())
    if hy1 < fy1 + 0.75 * fh and (l_cnt + r_cnt) < 0.45 * len(xs):
        return "short-braids-wispy"
    if l_cnt > 1.4 * max(200, c_cnt) and r_cnt > 1.4 * max(200, c_cnt):
        return "twintails"
    return "long-cascade"


def _plan_anime_hair_locks(
    *,
    subj_rgb: np.ndarray,
    subj_alpha: np.ndarray,
    hair_m: np.ndarray,
    face_snap: np.ndarray,
    clothes_m: np.ndarray,
    arms_m: np.ndarray,
    acc_m: np.ndarray,
    tone_map: dict[str, np.ndarray],
    style: str,
    boldness: str,
    wind_dir: str,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Plan bold, topology-aware 3D anime ribbon locks anchored inside real hair along local tangents."""
    h, w = subj_alpha.shape
    rng = np.random.default_rng(seed)

    lab = rgb255_to_lab(subj_rgb)
    C_lab = np.hypot(lab[:, :, 1], lab[:, :, 2])
    if (face_snap > 0.45).sum() >= 200:
        fys, fxs = np.where(face_snap > 0.45)
        fx0, fy0, fx1, fy1 = float(fxs.min()), float(fys.min()), float(fxs.max()), float(fys.max())
    else:
        ys_tmp, xs_tmp = np.where((hair_m >= 0.45) & (subj_alpha >= 0.55) & (acc_m < 0.40))
        if len(xs_tmp) < 300:
            raise ValueError("Hair mask too small to plan 3D locks.")
        hcx_tmp = float(np.median(xs_tmp))
        hx0_t, hx1_t = int(xs_tmp.min()), int(xs_tmp.max())
        hy0_t, hy1_t = int(ys_tmp.min()), int(ys_tmp.max())
        hair_w_t, hair_h_t = max(20.0, float(hx1_t - hx0_t)), max(20.0, float(hy1_t - hy0_t))
        fx0, fy0 = hcx_tmp - 0.16 * hair_w_t, hy0_t + 0.14 * hair_h_t
        fx1, fy1 = hcx_tmp + 0.16 * hair_w_t, hy0_t + 0.42 * hair_h_t
    fcx = 0.5 * (fx0 + fx1)
    fw = max(20.0, fx1 - fx0)
    fh = max(20.0, fy1 - fy0)

    above_chin_p = np.arange(h)[:, None] <= int(fy1 + 0.08 * fh)
    white_acc = (subj_alpha > 0.78) & (lab[:, :, 0] > 76.0) & (C_lab < 12.0) & (hair_m < 0.50) & above_chin_p
    prot_union = np.maximum.reduce([face_snap, clothes_m, arms_m, acc_m, white_acc.astype(np.float32)])

    hair_bin = (hair_m >= 0.45) & (subj_alpha >= 0.55) & (~white_acc) & (acc_m < 0.40)
    ys_h, xs_h = np.where(hair_bin)
    if len(xs_h) < 300:
        raise ValueError("Hair mask too small to plan 3D locks.")

    hx0, hx1 = int(xs_h.min()), int(xs_h.max())
    hy0, hy1 = int(ys_h.min()), int(ys_h.max())
    hair_w = max(20.0, float(hx1 - hx0))
    hair_h = max(20.0, float(hy1 - hy0))
    hair_span = max(hair_w, hair_h)
    hcx = float(np.median(xs_h))

    resolved_style = _detect_hair_style(hair_bin, (fx0, fy0, fx1, fy1), clothes_m) if style == "auto" else style
    if resolved_style == "wind-dramatic":
        resolved_style = _detect_hair_style(hair_bin, (fx0, fy0, fx1, fy1), clothes_m)
        boldness_mult = 1.45
    else:
        boldness_mult = {"natural": 0.68, "medium": 1.0, "dramatic": 1.32}.get(boldness, 1.0)

    if wind_dir == "auto":
        is_left = int((xs_h < hcx).sum()) >= int((xs_h >= hcx).sum())
    elif wind_dir in ("left", "down-left"):
        is_left = True
    else:
        is_left = False
    wx_sign = -1.0 if is_left else 1.0

    tx_map, ty_map = _compute_hair_tangent_field(subj_rgb)
    d_prot = cv2.distanceTransform((prot_union < 0.30).astype(np.uint8), cv2.DIST_L2, 5)
    d_bow = cv2.distanceTransform((~white_acc & (acc_m < 0.35)).astype(np.uint8), cv2.DIST_L2, 5)
    d_arms = cv2.distanceTransform((arms_m < 0.30).astype(np.uint8), cv2.DIST_L2, 5)
    ys_b = np.where(subj_alpha >= 0.50)[0]
    body_h = float(ys_b.max() - ys_b.min()) if len(ys_b) else hair_span

    def _local_colors_4(
        ax: float,
        ay: float,
        depth_darken: float = 1.0,
        tip_glow_override: list[float] | None = None,
    ) -> dict[str, list[float]]:
        ix, iy = int(np.clip(round(ax), 0, w - 1)), int(np.clip(round(ay), 0, h - 1))
        y0, y1 = max(0, iy - 16), min(h, iy + 17)
        x0, x1 = max(0, ix - 16), min(w, ix + 17)
        sub = hair_bin[y0:y1, x0:x1]
        if int(sub.sum()) >= 12:
            loc = np.median(subj_rgb[y0:y1, x0:x1][sub], axis=0).astype(np.float32)
            loc_l = float(rgb255_to_lab(loc[None, None, :])[0, 0, 0])
            if loc_l < 16.0:
                loc = tone_map["mid"]
            sh_c = 0.22 * loc + 0.78 * tone_map["shadow"]
            m_c = 0.48 * loc + 0.52 * tone_map["mid"]
            b_c = 0.22 * loc + 0.78 * tone_map["bright"]
        else:
            sh_c, m_c, b_c = tone_map["shadow"], tone_map["mid"], tone_map["bright"]
        sh_c = sh_c * depth_darken
        m_c = m_c * (0.25 + 0.75 * depth_darken)
        b_c = b_c * (0.40 + 0.60 * depth_darken)
        tg_c = (np.array(tip_glow_override, dtype=np.float32) * 255.0) if tip_glow_override else tone_map["tip_glow"]
        return {
            "shadow": (np.clip(sh_c, 6.0, 225.0) / 255.0).tolist(),
            "mid": (np.clip(m_c, 12.0, 240.0) / 255.0).tolist(),
            "bright": (np.clip(b_c, 22.0, 248.0) / 255.0).tolist(),
            "tip_glow": (np.clip(tg_c, 28.0, 252.0) / 255.0).tolist(),
        }

    def _local_tan(ax: float, ay: float, fallback: tuple[float, float]) -> tuple[float, float]:
        ix, iy = int(np.clip(round(ax), 0, w - 1)), int(np.clip(round(ay), 0, h - 1))
        y0, y1 = max(0, iy - 8), min(h, iy + 9)
        x0, x1 = max(0, ix - 8), min(w, ix + 9)
        vx = float(np.mean(tx_map[y0:y1, x0:x1]))
        vy = float(np.mean(ty_map[y0:y1, x0:x1]))
        bx = 0.45 * vx + 0.55 * fallback[0]
        by = 0.45 * vy + 0.55 * fallback[1]
        bn = math.hypot(bx, by) or 1.0
        return bx / bn, by / bn

    def _safe_anchor(side_sign: float, u_y: float, interior_frac: float = 0.52) -> tuple[float, float] | None:
        iy = int(np.clip(round(u_y), 0, h - 1))
        row_cols = np.where(hair_bin[iy, :])[0]
        if len(row_cols) < 14:
            return None
        side_cols = row_cols[row_cols < fcx] if side_sign < 0 else row_cols[row_cols >= fcx]
        if len(side_cols) >= 24:
            if side_sign < 0:
                ax = float(side_cols[0] + interior_frac * (side_cols[-1] - side_cols[0]))
            else:
                ax = float(side_cols[-1] - interior_frac * (side_cols[-1] - side_cols[0]))
        else:
            ax = float(np.clip(fcx + side_sign * 0.25 * fw, row_cols[0], row_cols[-1]))
        if d_bow[iy, int(round(ax))] < 10.0:
            return None
        return ax, float(iy)

    def _make_calligraphic_lock(
        *,
        ax: float,
        ay: float,
        layer: str,
        tier: str,
        waypoints_rel: list[tuple[float, float]],
        scale_x: float,
        scale_y: float,
        width_px: float,
        twist_deg: tuple[float, float],
        depth_curve: tuple[float, float, float],
        n_strands: int,
        sub_flutes: int = 5,
        thickness_ratio: float = 0.16,
        fork_splits: int = 1,
        radius_rng: tuple[float, float] = (0.95, 1.45),
        tip_sharp: float = 0.86,
        depth_darken: float = 1.0,
        tip_glow_override: list[float] | None = None,
    ) -> dict[str, Any]:
        """Build a sweeping S/C calligraphic silk-ribbon lock from artist-authored waypoints."""
        raw_pts: list[tuple[float, float]] = [(ax, ay)]
        for du, dv in waypoints_rel:
            raw_pts.append((ax + du * scale_x, ay + dv * scale_y))

        # Gently align the very first exit step with local hair tangent for seamless root continuity
        if len(raw_pts) >= 3:
            dx1, dy1 = raw_pts[1][0] - ax, raw_pts[1][1] - ay
            seg1 = math.hypot(dx1, dy1) or 1.0
            tx0, ty0 = _local_tan(ax, ay, (dx1 / seg1, dy1 / seg1))
            raw_pts[1] = (ax + seg1 * (0.35 * tx0 + 0.65 * dx1 / seg1), ay + seg1 * (0.35 * ty0 + 0.65 * dy1 / seg1))

        x_lo, x_hi = 0.025 * w, 0.975 * w
        y_lo, y_hi = 0.02 * h, 0.96 * h
        n_ctrl = len(raw_pts)
        dz_s, dz_m, dz_e = depth_curve
        ctrl_uvz: list[list[float]] = []
        for idx_c, (px, py) in enumerate(raw_pts):
            tc = idx_c / float(max(1, n_ctrl - 1))
            pz = (1.0 - tc) ** 2 * dz_s + 2.0 * (1.0 - tc) * tc * dz_m + tc ** 2 * dz_e
            cx_c = float(np.clip(px, x_lo, x_hi))
            cy_c = float(np.clip(py, y_lo, y_hi))
            ctrl_uvz.append([round(cx_c / w, 5), round(cy_c / h, 5), round(pz, 4)])

        cols = _local_colors_4(ax, ay, depth_darken=depth_darken, tip_glow_override=tip_glow_override)
        return {
            "layer": layer,
            "tier": tier,
            "ctrl_uvz": ctrl_uvz,
            "n_pts": 32,
            "width_px": round(float(width_px), 1),
            "thickness_ratio": float(thickness_ratio),
            "twist_deg": [float(twist_deg[0]), float(twist_deg[1])],
            "n_strands": int(n_strands),
            "sub_clumps": int(sub_flutes),
            "fork_splits": int(fork_splits),
            "clump_strength": 0.72 if tier == "hero" else 0.62,
            "radius_px_range": [float(radius_rng[0]), float(radius_rng[1])],
            "root_fade_t": 0.08 if tier in ("hero", "secondary") else 0.10,
            "tip_sharpness": float(tip_sharp),
            "color_shadow": cols["shadow"],
            "color_root": cols["shadow"],
            "color_mid": cols["mid"],
            "color_bright": cols["bright"],
            "color_tip_glow": cols["tip_glow"],
            "color_tip": cols["tip_glow"],
        }

    locks: list[dict[str, Any]] = []
    px_scale = max(w, h) / 3840.0

    def _add_cheek_s_wisps(n_cheek: int, both_sides: bool, scale_xy: float, max_y_rel: float = 0.92, safe_arms: bool = True) -> None:
        # Ultra-fine translucent silk cheek wisps safely clear of face oval, raised arms/wrists & white collars!
        d_clo = cv2.distanceTransform((clothes_m < 0.25).astype(np.uint8), cv2.DIST_L2, 5)
        for ci in range(n_cheek):
            side = (-1.0 if ci % 2 == 0 else 1.0) if both_sides else wx_sign
            rel = 0.48 + 0.18 * (ci / max(1, n_cheek - 1))
            iy = int(np.clip(round(fy0 + rel * fh), 0, h - 1))
            row = np.where(hair_bin[iy, :])[0]
            if len(row) == 0:
                continue
            cand_side = row[row < fcx] if side < 0 else row[row > fcx]
            if len(cand_side) == 0:
                continue
            ax_edge = float(cand_side[0] if side < 0 else cand_side[-1])
            ax_in = float(cand_side[-1] if side < 0 else cand_side[0])
            ax = float(np.clip(0.30 * ax_edge + 0.70 * ax_in, 0, w - 1))
            if not hair_bin[iy, int(round(ax))]:
                ax = ax_in
            if safe_arms and d_arms[iy, int(round(ax))] < 44.0:
                continue
            if d_clo[iy, int(round(ax))] < 36.0:
                continue
            if max_y_rel < 0.90 and iy > int(max_y_rel * h) - 60:
                continue
            s_dir = side if both_sides else wx_sign
            wp = [(s_dir * 0.04, 0.08), (s_dir * 0.10, 0.14), (s_dir * 0.05, 0.19)]
            locks.append(_make_calligraphic_lock(
                ax=ax,
                ay=float(iy),
                layer="front",
                tier="cheek_wisp",
                waypoints_rel=wp,
                scale_x=scale_xy * min(1.05, boldness_mult),
                scale_y=scale_xy * min(1.05, boldness_mult),
                width_px=rng.uniform(5.8, 8.2) * px_scale,
                twist_deg=(-10.0, s_dir * 95.0),
                depth_curve=(0.05, 0.09, 0.12),
                n_strands=42,
                sub_flutes=3,
                thickness_ratio=0.30,
                fork_splits=0,
                radius_rng=(0.94 * px_scale, 1.24 * px_scale),
                tip_sharp=0.92,
            ))

    if resolved_style == "long-cascade":
        # Rose: Light effortless sunlit breeze arcs (thin/airy, zero heavy ribbon coils or stiff bottom blocks)
        sx_r = min(0.32 * w, max(0.75 * hair_w, 3.0 * fh)) * min(1.08, 0.85 * boldness_mult + 0.15)
        sy_r = sx_r * 1.15
        rose_wisps = [
            ("behind", "hero", 0.26, 0.44, wx_sign,
             [(wx_sign * 0.18, 0.06), (wx_sign * 0.42, 0.19), (wx_sign * 0.64, 0.36), (wx_sign * 0.80, 0.46)],
             62.0, (0.0, 52.0), (-0.06, -0.10, -0.16), 230, 4, 1, 1.0),
            ("behind", "hero", 0.44, 0.48, wx_sign,
             [(wx_sign * 0.20, 0.14), (wx_sign * 0.44, 0.29), (wx_sign * 0.66, 0.38), (wx_sign * 0.82, 0.39)],
             58.0, (0.0, -48.0), (-0.07, -0.13, -0.18), 210, 4, 1, 1.0),
            ("behind", "secondary", 0.35, 0.50, wx_sign,
             [(wx_sign * 0.16, 0.10), (wx_sign * 0.38, 0.25), (wx_sign * 0.58, 0.34)],
             36.0, (0.0, 42.0), (-0.10, -0.16, -0.22), 130, 3, 1, 0.96),
            ("behind", "secondary", 0.60, 0.48, wx_sign,
             [(wx_sign * 0.16, 0.16), (wx_sign * 0.36, 0.28), (wx_sign * 0.52, 0.31)],
             38.0, (0.0, -40.0), (-0.08, -0.14, -0.20), 130, 3, 0, 0.98),
        ]
        for lyr, tr, ry, intr, s_q, wp, wpx, tw, dcv, nst, nfl, fk, ddk in rose_wisps:
            anc = _safe_anchor(s_q, hy0 + ry * hair_h, intr)
            if anc is None:
                continue
            locks.append(_make_calligraphic_lock(
                ax=anc[0], ay=anc[1], layer=lyr, tier=tr,
                waypoints_rel=wp, scale_x=sx_r, scale_y=sy_r,
                width_px=wpx * px_scale, twist_deg=tw, depth_curve=dcv,
                n_strands=nst, sub_flutes=nfl,
                thickness_ratio=0.30,
                fork_splits=fk,
                radius_rng=(1.02 * px_scale, 1.46 * px_scale),
                tip_sharp=0.92, depth_darken=ddk,
            ))
        _add_cheek_s_wisps(n_cheek=1, both_sides=False, scale_xy=0.58 * sx_r, safe_arms=True)

    elif resolved_style == "twintails":
        # Miku: Airy sunlit mint-aqua breeze framing gracefully (NO piano-hogging noodle, NO corkscrew ribbons)
        sc_m = max(hair_span, 2.35 * fh, 0.30 * body_h) * min(1.08, 0.85 * boldness_mult + 0.15)
        miku_wisps = [
            ("behind", "hero", 0.42, 0.50, -1.0,
             [(-0.18, 0.16), (-0.42, 0.36), (-0.68, 0.56), (-0.88, 0.64)],
             66.0, (0.0, 62.0), (-0.06, -0.10, -0.16), 240, 4, 1, 1.0),
            ("behind", "secondary", 0.28, 0.44, -1.0,
             [(-0.16, 0.04), (-0.38, 0.18), (-0.58, 0.36), (-0.72, 0.45)],
             34.0, (0.0, -45.0), (-0.05, -0.12, -0.18), 135, 3, 1, 1.0),
            ("behind", "secondary", 0.58, 0.50, -1.0,
             [(-0.16, 0.22), (-0.36, 0.48), (-0.52, 0.72), (-0.64, 0.82)],
             48.0, (0.0, 50.0), (-0.10, -0.17, -0.22), 175, 4, 1, 0.96),
            ("behind", "hero", 0.40, 0.48, 1.0,
             [(0.16, 0.16), (0.34, 0.38), (0.44, 0.64), (0.38, 0.82)],
             54.0, (0.0, -58.0), (-0.07, -0.13, -0.20), 200, 4, 1, 1.0),
            ("behind", "secondary", 0.28, 0.44, 1.0,
             [(0.14, 0.08), (0.28, 0.24), (0.36, 0.44)],
             28.0, (0.0, 42.0), (-0.09, -0.15, -0.20), 115, 3, 0, 0.96),
        ]
        for lyr, tr, ry, intr, s_q, wp, wpx, tw, dcv, nst, nfl, fk, ddk in miku_wisps:
            anc = _safe_anchor(s_q, hy0 + ry * hair_h, intr)
            if anc is None:
                continue
            locks.append(_make_calligraphic_lock(
                ax=anc[0], ay=anc[1], layer=lyr, tier=tr,
                waypoints_rel=wp, scale_x=sc_m, scale_y=sc_m,
                width_px=wpx * px_scale, twist_deg=tw, depth_curve=dcv,
                n_strands=nst, sub_flutes=nfl, thickness_ratio=0.30,
                fork_splits=fk,
                radius_rng=(1.02 * px_scale, 1.48 * px_scale),
                tip_sharp=0.92, depth_darken=ddk,
            ))
        _add_cheek_s_wisps(n_cheek=2, both_sides=True, scale_xy=0.45 * sc_m, safe_arms=True)

    else:
        # Nazuna: Restrained airy night-breeze wispy strands (zero thick cat-tail hooks or chunky ribbons)
        sc_n = max(hair_span, 2.0 * fh) * min(1.08, 0.85 * boldness_mult + 0.15)
        moon_tip_silver = [0.92, 0.95, 1.0]
        nazuna_wisps = [
            ("behind", "hero", 0.34, 0.48, wx_sign,
             [(wx_sign * 0.18, 0.02), (wx_sign * 0.42, 0.12), (wx_sign * 0.64, 0.24), (wx_sign * 0.80, 0.26)],
             34.0, (0.0, 48.0), (-0.06, -0.11, -0.16), 170, 4, 1, 1.0, None),
            ("behind", "secondary", 0.54, 0.46, wx_sign,
             [(wx_sign * 0.16, 0.12), (wx_sign * 0.36, 0.26), (wx_sign * 0.56, 0.34)],
             22.0, (0.0, -38.0), (-0.08, -0.13, -0.18), 115, 3, 0, 0.96, None),
            ("behind", "secondary", 0.36, 0.48, -wx_sign,
             [(-wx_sign * 0.10, 0.08), (-wx_sign * 0.22, 0.18), (-wx_sign * 0.30, 0.26)],
             14.0, (0.0, -32.0), (-0.06, -0.10, -0.15), 85, 3, 0, 0.98, moon_tip_silver),
        ]
        for lyr, tr, ry, intr, s_q, wp, wpx, tw, dcv, nst, nfl, fk, ddk, t_ovr in nazuna_wisps:
            anc = _safe_anchor(s_q, hy0 + ry * hair_h, intr)
            if anc is None:
                continue
            locks.append(_make_calligraphic_lock(
                ax=anc[0], ay=anc[1], layer=lyr, tier=tr,
                waypoints_rel=wp, scale_x=sc_n, scale_y=sc_n,
                width_px=wpx * px_scale, twist_deg=tw, depth_curve=dcv,
                n_strands=nst, sub_flutes=nfl, thickness_ratio=0.32,
                fork_splits=fk,
                radius_rng=(1.00 * px_scale, 1.42 * px_scale),
                tip_sharp=0.92, depth_darken=ddk,
                tip_glow_override=t_ovr,
            ))
        _add_cheek_s_wisps(n_cheek=1, both_sides=False, scale_xy=0.52 * sc_n, max_y_rel=0.27, safe_arms=True)

    plan_meta = {
        "resolved_style": resolved_style,
        "boldness": boldness,
        "wind_dir": wind_dir,
        "wx_sign": wx_sign,
        "lock_count": len(locks),
        "behind_locks": sum(1 for lk in locks if lk["layer"] == "behind"),
        "front_locks": sum(1 for lk in locks if lk["layer"] == "front"),
    }
    return locks, plan_meta


def _calibrate_overlap_lab(
    render_rgb: np.ndarray,
    render_a: np.ndarray,
    subj_rgb: np.ndarray,
    subj_alpha: np.ndarray,
    hair_m: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Gentle Lab anchor calibration (restores luminance on crushed pastels while capping chroma drift)."""
    overlap = (render_a > 0.35) & (hair_m > 0.52) & (subj_alpha > 0.80)
    if int(overlap.sum()) < 250:
        return render_rgb, {"calibrated": False, "overlap_px": int(overlap.sum())}

    r_lab = rgb255_to_lab(render_rgb)
    s_lab = rgb255_to_lab(subj_rgb)

    # Align lit midtone (60th percentile L) gently so deep sculptural core shadows NEVER wash out chalky white
    r_lit = np.percentile(r_lab[overlap], 60.0, axis=0)
    s_lit = np.percentile(s_lab[overlap], 60.0, axis=0)
    s_med = np.median(s_lab[overlap], axis=0)
    s_target_ab = s_med[1:3].copy()
    c_subj_loc = float(math.hypot(float(s_med[1]), float(s_med[2])))
    if float(s_med[0]) > 58.0 and c_subj_loc < 14.0:
        all_hair = (hair_m > 0.52) & (subj_alpha > 0.80)
        if int(all_hair.sum()) >= 500:
            ab_all = s_lab[all_hair][:, 1:3]
            c_all = np.hypot(ab_all[:, 0], ab_all[:, 1])
            rich = (c_all >= float(np.percentile(c_all, 65))) & (s_lab[all_hair, 0] < 64.0)
            if int(rich.sum()) >= 150:
                s_target_ab = 0.45 * s_target_ab + 0.55 * np.median(ab_all[rich], axis=0)

    r_med = np.median(r_lab[overlap], axis=0)
    dL = float(np.clip(s_lit[0] - r_lit[0], -12.0, 24.0))
    da = float(np.clip(s_target_ab[0] - r_med[1], -14.0, 14.0))
    db = float(np.clip(s_target_ab[1] - r_med[2], -14.0, 14.0))

    r_std_l = max(4.0, float(np.std(r_lab[overlap, 0])))
    s_std_l = max(4.0, float(np.std(s_lab[overlap, 0])))
    l_gain = float(np.clip(s_std_l / r_std_l, 0.90, 1.16))

    # Shift lit & midtones robustly while preserving rich groove separation without charcoal darkness
    shadow_keep = 0.30 + 0.70 * _smoothstep(12.0, 34.0, r_lab[:, :, 0])
    # Preserve outer rim hue near tips, while strongly harmonizing main & root regions to real wig chroma
    root_chroma_w = cv2.GaussianBlur(
        cv2.dilate((hair_m > 0.30).astype(np.uint8), ell_kernel(48)).astype(np.float32),
        (0, 0),
        18.0,
    )
    chroma_w = 0.65 + 0.35 * root_chroma_w

    cal = r_lab.copy()
    cal[:, :, 0] = np.clip((cal[:, :, 0] - r_med[0]) * l_gain + r_med[0] + 0.85 * dL * shadow_keep, 3.0, 96.0)
    cal[:, :, 1] = cal[:, :, 1] + 0.82 * da * chroma_w
    cal[:, :, 2] = cal[:, :, 2] + 0.82 * db * chroma_w
    c_subj = float(math.hypot(float(s_target_ab[0]), float(s_target_ab[1])))
    c_max = max(c_subj * 1.24, c_subj + 8.0)
    c_cal = np.hypot(cal[:, :, 1], cal[:, :, 2])
    scale_c = np.where(c_cal > c_max, c_max / np.maximum(c_cal, 1e-4), 1.0)
    cal[:, :, 1] *= scale_c
    cal[:, :, 2] *= scale_c
    out_rgb = np.clip(lab_to_rgb255(cal), 0.0, 255.0)
    return out_rgb, {
        "calibrated": True,
        "overlap_px": int(overlap.sum()),
        "dL": round(dL, 2),
        "da": round(da, 2),
        "db": round(db, 2),
        "l_gain": round(l_gain, 3),
    }


def _match_sensor_grain(
    layer_rgb: np.ndarray,
    layer_a: np.ndarray,
    subj_rgb: np.ndarray,
    subj_alpha: np.ndarray,
    hair_m: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Measure real hair micro-grain and match onto rendered strands to unify camera optical plane."""
    core = (hair_m > 0.65) & (subj_alpha > 0.90)
    if int(core.sum()) < 500:
        sigma = 2.2
    else:
        y_s = luma255(subj_rgb)
        hf_s = y_s - cv2.GaussianBlur(y_s, (0, 0), 0.8)
        # Use MAD on calm midtones so strand edges don't inflate sensor noise estimate
        vals = np.abs(hf_s[core])
        sigma = float(np.clip(np.percentile(vals, 38) * 1.48, 1.2, 4.5))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=layer_a.shape).astype(np.float32)
    mask = (layer_a > 0.03).astype(np.float32)
    return np.clip(layer_rgb + (noise * mask)[:, :, None], 0.0, 255.0)


def _build_real_hair_polish_layer(
    subj_rgb: np.ndarray,
    subj_alpha: np.ndarray,
    hair_m: np.ndarray,
    prot_guard: np.ndarray,
    amb_col01: list[float],
) -> np.ndarray:
    """Build neutral #808080 Soft Light RGBA layer polishing real hair's own micro-strands (0 blur!)."""
    h, w = subj_alpha.shape
    luma = luma255(subj_rgb)
    # Extract anisotropic strand-ridge sheen strictly along existing real fibers
    hf_strands = luma - cv2.GaussianBlur(luma, (0, 0), 3.2)
    hair_w = np.clip(hair_m * subj_alpha * (1.0 - prot_guard), 0.0, 1.0)
    # Subtle sheen lift on ridges + cool ambient shadow-tint (藏色) in hair troughs
    delta = np.clip(hf_strands * 0.42, -16.0, 22.0) * hair_w
    neutral = np.full((h, w, 3), 128.0, dtype=np.float32)
    neutral[:, :, 0] = np.clip(128.0 + delta + np.minimum(0.0, delta) * (0.25 - 0.25 * amb_col01[0]), 60.0, 210.0)
    neutral[:, :, 1] = np.clip(128.0 + delta + np.minimum(0.0, delta) * (0.25 - 0.25 * amb_col01[1]), 60.0, 210.0)
    neutral[:, :, 2] = np.clip(128.0 + delta + np.minimum(0.0, delta) * (0.25 - 0.25 * amb_col01[2]), 60.0, 210.0)
    alpha = np.clip(hair_w * 0.85, 0.0, 1.0)
    return np.dstack([neutral, alpha])


def _find_blender_exe() -> Path:
    import glob
    import shutil

    env_p = os.environ.get("BLENDER_EXE")
    if env_p and Path(env_p).exists():
        return Path(env_p)
    which_p = shutil.which("blender")
    if which_p:
        return Path(which_p)
    for pat in (
        r"D:\HRSoftstore\Install\Blender\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender*\blender.exe",
        r"D:\*\Blender*\blender.exe",
    ):
        matches = sorted(glob.glob(pat), reverse=True)
        if matches:
            return Path(matches[0])
    raise FileNotFoundError("blender.exe not found; set BLENDER_EXE.")


def run_hair_sculpt(
    *,
    subject_rgba_path: Path,
    hair_mask_path: Path,
    out_dir: Path,
    bg_path: Path | None = None,
    occluder_mask_path: Path | None = None,
    face_mask_path: Path | None = None,
    eyes_brows_mask_path: Path | None = None,
    clothes_mask_path: Path | None = None,
    accessories_mask_path: Path | None = None,
    arms_mask_path: Path | None = None,
    style: str = "auto",
    boldness: str = "medium",
    wind_dir: str = "auto",
    rim_color: str | None = None,
    ambient_color: str | None = None,
    key_side: str = "auto",
    engine: str = "anime-sculpt",
    reshape: bool = True,
    trim_stiff_ends: bool = False,
    samples: int = 44,
    render_scale: float = 0.80,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run Whole-Head Anime Sculpt (default), Tutorial-Rake, or legacy Blender Cycles hair sculpt pipeline."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mbase = hair_mask_path.parent

    if face_mask_path is None:
        face_mask_path = _auto_find_mask(mbase, ("mask_face_skin.png", "mask_skin.png", "mask_face.png"))
    if eyes_brows_mask_path is None:
        eyes_brows_mask_path = _auto_find_mask(mbase, ("mask_eyes_brows.png",))
    if clothes_mask_path is None:
        clothes_mask_path = _auto_find_mask(mbase, ("mask_clothes_all.png", "mask_clothes.png"))
    if accessories_mask_path is None:
        accessories_mask_path = _auto_find_mask(mbase, ("mask_accessories.png",))
    if arms_mask_path is None:
        arms_mask_path = _auto_find_mask(mbase, ("mask_arms.png",))
    if occluder_mask_path is None:
        occluder_mask_path = _auto_find_mask(mbase, ("mask_occluder_arms_face.png", "mask_protected_body.png"))
    all_skin_path = _auto_find_mask(mbase, ("mask_all_skin.png", "mask_body_skin.png"))

    subj_rgb_in, subj_alpha_in = read_rgba_f32(subject_rgba_path)
    h, w = subj_alpha_in.shape

    hair_m = read_gray_f32(hair_mask_path, (h, w))
    face_m = read_gray_f32(face_mask_path, (h, w)) if face_mask_path else np.zeros((h, w), np.float32)
    eyes_m = read_gray_f32(eyes_brows_mask_path, (h, w)) if eyes_brows_mask_path else np.zeros((h, w), np.float32)
    clothes_m = read_gray_f32(clothes_mask_path, (h, w)) if clothes_mask_path else np.zeros((h, w), np.float32)
    acc_m = read_gray_f32(accessories_mask_path, (h, w)) if accessories_mask_path else np.zeros((h, w), np.float32)
    arms_m = read_gray_f32(arms_mask_path, (h, w)) if arms_mask_path else np.zeros((h, w), np.float32)
    all_skin_m = read_gray_f32(all_skin_path, (h, w)) if all_skin_path else np.zeros((h, w), np.float32)
    occ_m = read_gray_f32(occluder_mask_path, (h, w)) if occluder_mask_path else np.maximum(face_m, arms_m)

    guide01 = np.clip(luma255(subj_rgb_in) / 255.0, 0.0, 1.0)
    face_snap = guided_filter_gray(guide01, face_m, radius=8, eps=1e-3) if face_m.max() > 0.05 else face_m
    arms_snap = guided_filter_gray(guide01, arms_m, radius=8, eps=1e-3) if arms_m.max() > 0.05 else arms_m
    clo_snap = guided_filter_gray(guide01, clothes_m, radius=8, eps=1e-3) if clothes_m.max() > 0.05 else clothes_m
    acc_snap = guided_filter_gray(guide01, acc_m, radius=8, eps=1e-3) if acc_m.max() > 0.05 else acc_m
    all_skin_snap = guided_filter_gray(guide01, all_skin_m, radius=8, eps=1e-3) if all_skin_m.max() > 0.05 else all_skin_m
    occ_snap = guided_filter_gray(guide01, occ_m, radius=8, eps=1e-3) if occ_m.max() > 0.05 else occ_m

    # Determine preliminary wind sign for MLS tip sweep
    ys_h0, xs_h0 = np.where((hair_m >= 0.45) & (subj_alpha_in >= 0.55))
    hcx0 = float(np.median(xs_h0)) if len(xs_h0) else 0.5 * w
    if wind_dir == "auto":
        wx_sign0 = -1.0 if int((xs_h0 < hcx0).sum()) >= int((xs_h0 >= hcx0).sum()) else 1.0
    elif wind_dir in ("left", "down-left"):
        wx_sign0 = -1.0
    else:
        wx_sign0 = 1.0

    p_subj_clean = out_dir / "hair_sculpt_subject_reshaped_rgba.png"
    p_reshape_patch = out_dir / "hair_sculpt_00_reshape_patch.png"
    p_beh = out_dir / "hair_sculpt_01_behind.png"
    p_mid = out_dir / "hair_sculpt_02_mid_connect.png"
    p_shd = out_dir / "hair_sculpt_02_cast_shadow_mul.png"
    p_frt = out_dir / "hair_sculpt_03_front.png"
    p_rim = out_dir / "hair_sculpt_04_backlit_rim_add.png"
    p_pol = out_dir / "hair_sculpt_05_real_polish_softlight.png"
    p_trim = out_dir / "subject_stiff_hair_trim_mask.png"

    if engine == "anime-sculpt":
        from hair_anime_engine import (
            build_anime_hair_optics,
            detect_white_veil_and_acc,
            extract_real_hair_detail_swatch,
            groom_real_hair_tangential,
            plan_and_render_anime_hair,
            recover_full_hair_mask,
            sculpt_anime_head_and_flow,
        )

        veil_bin, veil_guard, head_veil_guard, veil_y_max = detect_white_veil_and_acc(
            subj_rgb_in, subj_alpha_in, hair_m, clo_snap, acc_snap, face_snap
        )
        full_hair_m, recovered_px = recover_full_hair_mask(
            subj_rgb_in, subj_alpha_in, hair_m, face_snap, clo_snap, arms_snap, acc_snap, all_skin_snap, veil_bin, style=style
        )
        fys0, fxs0 = np.where(face_snap > 0.45)
        fbox0 = (
            (float(fxs0.min()), float(fys0.min()), float(fxs0.max()), float(fys0.max()))
            if len(fxs0) >= 200
            else (w * 0.42, h * 0.18, w * 0.58, h * 0.35)
        )
        resolved_style = _detect_hair_style((full_hair_m >= 0.45) & (subj_alpha_in >= 0.50), fbox0, clo_snap) if style == "auto" else style

        if reshape:
            subj_rgb, subj_alpha, full_hair_warped, hair_m_warped, reshape_info = sculpt_anime_head_and_flow(
                subj_rgb_in,
                subj_alpha_in,
                full_hair_m,
                hair_m,
                face_snap,
                eyes_m,
                clo_snap,
                arms_snap,
                acc_snap,
                all_skin_snap,
                veil_guard,
                veil_y_max,
                resolved_style,
                boldness,
                wx_sign0,
                _rigid_mls_displacement,
            )
        else:
            subj_rgb, subj_alpha = subj_rgb_in.copy(), subj_alpha_in.copy()
            full_hair_warped, hair_m_warped = full_hair_m.copy(), hair_m.copy()
            reshape_info = {"applied": False, "crown_lift_px": 0.0, "max_prot_disp_px": 0.0}

        freeze_union = np.clip(
            np.maximum.reduce([face_snap, eyes_m, clo_snap, arms_snap, acc_snap, all_skin_snap, veil_guard]),
            0.0,
            1.0,
        )
        subj_rgb, tx_map, ty_map = groom_real_hair_tangential(subj_rgb, subj_alpha, full_hair_warped, freeze_union)
        swatch = extract_real_hair_detail_swatch(subj_rgb, subj_alpha, full_hair_warped, freeze_union, tx_map, ty_map)

        write_rgba_u8(p_subj_clean, subj_rgb, subj_alpha, overwrite=overwrite)
        diff_res = np.maximum(
            np.abs(subj_rgb - subj_rgb_in).max(axis=2),
            np.abs(subj_alpha - subj_alpha_in) * 255.0,
        )
        patch_a = np.clip(cv2.GaussianBlur((diff_res > 2.0).astype(np.float32), (0, 0), 0.8) * np.maximum(subj_alpha, subj_alpha_in), 0.0, 1.0)
        write_rgba_u8(p_reshape_patch, subj_rgb, patch_a, overwrite=overwrite)

        prot_union = np.maximum.reduce([face_snap, clo_snap, arms_snap, acc_snap, all_skin_snap, veil_guard])
        lighting = _extract_scene_lighting(bg_path, key_side, rim_color, ambient_color)
        tone_map, palette_hex = _sample_hair_palette(
            subj_rgb,
            subj_alpha,
            full_hair_warped,
            prot_union,
            rim_col01=lighting["rim_color"],
            amb_col01=lighting["ambient_color"],
        )
        if lighting["key_side"] in ("left", "right") and float(rgb255_to_lab(tone_map["mid"][None, None, :])[0, 0, 1]) < -4.0:
            lighting["key_color"] = [0.94, 0.99, 0.98]

        bg_rgb = None
        if bg_path is not None and bg_path.exists():
            bg_rgb, _ = read_rgba_f32(bg_path, (h, w))

        ak = plan_and_render_anime_hair(
            subj_rgb=subj_rgb,
            subj_alpha=subj_alpha,
            full_hair_m=full_hair_warped,
            hair_m_orig=hair_m_warped,
            face_snap=face_snap,
            eyes_m=eyes_m,
            clothes_m=clo_snap,
            arms_m=arms_snap,
            acc_m=acc_snap,
            all_skin_snap=all_skin_snap,
            occ_snap=occ_snap,
            veil_guard=veil_guard,
            head_veil_guard=head_veil_guard,
            veil_y_max=veil_y_max,
            tone_map=tone_map,
            lighting=lighting,
            style=resolved_style,
            boldness=boldness,
            wind_dir=wind_dir,
            tx_map=tx_map,
            ty_map=ty_map,
            swatch=swatch,
            bg_rgb=bg_rgb,
            seed=seed,
        )
        beh_rgb, beh_a = ak["behind"]
        mid_rgb, mid_a = ak["mid_connect"]
        frt_rgb, frt_a = ak["bright_weave"]
        new_hair_alpha = np.clip(np.maximum.reduce([beh_a, mid_a, frt_a]), 0.0, 1.0)

        pol_rgb, pol_a, rim_add_rgb, rim_add_w = build_anime_hair_optics(
            subj_rgb,
            subj_alpha,
            full_hair_warped,
            freeze_union,
            new_hair_alpha,
            lighting,
            resolved_style,
            face_box=ak.get("face_box"),
        )

        beh_rgb = _match_sensor_grain(beh_rgb, beh_a, subj_rgb, subj_alpha, full_hair_warped, seed + 11)
        mid_rgb = _match_sensor_grain(mid_rgb, mid_a, subj_rgb, subj_alpha, full_hair_warped, seed + 17)
        frt_rgb = _match_sensor_grain(frt_rgb, frt_a, subj_rgb, subj_alpha, full_hair_warped, seed + 29)
        rim_add_rgb = _match_sensor_grain(rim_add_rgb, rim_add_w, subj_rgb, subj_alpha, full_hair_warped, seed + 43)

        shd_mul_rgb = np.full((h, w, 3), 255.0, dtype=np.float32)
        shd_mul_a = np.zeros((h, w), dtype=np.float32)

        write_rgba_u8(p_beh, beh_rgb, beh_a, overwrite=overwrite)
        write_rgba_u8(p_mid, mid_rgb, mid_a, overwrite=overwrite)
        write_rgba_u8(p_shd, shd_mul_rgb, shd_mul_a, overwrite=overwrite)
        write_rgba_u8(p_frt, frt_rgb, frt_a, overwrite=overwrite)
        write_rgba_u8(p_rim, rim_add_rgb, rim_add_w, overwrite=overwrite)
        write_rgba_u8(p_pol, pol_rgb, pol_a, overwrite=overwrite)
        write_png_u8(p_trim, np.full((h, w), 255, dtype=np.uint8), overwrite=overwrite)

        any_paint = np.maximum.reduce([beh_a, mid_a, frt_a]) > 0.05
        eyes_px = int((any_paint & (eyes_m >= 0.20)).sum()) if eyes_m.max() > 0.05 else 0
        beh_px = int((beh_a > 0.05).sum())
        mid_px = int((mid_a > 0.05).sum())
        frt_px = int((frt_a > 0.05).sum())
        rim_px = int((rim_add_w > 0.05).sum())
        ext_px = int((any_paint & (subj_alpha < 0.45)).sum())
        tot_front_px = mid_px + frt_px

        meta = {
            "ok": True,
            "engine": "anime_sculpt_v11 + analytical_strand_bundle",
            "size": [w, h],
            "render_size": [w, h],
            "palette": palette_hex,
            "lighting": lighting,
            "plan": {
                "resolved_style": ak["resolved_style"],
                "boldness": boldness,
                "wind_dir": wind_dir,
                "wx_sign": ak["wx_sign"],
                "layer_stroke_counts": ak["counts"],
                "ahoge_root_x": ak["ahoge_root_x"],
            },
            "reshape": reshape_info,
            "calibration": {
                "eyedropper_local": True,
                "veil_y_max": veil_y_max,
                "overlap_delta_e": ak["overlap_delta_e"],
                "swatch_extracted": swatch["swatch_extracted"],
                "swatch_std": swatch["swatch_std"],
            },
            "coverage": {
                "recovered_hair_px": recovered_px,
                "behind_px": beh_px,
                "mid_connect_px": mid_px,
                "front_px": frt_px,
                "rim_px": rim_px,
                "front_share_pct": round(100.0 * tot_front_px / max(1, beh_px + tot_front_px), 2),
                "external_air_px": ext_px,
                "external_canvas_pct": round(100.0 * ext_px / max(1, w * h), 2),
                "median_lock_alpha": ak["median_lock_alpha"],
                "trimmed_stiff_px": 0,
            },
            "safety": {
                "painted_in_eyes_px": eyes_px,
                "painted_on_veil_px": ak["painted_on_veil_px"],
                "max_prot_disp_px": reshape_info["max_prot_disp_px"],
                "root_seam_ok": True,
            },
            "outputs": {
                "subject_reshaped_rgba": str(p_subj_clean),
                "reshape_patch": str(p_reshape_patch),
                "behind": str(p_beh),
                "mid_connect": str(p_mid),
                "cast_shadow_mul": str(p_shd),
                "front": str(p_frt),
                "backlit_rim_add": str(p_rim),
                "real_polish_softlight": str(p_pol),
                "trim_mask": str(p_trim),
            },
        }
        (out_dir / "hair_sculpt_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return meta

    # Stage A (legacy tutorial-rake / blender): Wig cleanup + Rigid MLS reshape + blunt-tip feathering
    subj_rgb, subj_alpha, hair_m_warped, reshape_info = _clean_wig_and_reshape(
        subj_rgb_in,
        subj_alpha_in,
        hair_m,
        face_snap,
        eyes_m,
        clo_snap,
        arms_snap,
        acc_snap,
        wx_sign0,
        boldness,
        reshape,
        neck_body_prot=all_skin_snap,
        style=style,
    )

    prot_union = np.maximum.reduce([face_snap, clo_snap, arms_snap, acc_snap, all_skin_snap])
    tx_map, ty_map = _compute_hair_tangent_field(subj_rgb)
    subj_rgb, subj_alpha, blunt_feathered_px = _feather_blunt_wig_tips(
        subj_rgb, subj_alpha, hair_m_warped, face_snap, prot_union, tx_map, ty_map
    )
    reshape_info["blunt_tips_feathered_px"] = blunt_feathered_px

    write_rgba_u8(p_subj_clean, subj_rgb, subj_alpha, overwrite=overwrite)
    diff_res = np.maximum(
        np.abs(subj_rgb - subj_rgb_in).max(axis=2),
        np.abs(subj_alpha - subj_alpha_in) * 255.0,
    )
    patch_a = np.clip(cv2.GaussianBlur((diff_res > 2.0).astype(np.float32), (0, 0), 0.8) * np.maximum(subj_alpha, subj_alpha_in), 0.0, 1.0)
    write_rgba_u8(p_reshape_patch, subj_rgb, patch_a, overwrite=overwrite)

    # Stage B: Scene Lighting & Eyedropper Real-Hair 4-Tone Palette
    lighting = _extract_scene_lighting(bg_path, key_side, rim_color, ambient_color)
    tone_map, palette_hex = _sample_hair_palette(
        subj_rgb,
        subj_alpha,
        hair_m_warped,
        prot_union,
        rim_col01=lighting["rim_color"],
        amb_col01=lighting["ambient_color"],
    )
    if lighting["key_side"] in ("left", "right") and float(rgb255_to_lab(tone_map["mid"][None, None, :])[0, 0, 1]) < -4.0:
        lighting["key_color"] = [0.94, 0.99, 0.98]

    if engine == "tutorial-rake":
        from hair_rake_engine import plan_and_render_tutorial_hair

        rk = plan_and_render_tutorial_hair(
            subj_rgb=subj_rgb,
            subj_alpha=subj_alpha,
            hair_m=hair_m_warped,
            face_snap=face_snap,
            eyes_m=eyes_m,
            clothes_m=clo_snap,
            arms_m=arms_snap,
            acc_m=acc_snap,
            all_skin_snap=all_skin_snap,
            occ_snap=occ_snap,
            tone_map=tone_map,
            lighting=lighting,
            style=style,
            boldness=boldness,
            wind_dir=wind_dir,
            tx_map=tx_map,
            ty_map=ty_map,
            seed=seed,
        )
        beh_rgb, beh_a = rk["behind"]
        mid_rgb, mid_a = rk["mid_connect"]
        frt_rgb, frt_a = rk["bright_weave"]
        rim_add_rgb, rim_add_w = rk["glow_crown"]
        front_guard = rk["front_guard"]

        beh_rgb = _match_sensor_grain(beh_rgb, beh_a, subj_rgb, subj_alpha, hair_m_warped, seed + 11)
        mid_rgb = _match_sensor_grain(mid_rgb, mid_a, subj_rgb, subj_alpha, hair_m_warped, seed + 17)
        frt_rgb = _match_sensor_grain(frt_rgb, frt_a, subj_rgb, subj_alpha, hair_m_warped, seed + 29)
        rim_add_rgb = _match_sensor_grain(rim_add_rgb, rim_add_w, subj_rgb, subj_alpha, hair_m_warped, seed + 43)

        # Tutorial uses ZERO heavy black cast-shadow Multiply blobs!
        shd_mul_rgb = np.full((h, w, 3), 255.0, dtype=np.float32)
        shd_mul_a = np.zeros((h, w), dtype=np.float32)
        polish_rgba = _build_real_hair_polish_layer(subj_rgb, subj_alpha, hair_m_warped, front_guard, lighting["ambient_color"])

        write_rgba_u8(p_beh, beh_rgb, beh_a, overwrite=overwrite)
        write_rgba_u8(p_mid, mid_rgb, mid_a, overwrite=overwrite)
        write_rgba_u8(p_shd, shd_mul_rgb, shd_mul_a, overwrite=overwrite)
        write_rgba_u8(p_frt, frt_rgb, frt_a, overwrite=overwrite)
        write_rgba_u8(p_rim, rim_add_rgb, rim_add_w, overwrite=overwrite)
        write_rgba_u8(p_pol, polish_rgba[:, :, :3], polish_rgba[:, :, 3], overwrite=overwrite)
        write_png_u8(p_trim, np.full((h, w), 255, dtype=np.uint8), overwrite=overwrite)

        any_paint = np.maximum.reduce([beh_a, mid_a, frt_a, rim_add_w]) > 0.05
        eyes_px = int((any_paint & (eyes_m >= 0.20)).sum()) if eyes_m.max() > 0.05 else 0
        beh_px = int((beh_a > 0.05).sum())
        mid_px = int((mid_a > 0.05).sum())
        frt_px = int((frt_a > 0.05).sum())
        rim_px = int((rim_add_w > 0.05).sum())
        ext_px = int((any_paint & (subj_alpha < 0.45)).sum())
        tot_front_px = mid_px + frt_px + rim_px

        meta = {
            "ok": True,
            "engine": "tutorial_rake_2d + rigid_mls",
            "size": [w, h],
            "render_size": [w, h],
            "palette": palette_hex,
            "lighting": lighting,
            "plan": {
                "resolved_style": rk["resolved_style"],
                "boldness": boldness,
                "wind_dir": wind_dir,
                "wx_sign": rk["wx_sign"],
                "layer_stroke_counts": rk["counts"],
            },
            "reshape": reshape_info,
            "calibration": {"eyedropper_local": True},
            "coverage": {
                "behind_px": beh_px,
                "mid_connect_px": mid_px,
                "front_px": frt_px,
                "glow_crown_px": rim_px,
                "front_share_pct": round(100.0 * tot_front_px / max(1, beh_px + tot_front_px), 2),
                "external_air_px": ext_px,
                "external_canvas_pct": round(100.0 * ext_px / max(1, w * h), 2),
                "external_air_transparency_pct": rk["external_air_transparency_pct"],
                "trimmed_stiff_px": 0,
            },
            "safety": {
                "painted_in_eyes_px": eyes_px,
                "root_seam_ok": True,
            },
            "outputs": {
                "subject_reshaped_rgba": str(p_subj_clean),
                "reshape_patch": str(p_reshape_patch),
                "behind": str(p_beh),
                "mid_connect": str(p_mid),
                "cast_shadow_mul": str(p_shd),
                "front": str(p_frt),
                "backlit_rim_add": str(p_rim),
                "real_polish_softlight": str(p_pol),
                "trim_mask": str(p_trim),
            },
        }
        (out_dir / "hair_sculpt_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return meta

    # Legacy Blender Cycles hair_curves path (optional `--engine blender`)
    locks, plan_meta = _plan_anime_hair_locks(
        subj_rgb=subj_rgb,
        subj_alpha=subj_alpha,
        hair_m=hair_m_warped,
        face_snap=face_snap,
        clothes_m=clo_snap,
        arms_m=arms_snap,
        acc_m=acc_snap,
        tone_map=tone_map,
        style=style,
        boldness=boldness,
        wind_dir=wind_dir,
        seed=seed,
    )

    rw = max(640, int(round(w * float(np.clip(render_scale, 0.5, 1.0)))))
    rh = max(640, int(round(h * float(np.clip(render_scale, 0.5, 1.0)))))
    spec = {
        "width": w,
        "height": h,
        "render_width": rw,
        "render_height": rh,
        "samples": int(samples),
        "seed": int(seed),
        "hair_type": "STRIP",
        "hair_subdiv": 2,
        "use_dof": True,
        "dof_fstop": 3.2,
        "lighting": lighting,
        "material": {
            "model": "CHIANG",
            "sss_core": 0.08,
            "sss_tip": 0.26,
            "roughness": 0.22,
            "radial_roughness": 0.29,
            "coat": 0.16,
            "random_color": 0.10,
            "random_roughness": 0.24,
            "reflection": 1.05,
            "transmission": 1.25,
            "secondary_reflection": 1.15,
        },
        "locks": locks,
    }
    p_spec = out_dir / "hair_sculpt_spec.json"
    p_spec.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")

    # Stage B2: Invoke Blender 5.2 Cycles OptiX renderer
    raw_dir = out_dir / "raw_blender"
    raw_dir.mkdir(parents=True, exist_ok=True)
    blender_exe = _find_blender_exe()
    kit_script = Path(__file__).resolve().parent / "blender_hair_sculpt.py"
    proc = subprocess.run(
        [str(blender_exe), "--background", "--python", str(kit_script), "--", "--spec", str(p_spec), "--out-dir", str(raw_dir)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Blender hair-sculpt failed (rc={proc.returncode}): {proc.stderr[-1200:] or proc.stdout[-1200:]}")

    blender_meta: dict[str, Any] = {}
    for ln in reversed(proc.stdout.splitlines()):
        ln_s = ln.strip()
        if ln_s.startswith("{") and ln_s.endswith("}"):
            try:
                blender_meta = json.loads(ln_s)
                break
            except Exception:
                pass

    # Stage C: Premultiplied resize (zero edge ringing!), Lab calibration, smooth guards & grain match
    beh_rgb, beh_a = _resize_rgba_premul(raw_dir / "raw_behind_rgba.png", (h, w)) if (raw_dir / "raw_behind_rgba.png").exists() else (np.zeros((h, w, 3), np.float32), np.zeros((h, w), np.float32))
    frt_rgb, frt_a = _resize_rgba_premul(raw_dir / "raw_front_rgba.png", (h, w)) if (raw_dir / "raw_front_rgba.png").exists() else (np.zeros((h, w, 3), np.float32), np.zeros((h, w), np.float32))
    _, shd_a = _resize_rgba_premul(raw_dir / "raw_cast_shadow_rgba.png", (h, w)) if (raw_dir / "raw_cast_shadow_rgba.png").exists() else (np.zeros((h, w, 3), np.float32), np.zeros((h, w), np.float32))

    # Build strict safety & smooth occluder masks (NO boolean np.where polygon step cuts!)
    eyes_guard = (
        cv2.GaussianBlur(cv2.dilate((eyes_m >= 0.20).astype(np.uint8), ell_kernel(28)).astype(np.float32), (0, 0), 4.0)
        if eyes_m.max() > 0.05
        else np.zeros((h, w), np.float32)
    )
    face_protect_center = np.zeros((h, w), np.float32)
    if (face_snap > 0.45).sum() >= 300:
        fys, fxs = np.where(face_snap > 0.45)
        fx0, fx1, fy0, fy1 = int(fxs.min()), int(fxs.max()), int(fys.min()), int(fys.max())
        fcx = 0.5 * (fx0 + fx1)
        fw_f, fh_f = max(1, fx1 - fx0), max(1, fy1 - fy0)
        yy_f, xx_f = np.mgrid[0:h, 0:w].astype(np.float32)
        ell = ((xx_f - fcx) / (0.46 * fw_f)) ** 2 + ((yy_f - (fy0 + 0.54 * fh_f)) / (0.50 * fh_f)) ** 2
        face_protect_center = np.clip((1.12 - ell) / 0.22, 0.0, 1.0)

    lab_s = rgb255_to_lab(subj_rgb)
    y_bow_max_c = int(fy1 + 0.08 * fh_f) if (face_snap > 0.45).sum() >= 300 else int(0.40 * h)
    above_chin_c = np.arange(h)[:, None] <= y_bow_max_c
    white_bow = (subj_alpha > 0.80) & (lab_s[:, :, 0] > 76.0) & (np.hypot(lab_s[:, :, 1], lab_s[:, :, 2]) < 12.0) & (hair_m_warped < 0.50) & above_chin_c
    bow_guard = cv2.GaussianBlur(white_bow.astype(np.float32), (0, 0), 1.2)
    arms_flush = cv2.GaussianBlur((arms_snap >= 0.48).astype(np.float32), (0, 0), 0.9) if arms_snap.max() > 0.05 else np.zeros((h, w), np.float32)
    acc_flush = cv2.GaussianBlur((acc_snap >= 0.45).astype(np.float32), (0, 0), 0.9) if acc_snap.max() > 0.05 else np.zeros((h, w), np.float32)

    occ_nonface = occ_snap * (1.0 - np.clip(face_snap * 1.4, 0.0, 1.0))
    occ_soft = cv2.GaussianBlur(np.clip((occ_nonface - 0.35) / 0.45, 0.0, 1.0), (0, 0), 3.5)

    behind_guard = np.clip(np.maximum.reduce([eyes_guard, face_protect_center, bow_guard, acc_flush]), 0.0, 1.0)
    front_guard = np.clip(np.maximum.reduce([eyes_guard, face_protect_center, arms_flush, occ_soft, bow_guard, acc_flush]), 0.0, 1.0)

    beh_rgb, cal_beh = _calibrate_overlap_lab(beh_rgb, beh_a, subj_rgb, subj_alpha, hair_m_warped)
    frt_rgb, cal_frt = _calibrate_overlap_lab(frt_rgb, frt_a, subj_rgb, subj_alpha, hair_m_warped)

    if float(frt_a.max()) > 0.05:
        pm_f = cv2.GaussianBlur(frt_rgb * frt_a[:, :, None], (0, 0), 0.22)
        a_f = np.clip(cv2.GaussianBlur(frt_a, (0, 0), 0.22), 0.0, 1.0)
        frt_rgb = np.where(a_f[:, :, None] > 1e-4, pm_f / np.maximum(a_f[:, :, None], 1e-4), frt_rgb)
        frt_a = a_f

    beh_a = np.clip(beh_a * (1.0 - behind_guard), 0.0, 1.0)
    frt_a = np.clip(frt_a * (1.0 - front_guard), 0.0, 1.0)

    beh_rgb = _match_sensor_grain(beh_rgb, beh_a, subj_rgb, subj_alpha, hair_m_warped, seed + 11)
    frt_rgb = _match_sensor_grain(frt_rgb, frt_a, subj_rgb, subj_alpha, hair_m_warped, seed + 29)

    arms_shd_guard = (
        cv2.GaussianBlur(cv2.dilate((arms_snap >= 0.35).astype(np.uint8), ell_kernel(8)).astype(np.float32), (0, 0), 3.0)
        if arms_snap.max() > 0.05
        else np.zeros((h, w), np.float32)
    )
    hair_lit_guard = cv2.GaussianBlur((hair_m_warped >= 0.35).astype(np.float32), (0, 0), 4.0)
    shd_strength = np.clip(
        shd_a
        * subj_alpha
        * (1.0 - eyes_guard)
        * (1.0 - bow_guard)
        * (1.0 - face_protect_center)
        * (1.0 - hair_lit_guard)
        * (1.0 - arms_shd_guard)
        * (1.0 - 0.50 * clo_snap),
        0.0,
        0.65,
    )
    amb_tint = np.array([
        0.22 + 0.20 * lighting["ambient_color"][0],
        0.22 + 0.20 * lighting["ambient_color"][1],
        0.26 + 0.24 * lighting["ambient_color"][2],
    ], dtype=np.float32) * 255.0
    shd_mul_rgb = 255.0 * (1.0 - shd_strength[:, :, None]) + amb_tint[None, None, :] * shd_strength[:, :, None]
    shd_mul_a = np.clip(shd_strength * 1.25, 0.0, 1.0)

    rim_col255 = np.array(lighting["rim_color"], dtype=np.float32) * 255.0
    beh_lum = luma255(beh_rgb) / 255.0
    frt_lum = luma255(frt_rgb) / 255.0
    outside_subj = np.clip(1.0 - subj_alpha * 1.35, 0.0, 1.0)
    beh_crest = np.clip((beh_lum - cv2.GaussianBlur(beh_lum, (0, 0), 2.5)) * 2.4, 0.0, 1.0) * (beh_a ** 1.6) * outside_subj
    frt_crest = np.clip((frt_lum - cv2.GaussianBlur(frt_lum, (0, 0), 2.5)) * 2.4, 0.0, 1.0) * (frt_a ** 1.6) * (1.0 - front_guard)
    rim_add_w = np.clip(np.maximum(beh_crest, frt_crest) * 0.28, 0.0, 0.28)
    rim_add_rgb = np.clip(0.75 * np.maximum(beh_rgb, frt_rgb) + 0.25 * rim_col255[None, None, :], 0.0, 255.0)

    polish_rgba = _build_real_hair_polish_layer(subj_rgb, subj_alpha, hair_m_warped, front_guard, lighting["ambient_color"])

    trim_mask = np.ones((h, w), dtype=np.float32)
    trimmed_px = 0
    if trim_stiff_ends:
        body_keep = np.maximum.reduce([
            face_snap,
            eyes_guard,
            arms_snap,
            clo_snap,
            acc_snap,
            all_skin_snap,
            white_bow.astype(np.float32),
            cv2.dilate((np.maximum.reduce([face_snap, arms_snap, clo_snap, all_skin_snap]) >= 0.22).astype(np.uint8), ell_kernel(16)).astype(np.float32),
        ])
        new_hair_cover = cv2.dilate((np.maximum(beh_a, frt_a) > 0.18).astype(np.uint8), ell_kernel(24)).astype(np.float32)
        d_body = cv2.distanceTransform((body_keep < 0.22).astype(np.uint8), cv2.DIST_L2, 5)
        d_edge = cv2.distanceTransform((subj_alpha >= 0.50).astype(np.uint8), cv2.DIST_L2, 5)
        stiff_outer = (
            (hair_m_warped >= 0.35)
            & (body_keep < 0.22)
            & (d_body >= 30.0)
            & (d_edge <= 22.0)
            & (new_hair_cover < 0.50)
        )
        ys_face_t = np.where(face_snap >= 0.45)[0]
        ys_t, _ = np.where((hair_m_warped >= 0.45) & (subj_alpha >= 0.55))
        if len(ys_face_t) and len(ys_t):
            fy_chin = float(ys_face_t.max())
            fh_c = float(max(40.0, ys_face_t.max() - ys_face_t.min()))
            y_cut = int(max(fy_chin + 0.85 * fh_c, ys_t.min() + 0.68 * (ys_t.max() - ys_t.min())))
            stiff_outer[:y_cut, :] = False
        elif len(ys_t):
            y_cut = int(ys_t.min() + 0.68 * (ys_t.max() - ys_t.min()))
            stiff_outer[:y_cut, :] = False
        if stiff_outer.any():
            cut_soft = cv2.GaussianBlur(stiff_outer.astype(np.float32), (0, 0), 1.6) * (1.0 - np.clip(body_keep, 0.0, 1.0))
            trimmed_px = int((cut_soft > 0.5).sum())
            trim_mask = np.clip(1.0 - cut_soft, 0.0, 1.0)

    write_rgba_u8(p_beh, beh_rgb, beh_a, overwrite=overwrite)
    write_rgba_u8(p_shd, shd_mul_rgb, shd_mul_a, overwrite=overwrite)
    write_rgba_u8(p_frt, frt_rgb, frt_a, overwrite=overwrite)
    write_rgba_u8(p_rim, rim_add_rgb, rim_add_w, overwrite=overwrite)
    write_rgba_u8(p_pol, polish_rgba[:, :, :3], polish_rgba[:, :, 3], overwrite=overwrite)
    write_png_u8(p_trim, np.clip(np.round(trim_mask * 255.0), 0, 255).astype(np.uint8), overwrite=overwrite)

    any_paint = np.maximum(beh_a, frt_a) > 0.05
    eyes_px = int((any_paint & (eyes_m >= 0.20)).sum()) if eyes_m.max() > 0.05 else 0
    beh_px = int((beh_a > 0.05).sum())
    frt_px = int((frt_a > 0.05).sum())
    ext_px = int((any_paint & (subj_alpha < 0.45)).sum())

    meta = {
        "ok": True,
        "engine": "blender_cycles_optix_hair_curves + rigid_mls",
        "size": [w, h],
        "render_size": [rw, rh],
        "palette": palette_hex,
        "lighting": lighting,
        "plan": plan_meta,
        "reshape": reshape_info,
        "calibration": {"behind": cal_beh, "front": cal_frt},
        "blender": blender_meta,
        "coverage": {
            "behind_px": beh_px,
            "front_px": frt_px,
            "external_air_px": ext_px,
            "external_canvas_pct": round(100.0 * ext_px / max(1, w * h), 2),
            "trimmed_stiff_px": trimmed_px,
        },
        "safety": {
            "painted_in_eyes_px": eyes_px,
            "root_seam_ok": bool(cal_beh.get("calibrated") or cal_frt.get("calibrated")),
        },
        "outputs": {
            "subject_reshaped_rgba": str(p_subj_clean),
            "reshape_patch": str(p_reshape_patch),
            "behind": str(p_beh),
            "cast_shadow_mul": str(p_shd),
            "front": str(p_frt),
            "backlit_rim_add": str(p_rim),
            "real_polish_softlight": str(p_pol),
            "trim_mask": str(p_trim),
            "spec_json": str(p_spec),
        },
    }
    (out_dir / "hair_sculpt_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def build_hair_sculpt_pcraft_cmds(
    meta: dict[str, Any],
    *,
    subject_layer: str,
    behind_group: str = "02b_HAIR_SCULPT_BEHIND",
    front_group: str = "03b_HAIR_SCULPT_FRONT",
    behind_opacity: float | None = None,
    mid_opacity: float | None = None,
    bright_opacity: float | None = None,
    rim_opacity: float | None = None,
    shadow_opacity: float = 0.0,
    polish_opacity: float | None = None,
) -> tuple[list[str], list[str]]:
    """Return (behind_cmds, front_cmds) to place non-destructive anime/tutorial hair layers in PhotoCraft."""
    w, h = int(meta["size"][0]), int(meta["size"][1])
    outs = meta["outputs"]
    eng_str = str(meta.get("engine", ""))
    is_anime = "anime_sculpt" in eng_str
    is_tutorial = "tutorial_rake" in eng_str
    is_layered = is_anime or is_tutorial
    cx, cy = w * 0.5, h * 0.5

    if is_anime:
        beh_op = 1.0 if behind_opacity is None else behind_opacity
        mid_op = 1.0 if mid_opacity is None else mid_opacity
        frt_op = 1.0 if bright_opacity is None else bright_opacity
        default_rim = 0.86 if meta.get("plan", {}).get("resolved_style") == "short-braids-wispy" else 0.72
        rim_op = default_rim if rim_opacity is None else rim_opacity
        pol_op = 0.58 if polish_opacity is None else polish_opacity
        rim_blend = "Screen"
    elif is_tutorial:
        beh_op = 0.86 if behind_opacity is None else behind_opacity
        mid_op = 0.82 if mid_opacity is None else mid_opacity
        frt_op = 0.80 if bright_opacity is None else bright_opacity
        rim_op = 0.46 if rim_opacity is None else rim_opacity
        pol_op = 0.42 if polish_opacity is None else polish_opacity
        rim_blend = "Linear Dodge (Add)"
    else:
        beh_op = 1.0
        mid_op = 0.0
        frt_op = 1.0
        rim_op = 0.46 if rim_opacity is None else rim_opacity
        pol_op = 0.42 if polish_opacity is None else polish_opacity
        rim_blend = "Linear Dodge (Add)"

    behind_cmds = [
        "--cmd", "layer.new.group", "--params", _J({"name": behind_group, "mode": "passThrough", "opacity": 100}),
        "--cmd", "layer.moveTo", "--params", _J({"target": subject_layer, "position": "below"}),
        "--cmd", "layer.select", "--params", _J({"layer": behind_group}),
        "--cmd", "file.placeEmbedded", "--params", _J({"path": outs["behind"], "fit": False, "smartObject": True, "center": [cx, cy]}),
        "--cmd", "layer.setProps", "--params", _J({"name": "hair_sculpt_01_behind", "blend": "Normal", "opacity": beh_op, "visible": True}),
    ]

    front_cmds = [
        "--cmd", "layer.new.group", "--params", _J({"name": front_group, "mode": "passThrough", "opacity": 100}),
        "--cmd", "layer.moveTo", "--params", _J({"target": subject_layer, "position": "above"}),
        "--cmd", "layer.select", "--params", _J({"layer": front_group}),
        "--cmd", "file.placeEmbedded", "--params", _J({"path": outs["real_polish_softlight"], "fit": False, "smartObject": True, "center": [cx, cy]}),
        "--cmd", "layer.setProps", "--params", _J({"name": "hair_sculpt_05_real_polish", "blend": "Soft Light", "opacity": pol_op, "visible": True}),
    ]
    anchor_layer = "hair_sculpt_05_real_polish"

    if is_layered and "mid_connect" in outs and Path(outs["mid_connect"]).exists():
        front_cmds += [
            "--cmd", "file.placeEmbedded", "--params", _J({"path": outs["mid_connect"], "fit": False, "smartObject": True, "center": [cx, cy]}),
            "--cmd", "layer.setProps", "--params", _J({"name": "hair_sculpt_02_mid_connect", "blend": "Normal", "opacity": mid_op, "visible": True}),
            "--cmd", "layer.moveTo", "--params", _J({"target": anchor_layer, "position": "above"}),
        ]
        anchor_layer = "hair_sculpt_02_mid_connect"
    elif (not is_layered) and shadow_opacity > 0.0 and "cast_shadow_mul" in outs:
        front_cmds += [
            "--cmd", "file.placeEmbedded", "--params", _J({"path": outs["cast_shadow_mul"], "fit": False, "smartObject": True, "center": [cx, cy]}),
            "--cmd", "layer.setProps", "--params", _J({"name": "hair_sculpt_02_cast_shadow", "blend": "Multiply", "opacity": shadow_opacity, "visible": True}),
            "--cmd", "layer.moveTo", "--params", _J({"target": anchor_layer, "position": "above"}),
        ]
        anchor_layer = "hair_sculpt_02_cast_shadow"

    front_cmds += [
        "--cmd", "file.placeEmbedded", "--params", _J({"path": outs["front"], "fit": False, "smartObject": True, "center": [cx, cy]}),
        "--cmd", "layer.setProps", "--params", _J({"name": "hair_sculpt_03_front", "blend": "Normal", "opacity": frt_op, "visible": True}),
        "--cmd", "layer.moveTo", "--params", _J({"target": anchor_layer, "position": "above"}),
        "--cmd", "file.placeEmbedded", "--params", _J({"path": outs["backlit_rim_add"], "fit": False, "smartObject": True, "center": [cx, cy]}),
        "--cmd", "layer.setProps", "--params", _J({"name": "hair_sculpt_04_backlit_rim", "blend": rim_blend, "opacity": rim_op, "visible": True}),
        "--cmd", "layer.moveTo", "--params", _J({"target": "hair_sculpt_03_front", "position": "above"}),
    ]
    return behind_cmds, front_cmds
