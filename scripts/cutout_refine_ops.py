"""Non-generative edge-integrity matting refiner: boundary high-res tiles, vectorized F/B solver & strand continuity."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def _read_rgb_f32(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _read_gray_f32(path: Path, size_wh: tuple[int, int] | None = None) -> np.ndarray:
    im = Image.open(path).convert("L")
    if size_wh is not None and im.size != size_wh:
        im = im.resize(size_wh, Image.Resampling.BILINEAR)
    return np.array(im, dtype=np.float32) / 255.0


def _write_png_u8(path: Path, arr: np.ndarray, mode: str, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.round(arr), 0, 255).astype(np.uint8), mode).save(path, "PNG")


def _dilate_radius(mask_bool: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0.0 or not bool(mask_bool.any()):
        return mask_bool.copy()
    return cv2.distanceTransform((~mask_bool).astype(np.uint8), cv2.DIST_L2, 5) <= float(radius)


def _erode_radius(mask_bool: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0.0 or not bool(mask_bool.any()):
        return np.zeros_like(mask_bool, dtype=bool)
    return cv2.distanceTransform(mask_bool.astype(np.uint8), cv2.DIST_L2, 5) > float(radius)


def _blur_fast(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma < 16.0:
        return cv2.GaussianBlur(arr, (0, 0), sigma)
    h, w = arr.shape[:2]
    ds = 8 if sigma >= 80.0 else 4
    sw, sh = max(8, w // ds), max(8, h // ds)
    small = cv2.resize(arr, (sw, sh), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(small, (0, 0), sigma / float(ds))
    out = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)
    if arr.ndim == 3 and out.ndim == 2:
        out = out[:, :, None]
    return out


def _run_birefnet_array(sess: Any, rgb_u8: np.ndarray) -> np.ndarray:
    h, w = rgb_u8.shape[:2]
    inp_meta = sess.get_inputs()[0]
    target_sz = int(inp_meta.shape[-1]) if isinstance(inp_meta.shape[-1], int) and inp_meta.shape[-1] > 0 else 1024
    rsz = cv2.resize(rgb_u8, (target_sz, target_sz), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    chw = np.transpose((rsz - mean) / std, (2, 0, 1))[None, ...]
    if "float16" in str(inp_meta.type):
        chw = chw.astype(np.float16)
    pred = np.asarray(sess.run(None, {inp_meta.name: chw})[-1], dtype=np.float32).squeeze()
    if pred.ndim == 3:
        pred = pred[-1]
    if float(pred.min()) < -0.05 or float(pred.max()) > 1.05:
        pred = 1.0 / (1.0 + np.exp(-np.clip(pred, -30.0, 30.0)))
    pred = np.clip(pred, 0.0, 1.0)
    if pred.shape != (h, w):
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
    return pred


def _grid_starts(dim: int, tile: int, step: int) -> list[int]:
    if dim <= tile:
        return [0]
    n = max(2, math.ceil((dim - tile) / max(1, step)) + 1)
    return [int(round(v)) for v in np.linspace(0, dim - tile, n)]


def _hi_res_boundary_tiles(
    rgb_u8: np.ndarray,
    alpha_coarse: np.ndarray,
    birefnet_model: Path,
    hair_mask: np.ndarray | None = None,
    tile_size: int = 1664,
    stride: int = 1360,
) -> tuple[np.ndarray, int]:
    """Run BiRefNet-lite on 1664px context boundary windows to recover thin hair & lace silhouette continuity."""
    from cuda_env import create_onnx_session

    h, w = alpha_coarse.shape[:2]
    if max(h, w) <= int(tile_size * 1.15):
        return alpha_coarse.copy(), 0

    sess = create_onnx_session(birefnet_model)

    # Perimeter unknown band + hair outline band (keeps solid interior & far background untouched)
    soft_band = (alpha_coarse > 0.015) & (alpha_coarse < 0.985)
    edge_grad = np.abs(cv2.Laplacian((alpha_coarse * 255.0).astype(np.float32), cv2.CV_32F)) > 6.0
    perim = _dilate_radius(soft_band | edge_grad, 37.0)
    if hair_mask is not None:
        hair_perim = _dilate_radius(hair_mask > 0.20, 27.0)
        perim = perim | (hair_perim & (alpha_coarse < 0.995))

    # Raised cosine spatial blending weight per tile
    wy = np.hanning(tile_size).astype(np.float32)
    wx = np.hanning(tile_size).astype(np.float32)
    win = np.maximum(np.outer(wy, wx), 1e-4)

    accum = np.zeros((h, w), dtype=np.float32)
    wsum = np.zeros((h, w), dtype=np.float32)
    tiles_run = 0

    ys = _grid_starts(h, tile_size, stride)
    xs = _grid_starts(w, tile_size, stride)

    for y0 in ys:
        y1 = min(h, y0 + tile_size)
        for x0 in xs:
            x1 = min(w, x0 + tile_size)
            tile_perim = perim[y0:y1, x0:x1]
            tile_coarse = alpha_coarse[y0:y1, x0:x1]
            # Must touch boundary AND contain sufficient coarse fg/bg context so BiRefNet recognizes the subject
            fg_frac = float((tile_coarse > 0.5).mean())
            min_perim = 0.015 if (hair_mask is not None and float((hair_mask[y0:y1, x0:x1] > 0.20).mean()) >= 0.005) else 0.025
            if float(tile_perim.mean()) < min_perim or fg_frac < 0.04 or fg_frac > 0.96:
                continue
            pred = _run_birefnet_array(sess, rgb_u8[y0:y1, x0:x1])
            th, tw = y1 - y0, x1 - x0
            tw_mat = win[:th, :tw]
            accum[y0:y1, x0:x1] += pred * tw_mat
            wsum[y0:y1, x0:x1] += tw_mat
            tiles_run += 1

    if tiles_run == 0:
        return alpha_coarse.copy(), 0

    tile_avg = np.where(wsum > 1e-4, accum / np.maximum(wsum, 1e-4), alpha_coarse)
    # Tightly confine perimeter refinement so BiRefNet tiles never hallucinate soft gray/cream background halos
    perim_w = cv2.GaussianBlur(perim.astype(np.float32), (0, 0), 8.0)
    solid_core = _erode_radius(alpha_coarse >= 0.99, 12.0)
    # Strict far-background gate: anything > 8px outside coarse foreground >= 0.04 stays 0 unless high-confidence (>0.65)
    near_coarse = _dilate_radius(alpha_coarse >= 0.04, 7.0)

    fused = 0.65 * tile_avg + 0.35 * alpha_coarse
    if hair_mask is not None:
        hm_ext = _dilate_radius(hair_mask > 0.25, 10.0).astype(np.float32)
        hm_ext = cv2.GaussianBlur(hm_ext, (0, 0), 4.0)
        # Keep continuous hair filaments cleanly while preventing low-confidence (<0.42) outer haze
        tile_clean = np.where(tile_avg > 0.38, tile_avg, tile_avg * 0.25)
        hair_fuse = np.maximum(fused, 0.88 * np.maximum(tile_clean, alpha_coarse))
        fused = fused * (1.0 - hm_ext * 0.60) + hair_fuse * (hm_ext * 0.60)

    out = alpha_coarse * (1.0 - perim_w) + fused * perim_w
    out = np.where((~near_coarse) & (out < 0.62), 0.0, out)
    # Tighten soft outer veil (> 6px from solid foreground) cleanly
    core_fg_dist = cv2.distanceTransform((alpha_coarse < 0.72).astype(np.uint8), cv2.DIST_L2, 5)
    soft_veil = (out < 0.36) & (core_fg_dist > 8.0)
    out = np.where(soft_veil, out * np.clip((out - 0.20) / 0.16, 0.0, 1.0), out)
    out[solid_core] = np.maximum(out[solid_core], alpha_coarse[solid_core])
    return np.clip(out, 0.0, 1.0), tiles_run


def estimate_foreground_background_ml(
    image_rgb01: np.ndarray,
    alpha01: np.ndarray,
    regularization: float = 1e-4,
    n_small_iters: int = 12,
    n_large_iters: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized coarse-to-fine multi-level solver for I = alpha*F + (1-alpha)*B (Germer et al. 2020 method).

    Pure NumPy/OpenCV implementation with zero Cython/Numba dependency, safe on Python 3.12+ / NumPy 2.x.
    """
    h0, w0 = alpha01.shape[:2]
    levels: list[tuple[np.ndarray, np.ndarray]] = [(image_rgb01.astype(np.float32), alpha01.astype(np.float32))]
    cur_h, cur_w = h0, w0
    while min(cur_h, cur_w) > 36:
        nh = max(16, (cur_h + 1) // 2)
        nw = max(16, (cur_w + 1) // 2)
        prev_i, prev_a = levels[-1]
        img_d = cv2.resize(prev_i, (nw, nh), interpolation=cv2.INTER_AREA)
        alp_d = cv2.resize(prev_a, (nw, nh), interpolation=cv2.INTER_AREA)
        levels.append((img_d, np.clip(alp_d, 0.0, 1.0)))
        cur_h, cur_w = nh, nw

    F = B = None
    # Red-black checkerboard Gauss-Seidel passes per pyramid scale
    for lvl_idx in range(len(levels) - 1, -1, -1):
        I_lvl, a_lvl = levels[lvl_idx]
        lh, lw = a_lvl.shape[:2]
        if F is None:
            F = I_lvl.copy()
            B = I_lvl.copy()
        else:
            F = cv2.resize(F, (lw, lh), interpolation=cv2.INTER_LINEAR)
            B = cv2.resize(B, (lw, lh), interpolation=cv2.INTER_LINEAR)

        a = a_lvl[:, :, None]
        inv_a = 1.0 - a
        a2 = a * a
        ia2 = inv_a * inv_a
        a_ia = a * inv_a

        n_it = n_small_iters if min(lh, lw) <= 256 else n_large_iters
        # Exact 4-neighbor average via 3x3 cross kernel divided by valid neighbor count (2..4)
        k_nb = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        deg = cv2.filter2D(np.ones((lh, lw), dtype=np.float32), -1, k_nb, borderType=cv2.BORDER_CONSTANT)[:, :, None]
        deg = np.maximum(deg, 1.0)
        reg = float(regularization)

        yy, xx = np.mgrid[0:lh, 0:lw]
        even = ((yy + xx) & 1) == 0
        odd = ~even

        a00 = a2 + reg
        a11 = ia2 + reg
        a01 = a_ia
        det = np.maximum(a00 * a11 - a01 * a01, 1e-8)

        for _ in range(n_it):
            for half_mask in (even, odd):
                F_nb = cv2.filter2D(F, -1, k_nb, borderType=cv2.BORDER_CONSTANT) / deg
                B_nb = cv2.filter2D(B, -1, k_nb, borderType=cv2.BORDER_CONSTANT) / deg
                b0 = a * I_lvl + reg * F_nb
                b1 = inv_a * I_lvl + reg * B_nb
                F_new = np.clip((a11 * b0 - a01 * b1) / det, 0.0, 1.0)
                B_new = np.clip((a00 * b1 - a01 * b0) / det, 0.0, 1.0)
                m3 = half_mask[:, :, None]
                F = np.where(m3, F_new, F)
                B = np.where(m3, B_new, B)

        # Anchor strictly opaque foreground & transparent background directly where alpha is unambiguous
        solid_f = (a_lvl >= 0.992)[:, :, None]
        solid_b = (a_lvl <= 0.008)[:, :, None]
        F = np.where(solid_f, I_lvl, F)
        B = np.where(solid_b, I_lvl, B)

    assert F is not None and B is not None
    return np.clip(F, 0.0, 1.0), np.clip(B, 0.0, 1.0)


def run_cutout_refine(
    *,
    input_path: Path,
    out_dir: Path,
    birefnet_model_path: Path,
    coarse_mask_path: Path | None = None,
    body_parse_dir: Path | None = None,
    hi_res_tiles: bool = True,
    clean_pockets: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """High-res boundary tile matting + multi-level F/B decontamination + hair strand integrity fix."""
    t0 = time.perf_counter()
    in_p = input_path.resolve()
    out_d = out_dir.resolve()
    out_d.mkdir(parents=True, exist_ok=True)

    rgb_f32 = _read_rgb_f32(in_p)
    h, w = rgb_f32.shape[:2]
    rgb_u8 = np.clip(np.round(rgb_f32 * 255.0), 0, 255).astype(np.uint8)

    if coarse_mask_path is not None and coarse_mask_path.exists():
        alpha_coarse = _read_gray_f32(coarse_mask_path.resolve(), (w, h))
    else:
        from cuda_env import create_onnx_session

        sess = create_onnx_session(birefnet_model_path)
        alpha_coarse = _run_birefnet_array(sess, rgb_u8)

    hair_mask = None
    prot_mask = None
    clothes_mask = None
    acc_mask = None
    skin_mask = None
    shoes_mask = None
    if body_parse_dir is not None and body_parse_dir.exists():
        def _opt_mask(fname: str) -> np.ndarray | None:
            p = body_parse_dir / fname
            return _read_gray_f32(p, (w, h)) if p.exists() else None

        hair_mask = _opt_mask("mask_hair.png")
        prot_mask = _opt_mask("mask_protected_body.png")
        clothes_mask = _opt_mask("mask_clothes_all.png")
        acc_mask = _opt_mask("mask_accessories.png")
        skin_mask = _opt_mask("mask_all_skin.png")
        shoes_mask = _opt_mask("mask_shoes.png")

    prot_parts = [m for m in (prot_mask, clothes_mask, acc_mask, skin_mask, shoes_mask) if m is not None]
    strict_protect = np.maximum.reduce(prot_parts) if prot_parts else None

    tiles_run = 0
    if hi_res_tiles:
        alpha_work, tiles_run = _hi_res_boundary_tiles(
            rgb_u8, alpha_coarse, birefnet_model_path, hair_mask=hair_mask
        )
    else:
        alpha_work = alpha_coarse.copy()

    # Guided edge snap on semi-transparent fringe using input luminance as crisp edge guide
    guide_gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
        alpha_guided = cv2.ximgproc.guidedFilter(guide=guide_gray, src=alpha_work.astype(np.float32), radius=7, eps=1.5e-4)
    else:
        alpha_guided = cv2.GaussianBlur(alpha_work, (0, 0), 1.0)
    fringe_band = cv2.GaussianBlur(((alpha_work > 0.015) & (alpha_work < 0.96)).astype(np.float32), (0, 0), 2.5)
    alpha_snapped = np.clip(alpha_work * (1.0 - 0.65 * fringe_band) + np.clip(alpha_guided, 0.0, 1.0) * (0.65 * fringe_band), 0.0, 1.0)

    # Prune detached tiny debris outside main foreground body (> 10px away and < 72px)
    fg_bin = (alpha_snapped >= 0.42).astype(np.uint8)
    n_cc, lbl, stats, _ = cv2.connectedComponentsWithStats(fg_bin, connectivity=8)
    specks_pruned = 0
    if n_cc > 2:
        areas = stats[1:, cv2.CC_STAT_AREA]
        main_id = int(np.argmax(areas)) + 1
        main_mask = (lbl == main_id).astype(np.uint8)
        dist_main = cv2.distanceTransform(1 - main_mask, cv2.DIST_L2, 5)
        keep_mask = np.ones((h, w), dtype=bool)
        for cid in range(1, n_cc):
            if cid == main_id:
                continue
            area = int(stats[cid, cv2.CC_STAT_AREA])
            cm = (lbl == cid)
            min_d = float(dist_main[cm].min())
            if area < 72 and min_d > 10.0:
                keep_mask[cm] = False
                specks_pruned += area
        if specks_pruned > 0:
            keep_f = cv2.GaussianBlur(keep_mask.astype(np.float32), (0, 0), 1.2)
            alpha_snapped *= keep_f

    # Bridge sub-4px hairline gaps strictly inside high-confidence hair regions (prevents chewed hair strands)
    if hair_mask is not None:
        hair_zone = (hair_mask > 0.35)
        bin_h = (alpha_snapped > 0.50).astype(np.uint8)
        closed_h = cv2.morphologyEx(bin_h, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        pinholes = hair_zone & (closed_h > 0) & (alpha_snapped < 0.72)
        if int(pinholes.sum()) > 0:
            ph_soft = cv2.GaussianBlur(pinholes.astype(np.float32), (0, 0), 1.2)
            alpha_snapped = np.clip(np.maximum(alpha_snapped, 0.86 * ph_soft), 0.0, 1.0)

    # Continuous optical trimap + color-line alpha matting + boundary spill decontamination
    pockets_cleaned = 0
    unknown_band = np.zeros((h, w), dtype=bool)
    local_hair_lab: np.ndarray | None = None
    conf = np.zeros((h, w), dtype=np.float32)
    alpha_opt = alpha_snapped.copy()

    def _nearest_fill(src_mask_bool: np.ndarray, val_img: np.ndarray, blur_sigma: float = 7.0) -> tuple[np.ndarray, np.ndarray]:
        dist, labels = cv2.distanceTransformWithLabels(
            (~src_mask_bool).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
        )
        ys_v, xs_v = np.nonzero(src_mask_bool)
        if ys_v.size == 0:
            return val_img.copy(), dist
        max_lbl = int(labels.max())
        lut = np.zeros((max_lbl + 1, val_img.shape[2]), dtype=np.float32)
        lut[labels[ys_v, xs_v]] = val_img[ys_v, xs_v]
        filled = lut[labels]
        if blur_sigma > 0.0:
            filled = cv2.GaussianBlur(filled, (0, 0), blur_sigma)
        return filled, dist

    if clean_pockets and hair_mask is not None:
        lab_in = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
        prot_guard = np.zeros((h, w), dtype=bool)
        if strict_protect is not None:
            prot_guard = cv2.dilate((strict_protect >= 0.48).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0

        solid_hair = (hair_mask > 0.50) & (alpha_snapped >= 0.90) & (~prot_guard)
        if int(solid_hair.sum()) >= 400:
            global_hair_lab = np.median(lab_in[solid_hair], axis=0).astype(np.float32)
            global_hair_rgb = np.median(rgb_f32[solid_hair], axis=0).astype(np.float32)
            # Extend solid hair reference across long twin-tails missed by SegFormer
            solid_hair_ext = solid_hair | (
                (alpha_snapped >= 0.92)
                & (~prot_guard)
                & (np.linalg.norm(lab_in - global_hair_lab[None, None, :], axis=2) < 26.0)
            )
            w_f = solid_hair_ext.astype(np.float32)
            w_f_s1 = _blur_fast(w_f, 40.0)
            w_f_s2 = _blur_fast(w_f, 150.0)
            den_f = (w_f_s1 + 0.25 * w_f_s2 + 0.02)[:, :, None]
            local_hair_lab = (
                _blur_fast(lab_in * w_f[:, :, None], 40.0)
                + 0.25 * _blur_fast(lab_in * w_f[:, :, None], 150.0)
                + 0.02 * global_hair_lab[None, None, :]
            ) / den_f
            local_hair_rgb = (
                _blur_fast(rgb_f32 * w_f[:, :, None], 40.0)
                + 0.25 * _blur_fast(rgb_f32 * w_f[:, :, None], 150.0)
                + 0.02 * global_hair_rgb[None, None, :]
            ) / den_f

            if clothes_mask is not None:
                solid_cloth = (clothes_mask > 0.55) & (alpha_snapped >= 0.90)
            else:
                solid_cloth = np.zeros((h, w), dtype=bool)
            if int(solid_cloth.sum()) >= 400:
                global_cloth_lab = np.median(lab_in[solid_cloth], axis=0).astype(np.float32)
                w_c = solid_cloth.astype(np.float32)
                num_c = _blur_fast(lab_in * w_c[:, :, None], 60.0) + 0.02 * global_cloth_lab[None, None, :]
                den_c = (_blur_fast(w_c, 60.0) + 0.02)[:, :, None]
                local_cloth_lab = num_c / den_c
                dist_cloth = np.linalg.norm(lab_in - local_cloth_lab, axis=2)
            else:
                dist_cloth = np.full((h, w), 999.0, dtype=np.float32)

            B_out_lab, dist_to_outer_bg = _nearest_fill(alpha_snapped <= 0.04, lab_in, blur_sigma=9.0)
            dist_hair = np.linalg.norm(lab_in - local_hair_lab, axis=2)
            cdist_hair = np.linalg.norm(lab_in[:, :, 1:3] - local_hair_lab[:, :, 1:3], axis=2)
            dist_bg_out = np.linalg.norm(lab_in - B_out_lab, axis=2)
            L_in = lab_in[:, :, 0] * (100.0 / 255.0)
            L_hair = local_hair_lab[:, :, 0] * (100.0 / 255.0)

            search_zone = (~prot_guard) & (alpha_snapped > 0.04) & (dist_to_outer_bg <= 160.0)
            not_hair = (dist_hair > 28.0) & ((cdist_hair > 12.0) | (np.abs(L_in - L_hair) > 24.0))
            not_cloth = (dist_cloth > 28.0)
            is_pocket_color = (dist_bg_out < 0.78 * dist_hair) | ((cdist_hair > 16.0) & (dist_hair > 40.0) & (L_in < L_hair - 18.0))
            cand_seed = search_zone & not_hair & not_cloth & is_pocket_color

            cand_u8 = cv2.morphologyEx(cand_seed.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            hair_region = _dilate_radius(solid_hair_ext | (hair_mask > 0.15), 70.0)
            subj_area = max(1, int((alpha_snapped > 0.5).sum()))
            npk, lpk, spk, _ = cv2.connectedComponentsWithStats(cand_u8, connectivity=8)
            bg_pockets = np.zeros((h, w), dtype=bool)
            for pid in range(1, npk):
                area = int(spk[pid, cv2.CC_STAT_AREA])
                if area < 64 or area > int(0.15 * subj_area):
                    continue
                comp = (lpk == pid)
                if int((comp & hair_region).sum()) >= 16:
                    bg_pockets |= comp
            pockets_cleaned = int(bg_pockets.sum())

            pocket_core = (
                cv2.erode(bg_pockets.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
                if pockets_cleaned > 0
                else np.zeros((h, w), dtype=bool)
            )
            def_bg = (alpha_snapped <= 0.03) | pocket_core
            pocket_band = (
                _dilate_radius(bg_pockets, 24.0)
                if pockets_cleaned > 0
                else np.zeros((h, w), dtype=bool)
            )
            outer_fringe = (dist_to_outer_bg <= 18.0) & (alpha_snapped > 0.02) & hair_region
            unknown_band = (pocket_band | outer_fringe) & (~prot_guard) & (~def_bg)

            if pockets_cleaned > 0 or bool(unknown_band.any()):
                B_local_rgb, _ = _nearest_fill(def_bg, rgb_f32, blur_sigma=5.0)
                F_local_rgb = local_hair_rgb
                dFB = F_local_rgb - B_local_rgb
                denom = np.sum(dFB * dFB, axis=2)
                proj = np.sum((rgb_f32 - B_local_rgb) * dFB, axis=2) / np.maximum(denom, 1e-4)
                alpha_opt = np.clip(proj, 0.0, 1.0)
                line_pt = B_local_rgb + alpha_opt[:, :, None] * dFB
                resid = np.linalg.norm(rgb_f32 - line_pt, axis=2)
                conf = np.clip((np.sqrt(denom) - 0.10) / 0.16, 0.0, 1.0) * np.clip(1.0 - (resid - 0.07) / 0.14, 0.0, 1.0)

                u_w = cv2.GaussianBlur(unknown_band.astype(np.float32), (0, 0), 2.0) * conf
                u_w = np.where(
                    bg_pockets & (~pocket_core),
                    np.maximum(u_w, 0.85 * np.clip((np.sqrt(denom) - 0.08) / 0.14, 0.0, 1.0)),
                    u_w,
                )
                alpha_new = alpha_snapped * (1.0 - u_w) + alpha_opt * u_w
                if int(pocket_core.sum()) > 0:
                    pc_soft = cv2.GaussianBlur(pocket_core.astype(np.float32), (0, 0), 1.2)
                    alpha_new = alpha_new * (1.0 - pc_soft)
                if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
                    alpha_pol = np.clip(
                        cv2.ximgproc.guidedFilter(guide=guide_gray, src=alpha_new.astype(np.float32), radius=5, eps=1.2e-4),
                        0.0,
                        1.0,
                    )
                    pol_w = cv2.GaussianBlur((unknown_band | bg_pockets).astype(np.float32), (0, 0), 1.5)
                    alpha_new = alpha_new * (1.0 - 0.55 * pol_w) + alpha_pol * (0.55 * pol_w)
                alpha_snapped = np.clip(alpha_new, 0.0, 1.0)

    # Solve pure F cleanly once with final continuous alpha so outer RGB edge holds 100% pure F pigment
    F_final, _ = estimate_foreground_background_ml(rgb_f32, alpha_snapped, regularization=1.2e-4, n_small_iters=10, n_large_iters=2)
    rgb_out = np.where((alpha_snapped >= 0.96)[:, :, None], rgb_f32, F_final)
    if local_hair_lab is not None and bool(unknown_band.any()):
        # Neutralize background chroma spill on translucent outer strands while preserving L* texture 100%
        spill_w = cv2.GaussianBlur(unknown_band.astype(np.float32), (0, 0), 2.0) * conf * np.clip((0.94 - alpha_opt) / 0.45, 0.0, 0.88)
        out_u8_tmp = np.clip(np.round(rgb_out * 255.0), 0, 255).astype(np.uint8)
        out_lab = cv2.cvtColor(out_u8_tmp, cv2.COLOR_RGB2LAB).astype(np.float32)
        cdist_out = np.linalg.norm(out_lab[:, :, 1:3] - local_hair_lab[:, :, 1:3], axis=2)
        chroma_spill = np.clip((cdist_out - 10.0) / 14.0, 0.0, 1.0)
        despill = np.clip(
            np.maximum(spill_w, cv2.GaussianBlur((unknown_band & (alpha_snapped > 0.05)).astype(np.float32), (0, 0), 2.0) * chroma_spill * 0.85),
            0.0,
            0.90,
        )
        out_lab[:, :, 1:3] = out_lab[:, :, 1:3] * (1.0 - despill[:, :, None]) + local_hair_lab[:, :, 1:3] * despill[:, :, None]
        rgb_out_u8 = cv2.cvtColor(np.clip(np.round(out_lab), 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
        rgb_out = rgb_out_u8.astype(np.float32) / 255.0
    else:
        rgb_out_u8 = np.clip(np.round(rgb_out * 255.0), 0, 255).astype(np.uint8)
    alpha_u8 = np.clip(np.round(alpha_snapped * 255.0), 0, 255).astype(np.uint8)
    rgba_u8 = np.dstack([rgb_out_u8, alpha_u8])

    p_rgba = out_d / "cutout_refined_rgba.png"
    p_alpha = out_d / "cutout_refined_alpha.png"
    p_fg = out_d / "cutout_fg_pure_rgb.png"
    _write_png_u8(p_rgba, rgba_u8, "RGBA", overwrite=overwrite)
    _write_png_u8(p_alpha, alpha_u8, "L", overwrite=overwrite)
    _write_png_u8(p_fg, rgb_out_u8, "RGB", overwrite=overwrite)

    # Half-res edge audit previews on bright white & dark studio slate
    a3 = alpha_snapped[:, :, None]
    on_white = rgb_out * a3 + np.array([0.97, 0.98, 0.99], dtype=np.float32) * (1.0 - a3)
    on_dark = rgb_out * a3 + np.array([0.09, 0.11, 0.14], dtype=np.float32) * (1.0 - a3)
    prev_w = max(640, min(1600, w // 2))
    prev_h = max(360, int(round(h * prev_w / w)))
    p_cw = out_d / "cutout_check_on_white.png"
    p_cd = out_d / "cutout_check_on_dark.png"
    Image.fromarray(np.clip(np.round(on_white * 255.0), 0, 255).astype(np.uint8)).resize((prev_w, prev_h), Image.Resampling.LANCZOS).save(p_cw, "PNG")
    Image.fromarray(np.clip(np.round(on_dark * 255.0), 0, 255).astype(np.uint8)).resize((prev_w, prev_h), Image.Resampling.LANCZOS).save(p_cd, "PNG")

    meta = {
        "ok": True,
        "input": str(in_p),
        "size": [w, h],
        "hi_res_tiles": hi_res_tiles,
        "tiles_run": tiles_run,
        "specks_pruned_px": specks_pruned,
        "pockets_cleaned_px": pockets_cleaned,
        "soft_edge_px": int(((alpha_u8 > 3) & (alpha_u8 < 252)).sum()),
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "outputs": {
            "cutout_refined_rgba": str(p_rgba),
            "cutout_refined_alpha": str(p_alpha),
            "cutout_fg_pure_rgb": str(p_fg),
            "cutout_check_on_white": str(p_cw),
            "cutout_check_on_dark": str(p_cd),
        },
    }
    (out_d / "cutout_refine_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta
