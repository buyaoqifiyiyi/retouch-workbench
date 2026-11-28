"""Tutorial-Faithful Soft-Core Bundled Clump & Multi-Strand COS Hair Engine (v3.1).

Synthesizes the Procreate 3-Brush 4-Layer tutorial + `@睡颜` real-hair restraint +
high-craft COS benchmark finals:
  - Three-stage organic strand hierarchy (ZERO hard 3D plastic tubes, ZERO faint spiderwebs):
      Stage 1: Soft-Core Bundled Hair Clumps (14-32px wide, 68-86% rich saturated core opacity
               built from a soft translucent fibrous heart + 14-20 clustered overlapping strands)
      Stage 2: Natural Mid-to-Tip Bifurcation (splits into fine 1-3px tapered strands with
               breathable open background air gaps)
      Stage 3: 8-12% Luminous Rim Highlights, Crown Fuzz, Bangs/Cheek Softening & 3-5 Front Threads.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def _smoothstep(edge0: float, edge1: float, x: np.ndarray | float) -> np.ndarray | float:
    t = np.clip((x - edge0) / max(1e-6, edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _catmull_rom_2d(ctrl_xy: np.ndarray, n_samples: int = 44) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate open 2D Catmull-Rom spline -> (pts [M,2], tangents [M,2], normals [M,2])."""
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
        p0 = 2.0 * ctrl_xy[0] - ctrl_xy[1]
        pn = 2.0 * ctrl_xy[-1] - ctrl_xy[-2]
        ext = np.vstack([p0[None, :], ctrl_xy, pn[None, :]]).astype(np.float64)
        n_seg = k - 1
        pts = np.zeros((n_samples, 2), dtype=np.float64)
        u_all = np.linspace(0.0, float(n_seg), n_samples)
        for i, u in enumerate(u_all):
            seg = min(n_seg - 1, int(math.floor(u)))
            t = u - seg
            P0, P1, P2, P3 = ext[seg], ext[seg + 1], ext[seg + 2], ext[seg + 3]
            t2, t3 = t * t, t * t * t
            pts[i] = 0.5 * (
                (2.0 * P1)
                + (-P0 + P2) * t
                + (2.0 * P0 - 5.0 * P1 + 4.0 * P2 - P3) * t2
                + (-P0 + 3.0 * P1 - 3.0 * P2 + P3) * t3
            )

    tang = np.zeros_like(pts)
    tang[0] = pts[1] - pts[0]
    tang[-1] = pts[-1] - pts[-2]
    if n_samples > 2:
        tang[1:-1] = pts[2:] - pts[:-2]
    norms_t = np.linalg.norm(tang, axis=1, keepdims=True)
    tang = tang / np.maximum(norms_t, 1e-8)
    norm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    return pts, tang, norm


def _rasterize_filament_to_canvas(
    canvas_pm: np.ndarray,
    canvas_a: np.ndarray,
    pts: np.ndarray,
    radii: np.ndarray,
    alphas: np.ndarray,
    colors_rgb: np.ndarray,
    additive: bool = False,
    soft_edge_mul: float = 1.0,
) -> None:
    """Sub-pixel analytical capsule-SDF rasterizer compositing one continuous filament atomically."""
    h, w = canvas_a.shape
    aa_half = 0.72 * soft_edge_mul
    max_r = float(np.max(radii)) + aa_half + 1.0
    x_min = max(0, int(math.floor(float(np.min(pts[:, 0])) - max_r)))
    x_max = min(w, int(math.ceil(float(np.max(pts[:, 0])) + max_r)) + 1)
    y_min = max(0, int(math.floor(float(np.min(pts[:, 1])) - max_r)))
    y_max = min(h, int(math.ceil(float(np.max(pts[:, 1])) + max_r)) + 1)
    if x_max <= x_min or y_max <= y_min:
        return

    bw = x_max - x_min
    bh = y_max - y_min
    cov = np.zeros((bh, bw), dtype=np.float32)
    rgb_num = np.zeros((bh, bw, 3), dtype=np.float32)
    w_sum = np.zeros((bh, bw), dtype=np.float32)

    pts_loc = pts.astype(np.float32).copy()
    pts_loc[:, 0] -= float(x_min)
    pts_loc[:, 1] -= float(y_min)

    n_pts = len(pts_loc)
    for s in range(n_pts - 1):
        ax, ay = float(pts_loc[s, 0]), float(pts_loc[s, 1])
        bx, by = float(pts_loc[s + 1, 0]), float(pts_loc[s + 1, 1])
        r0, r1 = float(radii[s]), float(radii[s + 1])
        a0, a1 = float(alphas[s]), float(alphas[s + 1])
        if max(a0, a1) < 0.008:
            continue
        rm = max(r0, r1) + aa_half + 0.5
        sx0 = max(0, int(math.floor(min(ax, bx) - rm)))
        sx1 = min(bw, int(math.ceil(max(ax, bx) + rm)) + 1)
        sy0 = max(0, int(math.floor(min(ay, by) - rm)))
        sy1 = min(bh, int(math.ceil(max(ay, by) + rm)) + 1)
        if sx1 <= sx0 or sy1 <= sy0:
            continue

        gx = np.arange(sx0, sx1, dtype=np.float32)[None, :]
        gy = np.arange(sy0, sy1, dtype=np.float32)[:, None]
        abx = bx - ax
        aby = by - ay
        ab2 = abx * abx + aby * aby
        if ab2 < 1e-6:
            t_seg = np.zeros((sy1 - sy0, sx1 - sx0), dtype=np.float32)
        else:
            t_seg = np.clip(((gx - ax) * abx + (gy - ay) * aby) / ab2, 0.0, 1.0)

        px = ax + abx * t_seg
        py = ay + aby * t_seg
        dist = np.hypot(gx - px, gy - py)
        r_loc = r0 + (r1 - r0) * t_seg
        a_loc = a0 + (a1 - a0) * t_seg

        outer = r_loc + aa_half
        inner = np.maximum(0.0, r_loc - aa_half)
        u = np.clip((outer - dist) / np.maximum(outer - inner, 1e-4), 0.0, 1.0)
        cov_seg = (u * u * (3.0 - 2.0 * u)) * a_loc

        sub_cov = cov[sy0:sy1, sx0:sx1]
        cov[sy0:sy1, sx0:sx1] = np.maximum(sub_cov, cov_seg)

        c0 = colors_rgb[s]
        c1 = colors_rgb[s + 1]
        c_loc = c0[None, None, :] * (1.0 - t_seg[:, :, None]) + c1[None, None, :] * t_seg[:, :, None]
        wt = cov_seg
        rgb_num[sy0:sy1, sx0:sx1] += c_loc * wt[:, :, None]
        w_sum[sy0:sy1, sx0:sx1] += wt

    mask = cov > 1e-4
    if not mask.any():
        return
    col_pix = np.where(w_sum[:, :, None] > 1e-5, rgb_num / np.maximum(w_sum[:, :, None], 1e-5), colors_rgb[0][None, None, :])
    col_pix = np.clip(col_pix, 0.0, 255.0)

    roi_pm = canvas_pm[y_min:y_max, x_min:x_max]
    roi_a = canvas_a[y_min:y_max, x_min:x_max]
    if additive:
        roi_pm[:] = np.clip(roi_pm + col_pix * cov[:, :, None], 0.0, 255.0)
        roi_a[:] = np.clip(1.0 - (1.0 - roi_a) * (1.0 - cov), 0.0, 1.0)
    else:
        inv_c = (1.0 - cov)[:, :, None]
        roi_pm[:] = col_pix * cov[:, :, None] + roi_pm * inv_c
        roi_a[:] = cov + roi_a * (1.0 - cov)


def _local_sample_rgb(
    subj_rgb: np.ndarray,
    hair_bin: np.ndarray,
    ax: float,
    ay: float,
    tone_rgb: np.ndarray,
    local_mix: float = 0.65,
) -> np.ndarray:
    """Eyedropper-only local hair color lookup blended with tonal anchor."""
    h, w = hair_bin.shape
    ix = int(np.clip(round(ax), 0, w - 1))
    iy = int(np.clip(round(ay), 0, h - 1))
    y0, y1 = max(0, iy - 16), min(h, iy + 17)
    x0, x1 = max(0, ix - 16), min(w, ix + 17)
    sub = hair_bin[y0:y1, x0:x1]
    if int(sub.sum()) >= 8:
        loc = np.median(subj_rgb[y0:y1, x0:x1][sub], axis=0).astype(np.float32)
        return np.clip(local_mix * loc + (1.0 - local_mix) * tone_rgb.astype(np.float32), 8.0, 248.0)
    return tone_rgb.astype(np.float32)


def _boost_mint_chroma(rgb: np.ndarray, boost: float = 1.22) -> np.ndarray:
    """Protect candy mint-aqua pigments from greying out over warm sunset skies."""
    lab = cv2.cvtColor(np.clip(rgb[None, None, :], 0, 255).astype(np.uint8), cv2.COLOR_RGB2Lab).astype(np.float32)
    a_v = float(lab[0, 0, 1] - 128.0)
    if a_v >= -3.0:
        return rgb
    lab[0, 0, 1] = np.clip(128.0 + a_v * boost, 20.0, 235.0)
    lab[0, 0, 2] = np.clip(128.0 + (lab[0, 0, 2] - 128.0) * (0.95 + 0.12 * (boost - 1.0)), 20.0, 235.0)
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2RGB).astype(np.float32)[0, 0]
    return out


def _filament_color_profile(
    base_rgb: np.ndarray,
    tip_rgb: np.ndarray,
    ts: np.ndarray,
    normals: np.ndarray,
    key_dir_xy: tuple[float, float],
    rng: np.random.Generator,
    is_glow: bool = False,
    crest_bias: float = 0.0,
) -> np.ndarray:
    """Compute smooth root->tip gradient + per-strand tonal relief + 2D key-light sheen."""
    lum_jit = float(rng.uniform(0.84, 1.16)) if not is_glow else float(rng.uniform(0.92, 1.08))
    hue_jit = rng.uniform(0.975, 1.025, size=3).astype(np.float32)
    b_col = np.clip(base_rgb * lum_jit * hue_jit * (1.0 + 0.12 * crest_bias), 8.0, 246.0)
    t_col = np.clip(tip_rgb * (0.92 + 0.16 * (lum_jit - 0.84) / 0.32) * hue_jit, 10.0, 248.0)

    blend = np.power(ts, 0.82)[:, None].astype(np.float32)
    grad = b_col[None, :] * (1.0 - blend) + t_col[None, :] * blend

    kx, ky = key_dir_xy
    ndotl = np.abs(normals[:, 0] * kx + normals[:, 1] * ky).astype(np.float32)
    sheen_gain = (1.0 + (0.14 if not is_glow else 0.18) * np.power(ndotl, 1.5))[:, None]
    cap = 238.0 if is_glow else 248.0
    return np.clip(grad * sheen_gain, 8.0, cap).astype(np.float32)


def _emit_soft_core_clump(
    canvas_pm: np.ndarray,
    canvas_a: np.ndarray,
    ctrl_xy: np.ndarray,
    *,
    clump_width_px: float,
    shadow_rgb: np.ndarray,
    mid_rgb: np.ndarray,
    bright_rgb: np.ndarray,
    tip_rgb: np.ndarray,
    key_dir_xy: tuple[float, float],
    px_scale: float,
    rng: np.random.Generator,
    opacity_scale: float = 1.0,
    n_filaments: int = 14,
    heart_alpha_scale: float = 1.0,
) -> int:
    """Emit one soft-core bundled lock: rich saturated fibrous heart -> split tapering strands."""
    n_samples = 48
    pts_c, _, norm_c = _catmull_rom_2d(ctrl_xy, n_samples=n_samples)
    ts = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)

    # 1. Soft translucent fibrous spine heart (holds saturated color in mid-belly, dissolves before tips!)
    for sub_off, col_a, col_b in (
        (-0.14 * clump_width_px, shadow_rgb, mid_rgb),
        (0.14 * clump_width_px, mid_rgb, bright_rgb),
    ):
        pts_heart = pts_c + sub_off * norm_c
        r_heart = np.maximum(1.5 * px_scale, 0.24 * clump_width_px * (1.0 - 0.72 * _smoothstep(0.32, 0.85, ts)))
        a_heart = np.clip(
            float(rng.uniform(0.42, 0.56))
            * opacity_scale
            * heart_alpha_scale
            * _smoothstep(0.0, 0.10, ts)
            * (1.0 - _smoothstep(0.38, 0.84, ts)),
            0.0,
            0.76,
        )
        cols_h = _filament_color_profile(col_a, col_b, ts, norm_c, key_dir_xy, rng, is_glow=False)
        _rasterize_filament_to_canvas(
            canvas_pm, canvas_a, pts_heart, r_heart.astype(np.float32), a_heart.astype(np.float32), cols_h,
            additive=False, soft_edge_mul=3.2,
        )

    # 2. Dense clustered micro-filaments overlapping over & splitting out of the soft heart
    # Gaussian-clustered lateral positions so strands coalesce smoothly instead of sparse comb teeth!
    u_core = np.clip(rng.normal(0.0, 0.28, size=max(6, int(round(0.75 * n_filaments)))), -0.58, 0.58)
    u_fringe = np.array([-0.54, -0.44, 0.44, 0.54], dtype=np.float64) + rng.uniform(-0.06, 0.06, size=4)
    u_all = np.concatenate([u_core, u_fringe])

    spread_env = (0.78 + 0.24 * _smoothstep(0.0, 0.20, ts)) * (1.0 - 0.58 * _smoothstep(0.45, 1.0, ts))
    wave_env = np.sin(math.pi * ts)
    emitted = 0

    for idx_u, u0 in enumerate(u_all):
        is_fringe = idx_u >= len(u_core)
        t_stop = 1.0 if (not is_fringe and rng.random() < 0.72) else float(rng.uniform(0.72, 0.98))
        m_idx = max(8, int(round(t_stop * n_samples)))
        ts_i = ts[:m_idx] / max(0.2, ts[m_idx - 1])
        pts_i = pts_c[:m_idx].copy()
        norm_i = norm_c[:m_idx]

        # Organic gentle split near tail so core clumps break into delicate bifurcated strands
        split_fan = (0.35 * math.copysign(1.0, u0) * _smoothstep(0.55, 1.0, ts_i)) if is_fringe else 0.0
        drift = float(rng.uniform(-0.14, 0.14)) * clump_width_px
        lat_off = ((u0 + split_fan) * clump_width_px * spread_env[:m_idx]) + drift * wave_env[:m_idx]
        pts_i = pts_i + lat_off[:, None] * norm_i

        r_peak = float(rng.uniform(1.15 * px_scale, 2.35 * px_scale)) if not is_fringe else float(rng.uniform(0.85 * px_scale, 1.65 * px_scale))
        rad_taper = (0.42 + 0.58 * _smoothstep(0.0, 0.12, ts_i)) * np.power(
            np.clip(1.0 - np.maximum(0.0, ts_i - 0.22) / 0.78, 0.04, 1.0), 0.75
        )
        radii = np.maximum(0.24 * px_scale, r_peak * rad_taper).astype(np.float32)

        op0 = (float(rng.uniform(0.58, 0.86)) if not is_fringe else float(rng.uniform(0.45, 0.72))) * opacity_scale
        root_a = _smoothstep(0.0, 0.12, ts_i)
        tip_a = 1.0 - _smoothstep(0.68, 1.0, ts_i)
        alphas = np.clip(op0 * root_a * tip_a, 0.0, 1.0).astype(np.float32)

        # Cross-strand shadow vs sunlit relief inside the clump
        crest_bias = float(np.clip(u0 * 1.4, -0.8, 0.9))
        base_c = shadow_rgb if crest_bias < -0.15 else (bright_rgb if crest_bias > 0.20 else mid_rgb)
        cols = _filament_color_profile(base_c, tip_rgb, ts_i, norm_i, key_dir_xy, rng, is_glow=False, crest_bias=crest_bias)
        _rasterize_filament_to_canvas(canvas_pm, canvas_a, pts_i, radii, alphas, cols, additive=False)
        emitted += 1
    return emitted


def _emit_rake_stroke(
    canvas_pm: np.ndarray,
    canvas_a: np.ndarray,
    ctrl_xy: np.ndarray,
    *,
    brush_kind: str,
    total_width_px: float,
    base_rgb: np.ndarray,
    tip_rgb: np.ndarray,
    key_dir_xy: tuple[float, float],
    px_scale: float,
    rng: np.random.Generator,
    opacity_scale: float = 1.0,
    root_fade: float = 0.18,
    tip_fade: float = 0.36,
    additive: bool = False,
    n_samples: int = 38,
) -> int:
    """Emit Procreate-style fine sub-rake or tapered single-strand wisps."""
    pts_c, _, norm_c = _catmull_rom_2d(ctrl_xy, n_samples=n_samples)
    ts = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)

    if brush_kind == "rake_wide":
        n_fil = int(rng.integers(6, 10))
        u_slots = np.clip(rng.normal(0.0, 0.30, size=n_fil), -0.56, 0.56)
        r_lo, r_hi = 1.10 * px_scale, 2.20 * px_scale
        base_alpha_rng = (0.50, 0.78)
        conv = float(rng.uniform(0.35, 0.58))
    elif brush_kind == "rake_fine":
        n_fil = int(rng.integers(3, 6))
        u_slots = np.clip(rng.normal(0.0, 0.32, size=n_fil), -0.55, 0.55)
        r_lo, r_hi = 0.85 * px_scale, 1.75 * px_scale
        base_alpha_rng = (0.50, 0.80)
        conv = float(rng.uniform(0.28, 0.52))
    else:
        n_fil = 1
        u_slots = np.array([0.0], dtype=np.float64)
        r_lo, r_hi = 0.72 * px_scale, 1.45 * px_scale
        base_alpha_rng = (0.58, 0.88)
        conv = 0.0

    spread_env = (0.72 + 0.28 * _smoothstep(0.0, 0.22, ts)) * (1.0 - conv * _smoothstep(0.45, 1.0, ts))
    wave_env = np.sin(math.pi * ts)

    emitted = 0
    for u0 in u_slots:
        t_stop = 1.0 if (n_fil == 1 or rng.random() < 0.72) else float(rng.uniform(0.76, 0.96))
        m_idx = max(6, int(round(t_stop * n_samples)))
        ts_i = ts[:m_idx] / max(0.2, ts[m_idx - 1])
        pts_i = pts_c[:m_idx].copy()
        norm_i = norm_c[:m_idx]

        drift = float(rng.uniform(-0.18, 0.18)) * max(total_width_px, 6.0 * px_scale)
        lat_off = (u0 * total_width_px * spread_env[:m_idx]) + drift * wave_env[:m_idx]
        pts_i = pts_i + lat_off[:, None] * norm_i

        r_peak = float(rng.uniform(r_lo, r_hi))
        rad_taper = (0.38 + 0.62 * _smoothstep(0.0, 0.12, ts_i)) * np.power(
            np.clip(1.0 - np.maximum(0.0, ts_i - 0.18) / 0.82, 0.04, 1.0), 0.72
        )
        radii = np.maximum(0.22 * px_scale, r_peak * rad_taper).astype(np.float32)

        op0 = float(rng.uniform(*base_alpha_rng)) * opacity_scale
        root_a = _smoothstep(0.0, max(0.04, root_fade), ts_i)
        tip_a = 1.0 - _smoothstep(max(0.15, 1.0 - tip_fade), 1.0, ts_i)
        alphas = np.clip(op0 * root_a * tip_a, 0.0, 1.0).astype(np.float32)

        cols = _filament_color_profile(
            base_rgb, tip_rgb, ts_i, normals=norm_i, key_dir_xy=key_dir_xy, rng=rng, is_glow=additive
        )
        _rasterize_filament_to_canvas(canvas_pm, canvas_a, pts_i, radii, alphas, cols, additive=additive)
        emitted += 1
    return emitted


def _finalize_canvas(pm: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.where(a[:, :, None] > 1e-4, pm / np.maximum(a[:, :, None], 1e-4), 0.0)
    return np.clip(rgb, 0.0, 255.0).astype(np.float32), np.clip(a, 0.0, 1.0).astype(np.float32)


def _apply_dof_psfs(
    rgb: np.ndarray,
    alpha: np.ndarray,
    d_hair: np.ndarray,
    near_sigma: float,
    far_sigma: float,
    far_dist: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Photographic 85mm distance-graded optical softening (crisp near focus, dreamy tips)."""
    if float(alpha.max()) < 1e-4:
        return rgb, alpha
    pm = rgb * alpha[:, :, None]
    pm_near = cv2.GaussianBlur(pm, (0, 0), near_sigma)
    a_near = cv2.GaussianBlur(alpha, (0, 0), near_sigma)
    pm_far = cv2.GaussianBlur(pm, (0, 0), far_sigma)
    a_far = cv2.GaussianBlur(alpha, (0, 0), far_sigma)

    w_far = _smoothstep(32.0, far_dist, d_hair).astype(np.float32)
    pm_mix = pm_near * (1.0 - w_far[:, :, None]) + pm_far * w_far[:, :, None]
    a_mix = np.clip(a_near * (1.0 - w_far) + a_far * w_far, 0.0, 1.0)
    out_rgb = np.where(a_mix[:, :, None] > 1e-4, pm_mix / np.maximum(a_mix[:, :, None], 1e-4), 0.0)
    return np.clip(out_rgb, 0.0, 255.0).astype(np.float32), a_mix


def plan_and_render_tutorial_hair(
    *,
    subj_rgb: np.ndarray,
    subj_alpha: np.ndarray,
    hair_m: np.ndarray,
    face_snap: np.ndarray,
    eyes_m: np.ndarray,
    clothes_m: np.ndarray,
    arms_m: np.ndarray,
    acc_m: np.ndarray,
    all_skin_snap: np.ndarray,
    occ_snap: np.ndarray,
    tone_map: dict[str, np.ndarray],
    lighting: dict[str, Any],
    style: str,
    boldness: str,
    wind_dir: str,
    tx_map: np.ndarray,
    ty_map: np.ndarray,
    seed: int = 42,
) -> dict[str, Any]:
    """Plan & render all 4 soft-core tutorial-faithful layers on 4K canvas."""
    h, w = subj_alpha.shape
    rng = np.random.default_rng(seed)
    px_scale = max(w, h) / 3840.0

    lab = cv2.cvtColor((np.clip(subj_rgb, 0, 255).astype(np.uint8)), cv2.COLOR_RGB2Lab).astype(np.float32)
    L_cv = lab[:, :, 0] * (100.0 / 255.0)
    C_cv = np.hypot(lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0)

    if (face_snap > 0.45).sum() >= 200:
        fys, fxs = np.where(face_snap > 0.45)
        fx0, fy0, fx1, fy1 = float(fxs.min()), float(fys.min()), float(fxs.max()), float(fys.max())
    else:
        ys_t, xs_t = np.where((hair_m >= 0.45) & (subj_alpha >= 0.55) & (acc_m < 0.40))
        hcx_t = float(np.median(xs_t))
        hw_t = max(40.0, float(xs_t.max() - xs_t.min()))
        hh_t = max(40.0, float(ys_t.max() - ys_t.min()))
        fx0, fy0 = hcx_t - 0.16 * hw_t, float(ys_t.min()) + 0.14 * hh_t
        fx1, fy1 = hcx_t + 0.16 * hw_t, float(ys_t.min()) + 0.42 * hh_t

    fcx = 0.5 * (fx0 + fx1)
    fw = max(24.0, fx1 - fx0)
    fh = max(24.0, fy1 - fy0)

    above_chin = np.arange(h)[:, None] <= int(fy1 + 0.08 * fh)
    white_acc = (subj_alpha > 0.78) & (L_cv > 76.0) & (C_cv < 12.0) & (hair_m < 0.50) & above_chin
    hair_bin = (hair_m >= 0.45) & (subj_alpha >= 0.55) & (~white_acc) & (acc_m < 0.40)

    ys_h, xs_h = np.where(hair_bin)
    if len(xs_h) < 300:
        raise ValueError("Hair mask too small for tutorial-rake hair generation.")
    hx0, hx1 = int(xs_h.min()), int(xs_h.max())
    hy0, hy1 = int(ys_h.min()), int(ys_h.max())
    hair_w = max(24.0, float(hx1 - hx0))
    hair_h = max(24.0, float(hy1 - hy0))
    hcx = float(np.median(xs_h))

    if style in ("auto", "wind-dramatic"):
        lower = hair_bin & (np.arange(h)[:, None] > fy1 + 0.15 * fh)
        l_cnt = int((lower & (np.arange(w)[None, :] < fcx - 0.55 * fw)).sum())
        r_cnt = int((lower & (np.arange(w)[None, :] > fcx + 0.55 * fw)).sum())
        c_cnt = int((lower & (np.abs(np.arange(w)[None, :] - fcx) <= 0.45 * fw)).sum())
        if hy1 < fy1 + 0.75 * fh and (l_cnt + r_cnt) < 0.45 * len(xs_h):
            resolved_style = "short-braids-wispy"
        elif l_cnt > 1.4 * max(200, c_cnt) and r_cnt > 1.4 * max(200, c_cnt):
            resolved_style = "twintails"
        else:
            resolved_style = "long-cascade"
    else:
        resolved_style = style

    if wind_dir == "auto":
        is_left = int((xs_h < hcx).sum()) >= int((xs_h >= hcx).sum())
    elif wind_dir in ("left", "down-left"):
        is_left = True
    else:
        is_left = False
    wx_sign = -1.0 if is_left else 1.0

    bold_scale = {"natural": 0.80, "medium": 1.0, "dramatic": 1.14}.get(boldness, 1.0)
    key_dir_xy = (0.72, -0.69) if lighting.get("key_side", "right") == "right" else (-0.72, -0.69)

    def _local_tan(ax: float, ay: float, fallback: tuple[float, float]) -> tuple[float, float]:
        ix, iy = int(np.clip(round(ax), 0, w - 1)), int(np.clip(round(ay), 0, h - 1))
        y0, y1 = max(0, iy - 10), min(h, iy + 11)
        x0, x1 = max(0, ix - 10), min(w, ix + 11)
        vx = float(np.mean(tx_map[y0:y1, x0:x1]))
        vy = float(np.mean(ty_map[y0:y1, x0:x1]))
        bx = 0.35 * vx + 0.65 * fallback[0]
        by = 0.35 * vy + 0.65 * fallback[1]
        bn = math.hypot(bx, by) or 1.0
        return bx / bn, by / bn

    d_bow = cv2.distanceTransform((~white_acc & (acc_m < 0.35)).astype(np.uint8), cv2.DIST_L2, 5)
    d_arms = cv2.distanceTransform((arms_m < 0.30).astype(np.uint8), cv2.DIST_L2, 5)
    d_clo = cv2.distanceTransform((clothes_m < 0.30).astype(np.uint8), cv2.DIST_L2, 5)
    d_hair = cv2.distanceTransform((~hair_bin).astype(np.uint8), cv2.DIST_L2, 5)
    k_ker = np.ones((21, 21), np.uint8)
    eyes_guard = (
        cv2.GaussianBlur(cv2.dilate((eyes_m >= 0.18).astype(np.uint8), k_ker).astype(np.float32), (0, 0), 4.0)
        if eyes_m.max() > 0.05
        else np.zeros((h, w), np.float32)
    )
    d_eyes = cv2.distanceTransform((eyes_guard < 0.12).astype(np.uint8), cv2.DIST_L2, 5)

    def _find_edge_anchor(side: float, y_pos: float, interior_px: float) -> tuple[float, float] | None:
        iy = int(np.clip(round(y_pos), 0, h - 1))
        cols = np.where(hair_bin[iy, :])[0]
        if len(cols) < 12:
            return None
        scols = cols[cols < fcx] if side < 0 else cols[cols >= fcx]
        if len(scols) < 10:
            scols = cols
        if side < 0:
            ax = float(min(scols[-1], scols[0] + interior_px))
        else:
            ax = float(max(scols[0], scols[-1] - interior_px))
        if d_bow[iy, int(round(ax))] < 8.0:
            return None
        return ax, float(iy)

    def _build_wavy_ctrl(
        ax: float,
        ay: float,
        dx_total: float,
        dy_total: float,
        arch_amp: float,
        s_wave: float = 0.0,
    ) -> np.ndarray:
        """Build a 5-point silky wave spline aligned with local root tangent (organic undulation!)."""
        L = max(14.0, math.hypot(dx_total, dy_total))
        ux, uy = dx_total / L, dy_total / L
        nx, ny = -uy, ux
        tx0, ty0 = _local_tan(ax, ay, (ux, uy))
        dir0x = 0.35 * tx0 + 0.65 * ux
        dir0y = 0.35 * ty0 + 0.65 * uy
        n0 = math.hypot(dir0x, dir0y) or 1.0
        dir0x, dir0y = dir0x / n0, dir0y / n0

        p0 = (ax, ay)
        p1 = (ax + 0.22 * L * dir0x, ay + 0.22 * L * dir0y)
        p2 = (ax + 0.48 * dx_total + (arch_amp + 0.65 * s_wave) * nx, ay + 0.48 * dy_total + (arch_amp + 0.65 * s_wave) * ny)
        p3 = (ax + 0.76 * dx_total + (0.85 * arch_amp - 0.55 * s_wave) * nx, ay + 0.76 * dy_total + (0.85 * arch_amp - 0.55 * s_wave) * ny)
        p4 = (ax + dx_total + (0.45 * arch_amp + 0.30 * s_wave) * nx, ay + dy_total + (0.45 * arch_amp + 0.30 * s_wave) * ny)
        pts = np.array([p0, p1, p2, p3, p4], dtype=np.float64)
        pts[:, 0] = np.clip(pts[:, 0], 0.02 * w, 0.98 * w)
        pts[:, 1] = np.clip(pts[:, 1], 0.02 * h, 0.97 * h)
        return pts

    l1_pm = np.zeros((h, w, 3), np.float32)
    l1_a = np.zeros((h, w), np.float32)
    l2_pm = np.zeros((h, w, 3), np.float32)
    l2_a = np.zeros((h, w), np.float32)
    l3_pm = np.zeros((h, w, 3), np.float32)
    l3_a = np.zeros((h, w), np.float32)
    l4_pm = np.zeros((h, w, 3), np.float32)
    l4_a = np.zeros((h, w), np.float32)

    counts = {
        "soft_core_clumps": 0,
        "deep_behind": 0,
        "mid_connect": 0,
        "bright_weave": 0,
        "glow_crown": 0,
        "front_shoulder_threads": 0,
    }
    glow_tint = np.clip(0.70 * tone_map["bright"] + 0.30 * np.array(lighting["rim_color"], np.float32) * 255.0, 28.0, 238.0)
    moon_silver = np.array([228.0, 233.0, 248.0], np.float32)
    is_mint_hair = bool(float(cv2.cvtColor(np.clip(tone_map["mid"][None, None, :], 0, 255).astype(np.uint8), cv2.COLOR_RGB2Lab)[0, 0, 1] - 128.0) < -4.0)

    # =========================================================================
    # A. MAIN SOFT-CORE FLOW CLUMPS BY HAIRSTYLE (65-75% SWEET SPOT)
    # =========================================================================
    main_streams: list[dict[str, Any]] = []
    if resolved_style == "long-cascade":
        # Rose: Golden silky waves scaled by true head silhouette width flowing smoothly past her waist!
        head_ref = min(1.65 * fh, max(fw, 0.90 * fh, 0.25 * hair_h))
        for i in range(13):
            t_r = i / 12.0
            y_pos = hy0 + (0.20 + 0.70 * t_r) * hair_h
            anc = _find_edge_anchor(wx_sign, y_pos, interior_px=float(rng.uniform(52.0, 92.0)) * px_scale)
            if anc is None:
                continue
            ext_x = wx_sign * head_ref * float(rng.uniform(0.75, 1.12) * (0.86 + 0.22 * math.sin(math.pi * t_r))) * bold_scale
            ext_y = abs(ext_x) * float(rng.uniform(1.12, 1.58))
            arch = -wx_sign * head_ref * float(rng.uniform(0.16, 0.34))
            s_w = wx_sign * head_ref * float(rng.uniform(0.12, 0.26))
            is_hero = (i % 2 == 1)
            main_streams.append({
                "ctrl": _build_wavy_ctrl(anc[0], anc[1], ext_x, ext_y, arch, s_w),
                "anchor": anc,
                "width": float(rng.uniform(28.0, 42.0) if is_hero else rng.uniform(17.0, 26.0)) * px_scale,
                "is_wind": True,
                "op": float(rng.uniform(0.84, 0.96) if is_hero else rng.uniform(0.72, 0.86)),
                "n_fil": 20 if is_hero else 14,
                "heart_alpha_scale": 1.32,
                "rich_core": True,
                "tip_col": None,
            })
        # Right front/shoulder straight edge: 3 delicate wavy softening locks breaking straight rod contour
        for t_r in (0.50, 0.64, 0.78):
            anc = _find_edge_anchor(-wx_sign, hy0 + t_r * hair_h, interior_px=42.0 * px_scale)
            if anc is None:
                continue
            ext_x = -wx_sign * head_ref * float(rng.uniform(0.22, 0.38))
            ext_y = abs(ext_x) * float(rng.uniform(1.50, 2.20))
            main_streams.append({
                "ctrl": _build_wavy_ctrl(anc[0], anc[1], ext_x, ext_y, wx_sign * head_ref * 0.12, -wx_sign * head_ref * 0.08),
                "anchor": anc,
                "width": float(rng.uniform(14.0, 22.0)) * px_scale,
                "is_wind": False,
                "op": 0.78,
                "n_fil": 12,
                "heart_alpha_scale": 1.24,
                "rich_core": True,
                "tip_col": None,
            })

    elif resolved_style == "twintails":
        # Miku: Saturated candy-mint soft-core wavy twintails (1.2-1.55 fw left, >=380px clear of piano!)
        for i in range(10):
            t_r = i / 9.0
            y_pos = hy0 + (0.34 + 0.56 * t_r) * hair_h
            anc = _find_edge_anchor(-1.0, y_pos, interior_px=float(rng.uniform(44.0, 76.0)) * px_scale)
            if anc is None:
                continue
            ext_x = -fw * float(rng.uniform(1.18, 1.56) * (0.85 + 0.20 * math.sin(math.pi * t_r))) * bold_scale
            ext_y = abs(ext_x) * float(rng.uniform(1.25, 1.75))
            arch = fw * float(rng.uniform(0.20, 0.42))
            s_w = -fw * float(rng.uniform(0.16, 0.30))
            is_hero = (i % 3 != 0)
            main_streams.append({
                "ctrl": _build_wavy_ctrl(anc[0], anc[1], ext_x, ext_y, arch, s_w),
                "anchor": anc,
                "width": float(rng.uniform(24.0, 34.0) if is_hero else rng.uniform(13.0, 20.0)) * px_scale,
                "is_wind": True,
                "op": float(rng.uniform(0.78, 0.92) if is_hero else rng.uniform(0.64, 0.78)),
                "n_fil": 18 if is_hero else 12,
                "tip_col": None,
            })
        # Right balancing mint wave clump strictly behind right shoulder/arm
        for i in range(6):
            t_r = i / 5.0
            y_pos = hy0 + (0.38 + 0.50 * t_r) * hair_h
            anc = _find_edge_anchor(1.0, y_pos, interior_px=float(rng.uniform(40.0, 70.0)) * px_scale)
            if anc is None:
                continue
            ext_x = fw * float(rng.uniform(0.72, 1.15)) * bold_scale
            ext_y = abs(ext_x) * float(rng.uniform(1.35, 1.95))
            arch = -fw * float(rng.uniform(0.15, 0.32))
            s_w = fw * float(rng.uniform(0.10, 0.22))
            is_hero = (i % 2 == 0)
            main_streams.append({
                "ctrl": _build_wavy_ctrl(anc[0], anc[1], ext_x, ext_y, arch, s_w),
                "anchor": anc,
                "width": float(rng.uniform(20.0, 28.0) if is_hero else rng.uniform(12.0, 17.0)) * px_scale,
                "is_wind": False,
                "op": float(rng.uniform(0.74, 0.88)),
                "n_fil": 15 if is_hero else 10,
                "tip_col": None,
            })

    else:
        # Nazuna: Short-hair restraint (~0.58-0.88 fw) with 4 distinct soft-core wispy clumps + moon-rim wisps!
        for i in range(6):
            t_r = i / 5.0
            y_pos = hy0 + (0.28 + 0.56 * t_r) * hair_h
            anc = _find_edge_anchor(wx_sign, y_pos, interior_px=float(rng.uniform(32.0, 56.0)) * px_scale)
            if anc is None:
                continue
            ext_x = wx_sign * fw * float(rng.uniform(0.56, 0.88)) * bold_scale
            ext_y = abs(ext_x) * float(rng.uniform(0.42, 0.95))
            arch = -wx_sign * fw * float(rng.uniform(0.10, 0.22))
            s_w = wx_sign * fw * float(rng.uniform(0.06, 0.14))
            is_hero = (i < 4)
            main_streams.append({
                "ctrl": _build_wavy_ctrl(anc[0], anc[1], ext_x, ext_y, arch, s_w),
                "anchor": anc,
                "width": float(rng.uniform(14.0, 22.0) if is_hero else rng.uniform(9.0, 14.0)) * px_scale,
                "is_wind": True,
                "op": float(rng.uniform(0.74, 0.88) if is_hero else 0.66),
                "n_fil": 14 if is_hero else 9,
                "tip_col": None,
            })
        # Upwind Moon side: crisp short cold-silver backlit wisps kissing moonlight (<= 0.36 fw)
        for t_r in (0.30, 0.44, 0.58):
            anc = _find_edge_anchor(-wx_sign, hy0 + t_r * hair_h, interior_px=28.0 * px_scale)
            if anc is None:
                continue
            ext_x = -wx_sign * fw * float(rng.uniform(0.24, 0.36))
            ext_y = abs(ext_x) * float(rng.uniform(0.95, 1.35))
            main_streams.append({
                "ctrl": _build_wavy_ctrl(anc[0], anc[1], ext_x, ext_y, wx_sign * fw * 0.08, 0.0),
                "anchor": anc,
                "width": 10.5 * px_scale,
                "is_wind": False,
                "op": 0.72,
                "n_fil": 8,
                "tip_col": moon_silver,
            })

    # Render each stream across the layers
    for idx_s, st in enumerate(main_streams):
        ax, ay = st["anchor"]
        ctrl = st["ctrl"]
        w_st = st["width"]
        op_s = st["op"]
        if st.get("rich_core"):
            d_col = _local_sample_rgb(subj_rgb, hair_bin, ax, ay, tone_map["shadow"], 0.30)
            m_col = _local_sample_rgb(subj_rgb, hair_bin, ax, ay, tone_map["mid"], 0.38)
            b_col = _local_sample_rgb(subj_rgb, hair_bin, ax, ay, tone_map["bright"], 0.45) * 0.94
        else:
            d_col = _local_sample_rgb(subj_rgb, hair_bin, ax, ay, tone_map["shadow"], 0.48)
            m_col = _local_sample_rgb(subj_rgb, hair_bin, ax, ay, tone_map["mid"], 0.55)
            b_col = _local_sample_rgb(subj_rgb, hair_bin, ax, ay, tone_map["bright"], 0.52)
        if is_mint_hair:
            d_col = _boost_mint_chroma(d_col, 1.24)
            m_col = _boost_mint_chroma(m_col, 1.24)
            b_col = _boost_mint_chroma(b_col, 1.18)
        t_col = st["tip_col"] if st["tip_col"] is not None else glow_tint

        # L1 BEHIND: Soft-core bundled lock cleanly behind arms/clothes/curls!
        counts["soft_core_clumps"] += _emit_soft_core_clump(
            l1_pm, l1_a, ctrl,
            clump_width_px=w_st,
            shadow_rgb=d_col, mid_rgb=m_col, bright_rgb=b_col, tip_rgb=t_col,
            key_dir_xy=key_dir_xy, px_scale=px_scale, rng=rng,
            opacity_scale=op_s, n_filaments=st["n_fil"],
            heart_alpha_scale=float(st.get("heart_alpha_scale", 1.0)),
        )

        # L2 MID CONNECT (Short root bridge ONLY over the inner 38% inside hair + clean tips!)
        # NEVER cross semi-transparently over dark foreground curls like crop_02!
        c_bridge = ctrl.copy()
        c_bridge[2:] = c_bridge[0] + (c_bridge[2:] - c_bridge[0]) * 0.52
        c_bridge = c_bridge[:4]
        counts["mid_connect"] += _emit_rake_stroke(
            l2_pm, l2_a, c_bridge,
            brush_kind="rake_fine",
            total_width_px=w_st * 0.72,
            base_rgb=m_col, tip_rgb=b_col, key_dir_xy=key_dir_xy, px_scale=px_scale, rng=rng,
            opacity_scale=op_s * 0.78, root_fade=0.18, tip_fade=0.45,
        )

        # L3 BRIGHT WEAVE: Crisp outer lit fibers + rebellious cross-weave strands BEHIND or outside main curls!
        # Put lit sub-clumps in L1 BEHIND (so arms/bows/curls stay cleanly on top) and reserve L3 for light edge shimmer
        counts["deep_behind"] += _emit_rake_stroke(
            l1_pm, l1_a, ctrl,
            brush_kind="rake_fine",
            total_width_px=w_st * 0.58,
            base_rgb=b_col, tip_rgb=t_col, key_dir_xy=key_dir_xy, px_scale=px_scale, rng=rng,
            opacity_scale=op_s * 0.72, root_fade=0.22, tip_fade=0.38,
        )
        if idx_s % 2 == 0 or st["is_wind"]:
            c_cross = ctrl.copy()
            ang_d = math.radians(float(rng.choice([-1.0, 1.0]) * rng.uniform(5.0, 11.5)))
            cs, sn = math.cos(ang_d), math.sin(ang_d)
            rel_pts = c_cross - c_cross[0:1]
            c_cross = c_cross[0:1] + np.column_stack([rel_pts[:, 0] * cs - rel_pts[:, 1] * sn, rel_pts[:, 0] * sn + rel_pts[:, 1] * cs])
            counts["bright_weave"] += _emit_rake_stroke(
                l1_pm, l1_a, c_cross,
                brush_kind="single_strand",
                total_width_px=2.8 * px_scale,
                base_rgb=b_col, tip_rgb=t_col, key_dir_xy=key_dir_xy, px_scale=px_scale, rng=rng,
                opacity_scale=0.66, root_fade=0.25, tip_fade=0.44,
            )

        # L4 LUMINOUS ADD: Selective 10% crest glow strands & gossamer outer flyaways
        c_gl = ctrl.copy()
        c_gl[1:] += rng.normal(0.0, 7.0 * px_scale, size=(len(ctrl) - 1, 2))
        counts["glow_crown"] += _emit_rake_stroke(
            l4_pm, l4_a, c_gl,
            brush_kind="single_strand",
            total_width_px=2.0 * px_scale,
            base_rgb=0.45 * b_col + 0.55 * t_col, tip_rgb=t_col, key_dir_xy=key_dir_xy, px_scale=px_scale, rng=rng,
            opacity_scale=0.76 if st["is_wind"] else 0.58, root_fade=0.30, tip_fade=0.42, additive=True,
        )
        if st["is_wind"] and idx_s % 2 == 1:
            c_fly = ctrl.copy()
            span_fly = (c_fly[-1] - c_fly[0]) * float(rng.uniform(1.06, 1.22))
            c_fly[-1] = np.clip(c_fly[0] + span_fly + np.array([wx_sign * 0.12 * fw, -0.08 * fh]), [0.02 * w, 0.02 * h], [0.98 * w, 0.96 * h])
            counts["glow_crown"] += _emit_rake_stroke(
                l4_pm, l4_a, c_fly,
                brush_kind="single_strand",
                total_width_px=1.6 * px_scale,
                base_rgb=t_col, tip_rgb=t_col, key_dir_xy=key_dir_xy, px_scale=px_scale, rng=rng,
                opacity_scale=0.64, root_fade=0.26, tip_fade=0.44, additive=True,
            )

    # =========================================================================
    # B. REAL-HAIR SOFTENING: BANGS TIPS & CHEEK CRESCENT WISPS
    # =========================================================================
    bangs_band = (
        hair_bin
        & (np.arange(h)[:, None] >= fy0 - 0.08 * fh)
        & (np.arange(h)[:, None] <= fy0 + 0.46 * fh)
        & (np.arange(w)[None, :] >= fx0 + 0.06 * fw)
        & (np.arange(w)[None, :] <= fx1 - 0.06 * fw)
        & (d_eyes >= 18.0 * px_scale)
    )
    bx_cand = []
    for x_col in range(int(fx0 + 0.08 * fw), int(fx1 - 0.08 * fw), max(8, int(round(fw / 18)))):
        ys_b = np.where(bangs_band[:, x_col])[0]
        if len(ys_b) >= 6:
            yb = int(ys_b[-1]) - int(10 * px_scale)
            if yb > 0 and hair_bin[yb, x_col] and d_eyes[yb, x_col] >= 20.0 * px_scale:
                bx_cand.append((float(x_col), float(yb)))
    if bx_cand:
        pick_b = np.linspace(0, len(bx_cand) - 1, min(8, len(bx_cand))).round().astype(int)
        for bi in pick_b:
            ax, ay = bx_cand[int(bi)]
            side_b = -1.0 if ax < fcx else 1.0
            slen = float(rng.uniform(26.0, 52.0)) * px_scale
            dx_b = side_b * slen * float(rng.uniform(0.16, 0.36)) + wx_sign * 0.12 * slen
            dy_b = slen * 0.94
            c_b = _build_wavy_ctrl(ax, ay, dx_b, dy_b, side_b * 0.10 * slen, 0.0)
            m_c = _local_sample_rgb(subj_rgb, hair_bin, ax, ay, tone_map["mid"], 0.82)
            b_c = _local_sample_rgb(subj_rgb, hair_bin, ax, ay, tone_map["bright"], 0.75)
            counts["mid_connect"] += _emit_rake_stroke(
                l2_pm, l2_a, c_b,
                brush_kind="single_strand",
                total_width_px=2.1 * px_scale,
                base_rgb=m_c, tip_rgb=b_c, key_dir_xy=key_dir_xy, px_scale=px_scale, rng=rng,
                opacity_scale=float(rng.uniform(0.30, 0.48)), root_fade=0.22, tip_fade=0.45, n_samples=22,
            )

    for side_c in (-1.0, 1.0):
        for rel_f in (0.44, 0.60, 0.74):
            iy = int(np.clip(round(fy0 + rel_f * fh), 0, h - 1))
            cols_c = np.where(hair_bin[iy, :])[0]
            sc_c = cols_c[cols_c < fcx] if side_c < 0 else cols_c[cols_c >= fcx]
            if len(sc_c) < 6:
                continue
            ax_in = float(sc_c[-1] if side_c < 0 else sc_c[0])
            ax_out = float(sc_c[0] if side_c < 0 else sc_c[-1])
            ax = float(np.clip(0.84 * ax_in + 0.16 * ax_out, 0, w - 1))
            if d_arms[iy, int(round(ax))] < 44.0 * px_scale or d_clo[iy, int(round(ax))] < 38.0 * px_scale:
                continue
            if d_eyes[iy, int(round(ax))] < 24.0 * px_scale:
                continue
            slen = float(rng.uniform(48.0, 90.0)) * px_scale
            inward = -side_c * 0.14 * slen + wx_sign * 0.18 * slen
            c_chk = _build_wavy_ctrl(ax, float(iy), inward, slen * 0.94, side_c * 0.12 * slen, 0.0)
            m_c = _local_sample_rgb(subj_rgb, hair_bin, ax, float(iy), tone_map["mid"], 0.80)
            b_c = _local_sample_rgb(subj_rgb, hair_bin, ax, float(iy), tone_map["bright"], 0.76)
            counts["mid_connect"] += _emit_rake_stroke(
                l2_pm, l2_a, c_chk,
                brush_kind="rake_fine",
                total_width_px=float(rng.uniform(6.5, 11.0)) * px_scale,
                base_rgb=m_c, tip_rgb=b_c, key_dir_xy=key_dir_xy, px_scale=px_scale, rng=rng,
                opacity_scale=float(rng.uniform(0.24, 0.44)), root_fade=0.22, tip_fade=0.48, n_samples=26,
            )

    # =========================================================================
    # C. NATURAL CROWN HALO (NO WHITE STATIC FRIZZ!) + FOREGROUND THREADS
    # =========================================================================
    crown_xs = np.linspace(hx0 + 0.20 * hair_w, hx1 - 0.20 * hair_w, 12)
    for x_cr in crown_xs:
        ix = int(np.clip(round(x_cr), 0, w - 1))
        ys_cr = np.where(hair_bin[:, ix] & (np.arange(h) <= fy0 + 0.22 * fh))[0]
        if len(ys_cr) < 8:
            continue
        iy = int(ys_cr[0]) + int(round(10.0 * px_scale))
        if iy >= h or d_bow[iy, ix] < 26.0 * px_scale or d_eyes[iy, ix] < 28.0 * px_scale:
            continue
        slope = (float(ix) - fcx) / max(1.0, 0.5 * hair_w)
        flen = float(rng.uniform(20.0, 42.0)) * px_scale
        # Smooth contour-hugging crown wisps (never upright curly static fuzz!)
        dx_f = (wx_sign * 0.70 + 0.30 * slope) * flen
        dy_f = -0.14 * flen + 0.28 * abs(slope) * flen
        c_cr = np.array([
            [float(ix), float(iy)],
            [float(ix) + 0.40 * dx_f, float(iy) + 0.35 * dy_f - 3.0 * px_scale],
            [float(ix) + dx_f, float(iy) + dy_f],
        ], dtype=np.float64)
        b_c = _local_sample_rgb(subj_rgb, hair_bin, float(ix), float(iy), tone_map["bright"], 0.72)
        c_crown_tip = 0.72 * b_c + 0.28 * glow_tint
        counts["glow_crown"] += _emit_rake_stroke(
            l3_pm, l3_a, c_cr,
            brush_kind="single_strand",
            total_width_px=1.4 * px_scale,
            base_rgb=b_c, tip_rgb=c_crown_tip, key_dir_xy=key_dir_xy, px_scale=px_scale, rng=rng,
            opacity_scale=float(rng.uniform(0.30, 0.48)), root_fade=0.24, tip_fade=0.48, additive=False, n_samples=18,
        )

    # Tutorial 01:16-01:18 secret: 2-4 delicate foreground strands over shoulder/collar (clear of hands!)
    shoulder_cand = (
        hair_bin
        & (np.arange(h)[:, None] >= fy1 + 0.05 * fh)
        & (np.arange(h)[:, None] <= fy1 + 0.95 * fh)
        & (d_arms >= 30.0 * px_scale)
        & (d_eyes >= 36.0 * px_scale)
        & (d_bow >= 18.0 * px_scale)
    )
    sy_pts, sx_pts = np.where(shoulder_cand)
    shoulder_mask_bypass = np.zeros((h, w), dtype=np.float32)
    if len(sx_pts) >= 30:
        n_sh = 4 if resolved_style != "short-braids-wispy" else 3
        idx_sh = np.linspace(0, len(sx_pts) - 1, n_sh).round().astype(int)
        for si in idx_sh:
            ax, ay = float(sx_pts[int(si)]), float(sy_pts[int(si)])
            slen = float(rng.uniform(75.0, 145.0)) * px_scale
            dx_s = wx_sign * 0.42 * slen + float(rng.uniform(-0.14, 0.14)) * slen
            dy_s = slen * 0.88
            c_sh = _build_wavy_ctrl(ax, ay, dx_s, dy_s, -wx_sign * 0.14 * slen, wx_sign * 0.06 * slen)
            m_c = _local_sample_rgb(subj_rgb, hair_bin, ax, ay, tone_map["mid"], 0.72)
            b_c = _local_sample_rgb(subj_rgb, hair_bin, ax, ay, tone_map["bright"], 0.68)
            tmp_pm = np.zeros((h, w, 3), np.float32)
            tmp_a = np.zeros((h, w), np.float32)
            em = _emit_rake_stroke(
                tmp_pm, tmp_a, c_sh,
                brush_kind="single_strand",
                total_width_px=1.9 * px_scale,
                base_rgb=m_c, tip_rgb=0.5 * b_c + 0.5 * glow_tint,
                key_dir_xy=key_dir_xy, px_scale=px_scale, rng=rng,
                opacity_scale=float(rng.uniform(0.58, 0.80)), root_fade=0.18, tip_fade=0.40, n_samples=26,
            )
            if em > 0:
                l3_pm = l3_pm * (1.0 - tmp_a[:, :, None]) + tmp_pm
                l3_a = tmp_a + l3_a * (1.0 - tmp_a)
                shoulder_mask_bypass = np.maximum(shoulder_mask_bypass, np.clip((tmp_a > 0.04).astype(np.float32), 0.0, 1.0))
                counts["front_shoulder_threads"] += 1

    # =========================================================================
    # D. UNPREMULTIPLY, PHOTOGRAPHIC DOF SOFTENING & ZERO-DILATION CARVE-OUT
    # =========================================================================
    l1_rgb, l1_al = _finalize_canvas(l1_pm, l1_a)
    l2_rgb, l2_al = _finalize_canvas(l2_pm, l2_a)
    l3_rgb, l3_al = _finalize_canvas(l3_pm, l3_a)
    l4_rgb, l4_al = _finalize_canvas(l4_pm, l4_a)

    l1_rgb, l1_al = _apply_dof_psfs(l1_rgb, l1_al, d_hair, 0.30 * px_scale, 0.95 * px_scale, 260.0 * px_scale)
    l2_rgb, l2_al = _apply_dof_psfs(l2_rgb, l2_al, d_hair, 0.25 * px_scale, 0.72 * px_scale, 220.0 * px_scale)
    l3_rgb, l3_al = _apply_dof_psfs(l3_rgb, l3_al, d_hair, 0.24 * px_scale, 0.76 * px_scale, 220.0 * px_scale)
    l4_rgb, l4_al = _apply_dof_psfs(l4_rgb, l4_al, d_hair, 0.22 * px_scale, 0.65 * px_scale, 240.0 * px_scale)

    face_protect_center = np.zeros((h, w), np.float32)
    if (face_snap > 0.45).sum() >= 300:
        yy_f, xx_f = np.mgrid[0:h, 0:w].astype(np.float32)
        ell = ((xx_f - fcx) / (0.44 * fw)) ** 2 + ((yy_f - (fy0 + 0.54 * fh)) / (0.48 * fh)) ** 2
        face_protect_center = np.clip((1.10 - ell) / 0.22, 0.0, 1.0)

    bow_guard = cv2.GaussianBlur(white_acc.astype(np.float32), (0, 0), 1.2)
    arms_flush = cv2.GaussianBlur((arms_m >= 0.46).astype(np.float32), (0, 0), 0.8) if arms_m.max() > 0.05 else np.zeros((h, w), np.float32)
    acc_flush = cv2.GaussianBlur((acc_m >= 0.45).astype(np.float32), (0, 0), 0.8) if acc_m.max() > 0.05 else np.zeros((h, w), np.float32)
    clo_flush = cv2.GaussianBlur((clothes_m >= 0.48).astype(np.float32), (0, 0), 0.8) if clothes_m.max() > 0.05 else np.zeros((h, w), np.float32)

    occ_nonface = occ_snap * (1.0 - np.clip(face_snap * 1.4, 0.0, 1.0))
    occ_soft = cv2.GaussianBlur(np.clip((occ_nonface - 0.35) / 0.45, 0.0, 1.0), (0, 0), 2.5)

    behind_guard = np.clip(np.maximum.reduce([eyes_guard, face_protect_center, bow_guard, acc_flush]), 0.0, 1.0)
    front_guard_strict = np.clip(np.maximum.reduce([eyes_guard, face_protect_center, arms_flush, occ_soft, bow_guard, acc_flush, clo_flush]), 0.0, 1.0)
    shoulder_bypass_soft = cv2.GaussianBlur(shoulder_mask_bypass, (0, 0), 0.6)
    weave_guard = np.clip(
        np.maximum.reduce([
            eyes_guard,
            face_protect_center,
            arms_flush,
            bow_guard,
            acc_flush,
            clo_flush * (1.0 - shoulder_bypass_soft),
        ]),
        0.0,
        1.0,
    )

    l1_al = np.clip(l1_al * (1.0 - behind_guard), 0.0, 1.0)
    l2_al = np.clip(l2_al * (1.0 - front_guard_strict), 0.0, 1.0)
    l3_al = np.clip(l3_al * (1.0 - weave_guard), 0.0, 1.0)
    l4_al = np.clip(l4_al * (1.0 - front_guard_strict), 0.0, 1.0)

    beh_ext = (l1_al > 0.03) & (subj_alpha < 0.40)
    if int(beh_ext.sum()) >= 50:
        ys_e, xs_e = np.where(beh_ext)
        ex0, ex1, ey0, ey1 = int(xs_e.min()), int(xs_e.max()), int(ys_e.min()), int(ys_e.max())
        hull_approx = cv2.dilate(beh_ext[ey0 : ey1 + 1, ex0 : ex1 + 1].astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
        ext_sub = (l1_al[ey0 : ey1 + 1, ex0 : ex1 + 1] > 0.08) & (subj_alpha[ey0 : ey1 + 1, ex0 : ex1 + 1] < 0.40)
        air_gap_pct = round(100.0 * max(0.0, 1.0 - float(ext_sub.sum()) / max(1, int(hull_approx.sum()))), 1)
    else:
        air_gap_pct = 100.0

    return {
        "resolved_style": resolved_style,
        "wx_sign": wx_sign,
        "behind": (l1_rgb, l1_al),
        "mid_connect": (l2_rgb, l2_al),
        "bright_weave": (l3_rgb, l3_al),
        "glow_crown": (l4_rgb, l4_al),
        "front_guard": front_guard_strict,
        "white_acc": white_acc,
        "counts": counts,
        "external_air_transparency_pct": air_gap_pct,
    }
