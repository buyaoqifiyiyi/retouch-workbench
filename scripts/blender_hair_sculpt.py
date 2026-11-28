"""Blender 5.2+ Cycles OptiX GPU physical hair-curve renderer for anime-flowing 3D hair locks.

Round 2 Architecture (Real Micro-Strands × Bold Anime Silhouette):
  - Photographic 3-point lighting + AgX view transform (zero 255 blowout / plastic bands).
  - Sparse, high-contrast sub-clumped hair_curves (wide per-strand brightness jitter &
    deep inter-strand shadow crevices so every lock reads as real optical hair fibers).
  - Geometric dual-taper + short inner-root fade + 48 samples + OIDN RGB_ALBEDO_NORMAL
    (zero stippled sawtooth tips or chroma fireflies).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from mathutils import Vector


CAM_DIST = 5.0
LENS_MM = 85.0
SENSOR_W_MM = 36.0


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(description="Blender Cycles OptiX Physical Anime Hair Sculpt Renderer")
    ap.add_argument("--spec", required=True, help="Path to hair_sculpt_spec.json")
    ap.add_argument("--out-dir", required=True, help="Output directory for rendered RGBA layers")
    return ap.parse_args(argv)


def _srgb_to_linear_channel(c: float) -> float:
    c = max(0.0, min(1.0, float(c)))
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def srgb_to_linear_rgba(rgb01: list[float] | tuple[float, ...]) -> tuple[float, float, float, float]:
    return (
        _srgb_to_linear_channel(rgb01[0]),
        _srgb_to_linear_channel(rgb01[1]),
        _srgb_to_linear_channel(rgb01[2]),
        1.0,
    )


def uvz_to_world(u: float, v: float, dz: float, width: int, height: int) -> tuple[np.ndarray, float]:
    """Convert normalized (u, v in 0..1 top-left origin) + depth offset dz (m, + = toward cam) to world XYZ + px_world."""
    dist = max(0.8, CAM_DIST - float(dz))
    half_w = dist * (SENSOR_W_MM / (2.0 * LENS_MM))
    half_h = half_w * (float(height) / max(1.0, float(width)))
    x = (2.0 * float(u) - 1.0) * half_w
    y = -float(dz)
    z = (1.0 - 2.0 * float(v)) * half_h
    px_world = (2.0 * half_w) / max(1.0, float(width))
    return np.array([x, y, z], dtype=np.float64), float(px_world)


def _catmull_rom_sample(ctrl_pts: np.ndarray, n_samples: int) -> np.ndarray:
    """Evaluate open Catmull-Rom spline through ctrl_pts (K, 3) -> (n_samples, 3)."""
    k = len(ctrl_pts)
    if k < 2:
        return np.repeat(ctrl_pts[:1], n_samples, axis=0)
    if k == 2:
        ts = np.linspace(0.0, 1.0, n_samples)[:, None]
        return ctrl_pts[0:1] * (1.0 - ts) + ctrl_pts[1:2] * ts

    p0 = 2.0 * ctrl_pts[0] - ctrl_pts[1]
    pn = 2.0 * ctrl_pts[-1] - ctrl_pts[-2]
    ext = np.vstack([p0[None, :], ctrl_pts, pn[None, :]])
    n_seg = k - 1
    out = np.zeros((n_samples, 3), dtype=np.float64)
    for i in range(n_samples):
        u = (float(i) / max(1, n_samples - 1)) * n_seg
        seg = min(n_seg - 1, int(math.floor(u)))
        t = u - seg
        P0, P1, P2, P3 = ext[seg], ext[seg + 1], ext[seg + 2], ext[seg + 3]
        t2, t3 = t * t, t * t * t
        out[i] = 0.5 * (
            (2.0 * P1)
            + (-P0 + P2) * t
            + (2.0 * P0 - 5.0 * P1 + 4.0 * P2 - P3) * t2
            + (-P0 + 3.0 * P1 - 3.0 * P2 + P3) * t3
        )
    return out


def _bishop_frames(spine: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute twist-free parallel-transport (tangent, binormal, normal) frames along spine (S, 3)."""
    n = len(spine)
    tang = np.zeros_like(spine)
    tang[0] = spine[1] - spine[0]
    tang[-1] = spine[-1] - spine[-2]
    if n > 2:
        tang[1:-1] = spine[2:] - spine[:-2]
    norms = np.linalg.norm(tang, axis=1, keepdims=True)
    tang = tang / np.maximum(norms, 1e-8)

    ref = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(tang[0], ref))) > 0.92:
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    b0 = np.cross(ref, tang[0])
    b0 /= max(1e-8, float(np.linalg.norm(b0)))
    n0 = np.cross(tang[0], b0)
    n0 /= max(1e-8, float(np.linalg.norm(n0)))

    B = np.zeros_like(spine)
    N = np.zeros_like(spine)
    B[0], N[0] = b0, n0

    for i in range(1, n):
        t_prev, t_cur = tang[i - 1], tang[i]
        axis = np.cross(t_prev, t_cur)
        ax_len = float(np.linalg.norm(axis))
        if ax_len < 1e-7:
            B[i], N[i] = B[i - 1], N[i - 1]
        else:
            axis /= ax_len
            cos_a = max(-1.0, min(1.0, float(np.dot(t_prev, t_cur))))
            ang = math.acos(cos_a)
            v = B[i - 1]
            b_rot = v * math.cos(ang) + np.cross(axis, v) * math.sin(ang) + axis * np.dot(axis, v) * (1.0 - math.cos(ang))
            b_rot -= tang[i] * np.dot(b_rot, tang[i])
            b_rot /= max(1e-8, float(np.linalg.norm(b_rot)))
            n_rot = np.cross(tang[i], b_rot)
            n_rot /= max(1e-8, float(np.linalg.norm(n_rot)))
            B[i], N[i] = b_rot, n_rot
    return tang, B, N


def _anime_width_envelope(t: np.ndarray, tip_sharpness: float = 0.86) -> np.ndarray:
    """Calligraphic ribbon envelope: deep solid root -> luscious silk belly -> crisp tapering brush point."""
    root_tuck = 0.78 + 0.22 * np.clip(t / 0.12, 0.0, 1.0)
    belly = np.where(
        t <= 0.42,
        1.0,
        np.power(np.clip((1.0 - t) / 0.58, 0.0, 1.0), tip_sharpness),
    )
    return np.maximum(0.015, root_tuck * belly)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(1e-6, edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def build_lock_curves(
    lock: dict[str, Any],
    width: int,
    height: int,
    render_scale: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Generate 80/15/5 liquid-silk calligraphic hair_curves with flat Möbius twist & convergent brush tips."""
    ctrl_uvz = lock["ctrl_uvz"]
    world_ctrl = []
    px_scales = []
    for pt in ctrl_uvz:
        wpt, pxs = uvz_to_world(pt[0], pt[1], pt[2], width, height)
        world_ctrl.append(wpt)
        px_scales.append(pxs)
    ctrl_arr = np.array(world_ctrl, dtype=np.float64)
    avg_px_world = float(np.mean(px_scales))

    n_pts = int(lock.get("n_pts", 32))
    n_dense = 96
    spine_dense = _catmull_rom_sample(ctrl_arr, n_dense)
    _, B_d, N_d = _bishop_frames(spine_dense)
    ts_d = np.linspace(0.0, 1.0, n_dense, dtype=np.float64)

    twist_start, twist_end = [math.radians(float(a)) for a in lock.get("twist_deg", [0.0, 180.0])]
    twist_ang = twist_start + (twist_end - twist_start) * _smoothstep(0.06, 0.92, ts_d)
    cos_tw = np.cos(twist_ang)[:, None]
    sin_tw = np.sin(twist_ang)[:, None]
    B_tw_d = B_d * cos_tw + N_d * sin_tw
    N_tw_d = -B_d * sin_tw + N_d * cos_tw

    width_px = float(lock.get("width_px", 84.0))
    thick_ratio = float(lock.get("thickness_ratio", 0.16))
    w_half_d = 0.5 * width_px * avg_px_world * _anime_width_envelope(ts_d, float(lock.get("tip_sharpness", 0.86)))
    h_half_d = w_half_d * thick_ratio

    n_strands = int(lock.get("n_strands", 360))
    n_flutes = max(3, int(lock.get("sub_clumps", 5)))
    fork_splits = int(lock.get("fork_splits", 1))
    clump_strength = float(lock.get("clump_strength", 0.72))

    r_lo_raw, r_hi_raw = lock.get("radius_px_range", [0.95, 1.45])
    # Slightly thicker minimum fiber radius eliminates outer dotted stipple noise at 0.80 scale
    min_r_full = 1.12 / max(0.5, float(render_scale))
    r_lo = max(min_r_full, float(r_lo_raw))
    r_hi = max(r_lo + 0.30, float(r_hi_raw))
    root_fade_t = float(min(0.10, lock.get("root_fade_t", 0.08)))

    def _albedo_bounce(lin: np.ndarray) -> np.ndarray:
        """Compensate Cycles multi-fiber absorption on bright/pastel pigments while keeping deep grooves rich."""
        lum = 0.2126 * float(lin[0]) + 0.7152 * float(lin[1]) + 0.0722 * float(lin[2])
        if lum <= 0.08:
            return lin
        gain = 1.0 + 0.42 * min(1.0, (lum - 0.08) / 0.28)
        out = lin.copy()
        out[:3] = np.clip(out[:3] * gain, 0.0, 0.96)
        return out

    c_shad_lin = _albedo_bounce(np.array(srgb_to_linear_rgba(lock.get("color_shadow", lock["color_root"])), dtype=np.float32))
    c_mid_lin = _albedo_bounce(np.array(srgb_to_linear_rgba(lock["color_mid"]), dtype=np.float32))
    c_brt_lin = _albedo_bounce(np.array(srgb_to_linear_rgba(lock.get("color_bright", lock["color_tip"])), dtype=np.float32))
    c_glow_lin = _albedo_bounce(np.array(srgb_to_linear_rgba(lock.get("color_tip_glow", lock["color_tip"])), dtype=np.float32))

    n_core = max(8, int(round(0.80 * n_strands)))
    n_edge = max(4, int(round(0.15 * n_strands)))
    n_comp = max(5, n_strands - n_core - n_edge)
    n_total = n_core + n_edge + n_comp

    pos_out = np.zeros((n_total, n_pts, 3), dtype=np.float32)
    rad_out = np.zeros((n_total, n_pts), dtype=np.float32)
    col_r = np.zeros((n_total, 4), dtype=np.float32)
    col_m = np.zeros((n_total, 4), dtype=np.float32)
    col_t = np.zeros((n_total, 4), dtype=np.float32)
    fade_out = np.full((n_total,), root_fade_t, dtype=np.float32)

    u_core_base = np.linspace(-0.94, 0.94, n_core, dtype=np.float64)
    rng.shuffle(u_core_base)
    fork_side = 1.0 if rng.random() > 0.5 else -1.0
    fork_t = float(rng.uniform(0.56, 0.68))

    for i in range(n_total):
        role = "core" if i < n_core else ("edge" if i < n_core + n_edge else "comp")
        if role == "core":
            u0 = float(np.clip(u_core_base[i] + rng.uniform(-0.04, 0.04), -0.96, 0.96))
            v_cup = -0.22 * (u0 ** 2)
            v0 = float(np.clip(v_cup + rng.normal(0.0, 0.34), -0.95, 0.95))
            t_end = 1.0 if rng.random() < 0.85 else float(rng.uniform(0.86, 0.99))
            r_px = float(rng.uniform(r_lo, r_hi))
        elif role == "edge":
            side_e = -1.0 if (i % 2 == 0) else 1.0
            u0 = float(side_e * rng.uniform(0.76, 1.06))
            v0 = float(np.clip(-0.18 * (u0 ** 2) + rng.normal(0.0, 0.28), -0.95, 0.95))
            t_end = float(rng.uniform(0.72, 0.97))
            r_px = float(rng.uniform(0.92 * r_lo, 0.88 * r_hi))
        else:
            side_c = 1.0 if ((i - n_core - n_edge) % 2 == 0) else -1.0
            u0 = float(side_c * rng.uniform(0.82, 1.05))
            v0 = float(rng.uniform(-0.35, 0.45))
            t_end = float(rng.uniform(0.85, 1.0))
            r_px = float(rng.uniform(0.78 * min_r_full, 1.05 * min_r_full))

        ts_i = np.linspace(0.0, t_end, n_pts, dtype=np.float64)
        sp_i = np.column_stack([np.interp(ts_i, ts_d, spine_dense[:, k]) for k in range(3)])
        B_i = np.column_stack([np.interp(ts_i, ts_d, B_tw_d[:, k]) for k in range(3)])
        N_i = np.column_stack([np.interp(ts_i, ts_d, N_tw_d[:, k]) for k in range(3)])
        wh_i = np.interp(ts_i, ts_d, w_half_d)
        hh_i = np.interp(ts_i, ts_d, h_half_d)

        # Gentle sculptural flutes + strong calligraphic tip convergence (zero frayed broom!)
        flute_u = math.sin(n_flutes * 0.5 * math.pi * u0) * 0.06 * clump_strength
        conv_w = 1.0 - 0.86 * _smoothstep(0.58, 1.0, ts_i)
        u_t = (u0 + flute_u * _smoothstep(0.10, 0.50, ts_i)) * conv_w
        v_t = v0 * (1.0 - 0.72 * _smoothstep(0.62, 1.0, ts_i))

        wave_env = np.sin(math.pi * (ts_i / max(0.2, t_end)))
        u_t = u_t + float(rng.uniform(-0.025, 0.025)) * wave_env
        pts_i = sp_i + (u_t * wh_i)[:, None] * B_i + (v_t * hh_i)[:, None] * N_i

        # Subtle sculptural secondary tip fork on edge ribbon without messy radial spray
        if role == "edge" and fork_splits >= 1 and (u0 * fork_side) > 0.82:
            fw = np.power(_smoothstep(fork_t, 1.0, ts_i), 1.35)[:, None]
            arch = float(rng.uniform(0.22, 0.42)) * width_px * avg_px_world * fork_side
            pts_i = pts_i + fw * (arch * B_i + 0.14 * abs(arch) * N_i)
            pts_i = pts_i - np.power(_smoothstep(0.84, 1.0, ts_i), 1.4)[:, None] * (0.28 * arch * B_i)

        # 5% smooth harmonic companion filaments peeling off ribbon waist with crisp continuous thickness
        if role == "comp":
            arch_amp = float(rng.uniform(0.24, 0.56)) * width_px * avg_px_world * math.copysign(1.0, u0)
            depth_amp = float(rng.uniform(-0.12, 0.16)) * width_px * avg_px_world
            arc_w = np.sin(math.pi * np.power(ts_i / max(0.2, t_end), 0.85))[:, None]
            pts_i = pts_i + arc_w * (arch_amp * B_i + depth_amp * N_i)
            r_px = max(r_px, 0.94 * min_r_full)
            fade_out[i] = min(0.12, root_fade_t + 0.02)

        # Crisp geometric calligraphic taper: tucked root -> full ribbon body -> razor sword tip
        u_norm = ts_i / max(0.2, t_end)
        rad_env = (0.24 + 0.76 * np.clip(u_norm / 0.10, 0.0, 1.0)) * np.where(
            u_norm <= 0.52,
            1.0,
            0.04 + 0.96 * np.power(np.clip((1.0 - u_norm) / 0.48, 0.0, 1.0), 0.78),
        )
        pos_out[i] = pts_i.astype(np.float32)
        rad_out[i] = (r_px * avg_px_world * rad_env).astype(np.float32)

        # Coherent cross-ribbon sculptural shading + high-frequency longitudinal strand texture in wide bellies
        flute_crest = 0.50 + 0.45 * math.cos(n_flutes * 2.0 * math.pi * u0)
        crest_w = float(np.clip(0.35 + 0.42 * v0 + 0.40 * flute_crest, 0.0, 1.0))
        val_jit = float(rng.uniform(0.76, 1.24))
        hue_jit = rng.uniform(0.97, 1.03, size=3).astype(np.float32)
        scale_c = np.array([val_jit * hue_jit[0], val_jit * hue_jit[1], val_jit * hue_jit[2], 1.0], np.float32)

        root_mix = c_shad_lin * (1.0 - 0.65 * crest_w) + c_mid_lin * (0.65 * crest_w)
        mid_mix = c_shad_lin * (0.18 * (1.0 - crest_w)) + c_mid_lin * (0.82 * (1.0 - 0.42 * crest_w)) + c_brt_lin * (0.42 * crest_w)
        if role == "comp":
            tip_mix = 0.30 * c_brt_lin + 0.70 * c_glow_lin
        elif role == "edge":
            tip_mix = 0.52 * c_brt_lin + 0.48 * c_glow_lin
        else:
            tip_mix = c_mid_lin * (0.22 * (1.0 - crest_w)) + c_brt_lin * (0.56 + 0.22 * crest_w) + c_glow_lin * 0.22

        col_r[i] = np.clip(root_mix * scale_c, 0.0, 1.0)
        col_m[i] = np.clip(mid_mix * scale_c, 0.0, 1.0)
        col_t[i] = np.clip(tip_mix * scale_c, 0.0, 1.0)

    return pos_out, rad_out, col_r, col_m, col_t, fade_out, n_total, n_pts


def create_hair_material(name: str, mat_cfg: dict[str, Any]) -> Any:
    """Create liquid-silk Chiang Principled Hair BSDF with intercept-gated translucent tip glow."""
    mat = bpy.data.materials.new(name)
    if getattr(mat, "node_tree", None) is None:
        mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (920, 0)

    mix_root = nt.nodes.new("ShaderNodeMixShader")
    mix_root.location = (680, 0)
    tr_bsdf = nt.nodes.new("ShaderNodeBsdfTransparent")
    tr_bsdf.location = (440, -220)

    mix_sss = nt.nodes.new("ShaderNodeMixShader")
    mix_sss.location = (460, 80)

    hair_bsdf = nt.nodes.new("ShaderNodeBsdfHairPrincipled")
    hair_bsdf.location = (220, 160)
    hair_bsdf.model = str(mat_cfg.get("model", "CHIANG"))
    hair_bsdf.parametrization = "COLOR"

    def _set_bsdf(inp_name: str, val: float) -> None:
        if inp_name in hair_bsdf.inputs:
            hair_bsdf.inputs[inp_name].default_value = float(val)

    _set_bsdf("Roughness", mat_cfg.get("roughness", 0.19))
    _set_bsdf("Radial Roughness", mat_cfg.get("radial_roughness", 0.26))
    _set_bsdf("Coat", mat_cfg.get("coat", 0.24))
    _set_bsdf("IOR", 1.55)
    _set_bsdf("Offset", 0.04)
    _set_bsdf("Random Color", mat_cfg.get("random_color", 0.11))
    _set_bsdf("Random Roughness", mat_cfg.get("random_roughness", 0.24))
    _set_bsdf("Reflection", mat_cfg.get("reflection", 1.15))
    _set_bsdf("Transmission", mat_cfg.get("transmission", 1.25))
    _set_bsdf("Secondary Reflection", mat_cfg.get("secondary_reflection", 1.25))

    translucent_bsdf = nt.nodes.new("ShaderNodeBsdfTranslucent")
    translucent_bsdf.location = (220, -120)

    hi = nt.nodes.new("ShaderNodeHairInfo")
    hi.location = (-560, 160)

    att_r = nt.nodes.new("ShaderNodeAttribute")
    att_r.attribute_name = "col_root"
    att_r.location = (-560, -40)
    att_m = nt.nodes.new("ShaderNodeAttribute")
    att_m.attribute_name = "col_mid"
    att_m.location = (-560, -200)
    att_t = nt.nodes.new("ShaderNodeAttribute")
    att_t.attribute_name = "col_tip"
    att_t.location = (-560, -360)
    att_fade = nt.nodes.new("ShaderNodeAttribute")
    att_fade.attribute_name = "root_fade"
    att_fade.location = (-340, 340)

    mr_rm = nt.nodes.new("ShaderNodeMapRange")
    mr_rm.location = (-280, 120)
    mr_rm.inputs["From Min"].default_value = 0.04
    mr_rm.inputs["From Max"].default_value = 0.34
    mr_rm.clamp = True

    mr_mt = nt.nodes.new("ShaderNodeMapRange")
    mr_mt.location = (-280, -40)
    mr_mt.inputs["From Min"].default_value = 0.34
    mr_mt.inputs["From Max"].default_value = 0.90
    mr_mt.clamp = True

    # Core silk stays opaque/glossy (2%); tapering outer tips glow luminously (up to 28% Translucent)
    mr_sss = nt.nodes.new("ShaderNodeMapRange")
    mr_sss.location = (180, -280)
    mr_sss.inputs["From Min"].default_value = 0.45
    mr_sss.inputs["From Max"].default_value = 0.96
    mr_sss.inputs["To Min"].default_value = float(mat_cfg.get("sss_core", 0.02))
    mr_sss.inputs["To Max"].default_value = float(mat_cfg.get("sss_tip", 0.28))
    mr_sss.clamp = True

    mix_rm = nt.nodes.new("ShaderNodeMix")
    mix_rm.data_type = "RGBA"
    mix_rm.location = (-60, 20)

    mix_mt = nt.nodes.new("ShaderNodeMix")
    mix_mt.data_type = "RGBA"
    mix_mt.location = (60, 20)

    div_fade = nt.nodes.new("ShaderNodeMath")
    div_fade.operation = "DIVIDE"
    div_fade.use_clamp = True
    div_fade.location = (80, 300)

    nt.links.new(hi.outputs["Intercept"], mr_rm.inputs["Value"])
    nt.links.new(hi.outputs["Intercept"], mr_mt.inputs["Value"])
    nt.links.new(hi.outputs["Intercept"], mr_sss.inputs["Value"])
    nt.links.new(mr_rm.outputs["Result"], mix_rm.inputs["Factor"])
    nt.links.new(att_r.outputs["Color"], mix_rm.inputs["A"])
    nt.links.new(att_m.outputs["Color"], mix_rm.inputs["B"])

    nt.links.new(mr_mt.outputs["Result"], mix_mt.inputs["Factor"])
    nt.links.new(mix_rm.outputs["Result"], mix_mt.inputs["A"])
    nt.links.new(att_t.outputs["Color"], mix_mt.inputs["B"])
    nt.links.new(mix_mt.outputs["Result"], hair_bsdf.inputs["Color"])
    nt.links.new(att_t.outputs["Color"], translucent_bsdf.inputs["Color"])

    nt.links.new(mr_sss.outputs["Result"], mix_sss.inputs["Fac"])
    nt.links.new(hair_bsdf.outputs["BSDF"], mix_sss.inputs[1])
    nt.links.new(translucent_bsdf.outputs["BSDF"], mix_sss.inputs[2])

    nt.links.new(hi.outputs["Intercept"], div_fade.inputs[0])
    nt.links.new(att_fade.outputs["Fac"], div_fade.inputs[1])
    nt.links.new(div_fade.outputs["Value"], mix_root.inputs["Fac"])
    nt.links.new(tr_bsdf.outputs["BSDF"], mix_root.inputs[1])
    nt.links.new(mix_sss.outputs["Shader"], mix_root.inputs[2])
    nt.links.new(mix_root.outputs["Shader"], out.inputs["Surface"])
    return mat


def create_curves_object(
    name: str,
    locks: list[dict[str, Any]],
    width: int,
    height: int,
    render_scale: float,
    material: Any,
    collection: Any,
    seed: int,
) -> Any | None:
    if not locks:
        return None
    rng = np.random.default_rng(seed)
    all_pos, all_rad, all_cr, all_cm, all_ct, all_fade = [], [], [], [], [], []
    sizes: list[int] = []

    for lock in locks:
        pos, rad, cr, cm, ct, fd, n_s, n_p = build_lock_curves(lock, width, height, render_scale, rng)
        all_pos.append(pos.reshape(-1, 3))
        all_rad.append(rad.reshape(-1))
        all_cr.append(cr)
        all_cm.append(cm)
        all_ct.append(ct)
        all_fade.append(fd)
        sizes.extend([n_p] * n_s)

    cd = bpy.data.hair_curves.new(name)
    cd.add_curves(sizes)
    cd.set_types(type="CATMULL_ROM")

    cd.attributes["position"].data.foreach_set("vector", np.concatenate(all_pos, axis=0).ravel())
    ra = cd.attributes.new("radius", "FLOAT", "POINT")
    ra.data.foreach_set("value", np.concatenate(all_rad, axis=0).ravel())

    ca_r = cd.attributes.new("col_root", "FLOAT_COLOR", "CURVE")
    ca_r.data.foreach_set("color", np.concatenate(all_cr, axis=0).ravel())
    ca_m = cd.attributes.new("col_mid", "FLOAT_COLOR", "CURVE")
    ca_m.data.foreach_set("color", np.concatenate(all_cm, axis=0).ravel())
    ca_t = cd.attributes.new("col_tip", "FLOAT_COLOR", "CURVE")
    ca_t.data.foreach_set("color", np.concatenate(all_ct, axis=0).ravel())
    fa = cd.attributes.new("root_fade", "FLOAT", "CURVE")
    fa.data.foreach_set("value", np.concatenate(all_fade, axis=0).ravel())

    cd.materials.append(material)
    obj = bpy.data.objects.new(name, cd)
    collection.objects.link(obj)
    return obj


def _add_area_light(
    name: str,
    loc: tuple[float, float, float],
    target: tuple[float, float, float],
    energy: float,
    size: float,
    color_srgb: list[float],
    col: Any,
) -> Any:
    ld = bpy.data.lights.new(name, type="AREA")
    ld.energy = float(energy)
    ld.size = float(size)
    lin = srgb_to_linear_rgba(color_srgb)
    ld.color = (lin[0], lin[1], lin[2])
    obj = bpy.data.objects.new(name, ld)
    col.objects.link(obj)
    obj.location = loc
    direction = Vector(target) - Vector(loc)
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return obj


def setup_lights_and_world(spec: dict[str, Any], scene: Any) -> None:
    lcol = bpy.data.collections.new("HairLights")
    scene.collection.children.link(lcol)

    lcfg = spec.get("lighting", {})
    key_side = str(lcfg.get("key_side", "right")).lower()
    kx = 2.6 if "right" in key_side else -2.6
    rx = -kx * 1.05

    key_col = lcfg.get("key_color", [1.0, 0.96, 0.92])
    rim_col = lcfg.get("rim_color", [1.0, 0.92, 0.80])
    fill_col = lcfg.get("fill_color", [0.80, 0.86, 0.96])
    amb_col = lcfg.get("ambient_color", [0.46, 0.48, 0.56])

    # Balanced luminous directional lighting: keeps crisp shadow crevices WITHOUT darkening hair into mud!
    _add_area_light("HairKeyLight", (kx, -3.2, 2.2), (0.0, 0.0, 0.1), float(lcfg.get("key_energy", 250.0)), 1.20, key_col, lcol)
    _add_area_light("HairRimBack", (rx, 2.2, 1.8), (0.0, -0.05, 0.0), float(lcfg.get("rim_energy", 285.0)), 1.35, rim_col, lcol)
    _add_area_light("HairSoftFill", (-kx * 0.6, -4.4, -0.4), (0.0, 0.0, 0.0), float(lcfg.get("fill_energy", 58.0)), 3.2, fill_col, lcol)

    world = bpy.data.worlds.new("HairWorld")
    scene.world = world
    if getattr(world, "node_tree", None) is None:
        world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        amb_lin = srgb_to_linear_rgba(amb_col)
        bg.inputs["Color"].default_value = amb_lin
        bg.inputs["Strength"].default_value = float(lcfg.get("world_strength", 0.16))


def main() -> None:
    t0 = time.time()
    args = parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    full_w = int(spec["width"])
    full_h = int(spec["height"])
    rw = int(spec.get("render_width", full_w))
    rh = int(spec.get("render_height", full_h))
    render_scale = float(rw) / max(1.0, float(full_w))
    samples = int(spec.get("samples", 44))

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"

    prefs = bpy.context.preferences.addons["cycles"].preferences
    gpu_backend = "CPU"
    for btype in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = btype
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type == btype]
            if gpus:
                for d in prefs.devices:
                    d.use = (d.type == btype)
                scene.cycles.device = "GPU"
                gpu_backend = btype
                break
        except Exception:
            pass

    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = float(spec.get("adaptive_threshold", 0.018))
    scene.cycles.use_denoising = True
    try:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
        scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    except Exception:
        pass
    scene.cycles.sample_clamp_direct = 4.0
    scene.cycles.sample_clamp_indirect = 1.6
    scene.cycles.max_bounces = 10
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 4
    scene.cycles.transmission_bounces = 8
    scene.cycles.transparent_max_bounces = 32
    scene.render.hair_type = str(spec.get("hair_type", "STRIP"))
    scene.render.hair_subdiv = int(spec.get("hair_subdiv", 2))

    scene.render.resolution_x = rw
    scene.render.resolution_y = rh
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"

    vt_items = [i.identifier for i in scene.view_settings.bl_rna.properties["view_transform"].enum_items]
    if "AgX" in vt_items:
        scene.view_settings.view_transform = "AgX"
    elif "Filmic" in vt_items:
        scene.view_settings.view_transform = "Filmic"
    scene.view_settings.exposure = float(spec.get("exposure", 0.0))

    cam_data = bpy.data.cameras.new("HairCam")
    cam_data.lens = LENS_MM
    cam_data.sensor_width = SENSOR_W_MM
    cam_data.sensor_fit = "HORIZONTAL"
    if bool(spec.get("use_dof", True)):
        cam_data.dof.use_dof = True
        cam_data.dof.focus_distance = CAM_DIST
        cam_data.dof.aperture_fstop = float(spec.get("dof_fstop", 3.6))
    cam_obj = bpy.data.objects.new("HairCam", cam_data)
    scene.collection.objects.link(cam_obj)
    cam_obj.location = (0.0, -CAM_DIST, 0.0)
    cam_obj.rotation_euler = (math.pi * 0.5, 0.0, 0.0)
    scene.camera = cam_obj

    setup_lights_and_world(spec, scene)

    mat = create_hair_material("PhysicalAnimeHair", spec.get("material", {}))
    locks = spec.get("locks", [])
    behind_locks = [lk for lk in locks if lk.get("layer", "behind") == "behind"]
    front_locks = [lk for lk in locks if lk.get("layer", "behind") == "front"]

    col_behind = bpy.data.collections.new("HairBehind")
    col_front = bpy.data.collections.new("HairFront")
    col_catcher = bpy.data.collections.new("ShadowCatcher")
    scene.collection.children.link(col_behind)
    scene.collection.children.link(col_front)
    scene.collection.children.link(col_catcher)

    seed = int(spec.get("seed", 42))
    obj_behind = create_curves_object("LocksBehind", behind_locks, full_w, full_h, render_scale, mat, col_behind, seed)
    obj_front = create_curves_object("LocksFront", front_locks, full_w, full_h, render_scale, mat, col_front, seed + 101)

    _, px_w = uvz_to_world(0.5, 0.5, -0.02, full_w, full_h)
    plane_w = px_w * full_w * 1.15
    plane_h = px_w * full_h * 1.15
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.02, 0.0), rotation=(math.pi * 0.5, 0.0, 0.0))
    sc_plane = bpy.context.active_object
    sc_plane.name = "SubjectShadowPlane"
    sc_plane.scale = (plane_w, plane_h, 1.0)
    sc_plane.is_shadow_catcher = True
    for c in list(sc_plane.users_collection):
        c.objects.unlink(sc_plane)
    col_catcher.objects.link(sc_plane)

    p_behind = out_dir / "raw_behind_rgba.png"
    p_front = out_dir / "raw_front_rgba.png"
    p_shadow = out_dir / "raw_cast_shadow_rgba.png"

    col_behind.hide_render = False
    col_front.hide_render = True
    col_catcher.hide_render = True
    if obj_behind is not None:
        scene.render.filepath = str(p_behind)
        bpy.ops.render.render(write_still=True)

    col_behind.hide_render = True
    col_front.hide_render = False
    col_catcher.hide_render = True
    if obj_front is not None:
        scene.render.filepath = str(p_front)
        bpy.ops.render.render(write_still=True)

    if obj_front is not None:
        col_behind.hide_render = True
        col_front.hide_render = False
        col_catcher.hide_render = False
        obj_front.visible_camera = False
        obj_front.visible_diffuse = False
        obj_front.visible_glossy = False
        obj_front.visible_transmission = False
        obj_front.visible_shadow = True
        cam_data.dof.use_dof = False
        scene.cycles.samples = 16
        scene.render.filepath = str(p_shadow)
        bpy.ops.render.render(write_still=True)

    elapsed = round(time.time() - t0, 2)
    print(json.dumps({
        "ok": True,
        "gpu_backend": gpu_backend,
        "render_size": [rw, rh],
        "target_size": [full_w, full_h],
        "behind_locks": len(behind_locks),
        "front_locks": len(front_locks),
        "elapsed_sec": elapsed,
        "outputs": {
            "behind": str(p_behind) if p_behind.exists() else None,
            "front": str(p_front) if p_front.exists() else None,
            "cast_shadow": str(p_shadow) if p_shadow.exists() else None,
        },
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
