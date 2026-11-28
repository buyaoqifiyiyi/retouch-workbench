"""Optical Unmult compositing & face/title avoidance for user-rolled black-background AI VFX overlays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


def _J(obj: object) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _read_rgba_f32(path: Path, size_wh: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    im = Image.open(path).convert("RGBA")
    if im.size != size_wh:
        im = im.resize(size_wh, Image.Resampling.LANCZOS)
    arr = np.array(im, dtype=np.float32) / 255.0
    return arr[:, :, :3], arr[:, :, 3]


def _build_vfx_causal_normals(
    sa: np.ndarray,
    width: int,
    normal_map_path: Path | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return smoothed (nx, ny, nz) from Depth-Anything-V2 normal map or silhouette distance-field fallback."""
    if normal_map_path is not None and normal_map_path.exists():
        nm_im = Image.open(normal_map_path.resolve()).convert("RGB")
        if nm_im.size != (sa.shape[1], sa.shape[0]):
            nm_im = nm_im.resize((sa.shape[1], sa.shape[0]), Image.Resampling.BILINEAR)
        nmap = np.array(nm_im, dtype=np.float32) / 255.0 * 2.0 - 1.0
        nmap = cv2.GaussianBlur(nmap, (0, 0), 10.0)
        nlen = np.maximum(np.sqrt((nmap * nmap).sum(axis=2, keepdims=True)), 1e-4)
        nmap = nmap / nlen
        return nmap[:, :, 0], nmap[:, :, 1], nmap[:, :, 2]

    in_fg = (sa > 0.25).astype(np.uint8)
    dist_in = cv2.distanceTransform(in_fg, cv2.DIST_L2, 5)
    dist_sm = cv2.GaussianBlur(dist_in, (0, 0), max(8.0, width * 0.004))
    gy, gx = np.gradient(dist_sm)
    mag = np.maximum(np.hypot(gx, gy), 1e-4)
    rim_w = np.exp(-np.square(dist_in / max(24.0, width * 0.012))) * sa
    nx = (-gx / mag) * rim_w
    ny = (-gy / mag) * rim_w
    nz = np.maximum(0.25, 1.0 - 0.72 * rim_w)
    nlen = np.maximum(np.sqrt(nx * nx + ny * ny + nz * nz), 1e-4)
    return nx / nlen, ny / nlen, nz / nlen


def build_vfx_pcraft_cmds(
    meta: dict[str, Any],
    *,
    width: int,
    height: int,
    behind_group: str = "02_VFX_BEHIND_4K",
    subject_layer: str = "03_SUBJECT_4K",
    front_group: str = "04_VFX_FRONT_4K",
) -> list[str]:
    """Build PhotoCraft CLI commands placing behind/env spill, subject-clipped causal spill/rim, and front VFX."""
    layers = [l for l in meta.get("layers", []) if Path(l["path"]).exists()]
    if not layers:
        return []

    cmds: list[str] = []
    # Track last placed layer per placement bucket
    last_in_group: dict[str, str | None] = {
        "behind": None,
        "front": None,
        "subject_clip": subject_layer,
    }
    group_map = {"behind": behind_group, "front": front_group}

    for lay in layers:
        role = str(lay["placement"])
        lname = str(lay["layer_name"])
        blend = str(lay["blend"])
        op = float(lay.get("opacity", 1.0))
        fill_op = lay.get("fill_opacity")
        is_clip = role == "subject_clip"

        anchor = last_in_group[role] or group_map.get(role)
        if anchor:
            cmds += ["--cmd", "layer.select", "--params", _J({"layer": anchor})]
        cmds += [
            "--cmd",
            "file.placeEmbedded",
            "--params",
            _J({"path": str(lay["path"]), "fit": False, "smartObject": True, "center": [width / 2.0, height / 2.0]}),
            "--cmd",
            "layer.setProps",
            "--params",
            _J({"name": lname, "blend": blend, "opacity": round(op, 3), "clipped": bool(is_clip), "visible": True}),
        ]
        if fill_op is not None and int(round(float(fill_op))) < 100:
            cmds += ["--cmd", "layer.layerStyle.blendingOptions", "--params", _J({"fillOpacity": int(round(float(fill_op)))})]
        if last_in_group[role] is None and role in group_map:
            cmds += ["--cmd", "layer.moveTo", "--params", _J({"target": group_map[role], "position": "into"})]
        elif anchor:
            cmds += ["--cmd", "layer.moveTo", "--params", _J({"target": anchor, "position": "above"})]
        last_in_group[role] = lname
    return cmds


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
    fg_rgba_path: Path | None = None,
    normal_map_path: Path | None = None,
    garment_mask_path: Path | None = None,
    bg_path: Path | None = None,
    face_parse_dir: Path | None = None,
    title_box_xywh: tuple[int, int, int, int] | None = None,
    placement: str = "auto-split",
    black_floor: str = "auto",
    tint_hex: str | None = None,
    tint_strength: float = 0.20,
    spill_strength: float = 0.25,
    causal_strength: float = 0.75,
    env_spill_strength: float = 0.40,
    subject_front_keep: float = 0.10,
    density_gamma: float = 1.40,
    haze_suppress: float = 0.45,
    max_alpha: float = 0.85,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert black-background AI VFX plates into straight RGBA depth-split PhotoCraft layers + 2.5D normal-modulated causal spill."""
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
    sa_resolved: np.ndarray | None = None
    fg_rgb_resolved: np.ndarray | None = None
    if fg_rgba_path is not None and fg_rgba_path.exists():
        fg_rgb_resolved, sa_resolved = _read_rgba_f32(fg_rgba_path.resolve(), (width, height))
    if subject_alpha_path is not None and subject_alpha_path.exists():
        sa_resolved = _read_alpha_or_gray_f32(subject_alpha_path.resolve(), (width, height))

    if sa_resolved is not None:
        sa_m = sa_resolved
        core_dist = cv2.distanceTransform((sa_m > 0.35).astype(np.uint8), cv2.DIST_L2, 5)
        core_bin = (core_dist > max(12.0, width * 0.004)).astype(np.float32)
        subj_core_bin = core_bin > 0.5
        ds = 4
        core_sm = cv2.resize(core_bin, (max(8, width // ds), max(8, height // ds)), interpolation=cv2.INTER_AREA)
        core_sm = cv2.GaussianBlur(core_sm, (0, 0), max(6.0, (width * 0.011) / ds))
        subj_core = np.clip(cv2.resize(core_sm, (width, height), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)
        lum_pk = cleaned.max(axis=2)
        bokeh_lo = cv2.GaussianBlur(lum_pk, (0, 0), sigmaX=max(12.0, width * 0.005))
        spark_hf = np.clip((lum_pk - bokeh_lo * 0.90) / np.maximum(lum_pk, 1e-4), 0.0, 1.0)
        spark_sharp = np.clip(spark_hf * 2.2, 0.0, 1.0)
        hero_glint = np.power(np.clip((lum_pk - 0.84) / 0.16, 0.0, 1.0), 2.0) * spark_sharp
        limb_bokeh_guard = np.clip((1.0 - sa_m) + sa_m * spark_sharp, 0.0, 1.0)
        keep_interior = float(np.clip(subject_front_keep, 0.0, 1.0)) * hero_glint
        subj_front_mask = np.clip(((1.0 - subj_core) * limb_bokeh_guard) + subj_core * keep_interior, 0.0, 1.0)

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

    # 5. 2.5D Normal-Modulated Causal Local Relighting (Subject Broad Spill + Hot Specular Rim + Environment Spill)
    eye_ov_px = int(((a_front > 0.05) & eye_mask_bin).sum()) if eye_mask_bin is not None else 0
    body_cov_pct = (
        round(100.0 * float(((a_front > 0.08) & subj_core_bin).sum()) / max(1, int(subj_core_bin.sum())), 3)
        if subj_core_bin is not None
        else 0.0
    )

    layer_specs: list[dict[str, Any]] = []
    if "vfx_behind_subject_rgba" in outputs:
        layer_specs.append({"key": "vfx_behind", "path": outputs["vfx_behind_subject_rgba"], "layer_name": "02a_VFX_Behind_Subject", "placement": "behind", "blend": "Screen", "opacity": 0.65, "fill_opacity": 100})

    cs = float(np.clip(causal_strength, 0.0, 1.5))
    if sa_resolved is not None and (cs > 0.01 or spill_strength > 0.0):
        sa = sa_resolved
        nx, ny, nz = _build_vfx_causal_normals(sa, width, normal_map_path)
        in_fg = (sa > 0.25).astype(np.uint8)
        edge_dist = cv2.distanceTransform(in_fg, cv2.DIST_L2, 5)
        out_dist = cv2.distanceTransform((sa < 0.25).astype(np.uint8), cv2.DIST_L2, 5)
        rim_med = np.exp(-np.square(edge_dist / max(18.0, width * 0.007))) * sa
        fresnel = np.power(np.clip(1.0 - nz, 0.0, 1.0), 1.45)

        e_near = cv2.GaussianBlur(cleaned, (0, 0), max(14.0, width * 0.0068))
        e_mid = cv2.GaussianBlur(cleaned, (0, 0), max(48.0, width * 0.022))
        e_far = cv2.GaussianBlur(cleaned, (0, 0), max(110.0, width * 0.048))
        lum_mid = 0.2126 * e_mid[:, :, 0] + 0.7152 * e_mid[:, :, 1] + 0.0722 * e_mid[:, :, 2]
        gx = cv2.Sobel(lum_mid, cv2.CV_32F, 1, 0, ksize=5)
        gy = cv2.Sobel(lum_mid, cv2.CV_32F, 0, 1, ksize=5)
        gmag = np.maximum(np.hypot(gx, gy), 1e-4)
        vfx_dir_w = np.where(gmag > 8e-4, 0.28 + 0.72 * np.clip(nx * (gx / gmag) + ny * (gy / gmag), 0.0, 1.0), 0.50)

        # Guard matte dark clothing interiors vs glossy specular folds & guard eyes/inner face
        garment_m = _read_gray_f32(garment_mask_path.resolve(), (width, height)) if (garment_mask_path and garment_mask_path.exists()) else np.zeros((height, width), dtype=np.float32)
        garment_soft = np.clip(cv2.GaussianBlur(garment_m, (0, 0), max(5.0, width * 0.002)), 0.0, 1.0)
        k_g_er = max(13, int(round(width * 0.006)) | 1)
        garment_core = cv2.GaussianBlur(
            cv2.erode((garment_m > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_g_er, k_g_er))).astype(np.float32),
            (0, 0),
            max(6.0, width * 0.0025),
        )
        if fg_rgb_resolved is not None:
            fg_luma = 0.2126 * fg_rgb_resolved[:, :, 0] + 0.7152 * fg_rgb_resolved[:, :, 1] + 0.0722 * fg_rgb_resolved[:, :, 2]
            fg_spec = np.power(np.clip((cv2.GaussianBlur(fg_luma, (0, 0), 2.2) - 0.08) / 0.44, 0.0, 1.0), 0.85)
        else:
            fg_spec = np.zeros((height, width), dtype=np.float32)

        mat_broad_gate = (1.0 - garment_soft) + garment_soft * np.clip(rim_med * 0.88 + fg_spec * 0.72, 0.0, 1.0)
        mat_spec_gate = (1.0 - garment_soft) * (0.55 * rim_med + 0.45 * fresnel * rim_med) + garment_soft * np.clip(rim_med * 0.90 + fg_spec * (1.0 - 0.58 * garment_core), 0.0, 1.0)

        # Preserve delicate front eye/lip makeup while keeping outer cheek, chin underside & jaw active
        makeup_guard = np.ones((height, width), dtype=np.float32)
        if face_parse_dir is not None and face_parse_dir.exists():
            eb_p = face_parse_dir / "mask_eyes_brows.png"
            fs_p = face_parse_dir / "mask_skin.png"
            if eb_p.exists():
                eb_m = _read_gray_f32(eb_p, (width, height))
                eb_g = np.clip(cv2.GaussianBlur(cv2.dilate((eb_m > 0.2).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))).astype(np.float32), (0, 0), 10.0), 0.0, 1.0)
                makeup_guard *= (1.0 - 0.96 * eb_g)
            if fs_p.exists():
                fs_m = _read_gray_f32(fs_p, (width, height))
                fs_er = cv2.GaussianBlur(cv2.erode((fs_m > 0.35).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))).astype(np.float32), (0, 0), 18.0)
                makeup_guard *= (1.0 - 0.72 * fs_er)

        broad_field = np.clip(0.42 * e_near + 0.40 * e_mid + 0.18 * e_far, 0.0, 1.0)
        hot_field = np.clip(0.68 * e_near + 0.32 * e_mid, 0.0, 1.0)

        scale_c = max(0.35, cs if cs > 0.01 else float(spill_strength) * 2.5)
        broad_raw = broad_field.max(axis=2) * 1.55 * scale_c * sa * vfx_dir_w * mat_broad_gate * makeup_guard
        broad_a = np.clip(broad_raw / (1.0 + 0.55 * broad_raw), 0.0, 1.0)
        broad_rgb = np.where(broad_field.max(axis=2, keepdims=True) > 1e-4, np.clip(broad_field / np.maximum(broad_field.max(axis=2, keepdims=True), 1e-4), 0.0, 1.0), 0.0)

        hot_raw = hot_field.max(axis=2) * 1.45 * scale_c * sa * vfx_dir_w * mat_spec_gate * makeup_guard
        hot_a = np.clip(hot_raw / (1.0 + 0.65 * hot_raw), 0.0, 1.0)
        hot_rgb = np.where(hot_field.max(axis=2, keepdims=True) > 1e-4, np.clip(hot_field / np.maximum(hot_field.max(axis=2, keepdims=True), 1e-4), 0.0, 1.0), 0.0)

        p_broad = out_d / "vfx_causal_broad_spill_screen.png"
        p_hot = out_d / "vfx_causal_hot_rim_add.png"
        p_spill_legacy = out_d / "vfx_subject_spill_screen.png"
        _save_rgba(p_broad, broad_rgb, broad_a, overwrite=overwrite)
        _save_rgba(p_hot, hot_rgb, hot_a, overwrite=overwrite)
        _save_rgba(p_spill_legacy, broad_rgb, broad_a, overwrite=overwrite)
        outputs["vfx_causal_broad_spill_screen"] = str(p_broad)
        outputs["vfx_causal_hot_rim_add"] = str(p_hot)
        outputs["vfx_subject_spill_screen"] = str(p_spill_legacy)
        layer_specs.extend([
            {"key": "causal_broad_spill", "path": str(p_broad), "layer_name": "03h_VFX_Causal_BroadSpill", "placement": "subject_clip", "blend": "Screen", "opacity": 0.36, "fill_opacity": 100},
            {"key": "causal_hot_rim", "path": str(p_hot), "layer_name": "03h2_VFX_Causal_HotRim", "placement": "subject_clip", "blend": "Linear Dodge (Add)", "opacity": 0.48, "fill_opacity": 44},
        ])

        # Nearby environment bounce light cast outside subject onto ground/railings/architecture
        p_env: Path | None = None
        if float(env_spill_strength) > 0.01:
            env_field = np.clip(0.50 * e_mid + 0.50 * e_far, 0.0, 1.0)
            outside = np.clip(1.0 - sa, 0.0, 1.0)
            prox = 0.35 + 0.65 * np.exp(-np.square(out_dist / max(120.0, width * 0.08)))
            if bg_path is not None and bg_path.exists():
                bg_im = Image.open(bg_path.resolve()).convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
                bg_a = np.array(bg_im, dtype=np.float32) / 255.0
                bg_l = 0.2126 * bg_a[:, :, 0] + 0.7152 * bg_a[:, :, 1] + 0.0722 * bg_a[:, :, 2]
                surf_gate = np.clip(0.28 + 0.90 * bg_l, 0.20, 1.0)
            else:
                surf_gate = np.ones((height, width), dtype=np.float32)
            env_raw = env_field.max(axis=2) * 1.40 * float(env_spill_strength) * outside * prox * surf_gate
            env_a = np.clip(env_raw / (1.0 + 0.55 * env_raw), 0.0, 1.0)
            env_rgb = np.where(env_field.max(axis=2, keepdims=True) > 1e-4, np.clip(env_field / np.maximum(env_field.max(axis=2, keepdims=True), 1e-4), 0.0, 1.0), 0.0)
            p_env = out_d / "vfx_causal_env_spill_screen.png"
            _save_rgba(p_env, env_rgb, env_a, overwrite=overwrite)
            outputs["vfx_causal_env_spill_screen"] = str(p_env)
            layer_specs.append({"key": "causal_env_spill", "path": str(p_env), "layer_name": "02d_VFX_Causal_EnvSpill", "placement": "behind", "blend": "Screen", "opacity": 0.56, "fill_opacity": 100})

        # 3-panel causal visual preview
        if fg_rgb_resolved is not None:
            ys_f, xs_f = np.nonzero(sa > 0.1)
            bx0, by0 = max(0, int(xs_f.min()) - 40), max(0, int(ys_f.min()) - 40)
            bx1, by1 = min(width, int(xs_f.max()) + 40), min(height, int(ys_f.max()) + 40)
            backdrop = np.array([0.08, 0.10, 0.14], dtype=np.float32)[None, None, :]
            fg_prev = fg_rgb_resolved * sa[:, :, None] + backdrop * (1.0 - sa[:, :, None])
            relit_prev = 1.0 - (1.0 - fg_prev) * (1.0 - broad_rgb * (broad_a[:, :, None] * 0.36))
            relit_prev = np.clip(relit_prev + hot_rgb * (hot_a[:, :, None] * 0.28), 0.0, 1.0)
            map_vis = broad_rgb * broad_a[:, :, None] + hot_rgb * (hot_a[:, :, None] * 0.55)
            map_vis = np.clip(map_vis, 0.0, 1.0)
            cw_p = min(560, bx1 - bx0)
            ch_p = max(240, int(round((by1 - by0) * cw_p / max(1, bx1 - bx0))))
            pan_v = [
                Image.fromarray(np.clip(np.round(cleaned[by0:by1, bx0:bx1] * 255.0), 0, 255).astype(np.uint8)).resize((cw_p, ch_p), Image.Resampling.LANCZOS),
                Image.fromarray(np.clip(np.round(map_vis[by0:by1, bx0:bx1] * 255.0), 0, 255).astype(np.uint8)).resize((cw_p, ch_p), Image.Resampling.LANCZOS),
                Image.fromarray(np.clip(np.round(relit_prev[by0:by1, bx0:bx1] * 255.0), 0, 255).astype(np.uint8)).resize((cw_p, ch_p), Image.Resampling.LANCZOS),
            ]
            cmp_im = Image.new("RGB", (cw_p * 3 + 48, ch_p + 56), (14, 16, 22))
            dr = ImageDraw.Draw(cmp_im)
            for idx, (pim, ttl) in enumerate(zip(pan_v, ("1. Unmulted VFX", "2. 2.5D Causal Spill Map", "3. Subject Causal Relit"))):
                px = 12 + idx * (cw_p + 12)
                dr.text((px, 12), ttl, fill=(230, 236, 245))
                cmp_im.paste(pim, (px, 40))
            p_cprev = out_d / "vfx_causal_preview.png"
            cmp_im.save(p_cprev, "PNG")
            outputs["vfx_causal_preview"] = str(p_cprev)

    if "vfx_front_subject_rgba" in outputs:
        layer_specs.append({"key": "vfx_front", "path": outputs["vfx_front_subject_rgba"], "layer_name": "04c_VFX_Front_Subject", "placement": "front", "blend": "Linear Dodge (Add)", "opacity": 0.72, "fill_opacity": 82})

    meta = {
        "ok": True,
        "input": str(in_p),
        "size": [width, height],
        "placement": placement,
        "black_floor_subtracted": round(floor_scalar, 4),
        "front_eye_overlap_px": eye_ov_px,
        "front_subject_body_coverage_pct": body_cov_pct,
        "causal_enabled": bool(sa_resolved is not None and (cs > 0.01 or spill_strength > 0.0)),
        "layers": layer_specs,
        "outputs": outputs,
    }
    p_meta = out_d / "vfx_meta.json"
    p_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["vfx_meta"] = str(p_meta)
    return meta
