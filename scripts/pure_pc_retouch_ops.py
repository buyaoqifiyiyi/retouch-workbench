"""Pure PC (non-generative) portrait & COS retouching engine: Rolling-Guidance Skin FS, Studio Brushes, 478-landmark Makeup Paint, and Pose-driven Body Liquify."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from workbench_ops import ensure_hair_brush_tips, guided_filter_gray, lab_to_rgb255, rgb255_to_lab


def _read_rgb_f32(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32)


def _read_gray_f32(path: Path, size_wh: tuple[int, int]) -> np.ndarray:
    im = Image.open(path).convert("L")
    if im.size != size_wh:
        im = im.resize(size_wh, Image.Resampling.BILINEAR)
    return np.array(im, dtype=np.float32) / 255.0


def _write_png_u8(path: Path, arr: np.ndarray, mode: str, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.round(arr), 0, 255).astype(np.uint8), mode).save(path, "PNG")


def _J(obj: object) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def ensure_studio_brush_tips(brushes_dir: Path | None = None, force: bool = False) -> dict[str, Path]:
    """Generate 7 non-generative grayscale studio brush tips alongside hair_01..03 (black=ink, white=transparent)."""
    out = ensure_hair_brush_tips(brushes_dir=brushes_dir, force=force)
    bdir = brushes_dir or (Path(__file__).resolve().parent.parent / "resources" / "brushes")
    bdir.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0:256, 0:256].astype(np.float32)

    def _save_tip(name: str, ink01: np.ndarray) -> None:
        p = bdir / f"{name}.png"
        if force or not p.exists():
            arr = np.clip(np.round(255.0 * (1.0 - np.clip(ink01, 0.0, 1.0))), 0, 255).astype(np.uint8)
            _write_png_u8(p, arr, "L", overwrite=True)
        out[name] = p

    r = np.hypot(xx - 127.5, yy - 127.5) / 118.0
    # 1. Soft airbrush
    _save_tip("makeup_01_soft_airbrush", np.exp(-3.2 * (r ** 2)) * np.clip(1.0 - r, 0.0, 1.0))
    # 2. Tapered lash filament
    dx_l, dy_l = (xx - 127.5) / 14.0, (yy - 127.5) / 72.0
    _save_tip("makeup_02_lash_taper", np.exp(-2.6 * (dx_l ** 2 + dy_l ** 2)))
    # 3. Blush stippled soft cloud
    cloud = np.zeros((256, 256), dtype=np.float32)
    rng = np.random.default_rng(77)
    for _ in range(18):
        cx, cy = float(rng.uniform(76, 180)), float(rng.uniform(76, 180))
        sr = float(rng.uniform(22, 42))
        w = float(rng.uniform(0.35, 0.85))
        cloud = np.maximum(cloud, w * np.exp(-2.2 * (((xx - cx) ** 2 + (yy - cy) ** 2) / (sr ** 2))))
    _save_tip("makeup_03_blush_cloud", cloud * np.clip(1.0 - r, 0.0, 1.0))
    # 4. Liquid lip-gloss specular crescent
    dx_g, dy_g = (xx - 127.5) / 68.0, (yy - 134.0) / 22.0
    arc = np.exp(-2.8 * (dx_g ** 2 + (dy_g + 0.38 * (dx_g ** 2)) ** 2))
    _save_tip("makeup_04_gloss_specular", arc)
    # 5. Pearl sparkle & pinpoint catchlight
    star = np.exp(-((np.hypot(xx - 127.5, yy - 127.5) / 18.0) ** 1.6))
    cross = np.maximum(
        np.exp(-np.abs(yy - 127.5) / 4.2) * np.exp(-np.abs(xx - 127.5) / 48.0),
        np.exp(-np.abs(xx - 127.5) / 4.2) * np.exp(-np.abs(yy - 127.5) / 48.0),
    )
    _save_tip("makeup_05_pearl_star", np.clip(np.maximum(star, 0.72 * cross), 0.0, 1.0))
    # 6. Ultra-smooth cosine-bell D&B feather tip
    cos_bell = np.where(r < 1.0, 0.5 * (1.0 + np.cos(math.pi * r)), 0.0) ** 1.4
    _save_tip("retouch_01_db_feather", cos_bell)
    # 7. Crisp micro liner tip
    _save_tip("retouch_02_micro_liner", np.exp(-2.8 * ((np.hypot(xx - 127.5, yy - 127.5) / 28.0) ** 2)))
    return out


def _render_pc_strokes_multi(
    *,
    pc_exe: str,
    width: int,
    height: int,
    brush_tips: dict[str, Path],
    strokes: list[dict[str, Any]],
    out_png: Path,
    tmp_pcraft: Path,
    base_fill_hex: str | None = None,
) -> None:
    """Render strokes natively through PhotoCraft CLI paint.stroke with chunking."""
    if not strokes:
        if base_fill_hex is None:
            _write_png_u8(out_png, np.zeros((height, width, 4), dtype=np.float32), "RGBA", overwrite=True)
        else:
            bg = np.full((height, width, 3), 128, dtype=np.uint8)
            _write_png_u8(out_png, bg, "RGB", overwrite=True)
        return

    chunk_size = 14
    chunks = [strokes[i : i + chunk_size] for i in range(0, len(strokes), chunk_size)]
    for ci, ch in enumerate(chunks):
        if ci == 0:
            bg_spec = "transparent" if base_fill_hex is None else base_fill_hex
            base_args = [
                pc_exe,
                "run",
                "--new",
                _J({"width": width, "height": height, "mode": "rgb", "depth": 8, "background": bg_spec}),
            ]
        else:
            base_args = [pc_exe, "run", str(tmp_pcraft)]

        cmds: list[str] = []
        for tip_name in sorted({str(s["tip"]) for s in ch}):
            cmds += [
                "--cmd",
                "brush.presets.importImage",
                "--params",
                _J({"path": str(brush_tips[tip_name]), "name": tip_name, "select": False}),
            ]

        for s in ch:
            tname = str(s["tip"])
            cmds += ["--cmd", "brush.presets.setCurrent", "--params", _J({"name": tname})]
            cmds += [
                "--cmd",
                "tools.setBrush",
                "--params",
                _J({
                    "size": float(s["size"]),
                    "spacing": float(s.get("spacing", 0.08)),
                    "shapeDynamics": {
                        "enabled": True,
                        "size": {"control": "penPressure", "minimum": float(s.get("size_min", 0.15))},
                        "angle": {"control": "direction" if s.get("follow_dir", False) else "off"},
                    },
                    "transfer": {
                        "enabled": True,
                        "opacity": {"control": "penPressure", "minimum": 0.0},
                    },
                }),
            ]
            cmds += [
                "--cmd",
                "paint.stroke",
                "--params",
                _J({
                    "points": s["points"],
                    "size": float(s["size"]),
                    "color": str(s["color"]),
                    "opacity": float(s["opacity"]),
                    "flow": float(s.get("flow", 0.75)),
                    "smoothing": float(s.get("smoothing", 0.30)),
                }),
            ]

        is_last = ci == len(chunks) - 1
        out_args = ["--out", str(out_png)] if is_last else ["--out", str(tmp_pcraft)] + (["--in-place"] if ci > 0 else [])
        r = subprocess.run(base_args + cmds + out_args, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            raise RuntimeError(f"PhotoCraft paint.stroke chunk {ci} failed (rc={r.returncode}): {r.stderr[-600:] or r.stdout[-600:]}")
    if tmp_pcraft.exists():
        tmp_pcraft.unlink(missing_ok=True)


def _detect_mediapipe_face478(
    rgb_u8: np.ndarray,
    face_task_path: Path,
    faces_json_path: Path | None = None,
) -> np.ndarray | None:
    if not face_task_path.exists():
        return None
    import mediapipe as mp  # type: ignore
    from mediapipe.tasks import python as mp_python  # type: ignore
    from mediapipe.tasks.python import vision as mp_vision  # type: ignore

    h, w = rgb_u8.shape[:2]
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=face_task_path.read_bytes()),
        num_faces=1,
        min_face_detection_confidence=0.25,
        min_face_presence_confidence=0.25,
    )
    with mp_vision.FaceLandmarker.create_from_options(opts) as det:
        # 1. Try ROI cropped & de-tilted from faces.json first (robust on 4K-5K full-body tilted COS photos!)
        if faces_json_path and faces_json_path.exists():
            try:
                fdat = json.loads(faces_json_path.read_text(encoding="utf-8"))
                if fdat.get("faces"):
                    f0 = fdat["faces"][0]
                    bx0, by0, bx1, by1 = [float(v) for v in f0["bbox"]]
                    bw, bh = bx1 - bx0, by1 - by0
                    cx, cy = 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)
                    pad = max(bw, bh) * 0.85
                    x0, y0 = max(0, int(round(cx - pad))), max(0, int(round(cy - pad)))
                    x1, y1 = min(w, int(round(cx + pad))), min(h, int(round(cy + pad)))
                    crop = rgb_u8[y0:y1, x0:x1]
                    ch, cw = crop.shape[:2]
                    # Compute upright de-tilt angle from landmarks_5 eyes
                    lm5 = f0.get("landmarks_5")
                    ang_deg = 0.0
                    if lm5 and len(lm5) >= 2:
                        dx_e = float(lm5[1][0]) - float(lm5[0][0])
                        dy_e = float(lm5[1][1]) - float(lm5[0][1])
                        ang_deg = math.degrees(math.atan2(dy_e, dx_e))
                    M = cv2.getRotationMatrix2D((cw * 0.5, ch * 0.5), ang_deg, 1.0)
                    rot_crop = cv2.warpAffine(crop, M, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rot_crop))
                    res = det.detect(mp_img)
                    if res.face_landmarks:
                        Minv = cv2.invertAffineTransform(M)
                        out_pts = []
                        for lm in res.face_landmarks[0]:
                            rx, ry = lm.x * cw, lm.y * ch
                            gx = float(Minv[0, 0] * rx + Minv[0, 1] * ry + Minv[0, 2]) + x0
                            gy = float(Minv[1, 0] * rx + Minv[1, 1] * ry + Minv[1, 2]) + y0
                            out_pts.append([gx, gy])
                        return np.array(out_pts, dtype=np.float32)
            except Exception:
                pass

        # 2. Fallback: full-frame detection
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_u8))
        res = det.detect(mp_img)
        if res.face_landmarks:
            return np.array([[lm.x * w, lm.y * h] for lm in res.face_landmarks[0]], dtype=np.float32)
    return None


def _detect_mediapipe_pose33(rgb_u8: np.ndarray, pose_task_path: Path) -> np.ndarray | None:
    if not pose_task_path.exists():
        return None
    import mediapipe as mp  # type: ignore
    from mediapipe.tasks import python as mp_python  # type: ignore
    from mediapipe.tasks.python import vision as mp_vision  # type: ignore

    h, w = rgb_u8.shape[:2]
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=pose_task_path.read_bytes()),
        num_poses=1,
        min_pose_detection_confidence=0.35,
    )
    with mp_vision.PoseLandmarker.create_from_options(opts) as det:
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_u8))
        res = det.detect(mp_img)
    if not res.pose_landmarks:
        return None
    lms = res.pose_landmarks[0]
    return np.array([[lm.x * w, lm.y * h, getattr(lm, "visibility", 1.0)] for lm in lms], dtype=np.float32)


def run_skin_fs(
    *,
    input_path: Path,
    out_dir: Path,
    pc_exe: str,
    face_parse_dir: Path,
    body_parse_dir: Path | None = None,
    alpha_mask_path: Path | None = None,
    face_task_path: Path | None = None,
    smooth_tone: float = 0.62,
    porcelain_lift: float = 0.38,
    warmth_balance: float = 0.45,
    db_strength: float = 0.42,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Pure non-generative Rolling-Guidance CIE Lab frequency separation + PhotoCraft native Soft Light D&B."""
    in_p = input_path.resolve()
    out_d = out_dir.resolve()
    out_d.mkdir(parents=True, exist_ok=True)

    rgb_f32 = _read_rgb_f32(in_p)
    h, w = rgb_f32.shape[:2]
    rgb_u8 = np.clip(np.round(rgb_f32), 0, 255).astype(np.uint8)
    lab = rgb255_to_lab(rgb_f32)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    # Build strict continuous safe skin mask (face + neck/chest/arms + bare legs, never eyes/lips/hair/clothes)
    f_skin = _read_gray_f32(face_parse_dir / "mask_skin.png", (w, h))
    f_eyes = _read_gray_f32(face_parse_dir / "mask_eyes_brows.png", (w, h))
    f_lips = _read_gray_f32(face_parse_dir / "mask_lips.png", (w, h))
    f_hair = _read_gray_f32(face_parse_dir / "mask_hair_head.png", (w, h))

    skin_union = f_skin.copy()
    hair_guard = f_hair.copy()
    clothes_guard = np.zeros((h, w), dtype=np.float32)
    if body_parse_dir is not None and body_parse_dir.exists():
        b_arms = _read_gray_f32(body_parse_dir / "mask_arms.png", (w, h))
        b_bskin = _read_gray_f32(body_parse_dir / "mask_body_skin.png", (w, h))
        b_legs = _read_gray_f32(body_parse_dir / "mask_legs.png", (w, h))
        b_hair = _read_gray_f32(body_parse_dir / "mask_hair.png", (w, h))
        b_clothes = _read_gray_f32(body_parse_dir / "mask_clothes_all.png", (w, h))

        chroma = np.hypot(a, b)
        # Only count bare leg skin with real warm flesh chroma (protects white/sheer stockings)
        bare_legs = b_legs * np.clip((a - 6.0) / 4.0, 0.0, 1.0) * np.clip((chroma - 13.0) / 6.0, 0.0, 1.0)
        # Avoid white clothing bleeding into neck/chest mask_body_skin
        real_bskin = b_bskin * np.clip((chroma - 9.0) / 5.0, 0.0, 1.0) * np.clip((92.0 - L) / 6.0, 0.0, 1.0)
        skin_union = np.maximum.reduce([f_skin, b_arms, real_bskin, bare_legs])
        # Never let coarse full-body SegFormer hair mask punch false holes inside high-res BiSeNet f_skin cheeks!
        hair_guard = np.maximum(f_hair, b_hair * np.clip(1.0 - f_skin * 1.4, 0.0, 1.0))
        clothes_guard = b_clothes

    eyes_g = cv2.dilate((f_eyes > 0.12).astype(np.float32), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)))
    lips_g = cv2.dilate((f_lips > 0.12).astype(np.float32), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)))
    hair_g_raw = cv2.dilate((hair_guard > 0.40).astype(np.float32), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    hair_g = np.where(f_skin > 0.65, f_hair, hair_g_raw)
    clothes_g = cv2.dilate((clothes_guard > 0.42).astype(np.float32), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))

    fg = _read_gray_f32(alpha_mask_path.resolve(), (w, h)) if alpha_mask_path and alpha_mask_path.exists() else np.ones((h, w), dtype=np.float32)
    skin_safe = skin_union * (1.0 - eyes_g) * (1.0 - lips_g) * (1.0 - 0.90 * hair_g) * (1.0 - 0.92 * clothes_g) * fg
    skin_safe = np.clip(cv2.GaussianBlur(skin_safe, (0, 0), 4.0), 0.0, 1.0)
    p_skin_mask = out_d / "skin_unified_mask.png"
    _write_png_u8(p_skin_mask, skin_safe * 255.0, "L", overwrite=overwrite)

    # 2-scale Rolling Guidance edge-preserving decomposition strictly on L01 + chroma
    L01 = np.clip(L / 100.0, 0.0, 1.0).astype(np.float32)
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "rollingGuidanceFilter"):
        L_mid_base = cv2.ximgproc.rollingGuidanceFilter(L01, d=-1, sigmaColor=0.055, sigmaSpace=5.0, numOfIter=3) * 100.0
        L_low_tone = cv2.ximgproc.rollingGuidanceFilter(L01, d=-1, sigmaColor=0.14, sigmaSpace=24.0, numOfIter=4) * 100.0
    else:
        L_mid_base = guided_filter_gray(L01, L01, radius=5, eps=0.0025) * 100.0
        L_low_tone = guided_filter_gray(L01, L01, radius=24, eps=0.018) * 100.0

    L_high_pores = L - L_mid_base          # fine 1-3px pores & skin micro-grain (kept 100% intact)
    L_mid_blotch = L_mid_base - L_low_tone # 4-24px uneven spots / mottling

    # Softly flatten blotchy mid-band & lift deep dull skin tone gently
    st = float(np.clip(smooth_tone, 0.0, 1.0))
    L_mid_clean = L_mid_blotch * (1.0 - 0.58 * st)
    L_low_smooth = cv2.bilateralFilter(L_low_tone.astype(np.float32), 11, 8.0, 18.0)
    L_low_clean = L_low_tone * (1.0 - 0.48 * st) + L_low_smooth * (0.48 * st)

    # Unify face & neck/arm/thigh skin tone toward luminous fair porcelain
    sel_f = (f_skin * (1.0 - eyes_g) * (1.0 - lips_g)) > 0.50
    sel_all = skin_safe > 0.50
    if int(sel_all.sum()) >= 500:
        ref_L = float(np.percentile(L_low_clean[sel_f if int(sel_f.sum()) >= 300 else sel_all], 62))
        target_L = min(78.5, max(ref_L + 4.5 * float(porcelain_lift), 68.0))
        pl = float(np.clip(porcelain_lift, 0.0, 1.0))
        # Pull darker limb/neck regions gently closer to airy bright skin L*
        delta_L = np.where(L_low_clean < target_L, (target_L - L_low_clean) * (0.42 * pl), 0.0)
        L_low_clean = np.clip(L_low_clean + delta_L, 4.0, 96.0)

        # Chroma smoothing & gentle reduction of muddy yellow/gray cast
        wb = float(np.clip(warmth_balance, 0.0, 1.0))
        a_sm = cv2.bilateralFilter(a.astype(np.float32), 9, 5.0, 14.0)
        b_sm = cv2.bilateralFilter(b.astype(np.float32), 9, 5.0, 14.0)
        a_out = a * (1.0 - 0.45 * st) + a_sm * (0.45 * st)
        b_out = b * (1.0 - 0.45 * st) + b_sm * (0.45 * st)
        # Keep rosy blush healthy, softly reduce excess warm/yellow cast in shadow skin
        b_out = b_out - 2.6 * wb * np.clip((b_out - 10.5) / 8.0, 0.0, 1.0)
    else:
        a_out, b_out = a.copy(), b.copy()

    # Low+Mid clean image & high-frequency pore layer for PhotoCraft Linear Light stack
    L_lm = np.clip(L_low_clean + L_mid_clean, 3.0, 97.5)
    rgb_lm = lab_to_rgb255(np.dstack([L_lm, a_out, b_out]))
    rgb_full_fs = lab_to_rgb255(np.dstack([np.clip(L_lm + L_high_pores, 2.0, 99.0), a_out, b_out]))
    out_rgb = np.clip(rgb_f32 * (1.0 - skin_safe[:, :, None]) + rgb_full_fs * skin_safe[:, :, None], 0.0, 255.0)

    # Export Linear-Light centered 128 high-freq layer + low-tone layer
    hi_gray_128 = np.clip(128.0 + L_high_pores * (255.0 / 100.0) * 0.5, 0.0, 255.0)
    hi_rgb_128 = np.dstack([hi_gray_128, hi_gray_128, hi_gray_128])
    p_low = out_d / "skin_fs_low_tone.png"
    p_hi = out_d / "skin_fs_high_pores_linearlight.png"
    p_res = out_d / "skin_fs_result.png"
    _write_png_u8(p_low, rgb_lm, "RGB", overwrite=overwrite)
    _write_png_u8(p_hi, hi_rgb_128, "RGB", overwrite=overwrite)

    # Native PhotoCraft 50%-gray Soft Light Dodge & Burn contour layer driven by MediaPipe 478 landmarks
    p_db = out_d / "skin_db_neutral_softlight.png"
    fj_cand = face_parse_dir / "faces.json"
    pts478 = _detect_mediapipe_face478(rgb_u8, face_task_path, fj_cand if fj_cand.exists() else None) if face_task_path else None
    tips = ensure_studio_brush_tips()
    db_strokes: list[dict[str, Any]] = []
    if pts478 is not None and db_strength > 0.02:
        l_eye = pts478[33]
        r_eye = pts478[263]
        eye_span = max(30.0, float(np.linalg.norm(r_eye - l_eye)))
        dbs = float(np.clip(db_strength, 0.0, 1.0))

        # Subtle highlights (#ffffff): nose bridge -> tip, forehead center, upper cheeks, chin dot
        bridge_pts = [[float(pts478[i][0]), float(pts478[i][1]), pr] for i, pr in ((168, 0.4), (6, 0.85), (197, 0.9), (195, 0.8), (5, 0.75), (4, 0.55))]
        db_strokes.append({"tip": "retouch_01_db_feather", "points": bridge_pts, "size": round(eye_span * 0.14, 1), "size_min": 0.4, "color": "#ffffff", "opacity": round(0.18 * dbs, 3), "flow": 0.65, "spacing": 0.08})

        # Cheek apple luminosity highlight
        for idx_list in ((50, 101, 118), (280, 330, 347)):
            c_pts = [[float(pts478[i][0]), float(pts478[i][1]), 0.75] for i in idx_list]
            db_strokes.append({"tip": "retouch_01_db_feather", "points": c_pts, "size": round(eye_span * 0.42, 1), "size_min": 0.5, "color": "#ffffff", "opacity": round(0.14 * dbs, 3), "flow": 0.55, "spacing": 0.10})

        # Gentle nasal-wall & jawline shadow sculpt (#1a1618)
        for idx_list in ((193, 188, 174), (417, 412, 399)):
            n_pts = [[float(pts478[i][0]), float(pts478[i][1]), 0.65] for i in idx_list]
            db_strokes.append({"tip": "retouch_01_db_feather", "points": n_pts, "size": round(eye_span * 0.13, 1), "size_min": 0.35, "color": "#20181a", "opacity": round(0.10 * dbs, 3), "flow": 0.55, "spacing": 0.08})

    _render_pc_strokes_multi(
        pc_exe=pc_exe,
        width=w,
        height=h,
        brush_tips=tips,
        strokes=db_strokes,
        out_png=p_db,
        tmp_pcraft=out_d / "_tmp_db.pcraft",
        base_fill_hex="#808080",
    )

    # Apply neutral-gray D&B in Soft Light onto out_rgb for ready-to-use skin_fs_result.png preview
    if p_db.exists():
        db_arr = np.array(Image.open(p_db).convert("RGB"), dtype=np.float32) / 255.0
        bg_01 = np.clip(out_rgb / 255.0, 0.0, 1.0)
        # Standard Photoshop Soft Light approximation
        soft = np.where(db_arr <= 0.5, bg_01 - (1.0 - 2.0 * db_arr) * bg_01 * (1.0 - bg_01), bg_01 + (2.0 * db_arr - 1.0) * (np.sqrt(np.maximum(bg_01, 1e-5)) - bg_01))
        out_rgb = np.clip((bg_01 * (1.0 - skin_safe[:, :, None]) + soft * skin_safe[:, :, None]) * 255.0, 0.0, 255.0)
    _write_png_u8(p_res, out_rgb, "RGB", overwrite=overwrite)

    meta = {
        "ok": True,
        "input": str(in_p),
        "skin_px": int((skin_safe > 0.4).sum()),
        "mediapipe_478_detected": pts478 is not None,
        "db_stroke_count": len(db_strokes),
        "outputs": {
            "skin_fs_result": str(p_res),
            "skin_fs_low_tone": str(p_low),
            "skin_fs_high_pores_linearlight": str(p_hi),
            "skin_db_neutral_softlight": str(p_db),
            "skin_unified_mask": str(p_skin_mask),
        },
    }
    (out_d / "skin_fs_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def run_makeup_paint(
    *,
    input_path: Path,
    out_dir: Path,
    pc_exe: str,
    face_task_path: Path,
    faces_json_path: Path | None = None,
    face_parse_dir: Path | None = None,
    style: str = "natural",
    intensity: float = 0.72,
    iris_color: str | None = None,
    lip_tone: str | None = None,
    eyeshadow_tone: str | None = None,
    blush_tone: str | None = None,
    lash_mode: str = "auto",
    aegyo_sal: float | None = None,
    skip_nose_db: bool = False,
    build_pcraft: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Pure-PC PhotoCraft optical multi-blend makeup layers v2 (delegates to makeup_retouch_ops)."""
    from makeup_retouch_ops import run_makeup_paint_v2

    return run_makeup_paint_v2(
        input_path=input_path,
        out_dir=out_dir,
        pc_exe=pc_exe,
        face_task_path=face_task_path,
        faces_json_path=faces_json_path,
        face_parse_dir=face_parse_dir,
        style=style,
        intensity=intensity,
        iris_color=iris_color,
        lip_tone=lip_tone,
        eyeshadow_tone=eyeshadow_tone,
        blush_tone=blush_tone,
        lash_mode=lash_mode,
        aegyo_sal=aegyo_sal,
        skip_nose_db=skip_nose_db,
        build_pcraft=build_pcraft,
        overwrite=overwrite,
    )


def run_body_liquify(
    *,
    input_path: Path,
    out_dir: Path,
    pc_exe: str,
    pose_task_path: Path,
    body_parse_dir: Path,
    faces_json_path: Path | None = None,
    shoulder_slim: float = 0.35,
    arm_slim: float = 0.35,
    waist_cinch: float = 0.25,
    cuff_smooth: float = 0.65,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Pose + contour-curvature driven PhotoCraft native filter.liquify with freeze masks & mirror sheet."""
    in_p = input_path.resolve()
    out_d = out_dir.resolve()
    out_d.mkdir(parents=True, exist_ok=True)

    rgb_u8 = np.array(Image.open(in_p).convert("RGB"), dtype=np.uint8)
    h, w = rgb_u8.shape[:2]
    pose = _detect_mediapipe_pose33(rgb_u8, pose_task_path)

    legs_m = _read_gray_f32(body_parse_dir / "mask_legs.png", (w, h))
    arms_m = _read_gray_f32(body_parse_dir / "mask_arms.png", (w, h))

    strokes: list[dict[str, Any]] = []
    # Freeze face & facial features from faces.json so facial geometry stays 100% locked
    if faces_json_path and faces_json_path.exists():
        fdat = json.loads(faces_json_path.read_text(encoding="utf-8"))
        for f0 in fdat.get("faces", [])[:1]:
            bx0, by0, bx1, by1 = [float(v) for v in f0.get("bbox", [0, 0, 0, 0])]
            if bx1 > bx0 and by1 > by0:
                pad_f = 0.12 * (bx1 - bx0)
                strokes.append({
                    "tool": "lassoMask",
                    "amount": 1,
                    "points": [
                        [round(bx0 - pad_f, 1), round(by0 - pad_f, 1)],
                        [round(bx1 + pad_f, 1), round(by0 - pad_f, 1)],
                        [round(bx1 + pad_f, 1), round(by1 + pad_f, 1)],
                        [round(bx0 - pad_f, 1), round(by1 + pad_f, 1)],
                    ],
                })

    def _add_warp(x0: float, y0: float, dx: float, dy: float, sz: float, pressure: float = 65.0) -> None:
        if math.hypot(dx, dy) < 1.5:
            return
        strokes.append({
            "tool": "forwardWarp",
            "size": int(np.clip(round(sz), 24, 900)),
            "density": 50,
            "pressure": int(np.clip(round(pressure), 15, 95)),
            "points": [[round(x0, 1), round(y0, 1)], [round(x0 + dx, 1), round(y0 + dy, 1)]],
        })

    ref_span = max(180.0, float(w) * 0.16)
    if pose is not None:
        # MediaPipe Pose indices: 11/12 shoulders, 13/14 elbows, 15/16 wrists, 23/24 hips
        l_sh, r_sh = pose[11], pose[12]
        l_el, r_el = pose[13], pose[14]
        l_wr, r_wr = pose[15], pose[16]
        l_hp, r_hp = pose[23], pose[24]
        if l_sh[2] > 0.3 and r_sh[2] > 0.3:
            ref_span = max(120.0, float(np.linalg.norm(l_sh[:2] - r_sh[:2])))

        # Smooth shoulder trapezius gently inward/downward
        for sh_pt, other_pt in ((l_sh, r_sh), (r_sh, l_sh)):
            if sh_pt[2] > 0.4 and shoulder_slim > 0.02:
                to_center = other_pt[:2] - sh_pt[:2]
                nrm = np.linalg.norm(to_center) + 1e-5
                ux, uy = float(to_center[0] / nrm), float(to_center[1] / nrm)
                shift = ref_span * 0.026 * float(shoulder_slim)
                _add_warp(float(sh_pt[0]) - 0.12 * ref_span * ux, float(sh_pt[1]) - 0.08 * ref_span, shift * ux, shift * 0.6, ref_span * 0.38)

        # Arm contour streamline (skip arms very close to wrists/hands)
        for sh_pt, el_pt, wr_pt, opp_sh in ((l_sh, l_el, l_wr, r_sh), (r_sh, r_el, r_wr, l_sh)):
            if el_pt[2] > 0.35 and sh_pt[2] > 0.35 and arm_slim > 0.02:
                mid = 0.52 * sh_pt[:2] + 0.48 * el_pt[:2]
                to_body = opp_sh[:2] - mid
                nrm = np.linalg.norm(to_body) + 1e-5
                ux, uy = float(to_body[0] / nrm), float(to_body[1] / nrm)
                shift = ref_span * 0.024 * float(arm_slim)
                _add_warp(float(mid[0] - 0.08 * ref_span * ux), float(mid[1]), shift * ux, shift * uy * 0.3, ref_span * 0.32)

        # Subtle waist cinch
        if l_hp[2] > 0.3 and r_hp[2] > 0.3 and l_sh[2] > 0.3 and r_sh[2] > 0.3 and waist_cinch > 0.02:
            w_l = 0.45 * l_sh[:2] + 0.55 * l_hp[:2]
            w_r = 0.45 * r_sh[:2] + 0.55 * r_hp[:2]
            mid_w = 0.5 * (w_l + w_r)
            for wp in (w_l, w_r):
                to_c = mid_w - wp
                nrm = np.linalg.norm(to_c) + 1e-5
                shift = ref_span * 0.022 * float(waist_cinch)
                _add_warp(float(wp[0]), float(wp[1]), float(to_c[0] / nrm * shift), 0.0, ref_span * 0.34)

    # Automatic outer thigh-cuff bump detection along mask_legs silhouette
    cuff_anchors: list[tuple[float, float]] = []
    ys_l, xs_l = np.where(legs_m > 0.5)
    if len(ys_l) >= 500 and cuff_smooth > 0.02:
        y_top, y_bot = int(ys_l.min()), int(ys_l.max())
        scan_y0 = y_top + int(0.08 * (y_bot - y_top))
        scan_y1 = y_top + int(0.42 * (y_bot - y_top))
        left_xs, right_xs, row_ys = [], [], []
        for ry in range(scan_y0, scan_y1, 8):
            cols = np.where(legs_m[ry] > 0.5)[0]
            if len(cols) >= 24:
                left_xs.append(float(cols.min()))
                right_xs.append(float(cols.max()))
                row_ys.append(float(ry))
        if len(row_ys) >= 7:
            ry_arr = np.array(row_ys)
            for xs_side, inward_sign in ((np.array(left_xs), 1.0), (np.array(right_xs), -1.0)):
                smooth_x = cv2.GaussianBlur(xs_side.reshape(-1, 1).astype(np.float32), (1, 15), 4.0).ravel()
                # Outward protrusion: left edge smaller than smooth line OR right edge larger than smooth line
                bump = (smooth_x - xs_side) if inward_sign > 0 else (xs_side - smooth_x)
                idx_max = int(np.argmax(bump))
                if 2 <= idx_max <= len(bump) - 3 and float(bump[idx_max]) > 4.0:
                    cx_b = float(xs_side[idx_max])
                    cy_b = float(ry_arr[idx_max])
                    cuff_anchors.append((cx_b, cy_b))
                    shift = float(np.clip(bump[idx_max] * 1.15 * float(cuff_smooth), 4.0, 26.0))
                    _add_warp(cx_b, cy_b, inward_sign * shift, 0.0, max(48.0, ref_span * 0.24), pressure=68.0)

    p_strokes = out_d / "liquify_strokes.json"
    p_strokes.write_text(json.dumps({"strokes": strokes}, indent=2, ensure_ascii=False), encoding="utf-8")

    p_pcraft = out_d / "body_liquify_smart.pcraft"
    p_out = out_d / "body_liquified.png"
    new_doc = {"width": w, "height": h, "mode": "rgb", "depth": 8, "background": "transparent", "name": "body_liquify"}
    cmds = [
        "--cmd", "file.placeEmbedded",
        "--params", _J({"path": str(in_p), "fit": False, "smartObject": True, "center": [w / 2, h / 2]}),
        "--cmd", "layer.setProps", "--params", _J({"name": "Subject_Body_Liquified"}),
        "--cmd", "layer.select", "--params", _J({"layer": "Layer 1"}),
        "--cmd", "layer.delete", "--params", "{}",
        "--cmd", "layer.select", "--params", _J({"layer": "Subject_Body_Liquified"}),
    ]
    if strokes:
        cmds += ["--cmd", "filter.liquify", "--params", _J({"strokes": strokes, "meshSize": 4 if w * h > 4_000_000 else 2})]

    r1 = subprocess.run([pc_exe, "run", "--new", _J(new_doc)] + cmds + ["--out", str(p_pcraft)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r1.returncode != 0:
        raise RuntimeError(f"PhotoCraft filter.liquify build failed (rc={r1.returncode}): {r1.stderr[-800:] or r1.stdout[-800:]}")
    r2 = subprocess.run([pc_exe, "run", str(p_pcraft), "--out", str(p_out)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r2.returncode != 0:
        raise RuntimeError(f"PhotoCraft liquify export failed (rc={r2.returncode}): {r2.stderr[-600:] or r2.stdout[-600:]}")

    # Export Original | Liquified | Horizontal Flip mirror inspection sheet
    orig_im = Image.open(in_p).convert("RGB")
    liq_im = Image.open(p_out).convert("RGB")
    flip_im = liq_im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    th = 1100
    tw = int(round(w * th / h))
    sheet = Image.new("RGB", (tw * 3 + 60, th + 90), (22, 24, 30))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", size=22)
    except Exception:
        font = ImageFont.load_default()
    for i, (im_s, lbl) in enumerate(((orig_im, "1. BEFORE"), (liq_im, "2. BODY LIQUIFY"), (flip_im, "3. MIRROR CHECK"))):
        x0 = 15 + i * (tw + 15)
        draw.text((x0, 14), lbl, font=font, fill=(232, 236, 242))
        sheet.paste(im_s.resize((tw, th), Image.Resampling.LANCZOS), (x0, 52))
    p_mirror = out_d / "check_liquify_mirror.png"
    sheet.save(p_mirror, "PNG")

    return {
        "ok": True,
        "input": str(in_p),
        "pose_detected": pose is not None,
        "cuff_bumps_detected": len(cuff_anchors),
        "stroke_count": len(strokes),
        "outputs": {
            "body_liquified": str(p_out),
            "body_liquify_smart_pcraft": str(p_pcraft),
            "check_liquify_mirror": str(p_mirror),
            "liquify_strokes": str(p_strokes),
        },
    }
