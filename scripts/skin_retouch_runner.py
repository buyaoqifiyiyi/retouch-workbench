"""Standalone runner for DAMO Apache-2.0 skin retouching (ABPN) and 106-landmark COS liquify."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _read_bgr_unicode(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"Empty or unreadable image file: {path}")
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image as BGR: {path}")
    return img


def _write_png_unicode(path: Path, arr: np.ndarray, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError(f"Failed to encode PNG: {path}")
    buf.tofile(str(path))


def _retouch_local_roi(image_t: Any, detection_net: Any, inpainting_net: Any, patch_size: int = 512) -> Any:
    import torch
    import torch.nn.functional as F
    from skin_retouching.utils import patch_aggregation_overlap, patch_partition_overlap

    sub_h, sub_w = image_t.shape[2:]
    sub_std = F.interpolate(image_t, size=(768, 768), mode="bilinear", align_corners=True)
    sub_mask = torch.sigmoid(detection_net(sub_std))
    sub_mask = F.interpolate(sub_mask, size=(sub_h, sub_w), mode="nearest")

    low_hard = (sub_mask >= 0.35).float()
    high_hard = (sub_mask >= 0.5).float()
    sub_mask = sub_mask * (1.0 - high_hard) + high_hard
    sub_mask = sub_mask * low_hard
    sub_mask = 1.0 - sub_mask

    pad_h = sub_h if sub_h % patch_size == 0 else (sub_h // patch_size + 1) * patch_size
    pad_w = sub_w if sub_w % patch_size == 0 else (sub_w // patch_size + 1) * patch_size

    img_pad = F.pad(image_t, pad=(0, pad_w - sub_w, 0, pad_h - sub_h, 0, 0), mode="constant", value=0)
    mask_pad = F.pad(sub_mask, pad=(0, pad_w - sub_w, 0, pad_h - sub_h, 0, 0), mode="constant", value=0)

    img_patches = patch_partition_overlap(img_pad, p1=patch_size, p2=patch_size)
    mask_patches = patch_partition_overlap(mask_pad, p1=patch_size, p2=patch_size)

    out_patches = []
    for i in range(img_patches.size(0)):
        w_img = img_patches[i : i + 1]
        w_msk = mask_patches[i : i + 1]
        masked_in = w_img * w_msk
        pred = inpainting_net(masked_in, w_msk)
        out_patches.append(masked_in + (1.0 - w_msk) * pred)

    comp = patch_aggregation_overlap(
        torch.cat(out_patches, dim=0),
        h=int(round(pad_h / patch_size)),
        w=int(round(pad_w / patch_size)),
    )
    return comp[:, :, :sub_h, :sub_w]


def _whiten_rgb_safe(
    rgb_u8: np.ndarray,
    skin_mask: np.ndarray,
    whitening_degree: float,
    flag_big_kernel: bool = False,
) -> np.ndarray:
    if skin_mask.ndim == 3:
        skin_mask = skin_mask[:, :, -1]
    ksize = 80 if flag_big_kernel else 30
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    m = cv2.dilate(skin_mask, k1, 1)
    m = cv2.erode(m, k2, 1)
    m = cv2.blur(m, (20, 20)).astype(np.float32) / 255.0
    m = cv2.resize(m, (rgb_u8.shape[1], rgb_u8.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Reduce green/blue lift ratio to 0.75 per official ms_wrapper/utils whitening curve
    mask3 = np.stack([m, m * 0.75, m * 0.75], axis=-1)
    whiten_mg = mask3 * 0.2 * float(whitening_degree) + 0.5
    out_f = rgb_u8.astype(np.float32) / 255.0
    out_f = (-2.0 * whiten_mg + 1.0) * out_f * out_f + 2.0 * whiten_mg * out_f
    return np.clip(out_f * 255.0, 0, 255).astype(np.uint8)


def run_skin_retouch(
    input_path: Path,
    out_path: Path,
    model_dir: Path,
    uniface_cache: Path,
    degree: float = 0.7,
    whitening_degree: float = 0.75,
    retouch_local: bool = True,
    overwrite: bool = False,
    out_skin_mask: Path | None = None,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from cuda_env import create_onnx_session
    from uniface import RetinaFace, set_cache_dir  # type: ignore

    from skin_retouching.detection_model.detection_unet_in import DetectionUNet
    from skin_retouching.inpainting_model.inpainting_unet import RetouchingNet
    from skin_retouching.unet_deploy import UNet
    from skin_retouching.utils import (
        gen_diffuse_mask,
        get_crop_bbox,
        get_roi_without_padding,
        preprocess_roi,
        resize_on_long_side,
        roi_to_tensor,
        smooth_border_mg,
    )

    in_path = input_path.resolve()
    out_resolved = out_path.resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input image not found: {in_path}")

    gen_ckpt_path = model_dir / "pytorch_model.pt"
    joint_ckpt_path = model_dir / "joint_20210926.pth"
    skin_onnx_path = model_dir / "model.onnx"
    for req in (gen_ckpt_path, skin_onnx_path):
        if not req.exists():
            raise FileNotFoundError(f"Missing skin-retouching model file: {req}")

    bgr = _read_bgr_unicode(in_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = UNet(3, 3).to(device)
    gen_ckpt = torch.load(str(gen_ckpt_path), map_location="cpu", weights_only=True)
    generator.load_state_dict(gen_ckpt["generator"])
    generator.eval()

    detection_net = None
    inpainting_net = None
    if retouch_local and joint_ckpt_path.exists():
        joint_ckpt = torch.load(str(joint_ckpt_path), map_location="cpu", weights_only=True)
        inpainting_net = RetouchingNet(in_channels=4, out_channels=3).to(device)
        detection_net = DetectionUNet(n_channels=3, n_classes=1).to(device)
        inpainting_net.load_state_dict(joint_ckpt["inpainting_net"])
        detection_net.load_state_dict(joint_ckpt["detection_net"])
        inpainting_net.eval()
        detection_net.eval()

    set_cache_dir(str(uniface_cache))
    detector = RetinaFace()
    faces = detector.detect(bgr)
    faces = sorted(faces, key=lambda f: float(f.confidence), reverse=True)

    det_results: list[dict[str, Any]] = []
    for f in faces:
        det_results.append(
            {
                "bbox": [int(round(v)) for v in f.bbox[:4]],
                "score": float(f.confidence),
                "landmarks": [[int(round(px)), int(round(py))] for px, py in f.landmarks],
            }
        )
    crop_bboxes = get_crop_bbox(det_results)

    skin_mask_arr = None
    if whitening_degree > 0:
        sess = create_onnx_session(skin_onnx_path)
        rgb_small, _ = resize_on_long_side(rgb, 800)
        in_name = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name
        skin_mask_arr = sess.run([out_name], {in_name: rgb_small.astype(np.float32)})[0]
        if out_skin_mask is not None:
            sm = skin_mask_arr[:, :, -1] if skin_mask_arr.ndim == 3 else skin_mask_arr
            sm_full = cv2.resize(sm.astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
            _write_png_unicode(out_skin_mask.resolve(), sm_full, overwrite=overwrite)

    diffuse_mask = torch.from_numpy(gen_diffuse_mask()).to(device).float().permute(2, 0, 1)[None, ...]
    out_rgb = rgb.copy()
    flag_big_kernel = False

    with torch.no_grad():
        for bbox in crop_bboxes:
            roi, _, tblr = get_roi_without_padding(rgb, bbox)
            if roi.size == 0:
                continue
            roi_t = preprocess_roi(roi_to_tensor(roi).to(device))
            if roi_t.shape[2] > 0.4 * h:
                flag_big_kernel = True
            if detection_net is not None and inpainting_net is not None:
                roi_t = _retouch_local_roi(roi_t, detection_net, inpainting_net, patch_size=512)

            img_512 = F.interpolate(roi_t, (512, 512), mode="bilinear")
            pred_mg = generator(img_512)
            pred_mg = ((pred_mg - 0.5) * float(degree) + 0.5).clamp(0.0, 1.0)
            pred_mg = F.interpolate(pred_mg, roi_t.shape[2:], mode="bilinear")[0].permute(1, 2, 0)
            if pred_mg.ndim == 2:
                pred_mg = pred_mg[..., None]
            pred_mg = smooth_border_mg(diffuse_mask, pred_mg)

            img_unit = (roi_t[0].permute(1, 2, 0) + 1.0) / 2.0
            pred_unit = (1.0 - 2.0 * pred_mg) * img_unit * img_unit + 2.0 * pred_mg * img_unit
            roi_u8 = (pred_unit.clamp(0.0, 1.0) * 255.0).byte().cpu().numpy()
            out_rgb[tblr[0] : tblr[1], tblr[2] : tblr[3]] = roi_u8

    if whitening_degree > 0 and skin_mask_arr is not None:
        out_rgb = _whiten_rgb_safe(out_rgb, skin_mask_arr, whitening_degree, flag_big_kernel=flag_big_kernel)

    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    _write_png_unicode(out_resolved, out_bgr, overwrite=overwrite)
    return {
        "ok": True,
        "input": str(in_path),
        "out": str(out_resolved),
        "width": int(w),
        "height": int(h),
        "faces_detected": len(crop_bboxes),
        "retouch_local": bool(retouch_local and detection_net is not None),
        "degree": float(degree),
        "whitening_degree": float(whitening_degree),
        "device": str(device),
    }


def run_cos_liquify(
    input_path: Path,
    faces_json_path: Path,
    out_path: Path,
    eye_scale: float = 0.09,
    nose_narrow: float = 0.08,
    nose_tip_narrow: float = 0.05,
    mouth_narrow: float = 0.09,
    chin_lift_px: float = 8.0,
    jaw_taper: float = 0.045,
    wig_fluff_px: float = 7.0,
    overwrite: bool = False,
) -> dict[str, Any]:
    in_path = input_path.resolve()
    out_resolved = out_path.resolve()
    fdat = json.loads(faces_json_path.read_text(encoding="utf-8"))
    if not fdat.get("faces"):
        raise ValueError(f"No faces in {faces_json_path}")

    f0 = fdat["faces"][0]
    pts = np.array(f0["landmarks_106"], dtype=np.float32)
    lm5 = np.array(f0["landmarks_5"], dtype=np.float32)

    img = _read_bgr_unicode(in_path)
    h, w = img.shape[:2]

    # Meshgrid in float32 for smooth vector-field remap
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = xx.copy()
    map_y = yy.copy()

    # Anchor coordinates derived directly from detected face landmarks
    # Identify left eye / right eye / nose / mouth clusters robustly via landmarks_5 proximity
    l_eye_c = lm5[0]
    r_eye_c = lm5[1]
    nose_c = lm5[2]
    m_left = lm5[3]
    m_right = lm5[4]
    mouth_c = 0.5 * (m_left + m_right)
    eye_vec = r_eye_c - l_eye_c
    eye_dist = float(np.linalg.norm(eye_vec))
    ux, uy = eye_vec / max(eye_dist, 1e-5)  # face-local right unit vector
    vx, vy = -uy, ux                        # face-local down unit vector

    # 1. Enlarge both eyes gently (slightly stronger face-local vertical boost for round COS gaze)
    if eye_scale > 0:
        sigma_e = 0.37 * eye_dist
        for ec in (l_eye_c, r_eye_c):
            dx = xx - ec[0]
            dy = yy - ec[1]
            lx = dx * ux + dy * uy
            ly = dx * vx + dy * vy
            r2 = lx * lx + ly * ly
            w_e = np.exp(-0.5 * r2 / (sigma_e * sigma_e))
            sx = -lx * (eye_scale * 0.78) * w_e
            sy = -ly * (eye_scale * 1.18) * w_e
            map_x += sx * ux + sy * vx
            map_y += sx * uy + sy * vy

    # 2. Narrow nose alae & refine nose tip inward toward face-local nasal midline
    if nose_narrow > 0 or nose_tip_narrow > 0:
        ala_c = nose_c + np.array([vx, vy], dtype=np.float32) * (0.08 * eye_dist)
        ala_half = 0.21 * eye_dist
        sigma_n = 0.25 * eye_dist
        dx = xx - ala_c[0]
        dy = yy - ala_c[1]
        lx = dx * ux + dy * uy
        ly = dx * vx + dy * vy
        w_wing = np.exp(-0.5 * (lx * lx + ly * ly) / (sigma_n * sigma_n))
        sx = np.sign(lx) * np.minimum(np.abs(lx), ala_half * 1.4) * nose_narrow * w_wing
        map_x += sx * ux
        map_y += sx * uy
        if nose_tip_narrow > 0:
            tip_c = nose_c + np.array([vx, vy], dtype=np.float32) * (0.04 * eye_dist)
            dxt = xx - tip_c[0]
            dyt = yy - tip_c[1]
            w_t = np.exp(-0.5 * (dxt * dxt + dyt * dyt) / ((0.18 * eye_dist) ** 2))
            map_x += dxt * nose_tip_narrow * w_t
            map_y += dyt * (nose_tip_narrow * 0.6) * w_t

    # 3. Contract mouth width subtly along mouth axis (keeping lip height/curves clean)
    if mouth_narrow > 0:
        dx = xx - mouth_c[0]
        dy = yy - mouth_c[1]
        lx = dx * ux + dy * uy
        ly = dx * vx + dy * vy
        sigma_mx = 0.42 * eye_dist
        sigma_my = 0.22 * eye_dist
        w_m = np.exp(-0.5 * ((lx / sigma_mx) ** 2 + (ly / sigma_my) ** 2))
        sx = lx * mouth_narrow * w_m
        map_x += sx * ux
        map_y += sx * uy

    # 4. Chin lift along face-local axis & smooth lower-jaw oval V-taper (zero cheek hollowing)
    chin_y = float(pts[:, 1].max())
    chin_x = float(pts[int(np.argmax(pts[:, 1])), 0])
    if chin_lift_px != 0:
        dx = xx - chin_x
        dy = yy - chin_y
        lx = dx * ux + dy * uy
        ly = dx * vx + dy * vy
        sigma_cx = 0.62 * eye_dist
        sigma_cy = 0.34 * eye_dist
        w_c = np.exp(-0.5 * ((lx / sigma_cx) ** 2 + (ly / sigma_cy) ** 2))
        below_guard = np.clip(1.0 - np.maximum(0.0, ly - 10.0) / 48.0, 0.0, 1.0)
        sy = chin_lift_px * w_c * below_guard
        map_x += sy * vx
        map_y += sy * vy

    if jaw_taper > 0:
        jaw_mid = 0.5 * (mouth_c + np.array([chin_x, chin_y], dtype=np.float32)) - np.array([vx, vy], dtype=np.float32) * (0.05 * eye_dist)
        dx = xx - jaw_mid[0]
        dy = yy - jaw_mid[1]
        lx = dx * ux + dy * uy
        ly = dx * vx + dy * vy
        sigma_jx = 0.92 * eye_dist
        sigma_jy = 0.46 * eye_dist
        w_j = np.exp(-0.5 * ((lx / sigma_jx) ** 2 + (ly / sigma_jy) ** 2))
        # Local y relative to nose and chin
        n_ly = (nose_c[0] - jaw_mid[0]) * vx + (nose_c[1] - jaw_mid[1]) * vy
        c_ly = (chin_x - jaw_mid[0]) * vx + (chin_y - jaw_mid[1]) * vy
        cheek_guard = np.clip((ly - (n_ly + 0.15 * eye_dist)) / (0.35 * eye_dist), 0.0, 1.0)
        collar_guard = np.clip(1.0 - np.maximum(0.0, ly - (c_ly + 12.0)) / 50.0, 0.0, 1.0)
        edge_weight = np.clip(np.abs(lx) / (0.45 * eye_dist), 0.0, 1.2)
        sx = np.sign(lx) * (eye_dist * jaw_taper) * w_j * cheek_guard * collar_guard * edge_weight
        map_x += sx * ux
        map_y += sx * uy

    # 5. Subtle outer upper-wig fluffing for 头包脸 balance (well inside canvas, strictly above shoulders)
    if wig_fluff_px > 0:
        wig_y = 0.5 * (l_eye_c[1] + r_eye_c[1])
        sigma_wy = 1.05 * eye_dist
        sigma_wx = 0.45 * eye_dist
        for side_sign, edge_x in ((-1.0, l_eye_c[0] - 1.15 * eye_dist), (1.0, r_eye_c[0] + 1.15 * eye_dist)):
            dx = xx - edge_x
            dy = yy - wig_y
            w_wig = np.exp(-0.5 * ((dx / sigma_wx) ** 2 + (dy / sigma_wy) ** 2))
            # Outward fluff -> sample slightly inward to push hair outward
            map_x -= side_sign * wig_fluff_px * w_wig

    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    disp = np.sqrt((map_x - xx) ** 2 + (map_y - yy) ** 2)
    _write_png_unicode(out_resolved, warped, overwrite=overwrite)
    return {
        "ok": True,
        "input": str(in_path),
        "out": str(out_resolved),
        "max_displacement_px": round(float(disp.max()), 2),
        "eye_dist_px": round(eye_dist, 1),
        "chin_anchor": [round(chin_x, 1), round(chin_y, 1)],
    }
