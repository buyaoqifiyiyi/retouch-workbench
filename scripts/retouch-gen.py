#!/usr/bin/env python3
"""
retouch-gen.py — Isolated CUDA generative relighting (IC-Light) and painterly style
transfer (Counterfeit-V3.0 + ControlNet-v1-1 Tile) for 6 GB VRAM (RTX 3060 Laptop).
Intended to run inside %LOCALAPPDATA%\\RetouchTools\\venv-gen\\Scripts\\python.exe.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

LOCAL_APP_DATA = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
TOOLS_ROOT = Path(os.environ.get("RETOUCH_TOOLS_ROOT", LOCAL_APP_DATA / "RetouchTools"))
MODELS_ROOT = Path(os.environ.get("RETOUCH_MODELS_DIR", TOOLS_ROOT / "models"))

RV51_DIR = MODELS_ROOT / "sd15-rv51"
ICLIGHT_DIR = MODELS_ROOT / "iclight"
CF3_DIR = MODELS_ROOT / "sd15-anime" / "counterfeit-v30-fp16"
CN_TILE_DIR = MODELS_ROOT / "controlnet-tile"


def _clear_vram() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _align8(x: int) -> int:
    return max(64, int(round(x / 8.0)) * 8)


def _read_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def _read_mask_gray(path: Path, size_wh: tuple[int, int]) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    arr = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise ValueError(f"Failed to read mask: {path}")
    w, h = size_wh
    if arr.shape != (h, w):
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
    return arr.astype(np.float32) / 255.0


def _make_light_gradient(width: int, height: int, direction: str) -> Image.Image:
    """Create official IC-Light initial directional lighting gradient (uint8 RGB)."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    d = direction.lower().strip()
    if d == "left":
        grad = np.linspace(255, 0, width, dtype=np.float32)[None, :, None]
        arr[:] = np.clip(grad, 0, 255).astype(np.uint8)
    elif d == "right":
        grad = np.linspace(0, 255, width, dtype=np.float32)[None, :, None]
        arr[:] = np.clip(grad, 0, 255).astype(np.uint8)
    elif d == "top":
        grad = np.linspace(255, 0, height, dtype=np.float32)[:, None, None]
        arr[:] = np.clip(grad, 0, 255).astype(np.uint8)
    elif d == "bottom":
        grad = np.linspace(0, 255, height, dtype=np.float32)[:, None, None]
        arr[:] = np.clip(grad, 0, 255).astype(np.uint8)
    else:
        # soft centered ambient vignette
        yy, xx = np.mgrid[0:height, 0:width]
        cx, cy = width * 0.5, height * 0.42
        dist = np.sqrt(((xx - cx) / (width * 0.6)) ** 2 + ((yy - cy) / (height * 0.6)) ** 2)
        val = np.clip(225.0 - dist * 135.0, 48, 245)[:, :, None]
        arr[:] = val.astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _encode_concat_conds(vae: Any, images_np: list[np.ndarray], device: str, dtype: Any) -> Any:
    """Official IC-Light numpy2pytorch + vae.encode * scaling_factor."""
    import torch

    tensors = []
    for im in images_np:
        arr = im.astype(np.float32) / 127.5 - 1.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        tensors.append(t)
    batch = torch.stack(tensors, dim=0).to(device=device, dtype=dtype)
    vae = vae.to(device=device, dtype=dtype)
    with torch.no_grad():
        latents = vae.encode(batch).latent_dist.mode() * vae.config.scaling_factor
    latents = latents.reshape((1, len(images_np) * 4, latents.shape[2], latents.shape[3])).detach().cpu()
    vae.to("cpu")
    _clear_vram()
    return latents


def cmd_doctor(_args: argparse.Namespace) -> int:
    import torch
    import diffusers
    import transformers
    import accelerate
    import safetensors

    cuda_ok = torch.cuda.is_available()
    gpu_info: dict[str, Any] = {"cuda_available": cuda_ok}
    if cuda_ok:
        free_b, total_b = torch.cuda.mem_get_info(0)
        gpu_info.update(
            {
                "device_name": torch.cuda.get_device_name(0),
                "vram_total_mb": round(total_b / (1024 * 1024)),
                "vram_free_mb": round(free_b / (1024 * 1024)),
            }
        )

    models = {
        "sd15_rv51": {"path": str(RV51_DIR), "exists": (RV51_DIR / "model_index.json").exists()},
        "iclight_fc": {
            "path": str(ICLIGHT_DIR / "iclight_sd15_fc.safetensors"),
            "exists": (ICLIGHT_DIR / "iclight_sd15_fc.safetensors").exists(),
        },
        "iclight_fbc": {
            "path": str(ICLIGHT_DIR / "iclight_sd15_fbc.safetensors"),
            "exists": (ICLIGHT_DIR / "iclight_sd15_fbc.safetensors").exists(),
        },
        "counterfeit_v3": {
            "path": str(CF3_DIR),
            "exists": (CF3_DIR / "model_index.json").exists()
            and (CF3_DIR / "unet" / "diffusion_pytorch_model.bin").exists(),
        },
        "controlnet_tile": {
            "path": str(CN_TILE_DIR),
            "exists": (CN_TILE_DIR / "config.json").exists()
            and (
                (CN_TILE_DIR / "diffusion_pytorch_model.bin").exists()
                or (CN_TILE_DIR / "diffusion_pytorch_model.safetensors").exists()
            ),
        },
    }
    payload = {
        "ok": bool(cuda_ok),
        "packages": {
            "torch": torch.__version__,
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
            "safetensors": safetensors.__version__,
        },
        "gpu": gpu_info,
        "models": models,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["ok"] else 1


def cmd_relight(args: argparse.Namespace) -> int:
    import torch
    from diffusers import (
        AutoencoderKL,
        DPMSolverMultistepScheduler,
        StableDiffusionImg2ImgPipeline,
        UNet2DConditionModel,
    )
    from diffusers.models.attention_processor import AttnProcessor2_0
    from safetensors.torch import load_file
    from transformers import CLIPTextModel, CLIPTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available in venv-gen")

    mode = args.mode.lower().strip()
    if mode not in ("fc", "fbc"):
        raise ValueError(f"Unsupported IC-Light mode: {mode}")

    ckpt_path = ICLIGHT_DIR / f"iclight_sd15_{mode}.safetensors"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing IC-Light checkpoint: {ckpt_path}")
    if not (RV51_DIR / "model_index.json").exists():
        raise FileNotFoundError(f"Missing SD1.5 RV5.1 diffusers dir: {RV51_DIR}")

    in_path = Path(args.input).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists (use --overwrite): {out_path}")

    orig_rgb = _read_rgb(in_path)
    orig_w, orig_h = orig_rgb.size
    mask_full = _read_mask_gray(Path(args.mask).resolve(), (orig_w, orig_h))[:, :, None]

    # Composite subject on neutral gray 127 (official IC-Light preprocessing)
    orig_np = np.array(orig_rgb, dtype=np.float32)
    fg_np_full = np.clip(orig_np * mask_full + 127.0 * (1.0 - mask_full), 0, 255).astype(np.uint8)
    fg_pil_full = Image.fromarray(fg_np_full, mode="RGB")

    w1, h1 = _align8(args.width), _align8(args.height)
    w2, h2 = _align8(args.hi_width), _align8(args.hi_height)

    dtype = torch.float16
    tokenizer = CLIPTokenizer.from_pretrained(str(RV51_DIR), subfolder="tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(
        str(RV51_DIR), subfolder="text_encoder", torch_dtype=dtype, local_files_only=True
    )
    vae = AutoencoderKL.from_pretrained(
        str(RV51_DIR), subfolder="vae", torch_dtype=dtype, local_files_only=True
    )
    unet = UNet2DConditionModel.from_pretrained(
        str(RV51_DIR), subfolder="unet", torch_dtype=dtype, local_files_only=True
    )
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        str(RV51_DIR),
        subfolder="scheduler",
        local_files_only=True,
        algorithm_type="sde-dpmsolver++",
        use_karras_sigmas=True,
    )

    # Patch UNet conv_in for 8 (fc) or 12 (fbc) channels and add IC-Light offsets
    in_ch = 8 if mode == "fc" else 12
    with torch.no_grad():
        old_conv = unet.conv_in
        new_conv = torch.nn.Conv2d(
            in_ch,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dtype=old_conv.weight.dtype,
        )
        new_conv.weight.zero_()
        new_conv.weight[:, :4, :, :].copy_(old_conv.weight)
        new_conv.bias = old_conv.bias
        unet.conv_in = new_conv
    unet.config["in_channels"] = in_ch

    sd_offset = load_file(str(ckpt_path))
    sd_origin = unet.state_dict()
    for k, delta in sd_offset.items():
        if k in sd_origin:
            sd_origin[k] = sd_origin[k] + delta.to(dtype=sd_origin[k].dtype)
    unet.load_state_dict(sd_origin, strict=True)
    del sd_offset, sd_origin

    unet.set_attn_processor(AttnProcessor2_0())
    vae.set_attn_processor(AttnProcessor2_0())
    vae.enable_slicing()
    vae.enable_tiling()

    unet_orig_forward = unet.forward

    def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
        kw = dict(kwargs)
        c_arg = dict(kw.get("cross_attention_kwargs") or {})
        c_concat = c_arg.pop("concat_conds", None)
        if c_concat is not None:
            c_concat = c_concat.to(device=sample.device, dtype=sample.dtype)
            if c_concat.shape[0] != sample.shape[0]:
                c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
            sample = torch.cat([sample, c_concat], dim=1)
        kw["cross_attention_kwargs"] = c_arg
        return unet_orig_forward(sample, timestep, encoder_hidden_states, **kw)

    unet.forward = hooked_unet_forward

    # Pre-encode BOTH Pass 1 and Pass 2 concat_conds BEFORE enable_model_cpu_offload()
    # so Accelerate device hooks on VAE are never disturbed.
    fg1_np = np.array(fg_pil_full.resize((w1, h1), Image.Resampling.LANCZOS), dtype=np.uint8)
    conds_list1 = [fg1_np]
    if mode == "fbc":
        if not args.bg_image:
            raise ValueError("--bg-image is required when --mode fbc")
        bg_pil = _read_rgb(Path(args.bg_image).resolve()).resize((w1, h1), Image.Resampling.LANCZOS)
        conds_list1.append(np.array(bg_pil, dtype=np.uint8))
    concat_conds1 = _encode_concat_conds(vae, conds_list1, "cuda", dtype)

    concat_conds2 = None
    if w2 > w1 or h2 > h1:
        fg2_np = np.array(fg_pil_full.resize((w2, h2), Image.Resampling.LANCZOS), dtype=np.uint8)
        conds_list2 = [fg2_np]
        if mode == "fbc" and args.bg_image:
            bg2_pil = _read_rgb(Path(args.bg_image).resolve()).resize((w2, h2), Image.Resampling.LANCZOS)
            conds_list2.append(np.array(bg2_pil, dtype=np.uint8))
        concat_conds2 = _encode_concat_conds(vae, conds_list2, "cuda", dtype)

    pipe = StableDiffusionImg2ImgPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipe.enable_model_cpu_offload()

    init_bg1 = (
        _read_rgb(Path(args.bg_image).resolve()).resize((w1, h1), Image.Resampling.LANCZOS)
        if (mode == "fbc" and args.bg_image)
        else _make_light_gradient(w1, h1, args.direction)
    )

    gen = torch.Generator(device="cuda").manual_seed(int(args.seed))
    res1 = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=init_bg1,
        strength=0.95,
        num_inference_steps=int(args.steps),
        guidance_scale=float(args.cfg),
        generator=gen,
        cross_attention_kwargs={"concat_conds": concat_conds1},
    ).images[0]

    del concat_conds1
    _clear_vram()

    # Pass 2 highres refine (if requested > w1/h1)
    if concat_conds2 is not None:
        res1_up = res1.resize((w2, h2), Image.Resampling.LANCZOS)
        gen2 = torch.Generator(device="cuda").manual_seed(int(args.seed))
        final_img = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=res1_up,
            strength=float(args.highres_denoise),
            num_inference_steps=int(args.steps),
            guidance_scale=float(args.cfg),
            generator=gen2,
            cross_attention_kwargs={"concat_conds": concat_conds2},
        ).images[0]
        del concat_conds2
    else:
        final_img = res1

    # Resize back cleanly to original canvas size if requested or save native resolution
    if args.match_input_size and final_img.size != (orig_w, orig_h):
        final_img = final_img.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

    final_img.save(out_path)
    del pipe, unet, vae, text_encoder
    _clear_vram()

    print(
        json.dumps(
            {
                "ok": True,
                "mode": mode,
                "direction": args.direction,
                "pass1_size": [w1, h1],
                "pass2_size": [w2, h2],
                "out_size": list(final_img.size),
                "seed": int(args.seed),
                "cfg": float(args.cfg),
                "out": str(out_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def cmd_painterly(args: argparse.Namespace) -> int:
    import torch
    from diffusers import (
        ControlNetModel,
        DPMSolverMultistepScheduler,
        StableDiffusionControlNetImg2ImgPipeline,
        StableDiffusionPipeline,
    )
    from diffusers.models.attention_processor import AttnProcessor2_0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available in venv-gen")
    if not (CF3_DIR / "model_index.json").exists():
        raise FileNotFoundError(f"Missing Counterfeit-V3.0 diffusers dir: {CF3_DIR}")
    if not (CN_TILE_DIR / "config.json").exists():
        raise FileNotFoundError(f"Missing ControlNet Tile dir: {CN_TILE_DIR}")

    in_path = Path(args.input).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists (use --overwrite): {out_path}")

    orig_rgb = _read_rgb(in_path)
    orig_w, orig_h = orig_rgb.size
    tw, th = _align8(args.width), _align8(args.height)
    init_pil = orig_rgb.resize((tw, th), Image.Resampling.LANCZOS)

    dtype = torch.float16
    controlnet = ControlNetModel.from_pretrained(
        str(CN_TILE_DIR), torch_dtype=dtype, local_files_only=True
    )
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        str(CF3_DIR),
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
    )

    # Pre-encode prompt_embeds with clip_skip BEFORE enable_model_cpu_offload()
    # to handle transformers 5.x flattened CLIPTextModel (no .text_model wrapper).
    clip_skip = max(1, int(args.clip_skip))
    tok_pos = pipe.tokenizer(
        args.prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    tok_neg = pipe.tokenizer(
        args.negative_prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    te = pipe.text_encoder.to("cuda", dtype=dtype)
    final_ln = getattr(te, "final_layer_norm", None) or te.text_model.final_layer_norm
    with torch.no_grad():
        out_pos = te(tok_pos.input_ids.to("cuda"), output_hidden_states=True)
        out_neg = te(tok_neg.input_ids.to("cuda"), output_hidden_states=True)
        pos_emb = final_ln(out_pos.hidden_states[-clip_skip]).to(dtype=dtype).cpu()
        neg_emb = final_ln(out_neg.hidden_states[-clip_skip]).to(dtype=dtype).cpu()
    te.to("cpu")
    _clear_vram()

    pipe.unet.set_attn_processor(AttnProcessor2_0())
    pipe.controlnet.set_attn_processor(AttnProcessor2_0())
    pipe.vae.set_attn_processor(AttnProcessor2_0())
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload()

    gen = torch.Generator(device="cuda").manual_seed(int(args.seed))
    out_img = pipe(
        prompt_embeds=pos_emb.to("cuda"),
        negative_prompt_embeds=neg_emb.to("cuda"),
        image=init_pil,
        control_image=init_pil,
        strength=float(args.strength),
        controlnet_conditioning_scale=float(args.control_scale),
        guidance_scale=float(args.guidance),
        num_inference_steps=int(args.steps),
        generator=gen,
    ).images[0]

    if args.match_input_size and out_img.size != (orig_w, orig_h):
        out_img = out_img.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
    out_img.save(out_path)

    del pipe, controlnet
    _clear_vram()

    print(
        json.dumps(
            {
                "ok": True,
                "gen_size": [tw, th],
                "out_size": list(out_img.size),
                "strength": float(args.strength),
                "control_scale": float(args.control_scale),
                "guidance": float(args.guidance),
                "steps": int(args.steps),
                "seed": int(args.seed),
                "out": str(out_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def cmd_tile_refine(args: argparse.Namespace) -> int:
    """Overlapping tiled ControlNet-Tile photographic detail refinement (6 GB VRAM safe)."""
    import torch
    from diffusers import (
        ControlNetModel,
        DPMSolverMultistepScheduler,
        StableDiffusionControlNetImg2ImgPipeline,
    )
    from diffusers.models.attention_processor import AttnProcessor2_0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available in venv-gen")
    if not (RV51_DIR / "model_index.json").exists():
        raise FileNotFoundError(f"Missing SD1.5 RV5.1 diffusers dir: {RV51_DIR}")
    if not (CN_TILE_DIR / "config.json").exists():
        raise FileNotFoundError(f"Missing ControlNet Tile dir: {CN_TILE_DIR}")

    in_path = Path(args.input).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists (use --overwrite): {out_path}")

    orig_rgb = _read_rgb(in_path)
    orig_w, orig_h = orig_rgb.size
    orig_np = np.array(orig_rgb, dtype=np.float32)

    dtype = torch.float16
    controlnet = ControlNetModel.from_pretrained(str(CN_TILE_DIR), torch_dtype=dtype, local_files_only=True)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        str(RV51_DIR),
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
    )
    pipe.unet.set_attn_processor(AttnProcessor2_0())
    pipe.controlnet.set_attn_processor(AttnProcessor2_0())
    pipe.vae.set_attn_processor(AttnProcessor2_0())
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload()

    tile_sz = _align8(args.tile_size)
    ov = int(args.overlap)
    step = max(64, tile_sz - ov)
    xs = [0] if orig_w <= tile_sz else sorted(set(list(range(0, max(1, orig_w - tile_sz + 1), step)) + [orig_w - tile_sz]))
    ys = [0] if orig_h <= tile_sz else sorted(set(list(range(0, max(1, orig_h - tile_sz + 1), step)) + [orig_h - tile_sz]))

    accum = np.zeros_like(orig_np)
    w_sum = np.zeros((orig_h, orig_w), dtype=np.float32)
    gen = torch.Generator(device="cuda").manual_seed(int(args.seed))

    for idx_y, y0 in enumerate(ys):
        for idx_x, x0 in enumerate(xs):
            x1 = min(orig_w, x0 + tile_sz)
            y1 = min(orig_h, y0 + tile_sz)
            crop = orig_rgb.crop((x0, y0, x1, y1))
            cw, ch = crop.size
            rw, rh = _align8(cw), _align8(ch)
            crop_fit = crop.resize((rw, rh), Image.Resampling.LANCZOS) if (rw, rh) != (cw, ch) else crop
            res = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                image=crop_fit,
                control_image=crop_fit,
                strength=float(args.denoise),
                controlnet_conditioning_scale=float(args.control_scale),
                guidance_scale=float(args.guidance),
                num_inference_steps=int(args.steps),
                generator=gen,
            ).images[0]
            if res.size != (cw, ch):
                res = res.resize((cw, ch), Image.Resampling.LANCZOS)
            wx = np.ones(cw, dtype=np.float32)
            wy = np.ones(ch, dtype=np.float32)
            if idx_x > 0 and ov > 2:
                n = min(cw // 2, ov)
                wx[:n] = np.linspace(0.0, 1.0, n, dtype=np.float32)
            if idx_x < len(xs) - 1 and ov > 2:
                n = min(cw // 2, ov)
                wx[-n:] = np.linspace(1.0, 0.0, n, dtype=np.float32)
            if idx_y > 0 and ov > 2:
                n = min(ch // 2, ov)
                wy[:n] = np.linspace(0.0, 1.0, n, dtype=np.float32)
            if idx_y < len(ys) - 1 and ov > 2:
                n = min(ch // 2, ov)
                wy[-n:] = np.linspace(1.0, 0.0, n, dtype=np.float32)
            wt = np.outer(wy, wx)
            accum[y0:y1, x0:x1] += np.array(res, dtype=np.float32) * wt[:, :, None]
            w_sum[y0:y1, x0:x1] += wt

    out_np = np.clip(accum / np.maximum(w_sum[:, :, None], 1e-5), 0.0, 255.0).astype(np.uint8)
    Image.fromarray(out_np, mode="RGB").save(out_path)
    del pipe, controlnet
    _clear_vram()

    print(
        json.dumps(
            {
                "ok": True,
                "size": [orig_w, orig_h],
                "tiles": len(xs) * len(ys),
                "tile_size": tile_sz,
                "denoise": float(args.denoise),
                "out": str(out_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="RetouchTools CUDA Generative Helper")
    sub = ap.add_subparsers(dest="command", required=True)

    p_doc = sub.add_parser("doctor")
    p_doc.set_defaults(func=cmd_doctor)

    p_rl = sub.add_parser("relight")
    p_rl.add_argument("input")
    p_rl.add_argument("--mask", required=True)
    p_rl.add_argument("--out", required=True)
    p_rl.add_argument("--mode", default="fc", choices=["fc", "fbc"])
    p_rl.add_argument("--bg-image", help="Background image when --mode fbc")
    p_rl.add_argument("--direction", default="left", choices=["left", "right", "top", "bottom", "ambient"])
    p_rl.add_argument(
        "--prompt",
        default="1girl, soft eastern dragon palace misty celadon lighting, warm porcelain skin, serene atmospheric illumination, detailed silk garments, masterpiece",
    )
    p_rl.add_argument(
        "--negative-prompt",
        default="bad quality, lowres, harsh shadows, neon oversaturation, deformed face, muddy colors",
    )
    p_rl.add_argument("--width", type=int, default=576)
    p_rl.add_argument("--height", type=int, default=1024)
    p_rl.add_argument("--hi-width", type=int, default=864)
    p_rl.add_argument("--hi-height", type=int, default=1536)
    p_rl.add_argument("--highres-denoise", type=float, default=0.42)
    p_rl.add_argument("--steps", type=int, default=25)
    p_rl.add_argument("--cfg", type=float, default=2.0)
    p_rl.add_argument("--seed", type=int, default=42)
    p_rl.add_argument("--match-input-size", action="store_true")
    p_rl.add_argument("--overwrite", action="store_true")
    p_rl.set_defaults(func=cmd_relight)

    p_pt = sub.add_parser("painterly")
    p_pt.add_argument("input")
    p_pt.add_argument("--out", required=True)
    p_pt.add_argument(
        "--prompt",
        default="masterpiece, best quality, exquisite semi-realistic anime painterly illustration, soft impasto brushstrokes, creamy porcelain skin, luminous expressive eyes, cohesive hair clumps with cool cyan rim light, soft celadon eastern fantasy atmosphere",
    )
    p_pt.add_argument(
        "--negative-prompt",
        default="lowres, bad anatomy, extra fingers, distorted face, harsh sharpen artifacts, sticker eyes, gray muddy skin, flat plastic noise",
    )
    p_pt.add_argument("--strength", type=float, default=0.35)
    p_pt.add_argument("--control-scale", type=float, default=0.85)
    p_pt.add_argument("--guidance", type=float, default=6.5)
    p_pt.add_argument("--steps", type=int, default=24)
    p_pt.add_argument("--clip-skip", type=int, default=2)
    p_pt.add_argument("--width", type=int, default=768)
    p_pt.add_argument("--height", type=int, default=1344)
    p_pt.add_argument("--seed", type=int, default=42)
    p_pt.add_argument("--match-input-size", action="store_true")
    p_pt.add_argument("--overwrite", action="store_true")
    p_pt.set_defaults(func=cmd_painterly)

    p_tr = sub.add_parser("tile-refine")
    p_tr.add_argument("input")
    p_tr.add_argument("--out", required=True)
    p_tr.add_argument(
        "--prompt",
        default="masterpiece architectural photograph, crisp weathered natural limestone pores, delicate natural leaves and rose petals, realistic subtle mortar joints, sharp 85mm full-frame lens optical detail",
    )
    p_tr.add_argument(
        "--negative-prompt",
        default="oil painting, anime, illustration, CGI grid, heavy black outline, blurry, plastic, wax, distorted lines",
    )
    p_tr.add_argument("--denoise", type=float, default=0.30)
    p_tr.add_argument("--control-scale", type=float, default=0.90)
    p_tr.add_argument("--guidance", type=float, default=5.5)
    p_tr.add_argument("--steps", type=int, default=20)
    p_tr.add_argument("--tile-size", type=int, default=768)
    p_tr.add_argument("--overlap", type=int, default=160)
    p_tr.add_argument("--seed", type=int, default=42)
    p_tr.add_argument("--overwrite", action="store_true")
    p_tr.set_defaults(func=cmd_tile_refine)

    return ap


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
