#!/usr/bin/env python3
"""
download_models.py — Sync permissive open-source models, CUDA PyTorch wheels, and
Real-ESRGAN Vulkan binaries into %LOCALAPPDATA%\\RetouchTools\\ using fast
direct mirrors first (Aliyun / ModelScope / hf-mirror), with upstream fallbacks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

from fast_download import download_file

LOCAL_APP_DATA = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
ROOT = Path(os.environ.get("RETOUCH_TOOLS_ROOT") or (LOCAL_APP_DATA / "RetouchTools"))
MODELS = Path(os.environ.get("RETOUCH_MODELS_DIR") or (ROOT / "models"))
WHEELS = ROOT / "wheels"

RV51_FILES = [
    ("model_index.json", 100),
    ("scheduler/scheduler_config.json", 100),
    ("text_encoder/config.json", 100),
    ("text_encoder/model.safetensors", 50_000_000),
    ("tokenizer/merges.txt", 1000),
    ("tokenizer/special_tokens_map.json", 50),
    ("tokenizer/tokenizer_config.json", 50),
    ("tokenizer/vocab.json", 1000),
    ("unet/config.json", 100),
    ("unet/diffusion_pytorch_model.safetensors", 500_000_000),
    ("vae/config.json", 100),
    ("vae/diffusion_pytorch_model.safetensors", 50_000_000),
]

CF3_DIFFUSERS_FILES = [
    ("model_index.json", 100),
    ("scheduler/scheduler_config.json", 100),
    ("text_encoder/config.json", 100),
    ("text_encoder/pytorch_model.bin", 50_000_000),
    ("tokenizer/merges.txt", 1000),
    ("tokenizer/special_tokens_map.json", 50),
    ("tokenizer/tokenizer_config.json", 50),
    ("tokenizer/vocab.json", 1000),
    ("unet/config.json", 100),
    ("unet/diffusion_pytorch_model.bin", 500_000_000),
    ("vae/config.json", 100),
    ("vae/diffusion_pytorch_model.bin", 50_000_000),
]

SKIN_RETOUCH_FILES = [
    ("pytorch_model.pt", 50_000_000),
    ("joint_20210926.pth", 5_000_000),
    ("model.onnx", 1_000_000),
]

ASSETS: list[dict] = [
    # === Core phase: parsing, matting, inpainting, upscaling & skin retouch ===
    {
        "phase": "core",
        "name": "segformer/segformer_b2_clothes.onnx",
        "dest": MODELS / "segformer" / "segformer_b2_clothes.onnx",
        "urls": [
            "https://hf-mirror.com/mattmdjaga/segformer_b2_clothes/resolve/main/onnx/model.onnx",
            "https://hf-mirror.com/Xenova/segformer_b2_clothes/resolve/main/onnx/model.onnx",
            "https://huggingface.co/mattmdjaga/segformer_b2_clothes/resolve/main/onnx/model.onnx",
        ],
        "min_bytes": 90_000_000,
    },
    {
        "phase": "core",
        "name": "birefnet_lite.onnx",
        "dest": MODELS / "birefnet" / "birefnet_lite.onnx",
        "urls": [
            "https://modelscope.cn/models/onnx-community/BiRefNet_lite-ONNX/resolve/master/onnx/model.onnx",
            "https://hf-mirror.com/onnx-community/BiRefNet_lite-ONNX/resolve/main/onnx/model.onnx",
            "https://huggingface.co/onnx-community/BiRefNet_lite-ONNX/resolve/main/onnx/model.onnx",
        ],
        "min_bytes": 100_000_000,
    },
    {
        "phase": "core",
        "name": "selfie_multiclass_256x256.tflite",
        "dest": MODELS / "selfie_multiclass_256x256.tflite",
        "urls": [
            "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
        ],
        "min_bytes": 5_000_000,
    },
    {
        "phase": "core",
        "name": "mediapipe/face_landmarker.task",
        "dest": MODELS / "mediapipe" / "face_landmarker.task",
        "urls": [
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        ],
        "min_bytes": 2_000_000,
    },
    {
        "phase": "core",
        "name": "mediapipe/pose_landmarker_heavy.task",
        "dest": MODELS / "mediapipe" / "pose_landmarker_heavy.task",
        "urls": [
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
        ],
        "min_bytes": 20_000_000,
    },
    {
        "phase": "core",
        "name": "lama/big-lama.pt",
        "dest": MODELS / "lama" / "big-lama.pt",
        "urls": [
            "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
        ],
        "min_bytes": 150_000_000,
    },
    {
        "phase": "core",
        "name": "gfpgan/gfpgan_1.4.onnx",
        "dest": MODELS / "gfpgan" / "gfpgan_1.4.onnx",
        "urls": [
            "https://hf-mirror.com/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx",
            "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx",
        ],
        "min_bytes": 200_000_000,
    },
    {
        "phase": "core",
        "name": "realesrgan-ncnn-vulkan-windows.zip",
        "dest": ROOT / "realesrgan-ncnn-vulkan-20220424-windows.zip",
        "extract_dir": ROOT / "realesrgan-ncnn-vulkan",
        "urls": [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip",
        ],
        "min_bytes": 30_000_000,
    },
    {
        "phase": "core",
        "name": "pih/curves.onnx",
        "dest": MODELS / "pih" / "curves.onnx",
        "urls": [
            "https://hf-mirror.com/JeremySD99/PhotoBlaze-PIH/resolve/main/curves.onnx",
            "https://huggingface.co/JeremySD99/PhotoBlaze-PIH/resolve/main/curves.onnx",
        ],
        "min_bytes": 80_000_000,
    },
    {
        "phase": "core",
        "name": "pih/gain.onnx",
        "dest": MODELS / "pih" / "gain.onnx",
        "urls": [
            "https://hf-mirror.com/JeremySD99/PhotoBlaze-PIH/resolve/main/gain.onnx",
            "https://huggingface.co/JeremySD99/PhotoBlaze-PIH/resolve/main/gain.onnx",
        ],
        "min_bytes": 250_000_000,
    },
    {
        "phase": "core",
        "name": "depth/depth_anything_v2_small.onnx",
        "dest": MODELS / "depth" / "depth_anything_v2_small.onnx",
        "urls": [
            "https://hf-mirror.com/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx",
            "https://modelscope.cn/models/onnx-community/depth-anything-v2-small/resolve/master/onnx/model.onnx",
            "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx",
        ],
        "min_bytes": 90_000_000,
    },
    # === Wheels phase: PyTorch CUDA 12.4 for isolated venv-gen ===
    {
        "phase": "wheels",
        "name": "torch-2.6.0+cu124",
        "dest": WHEELS / "torch-2.6.0+cu124-cp312-cp312-win_amd64.whl",
        "urls": [
            "https://mirrors.aliyun.com/pytorch-wheels/cu124/torch-2.6.0%2Bcu124-cp312-cp312-win_amd64.whl",
            "https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp312-cp312-win_amd64.whl",
        ],
        "min_bytes": 1_500_000_000,
    },
    {
        "phase": "wheels",
        "name": "torchvision-0.21.0+cu124",
        "dest": WHEELS / "torchvision-0.21.0+cu124-cp312-cp312-win_amd64.whl",
        "urls": [
            "https://mirrors.aliyun.com/pytorch-wheels/cu124/torchvision-0.21.0%2Bcu124-cp312-cp312-win_amd64.whl",
            "https://download.pytorch.org/whl/cu124/torchvision-0.21.0%2Bcu124-cp312-cp312-win_amd64.whl",
        ],
        "min_bytes": 2_000_000,
    },
    # === Gen phase: IC-Light + ControlNet Tile + SD1.5 bases ===
    {
        "phase": "gen",
        "name": "iclight_sd15_fc.safetensors",
        "dest": MODELS / "iclight" / "iclight_sd15_fc.safetensors",
        "urls": [
            "https://modelscope.cn/models/lllyasviel/ic-light/resolve/master/iclight_sd15_fc.safetensors",
            "https://hf-mirror.com/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors",
            "https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors",
        ],
        "min_bytes": 1_000_000_000,
    },
    {
        "phase": "gen",
        "name": "iclight_sd15_fbc.safetensors",
        "dest": MODELS / "iclight" / "iclight_sd15_fbc.safetensors",
        "urls": [
            "https://modelscope.cn/models/lllyasviel/ic-light/resolve/master/iclight_sd15_fbc.safetensors",
            "https://hf-mirror.com/lllyasviel/ic-light/resolve/main/iclight_sd15_fbc.safetensors",
            "https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fbc.safetensors",
        ],
        "min_bytes": 1_000_000_000,
    },
    {
        "phase": "tile",
        "name": "controlnet-tile/config.json",
        "dest": MODELS / "controlnet-tile" / "config.json",
        "urls": [
            "https://modelscope.cn/models/lllyasviel/control_v11f1e_sd15_tile/resolve/master/config.json",
            "https://hf-mirror.com/lllyasviel/control_v11f1e_sd15_tile/resolve/main/config.json",
            "https://huggingface.co/lllyasviel/control_v11f1e_sd15_tile/resolve/main/config.json",
        ],
        "min_bytes": 100,
    },
    {
        "phase": "tile",
        "name": "controlnet-tile/diffusion_pytorch_model.bin",
        "dest": MODELS / "controlnet-tile" / "diffusion_pytorch_model.bin",
        "urls": [
            "https://modelscope.cn/models/lllyasviel/control_v11f1e_sd15_tile/resolve/master/diffusion_pytorch_model.bin",
            "https://hf-mirror.com/lllyasviel/control_v11f1e_sd15_tile/resolve/main/diffusion_pytorch_model.bin",
            "https://huggingface.co/lllyasviel/control_v11f1e_sd15_tile/resolve/main/diffusion_pytorch_model.bin",
        ],
        "min_bytes": 1_000_000_000,
    },
]

for rel, min_b in SKIN_RETOUCH_FILES:
    ASSETS.append(
        {
            "phase": "core",
            "name": f"skin-retouching/{rel}",
            "dest": MODELS / "skin-retouching" / rel,
            "urls": [
                f"https://modelscope.cn/models/damo/cv_unet_skin_retouching_torch/resolve/master/{rel}",
            ],
            "min_bytes": min_b,
        }
    )

for rel, min_b in RV51_FILES:
    ASSETS.append(
        {
            "phase": "tile",
            "name": f"sd15-rv51/{rel}",
            "dest": MODELS / "sd15-rv51" / rel,
            "urls": [
                f"https://modelscope.cn/models/AI-ModelScope/realistic-vision-v51/resolve/master/{rel}",
                f"https://hf-mirror.com/stablediffusionapi/realistic-vision-v51/resolve/main/{rel}",
                f"https://huggingface.co/stablediffusionapi/realistic-vision-v51/resolve/main/{rel}",
            ],
            "min_bytes": min_b,
        }
    )

for rel, min_b in CF3_DIFFUSERS_FILES:
    ASSETS.append(
        {
            "phase": "gen",
            "name": f"counterfeit-v30-fp16/{rel}",
            "dest": MODELS / "sd15-anime" / "counterfeit-v30-fp16" / rel,
            "urls": [
                f"https://modelscope.cn/models/stablediffusionapi/counterfeit-v30-fp16/resolve/master/{rel}",
                f"https://hf-mirror.com/stablediffusionapi/counterfeit-v30-fp16/resolve/main/{rel}",
                f"https://huggingface.co/stablediffusionapi/counterfeit-v30-fp16/resolve/main/{rel}",
            ],
            "min_bytes": min_b,
        }
    )


def _select_assets(phase: str) -> list[dict]:
    if phase == "all":
        return ASSETS
    if phase == "models":
        return [a for a in ASSETS if a["phase"] in {"core", "tile", "gen"}]
    if phase == "gen":
        return [a for a in ASSETS if a["phase"] in {"tile", "gen"}]
    if phase in {"core", "wheels", "tile"}:
        return [a for a in ASSETS if a["phase"] == phase]
    raise ValueError(f"Unknown phase: {phase!r}. Expected all|core|wheels|tile|gen|models.")


def _extract_archive_if_needed(zip_path: Path, target_dir: Path) -> None:
    exe_path = target_dir / "realesrgan-ncnn-vulkan.exe"
    if exe_path.exists():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download RetouchTools models and CUDA wheels.")
    parser.add_argument(
        "phase",
        nargs="?",
        default="all",
        choices=["all", "core", "wheels", "tile", "gen", "models"],
        help="Asset group to sync (default: all)",
    )
    parser.add_argument("--dry-run", "--list", dest="dry_run", action="store_true", help="Print manifest and destinations without downloading")
    args = parser.parse_args()

    selected = _select_assets(args.phase)
    if args.dry_run:
        plan = [
            {
                "phase": item["phase"],
                "name": item["name"],
                "dest": str(item["dest"]),
                "exists": item["dest"].exists(),
                "urls": item["urls"],
            }
            for item in selected
        ]
        print(json.dumps({"root": str(ROOT), "phase": args.phase, "count": len(plan), "assets": plan}, ensure_ascii=False, indent=2))
        return 0

    results = []
    failed = []
    for item in selected:
        dest = item["dest"]
        ok = False
        last_err = None
        for url in item["urls"]:
            try:
                res = download_file(
                    url,
                    dest,
                    workers=6,
                    timeout=35,
                    retries=4,
                    min_bytes=item["min_bytes"],
                )
                if "extract_dir" in item:
                    _extract_archive_if_needed(dest, item["extract_dir"])
                mb = res["bytes"] / (1024 * 1024)
                tag = "cached" if res["cached"] else f"{res.get('speed_mb_s', '?')} MB/s"
                print(f"[OK] {item['name']} ({mb:.1f} MB, {res['elapsed_s']}s, {tag})", flush=True)
                results.append({"name": item["name"], **res})
                ok = True
                break
            except Exception as e:
                last_err = str(e)
                print(f"[WARN] {item['name']} failed on {url}: {e}", file=sys.stderr, flush=True)
        if not ok:
            failed.append({"name": item["name"], "error": last_err})

    print(json.dumps({"phase": args.phase, "ok_count": len(results), "failed": failed}, ensure_ascii=False), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
