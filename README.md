# Retouch Workbench · 纯开源 Agent 人像与 COS 后期修图工作台

**Retouch Workbench** 是一个面向 AI Agent（Codex、Antigravity、Claude Code、OpenCode 等）与命令行的**纯开源人像及 COS 后期修图工作台**，以 **ArtCraft PhotoCraft**（分层精修、蒙版、骨相液化、混合画笔、PSD/PCRAFT 无损工程）与 **ArtCraft LightCraft**（批量 RAW 暗房、选片、32-bit 浮点色彩分级）为核心双引擎，配合本机可插拔开源 AI 侧车（`scripts/retouch-ai.py`）与实证修图方法库（`references/`）协同完成非破坏后期。

> **开源与合规边界**：本项目源码、外部核心引擎与默认预置模型均采用宽松开源或开放权重许可（**MIT / Apache-2.0 / BSD-3-Clause / CreativeML OpenRAIL-M**），不含任何商业闭源修图软件依赖、个人隐私原图或闭源/非商用限制抠图模型，亦不依赖重型本地大模型流水线。

---

## 核心架构与职责分工

| 模块 / 组件 | 开源许可 | 在工作台中的职责 | 关键产物 |
| :--- | :--- | :--- | :--- |
| **ArtCraft PhotoCraft** (`photocraft-cli`) | MIT OR Apache-2.0 | 单张分层精修、皮肤高低频、冻区骨相液化、图层/矢量蒙版、混合画笔、压感路径勾丝、`.pcraft`/`.psd` 读写 | 无损 `.pcraft`、兼容 `.psd`、4K 发布成片 |
| **ArtCraft LightCraft** (`lightcraft-cli`) | MIT OR Apache-2.0 | 批量导入、选片、白平衡/曲线/HSL/LUT 同组暗房同步、16-bit/32-bit 显影导出 | 无损调整日志、标准 16-bit PNG/TIFF 底图 |
| **核心 AI 侧车** (`scripts/retouch-ai.py` + `workbench_ops.py` + `cutout_refine_ops.py` + `pure_pc_retouch_ops.py` + `makeup_retouch_ops.py` + `hair_sculpt_ops.py` + `hair_anime_engine.py` + `handoff_ops.py` + `output_sharpen_ops.py` + `vfx_overlay_ops.py` + `relight_cinema_ops.py` + `cuda_env.py`) | MIT / Apache-2.0 / BSD | 106 关键点冻区与人脸 19 类语义（`face-parse`）、发丝精抠与多层光学去溢色（`matte` / `cutout-refine`）、全身 18 类分割（`body-parse`）、478 点纯 PC 三频奶油肌与光学多混合模式 `Blend-If` 补妆 v2/骨相液化（`skin-fs` / `makeup-paint` / `body-liquify`）、服饰分材质三频去雾（`garment-retouch`）与下肢统一（`skin-unify`）、实拍理发与 v11 逐根解析微簇发丝重塑（`hair-comp` / `hair-paint` / `hair-sculpt`）、人机协同白模硬暂停交接（`handoff-pack`）、Depth-Anything-V2 去场馆平光与多光源 2.5D 重打光（`relight-sculpt`，**零人物重绘**）、Blender/AgX `.cube` 3D LUT 电影分级（`cinema-grade`）、2.5D 因果特效溢光（`vfx-overlay`）、PIH 溶图（`harmonize`）、无白边焦点输出锐化（`output-sharpen`）、分块拼合、ROI 消除、Vulkan 超分与自动掩码/影调体检（`mask-audit` / `qa` / `compare`） | 严格互斥 PNG 蒙版、纯净发丝 Alpha、可编辑 PhotoCraft D&B/光学补妆/逐根发丝/重打光层、人机协同交接包、`.cube` 电影 LUT 分级与 1:1 体检报告 |
| **3D 白模、实景前景与物理特效侧车** (`scripts/blender_scene_kit.py` + `scripts/blender_fg_vfx_kit.py` + `scripts/blender-shadow-catcher.py`) | MIT | Cycles GPU 6 类高清场景白模 + 16-bit Z-Depth 与抽卡提示词（`scene-whitemodel`）；`f/1.18` 真实光学散景前景花叶铁艺暗框 +“后穿前绕”三维流光丝带及因果实光反射（`fg-vfx-3d`）；`Shadow Catcher` 同光向投影 | 精确透视白模、3D 前景遮挡层、贴身环绕光效与真投影层 |
| **可选 6GB 显存本地重打光/笔触通道** (`scripts/retouch-gen.py`) | Apache-2.0 / OpenRAIL-M | 在独立隔离 `venv-gen` 中运行 `IC-Light` 环境光包裹与 `Counterfeit-V3.0 + ControlNet Tile` 笔触参考 | 环境光迁移层与受控局部质感底图 |
| **像素头校验与本地裁切** (`scripts/inspect-image.mjs`) | MIT | 检查真实文件头编码（识别伪装 `.png` 后缀的 JPEG）、尺寸、位深及导出 1:1 局部像素对比图 | JSON 元数据、1:1 局部校验切片 |
| **COS 与人像方法库** (`SKILL.md` & `references/`) | — | `Beauty-First` 前置门禁、骨相液化五大原则、奶油瓷白肌 3 步调色、服饰纯明度精修与下肢统一、原生单丝束画发、Blender 白模协同抽卡+PIH 溶图及防糊斑避坑体系 | 可直接约束 Agent 行为的工程化方法卡 |

---

## 仓库目录结构

```text
retouch-workbench/
├── SKILL.md                          # Agent 技能主入口与命令速查表
├── LICENSE                           # MIT License
├── README.md
├── docs/
│   └── mcp.md                        # PhotoCraft / LightCraft MCP 与桌面桥接指南
├── resources/
│   ├── brushes/                      # 发丝/妆容/精修笔刷笔尖（hair_01~03 / makeup_01~05 / retouch_01~02）
│   └── luts/                         # 7 套 33³ 电影级 .cube 3D LUT（Blender AgX/PBR Neutral + 5 套定制电影分色预设）
├── references/
│   ├── workflows.md                  # 场照（field）/ 正片（formal）两阶段标准流程与故障排查
│   ├── liquify-guide.md              # 骨相液化五大原则、冻区设置与照镜子验收
│   ├── natural-beauty-guide.md       # 自然精修与 3 步不偏灰奶油瓷白肌调色
│   ├── garment-retouch-guide.md      # 服饰三频纯明度去雾、--out-dark-mask 与 alpha-only 去白边
│   ├── hair-painting-guide.md        # v11 逐根微簇真发丝 × 动漫整头融合指南与 hair-sculpt 实战
│   ├── human-ai-collaboration-guide.md # 人机协同与白模抽卡硬门禁规范（Gate 0–3 & handoff-pack）
│   ├── compositing-guide.md          # COS 专业合成溶图（Blender白模与3D前景/PIH/三层影/环绕因果光/反塑料）
│   ├── color-lighting-guide.md       # 调色、38张成片基准与 Depth-Anything-V2 重打光/电影 LUT 指南
│   ├── makeup-retouch-guide.md       # 商用人像与 COS 光学补妆精修圣经（Anti-Sticker & Blend-If）
│   ├── painterly-methods.md          # 2.5D COS 厚涂方法卡与锁罩回贴 SOP
│   ├── techniques-and-pitfalls.md    # 显影、高低频、去边、特效克制与输出避坑细则
│   └── learning-roadmap.md           # 12 类后期技能成长路线与公开参考索引
└── scripts/
    ├── retouch-ai.py                 # 主力本地开源 AI 侧车 CLI
    ├── workbench_ops.py              # hair-comp / hair-paint / garment-retouch / skin-unify / harmonize / tile 核心实现
    ├── cuda_env.py                   # CUDA DLL 预加载与 ONNX/PyTorch GPU 推理环境自举
    ├── cutout_refine_ops.py          # 高分边界 BiRefNet 分块与连续色彩线 F/B 光学前景解算精抠
    ├── pure_pc_retouch_ops.py        # 478 点纯 PC 三频奶油肌（skin-fs）与智能液化（body-liquify）
    ├── makeup_retouch_ops.py         # 9 层光学混合模式 + 原生 Blend-If 补妆引擎 v2（makeup-paint）
    ├── hair_sculpt_ops.py            # hair-sculpt 编排器（发根吸色/Rigid MLS/Sensor 颗粒/PhotoCraft 分层栈）
    ├── hair_anime_engine.py          # v11 亚像素解析逐根微簇发丝与动漫整头协同引擎（anime-sculpt）
    ├── hair_rake_engine.py           # 教程三笔刷柔和软芯排笔回退引擎（tutorial-rake）
    ├── blender_hair_sculpt.py        # Blender Cycles OptiX 3D 发丝重塑回退脚本（blender）
    ├── handoff_ops.py                # 人机协同白模硬暂停交接包生成器（handoff-pack）
    ├── output_sharpen_ops.py         # 无白边线性光 RL 反卷积与 4 级语义焦点输出锐化（output-sharpen）
    ├── vfx_overlay_ops.py            # 能量守恒自适应黑底光学 Unmult 与 2.5D 法线因果溢光模块
    ├── relight_cinema_ops.py         # Depth-Anything-V2 2.5D 去平光重打光、.cube 电影分级与前景因果光后处理
    ├── blender_scene_kit.py          # Blender GPU 6 大场景高清白模与 Z 深度/抽卡提示词生成器
    ├── blender_fg_vfx_kit.py         # Blender Cycles f/1.18 物理前景散景与“后穿前绕”三维流光丝带生成器
    ├── blender-shadow-catcher.py     # Blender Cycles Shadow Catcher 立牌投影与 ZDepth 生成脚本
    ├── retouch-gen.py                # 可选 6GB CUDA 显存隔离生图管线（IC-Light + Tile）
    ├── gfpgan_runner.py              # GFPGAN v1.4 ONNX 独立运行器
    ├── skin_retouch_runner.py        # DAMO ABPN 磨皮美白 + 106 关键点 COS 液化运行器
    ├── inspect-image.mjs             # 文件头真实格式检验与 1:1 像素裁切脚本
    ├── fast_download.py              # 绕过慢代理的多线程 Range 直连下载器
    ├── download_models.py            # 全量模型与 CUDA Wheel 资源清单与校验下载器
    ├── setup.ps1                     # Windows 一键分级环境安装脚本（-Tier core / full）
    ├── requirements-ai.txt           # 主力虚拟环境依赖（%LOCALAPPDATA%\RetouchTools\venv）
    └── requirements-gen.txt          # 可选生成虚拟环境依赖（%LOCALAPPDATA%\RetouchTools\venv-gen）
```

---

## 完整依赖、工具与模型下载地址清单

所有外部二进制程序、Python 虚拟环境及模型权重均隔离存放在 `%LOCALAPPDATA%\Programs\ArtCraft\` 与 `%LOCALAPPDATA%\RetouchTools\`，仓库本身保持纯源码轻量。

### 1. 主力修图引擎与系统运行环境

| 组件 | 推荐版本 / 要求 | 开源协议 | 官方下载地址 | 本机安装目录 |
| :--- | :--- | :--- | :--- | :--- |
| **PhotoCraft** | `v0.23.1` (Win x64) | MIT OR Apache-2.0 | [GitHub Releases](https://github.com/storytold/photocraft/releases/latest) · [MSI 安装包](https://github.com/storytold/photocraft/releases/download/v0.23.1/photocraft-0.23.1-windows-x64.msi) · [便携版](https://github.com/storytold/photocraft/releases/download/v0.23.1/photocraft-0.23.1-windows-x64-portable.zip) | `%LOCALAPPDATA%\Programs\ArtCraft\PhotoCraft\` |
| **LightCraft** | `v0.21.1` (Win x64) | MIT OR Apache-2.0 | [GitHub Releases](https://github.com/storytold/lightcraft/releases/latest) · [MSI 安装包](https://github.com/storytold/lightcraft/releases/download/v0.21.1/lightcraft-0.21.1-windows-x64.msi) · [便携版](https://github.com/storytold/lightcraft/releases/download/v0.21.1/lightcraft-0.21.1-windows-x64-portable.zip) | `%LOCALAPPDATA%\Programs\ArtCraft\LightCraft\` |
| **Real-ESRGAN ncnn-vulkan** | `v0.2.5.0` (Windows) | MIT | [windows.zip](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip) | `%LOCALAPPDATA%\RetouchTools\realesrgan-ncnn-vulkan\` |
| **Python & Node.js** | Python `3.12` + Node `>=22` | PSF / MIT | [python.org](https://www.python.org/downloads/) / [nodejs.org](https://nodejs.org/) | 系统全局 + `%LOCALAPPDATA%\RetouchTools\venv` |

### 2. Core 核心视觉解析与精修模型（`download_models.py core`）

> 说明：`uniface` 模型（RetinaFace / `2d_106` / `parsing_resnet18` / `ppmatting_v2`）在首次调用或运行 `retouch-ai.py` 时会自动缓存在 `%LOCALAPPDATA%\RetouchTools\models\uniface\`；`setup.ps1 -Tier core` 或 `download_models.py core` 会同步以下其余核心模型：

| 模型组件 | 开源协议 | 下载地址（主用镜像 / 官方原址） | 目标存放路径（相对 `%LOCALAPPDATA%\RetouchTools\`） |
| :--- | :--- | :--- | :--- |
| **BiRefNet_lite ONNX**（发丝与半透明饰品精抠） | MIT | [ModelScope](https://modelscope.cn/models/onnx-community/BiRefNet_lite-ONNX/resolve/master/onnx/model.onnx) · [hf-mirror](https://hf-mirror.com/onnx-community/BiRefNet_lite-ONNX/resolve/main/onnx/model.onnx) · [HuggingFace](https://huggingface.co/onnx-community/BiRefNet_lite-ONNX/resolve/main/onnx/model.onnx) | `models/birefnet/birefnet_lite.onnx` |
| **SegFormer-B2 Clothes ONNX**（18 类默认人体/服饰精准解析） | MIT | [hf-mirror (mattmdjaga)](https://hf-mirror.com/mattmdjaga/segformer_b2_clothes/resolve/main/onnx/model.onnx) · [hf-mirror (Xenova)](https://hf-mirror.com/Xenova/segformer_b2_clothes/resolve/main/onnx/model.onnx) | `models/segformer/segformer_b2_clothes.onnx` |
| **PhotoBlaze-PIH ONNX**（`curves.onnx` + `gain.onnx` 溶图调和） | Apache-2.0 | [hf-mirror](https://hf-mirror.com/JeremySD99/PhotoBlaze-PIH/tree/main) · [HuggingFace](https://huggingface.co/JeremySD99/PhotoBlaze-PIH/tree/main) | `models/pih/` |
| **Depth-Anything-V2-Small ONNX**（2.5D 法线重打光与因果反射深度引导） | Apache-2.0 | [hf-mirror](https://hf-mirror.com/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx) · [ModelScope](https://modelscope.cn/models/onnx-community/depth-anything-v2-small/resolve/master/onnx/model.onnx) · [HuggingFace](https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx) | `models/depth/depth_anything_v2_small.onnx` |
| **MediaPipe Selfie / Face(478) / Pose Landmarker**（全身分割回退、478 点眼唇精绘与 33 骨骼液化） | Apache-2.0 | [Selfie](https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite) · [Face Landmarker](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task) · [Pose Landmarker](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task) | `models/selfie_multiclass_256x256.tflite` & `models/mediapipe/` |
| **LaMa (`big-lama.pt`)**（ROI 局部无痕消除） | Apache-2.0 | [GitHub Release](https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt) | `models/lama/big-lama.pt` |
| **GFPGAN v1.4 ONNX**（眼睫与超分黑线微修） | Apache-2.0 | [hf-mirror](https://hf-mirror.com/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx) · [HuggingFace](https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx) | `models/gfpgan/gfpgan_1.4.onnx` |
| **DAMO ABPN 磨皮美白**（`cv_unet_skin_retouching_torch`） | Apache-2.0 | [ModelScope 模型仓库](https://www.modelscope.cn/models/damo/cv_unet_skin_retouching_torch/files)（`pytorch_model.pt`、`joint_20210926.pth`、`model.onnx`） | `models/skin-retouching/` |
| **UniFace (`4.3.0`)** | MIT / Apache-2.0 | [yakhyo/uniface](https://github.com/yakhyo/uniface)（随 `requirements-ai.txt` 安装并在 `UNIFACE_CACHE_DIR` 自动缓存） | `models/uniface/` |

### 3. 可选 6GB CUDA 生成辅助模型（`download_models.py wheels` & `gen`）

仅当需要本地离线运行 `IC-Light` 重打光或 `Counterfeit + ControlNet Tile` 时按需下载：

| 组件 / 模型 | 开源协议 | 下载地址（国内直连镜像 / 官方源） | 目标存放路径（相对 `%LOCALAPPDATA%\RetouchTools\`） |
| :--- | :--- | :--- | :--- |
| **PyTorch `2.6.0+cu124` + `torchvision 0.21.0`** | BSD-3-Clause | [阿里云镜像 torch](https://mirrors.aliyun.com/pytorch-wheels/cu124/torch-2.6.0%2Bcu124-cp312-cp312-win_amd64.whl) · [pytorch.org](https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp312-cp312-win_amd64.whl) | `wheels/` → 安装至 `venv-gen/` |
| **IC-Light (`fc` / `fbc`)** | Apache-2.0 / OpenRAIL-M | [ModelScope](https://modelscope.cn/models/lllyasviel/ic-light/files) · [HuggingFace](https://huggingface.co/lllyasviel/ic-light/tree/main) | `models/iclight/` |
| **ControlNet v1.1 Tile (SD1.5)** | Apache-2.0 | [ModelScope](https://modelscope.cn/models/lllyasviel/control_v11f1e_sd15_tile/files) · [HuggingFace](https://huggingface.co/lllyasviel/control_v11f1e_sd15_tile/tree/main) | `models/controlnet-tile/` |
| **Realistic Vision V5.1 (Diffusers)** | CreativeML OpenRAIL-M | [ModelScope](https://modelscope.cn/models/AI-ModelScope/realistic-vision-v51/files) · [hf-mirror](https://hf-mirror.com/stablediffusionapi/realistic-vision-v51) | `models/sd15-rv51/` |
| **Counterfeit-V3.0 fp16 (Diffusers)** | CreativeML OpenRAIL-M | [ModelScope](https://modelscope.cn/models/stablediffusionapi/counterfeit-v30-fp16/files) · [hf-mirror](https://hf-mirror.com/stablediffusionapi/counterfeit-v30-fp16) | `models/sd15-anime/counterfeit-v30-fp16/` |

---

## 快速安装与使用

### 一键初始化（Windows PowerShell）

```powershell
# 1. 先安装 PhotoCraft (v0.23.1) 与 LightCraft (v0.21.1) 到 %LOCALAPPDATA%\Programs\ArtCraft\

# 2. 预览将要安装的虚拟环境、依赖与 Core 模型清单（零下载、零改动）
.\scripts\setup.ps1 -Tier core -DryRun

# 3. 执行 Core 档安装（主解析/抠图/消除/超分/美肤环境）并软链接 Agent 技能
.\scripts\setup.ps1 -Tier core -SyncSkill

# （可选）加装 6GB 显存离线重打光与 Tile 生成环境：
.\scripts\setup.ps1 -Tier full -SyncSkill
```

查看全部资源清单（JSON 输出、不下载）：

```powershell
python scripts\download_models.py --dry-run
```

### 常用侧车与检查命令

```powershell
$py = "$env:LOCALAPPDATA\RetouchTools\venv\Scripts\python.exe"

# 环境与 GPU / 模型 / 画笔自检
& $py scripts\retouch-ai.py doctor

# 人脸 106 关键点 + 19 类语义分割 + 液化五官冻区导出
& $py scripts\retouch-ai.py face-parse photo.png --out-dir work\face

# 发丝与半透明饰品高精度前景抠图（BiRefNet_lite）
& $py scripts\retouch-ai.py matte photo.png --model birefnet-lite --out-mask work\matte.png

# 全身 18 类 SegFormer-B2 语义分割（配合抠图细化）
& $py scripts\retouch-ai.py body-parse photo.png --refine-with-matte work\matte.png --out-dir work\body

# 服饰分材质去雾 + 完整深色服装/鞋袜联合保护罩导出
& $py scripts\retouch-ai.py garment-retouch work\fg_rgba.png --body-parse-dir work\body --out work\fg_garment.png --out-dark-mask work\dark_full_mask.png --overwrite

# 下肢褪色炫光肤色统一 + 外缘去白边 + 裙/腿遮挡影
& $py scripts\retouch-ai.py skin-unify work\fg_garment.png --body-parse-dir work\body --out work\fg_skin.png --overwrite

# 实拍真发去杂边 + 漂白反光回暖 + 顺发流轻理顺
& $py scripts\retouch-ai.py hair-comp --subject-rgba work\fg_skin.png --subject-hair-mask work\body\mask_hair.png --protected-mask work\body\mask_protected_body.png --occluder-mask work\body\mask_occluder_arms_face.png --mode real-first --groom --out-dir work\hair_comp --overwrite

# PhotoBlaze-PIH ONNX 溶图调和（透传完整深色服装保护罩）
& $py scripts\retouch-ai.py harmonize --fg-rgba work\hair_comp\hair_comp_subject_rgba.png --bg work\bg.png --skin-mask work\body\mask_all_skin.png --garment-dark-mask work\dark_full_mask.png --grain-match --out-dir work\harmonize --overwrite

# PhotoCraft 原生极细单丝束前后分层飘逸画发
& $py scripts\retouch-ai.py hair-paint --subject-rgba work\harmonize\harmonized_fg_rgba.png --hair-mask work\body\mask_hair.png --occluder-mask work\body\mask_occluder_arms_face.png --face-mask work\body\mask_face_skin.png --eyes-brows-mask work\body\mask_eyes_brows.png --clothes-mask work\body\mask_clothes_all.png --wind-dir auto --intensity medium --out-dir work\hair_paint --overwrite

# 默认 v11 逐根解析微簇真发丝 × 动漫整头重塑（anime-sculpt，零碰撞分层 .pcraft）
& $py scripts\retouch-ai.py hair-sculpt --subject-rgba work\harmonize\harmonized_fg_rgba.png --hair-mask work\body\mask_hair.png --bg work\bg.png --occluder-mask work\body\mask_occluder_arms_face.png --face-mask work\body\mask_face_skin.png --eyes-brows-mask work\body\mask_eyes_brows.png --clothes-mask work\body\mask_clothes_all.png --engine anime-sculpt --style auto --out-dir work\hair_sculpt --overwrite

# 全项掩码与 1:1 切片体检（注意：亲眼核查切片后方可使用 --visual-reviewed）
& $py scripts\retouch-ai.py mask-audit final.png --out-dir work\audit --protected-mask work\body\mask_protected_body.png --garment-mask work\body\mask_skirt_dress.png --native-hair-mask work\body\mask_hair.png --eyes-brows-mask work\body\mask_eyes_brows.png --reference work\fg_rgb.png --reference-hair-mask work\body\mask_hair.png --neutral-garment --overwrite

# 发丝边界分块重组 + 连续色彩线光学去污精抠
& $py scripts\retouch-ai.py cutout-refine photo.png --body-parse-dir work\body --out-dir work\cutout --overwrite

# 纯 PC 非生成：骨相冻区液化、三频奶油肌与 478 点光学 Blend-If 补妆 v2
& $py scripts\retouch-ai.py body-liquify photo.png --body-parse-dir work\body --faces-json work\face\faces.json --out-dir work\liquify --overwrite
& $py scripts\retouch-ai.py skin-fs photo.png --face-parse-dir work\face --body-parse-dir work\body --out-dir work\skin_fs --overwrite
& $py scripts\retouch-ai.py makeup-paint photo.png --faces-json work\face\faces.json --style cos-anime --build-pcraft --out-dir work\makeup --overwrite

# Blender 建筑白模空镜（导出白模 + Z深度 + 双语抽卡提示词）与人机协同交接包
& $py scripts\retouch-ai.py scene-whitemodel --preset crystal-conservatory --out-dir work\scene_kit
& $py scripts\retouch-ai.py handoff-pack --out-dir work\scene_kit --whitemodel work\scene_kit\01_blender_whitemodel_empty.png

# 能量守恒黑底 AI 特效去底拆层（克制雾斑并避让人物正面/脸眼）
& $py scripts\retouch-ai.py vfx-overlay --vfx vfx_black.png --subject-alpha work\fg_cutout.png --face-parse-dir work\face --density-gamma 1.75 --haze-suppress 0.65 --out-dir work\vfx --overwrite

# 2.5D 深度法线体积重打光 + 3D 前景大光圈暗框/绕身因果流光 + 电影 3D LUT 分级 + 焦点输出锐化
& $py scripts\retouch-ai.py relight-sculpt --fg-rgba work\fg_harmonized.png --bg work\bg.png --face-parse-dir work\face --body-parse-dir work\body --out-dir work\relight --overwrite
& $py scripts\retouch-ai.py fg-vfx-3d --preset crystal-conservatory --subject-alpha work\fg_harmonized.png --face-parse-dir work\face --normal-map work\relight\normal_map_rgb.png --out-dir work\fg_vfx_3d --overwrite
& $py scripts\retouch-ai.py cinema-grade work\bg.png --mode bg-anchor --out-dir work\grade --overwrite
& $py scripts\retouch-ai.py cinema-grade work\comp.png --mode grade --preset crystal-sunset-cinema --out-dir work\grade --overwrite
& $py scripts\retouch-ai.py output-sharpen work\comp_graded.png --face-parse-dir work\face --body-parse-dir work\body --out work\comp_sharpened.png --overwrite

# 真实编码检验与 1:1 像素裁切
node scripts\inspect-image.mjs inspect photo.png
```

MCP 服务端配置与桌面端桥接方式详见 [docs/mcp.md](docs/mcp.md)。

---

## 许可说明

- 本仓库脚本与文档遵循 [LICENSE](LICENSE)（MIT License）。
- 上游引擎（PhotoCraft、LightCraft）与所有 AI 模型权重分别遵循各自仓库所列的 MIT、Apache-2.0、BSD-3-Clause 或 CreativeML OpenRAIL-M 许可协议；使用前请核对上游许可条款。
