# Retouch Workbench · 纯开源 Agent 人像与 COS 后期修图工作台

**Retouch Workbench** 是一个面向 AI Agent（Codex、Antigravity、Claude Code、OpenCode 等）与命令行的**纯开源人像及 COS 后期修图工作台**，以 **ArtCraft PhotoCraft**（分层精修、蒙版、骨相液化、混合画笔、PSD/PCRAFT 无损工程）与 **ArtCraft LightCraft**（批量 RAW 暗房、选片、32-bit 浮点色彩分级）为核心双引擎，配合本机可插拔开源 AI 侧车（`scripts/retouch-ai.py`）与实证修图方法库（`references/`）协同完成非破坏后期。

> **开源与合规边界**：本项目源码、外部核心引擎与默认预置模型均采用宽松开源或开放权重许可（**MIT / Apache-2.0 / BSD-3-Clause / CreativeML OpenRAIL-M**），不含任何商业闭源修图软件依赖、个人隐私原图或闭源/非商用限制抠图模型，亦不依赖重型本地大模型流水线。

---

## 核心架构与职责分工

| 模块 / 组件 | 开源许可 | 在工作台中的职责 | 关键产物 |
| :--- | :--- | :--- | :--- |
| **ArtCraft PhotoCraft** (`photocraft-cli`) | MIT OR Apache-2.0 | 单张分层精修、皮肤高低频、冻区骨相液化、图层/矢量蒙版、混合画笔、压感路径勾丝、`.pcraft`/`.psd` 读写 | 无损 `.pcraft`、兼容 `.psd`、4K 发布成片 |
| **ArtCraft LightCraft** (`lightcraft-cli`) | MIT OR Apache-2.0 | 批量导入、选片、白平衡/曲线/HSL/LUT 同组暗房同步、16-bit/32-bit 显影导出 | 无损调整日志、标准 16-bit PNG/TIFF 底图 |
| **核心 AI 侧车** (`scripts/retouch-ai.py`) | MIT / Apache-2.0 / BSD | 106 关键点冻区与 19 类人脸语义（`face-parse`）、发丝与半透明配饰抠图（`matte`：`BiRefNet_lite` / `PP-Matting V2`）、全身 6 类分割（`body-parse`）、ROI 局部消除（`inpaint`：`LaMa`）、Vulkan 写实超分（`upscale`：`Real-ESRGAN`）、面部清锐恢复（`face-restore`：`GFPGAN v1.4 ONNX`）、自然奶油肌（`skin-retouch`：`DAMO ABPN`）、像素对位与量化质检（`qa` / `compare`） | 严格互斥 PNG 蒙版、`faces.json` 冻区点位、透明局部补丁、4K 超分底图 |
| **可选 6GB 显存本地重打光/笔触通道** (`scripts/retouch-gen.py`) | Apache-2.0 / OpenRAIL-M | 在独立隔离 `venv-gen` 中运行 `IC-Light` 环境光包裹与 `Counterfeit-V3.0 + ControlNet Tile` 笔触参考 | 环境光迁移层与受控局部质感底图 |
| **像素头校验与本地裁切** (`scripts/inspect-image.mjs`) | MIT | 检查真实文件头编码（识别伪装 `.png` 后缀的 JPEG）、尺寸、位深及导出 1:1 局部像素对比图 | JSON 元数据、1:1 局部校验切片 |
| **COS 与人像方法库** (`SKILL.md` & `references/`) | — | `Beauty-First` 前置门禁、骨相液化五大原则、奶油瓷白肌 3 步调色、反贴纸眼唇、藏色发束与“走势底稿 + 强图模直出 + PhotoCraft 前景锁罩回贴”SOP | 可直接约束 Agent 行为的工程化方法卡 |

---

## 仓库目录结构

```text
retouch-workbench/
├── SKILL.md                          # Agent 技能主入口与命令速查表
├── LICENSE                           # MIT License
├── README.md
├── docs/
│   └── mcp.md                        # PhotoCraft / LightCraft MCP 与桌面桥接指南
├── references/
│   ├── workflows.md                  # 场照（field）/ 正片（formal）标准流程与防翻车自检
│   ├── liquify-guide.md              # 骨相液化五大原则、冻区设置与照镜子验收
│   ├── natural-beauty-guide.md       # 自然精修与 3 步不偏灰奶油瓷白肌调色
│   ├── painterly-methods.md          # 2.5D COS 厚涂方法卡与走势底稿+强图模+锁罩回贴 SOP
│   ├── techniques-and-pitfalls.md    # 显影、高低频、选区修边与输出避坑细则
│   └── learning-roadmap.md           # 12 类后期技能成长路线与公开参考索引
└── scripts/
    ├── retouch-ai.py                 # 主力本地开源 AI 侧车 CLI
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
| **MediaPipe Selfie Multiclass**（6 类全身分割） | Apache-2.0 | [Google Storage](https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite) | `models/selfie_multiclass_256x256.tflite` |
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

# 环境与 GPU / 模型自检
& $py scripts\retouch-ai.py doctor

# 人脸 106 关键点 + 19 类语义分割 + 液化五官冻区导出
& $py scripts\retouch-ai.py face-parse --input photo.png --out-dir work\face

# 发丝与半透明饰品高精度前景抠图（BiRefNet_lite）
& $py scripts\retouch-ai.py matte --input photo.png --model birefnet-lite --out-mask work\matte.png

# 全身 6 类分割（配合抠图细化）
& $py scripts\retouch-ai.py body-parse --input photo.png --refine-with-matte work\matte.png --out-dir work\body

# Real-ESRGAN 写实 4x 超分（COS 真人严禁使用 anime 模型）
& $py scripts\retouch-ai.py upscale --input photo.png --scale 4 --out work\photo_4k.png

# 真实编码检验与 1:1 像素切片
node scripts\inspect-image.mjs inspect photo.png
```

MCP 服务端配置与桌面端桥接方式详见 [docs/mcp.md](docs/mcp.md)。

---

## 许可说明

- 本仓库脚本与文档遵循 [LICENSE](LICENSE)（MIT License）。
- 上游引擎（PhotoCraft、LightCraft）与所有 AI 模型权重分别遵循各自仓库所列的 MIT、Apache-2.0、BSD-3-Clause 或 CreativeML OpenRAIL-M 许可协议；使用前请核对上游许可条款。
