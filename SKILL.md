---
name: retouch-workbench
description: 基于开源引擎 PhotoCraft + LightCraft（原生 MCP / CLI）及本地开源 AI 侧车（uniface / PP-Matting V2 / MediaPipe / LaMa / Real-ESRGAN-ncnn-vulkan）的本机人像与 COS 后期修图、厚涂重绘、局部精修与方法学习工作台技能；适用于场照返图、角色正片、二次元厚涂、局部修复与合成、以及修图技法复盘。
---

# Retouch Workbench · COS 与人像后期工作台（纯开源双引擎 + AI 侧车栈）

在当前会话接收修图需求与本机素材路径，优先通过已接入的 **LightCraft MCP**（批量 RAW 暗房/组图统一）与 **PhotoCraft MCP**（单张分层精修/厚涂重绘）配合本地宽松开源 AI 侧车 `retouch-ai.py` 执行非破坏编辑。

## 1. 核心原则与防呆护栏

1. **原图绝对只读**：绝不覆盖原图或用户已有交付文件。所有工程文件（`.pcraft` / `.psd`）、侧车输出与交付副本统一写入 `D:\ps\<项目子目录>\`。
2. **接图必先验真**：开修前运行检查脚本核对文件头真实编码（防范 JPEG 改名 `.png`）、像素尺寸、位深、色彩配置与图层树：
   ```powershell
   node "D:\ps\retouch-workbench\scripts\inspect-image.mjs" inspect "<原图路径>"
   ```
3. **先修美颜底图硬门禁（Beauty-First Gate）**：人像正片、美颜精修或 COS 二次元厚涂前，必须严格按 `LightCraft 暖润安全底 → 骨相液化+重跑蒙版 → 瑕疵/超分黑线清理 → 校准高低频与D&B → 3步奶油瓷白肌（全臂连续蒙版） → 自然非贴纸眼唇` 先产出并 1:1 验收 **`01_beauty_base.png` + `.pcraft`**，才允许继续换背景、重打光或厚涂风格化。
4. **关键修改前必留快照**：PhotoCraft 的 `command_batch` **不是原子事务**（中途失败时已完成的前序步骤会保留，不会自动回滚）。每轮批量修改前先调用 `history.newSnapshot` 留底；失败或跑偏时用 `history.selectSnapshot` 或 `edit.undo` 回退。
4. **整图 + 1:1 局部双重看图与量化验收**：
   - 整图光影与大形：`doc_render_preview`（最大边 2048）。
   - 皮肤纹理、眼妆边界、唇缝、假发边缘与衣纹 1:1 细节：先临时导出 PNG，再用 `inspect-image.mjs crop` 或 `retouch-ai.py compare` 生成带原像素局部对照的拼图看图，必要时配合 `retouch-ai.py qa` 核对溢出与肤色偏色。
   - 严禁把“命令返回成功”“建了空白修复层/中性灰层/分频组”或“仅看整图缩略图”说成修图完成。
5. **先存可编辑工程，再导出发布副本**：先通过 `doc_save` 保存可编辑 `.pcraft` / `.psd`，再视交付要求用 `edit.convertToProfile` 转换目标配置（如 `srgb`）并导出 JPEG/PNG，导出后重开复核尺寸、图层与观感。

## 2. 双引擎与侧车接入方式

### 2.1 LightCraft（批量 RAW 显影、选片与同组调色同步）
- **默认无头 MCP（`lightcraft`）**：适合场照多张选片、同光线批量白平衡/曝光/色彩分级、XMP/预设同步与快速初调，常用工具为 `import`、`query_photos`、`select_photos`、`list_controls`、`get_develop`、`set_develop`、`apply_preset`、`crop`、`render_photo`、`export`、`render_preview`；先用 LightCraft 统一底片再导出 16-bit TIFF/PNG 进 PhotoCraft 精修。
- **CLI 入口**：`%LOCALAPPDATA%\Programs\ArtCraft\LightCraft\lightcraft-cli.exe`（支持 `render`、`run`、`merge hdr|panorama`、`commands`、`controls`）。

### 2.2 PhotoCraft（单张分层精修、蒙版、液化与厚涂重绘）
- **默认无头 MCP（`photocraft`）**：已注册 26 个工具——常用入口为 `doc_open` / `doc_new`、`doc_inspect`、`command_list`、`command_run`、`command_batch`、`doc_render_preview`、`doc_place`、`doc_replace_smart_object`、`doc_save`、`doc_export`、`jobs_list`、`jobs_cancel`。
- **实时桌面协作模式**：启动 `photocraft.exe --control <PORT>` 并让 `photocraft-cli.exe mcp --bridge 127.0.0.1:<PORT>` 桥接，可额外使用 `ui_*` 工具。
- **CLI 入口**：`%LOCALAPPDATA%\Programs\ArtCraft\PhotoCraft\photocraft-cli.exe`（支持 `info`、`run`、`convert`、`batch` 及 `commands --filter <关键词> --json`）。

### 2.3 本机开源 AI 侧车（`retouch-ai.py`）
固定解释器：`%LOCALAPPDATA%\RetouchTools\venv\Scripts\python.exe`，输出结构化 JSON 与无损 PNG 蒙版/补丁：

| 子命令 | 作用与底层开源组件 | 核心输出与与 PhotoCraft 联动方式 |
| --- | --- | --- |
| `doctor` | 检查 `CUDAExecutionProvider`、`venv-gen` PyTorch CUDA 与全模型就绪状态 | 开工前确认 GPU/模型就绪 |
| `face-parse` | `uniface` RetinaFace + 106 关键点 + BiSeNet 19 类（CUDA） | 产出 `faces.json`（含 `landmarks_106` 与眼眉/唇 `liquify_freeze_polygons`）+ 严格互斥蒙版 `mask_skin.png` / `mask_eyes_brows.png` / `mask_lips.png` / `mask_hair_head.png` |
| `matte` | `PP-Matting V2`（Apache-2.0）、`BiRefNet_lite`（`--model birefnet-lite`，MIT，推荐正片精抠）或 `MODNet` | 产出含发丝与轻纱细缘的 `out_mask.png` 与可选 RGBA `out_fg.png` |
| `body-parse` | `MediaPipe` `selfie_multiclass_256x256`（Apache-2.0），支持 `--refine-with-matte` | 产出全身 `mask_body_hair.png` / `mask_body_skin.png` / `mask_face_skin.png` / `mask_all_skin.png` / `mask_clothes.png` / `mask_accessories.png` |
| `inpaint` | `LaMa`（`big-lama.pt`，Apache-2.0）仅裁剪 Mask ROI 推理，**掩码外零改动** | 加 `--patch-only` 输出透明局部补丁 PNG，直接用 `doc_place` 叠进 PhotoCraft 当非破坏修复层 |
| `upscale` | `Real-ESRGAN-ncnn-vulkan`（MIT，Vulkan GPU），支持 `--scale 2\|4` | 低像素/裁切素材无网格接缝放大；超分后须局部清理鼻孔等暗锐化黑线 |
| `relight` | 隔离 `venv-gen` 运行官方 `IC-Light`（`fc`/`fbc`，Apache-2.0 / OpenRAIL-M，6GB 显存安全） | 生成带方向光包裹与氛围背景的重打光底图，供 PhotoCraft 抽离低频环境光融回主体 |
| `painterly` | 隔离 `venv-gen` 运行 `Counterfeit-V3.0` + `ControlNet-v1-1 Tile`（OpenRAIL-M / Apache-2.0） | 双参数通道分区域赋予头发/衣褶厚涂笔触，同时严守五官与手部结构保真 |
| `qa` | 高光溢出 / 死黑 / 高饱和比例 / 清晰度 + 可选 `--skin-mask` Lab 均值 | 快速定位脸/臂肤色偏冷发灰（查 `b*`）与曝光溢出 |
| `compare` | 生成带全图 + `1:1` 原像素裁切对比的拼图 PNG | 视觉自检前后变化与细节 |

> **如何把侧车蒙版/补丁导入 PhotoCraft**：
> 1. 蒙版：用 `doc_place` 把 `mask_*.png` 临时置入、或用 `file.open` 打开后 `select.all` → `edit.copy` / `select.loadSelection` 载入为目标图层/组蒙版，确保磨皮或重绘严格避开眼眉唇；
> 2. 液化冻区：直接读取 `faces.json` 中的 `liquify_freeze_polygons` 点位传入 `filter.liquify` 的 `lassoMask` 笔画，开推前冻住五官；
> 3. 局部消除：优先 `inpaint --patch-only` 生成透明补丁层并用 `doc_place` 置入，保留可开关和蒙版擦回余地。

## 3. 常用修图动作 → PhotoCraft 命令速查

> 调用任何命令前，不确定的可选参数以 `command_list` 或 `photocraft-cli commands --filter <id> --json` 现场输出为准。注意不同命令的透明度量纲差异：
> - `layer.new.layer`、`layer.newAdjustmentLayer.*`、`layer.setProps`、`paint.stroke` 使用 **`0..1`**；
> - `layer.new.group` 使用 **`0..100`**；
> - `paint.dodge`、`paint.burn`、`paint.smudge`、`paint.spotHealing`、`paint.cloneStamp`、`paint.remove` 使用 **`1..100`**。

| 修图环节 | 常用 PhotoCraft 命令 ID | 关键注意点 |
| --- | --- | --- |
| 显影与调色 | `filter.cameraRaw`、`layer.newAdjustmentLayer.curves` / `hueSaturation` / `selectiveColor` / `colorBalance` / `levels` / `colorLookup` / `gradientMap`、`edit.assignProfile`、`edit.convertToProfile` | `curves` 支持 `points` 及通道分别调；区分“指定”与“转换”配置。 |
| 分频与图层管理 | `image.applyImage`、`filter.blur.gaussianBlur`、`filter.other.highPass`、`layer.new.layer`（`fillNeutral: true` 可建中性灰柔光层）、`layer.new.group`、`layer.moveTo`、`layer.setProps` | `layer.new.layer` 的 `opacity` 为 `0..1`，`layer.new.group` 为 `0..100`。 |
| 选区与蒙版 | `select.subject`、`select.refineEdge`（支持 `decontaminate: true`）、`select.rect`、`select.lasso`、`select.colorRange`、`select.modify.feather`/`expand`/`contract`、`layer.layerMask.revealSelection`/`hideSelection`/`revealAll`/`hideAll`、`layer.vectorMask.fromPath` | `select.subject` 只是初始蒙版，可先跑 `matte` + `body-parse` 补强。 |
| 绘画与厚涂 | `paint.stroke`（支持 `target: "pixels"\|"mask"` 与 `fill: true`）、`paint.mixerBrush`、`paint.dodge`、`paint.burn`、`paint.smudge`、`path.stroke`（支持 `simulatePressure: true` 模拟压感描发丝）、`brush.presets.importAbr` | 混色涂抹前确认 `sampleAllLayers`，防皮肤/衣色串入头发。 |
| 修复与清理 | `edit.contentAwareFill`、`paint.healingBrush`、`paint.spotHealing`、`paint.patch`、`paint.cloneStamp`、`paint.remove` | 跨材质或大块路人优先 `retouch-ai.py inpaint --patch-only` + `doc_place`。 |
| 塑形与液化 | `layer.smartObjects.convertToSmartObject`、`filter.liquify`（`forwardWarp`/`reconstruct`/`smooth`/`freeze`/`thaw`/`lassoMask`）、`edit.puppetWarp`、`edit.transform.warp`、`filter.distort.displace` | 先转智能对象再跑 `filter.liquify`，可用 `faces.json` 的 `lassoMask` 冻住五官。 |
| 合成、锐化与快照 | `doc_place`、`filter.sharpen.unsharpMask`、`filter.sharpen.smartSharpen`、`history.newSnapshot`、`history.selectSnapshot` | 批量改动前先 `history.newSnapshot`；超分图不默认再叠锐化。 |

## 4. 按需查阅参考手册

| 场景 | 读取文件 |
| --- | --- |
| 场照（`field`）/ 正片（`formal`）标准流程、美颜前置门禁与常见翻车排查 | [工作流程](references/workflows.md) |
| **人像与 COS 骨相液化实操（五大原则、冻区、照镜子验收、重跑蒙版）** | [液化指南](references/liquify-guide.md) |
| **商业自然精修与奶油瓷白肌调色（硬门禁、高低频校准、3 步不偏灰调色）** | [自然精修与奶油肌](references/natural-beauty-guide.md) |
| 二次元厚涂 COS（反贴纸眼唇、藏色发束与 `IC-Light` / `Counterfeit+Tile` 管线） | [厚涂方法卡](references/painterly-methods.md) |
| 显影色彩、高低频、选区修边、液化冻区、合成透视与输出避坑细则 | [技术与避坑细则](references/techniques-and-pitfalls.md) |
| COS 12 类技能学习地图、低风险对照实验与公开教程索引 | [学习路线与索引](references/learning-roadmap.md) |
