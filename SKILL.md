---
name: retouch-workbench
description: 基于开源引擎 PhotoCraft + LightCraft（原生 MCP / CLI）及本地开源 AI 侧车（uniface / BiRefNet / SegFormer / Depth-Anything-V2 / MediaPipe / LaMa / PhotoBlaze-PIH / Real-ESRGAN）与 Blender 3D/OCIO .cube 电影 LUT 的本机人像与 COS 后期修图、2.5D 重打光、3D 前景环绕特效与方法学习工作台技能；适用于场照返图、角色正片、二次元厚涂、局部修复与影视级合成。
---

# Retouch Workbench · COS 与人像后期工作台（纯开源双引擎 + AI 侧车栈）

在当前会话接收修图需求与本机素材路径，优先通过已接入的 **LightCraft MCP**（批量 RAW 暗房/组图统一）与 **PhotoCraft MCP**（单张分层精修/厚涂重绘）配合本地宽松开源 AI 侧车 `retouch-ai.py` 执行非破坏编辑。

## 1. 核心原则与防呆护栏

1. **人像零 AI 重绘 · 协同边界铁律（Pure-PC Person Retouch × AI Empty BG & Black-BG VFX）**：
   - **人物像素严禁 AI 重绘**：所有面部、五官、皮肤、妆容、发丝与服饰精修，必须 100% 使用本机 **LightCraft + PhotoCraft** 原生工具/画笔/液化/三频/D&B 与纯确定性数学算法完成（`retouch-ai.py skin-fs` / `makeup-paint` / `body-liquify` / `garment-retouch` / `hair-paint`）。本地神经模型**严格仅用于生成黑白选区蒙版、透明通道、去杂边光解算与关键点引导**（如 `face-parse` / `body-parse` / `cutout-refine`），绝不重绘或涂抹改变人物原生皮肤与材质像素。
   - **外部 AI 生图（用户抽卡）仅限两类图层**：① 按 `scene-whitemodel` 输出的白模底图与提示词抽出的**严格无人高清空镜背景**；② 按纯粹黑底（`#000000`）提示词抽出的**复杂梦幻粒子/光斑/流光特效叠层**（回传后由 `vfx-overlay` 执行光学 Unmult 去黑底、人物后/前纵深拆层与脸眼片名区避让后以图层覆盖）。
2. **原图绝对只读**：绝不覆盖原图或用户已有交付文件。所有工程文件（`.pcraft` / `.psd`）、侧车输出与交付副本统一写入 `D:\ps\<项目子目录>\`。
3. **接图必先验真**：开修前运行检查脚本核对文件头真实编码（防范 JPEG 改名 `.png`）、像素尺寸、位深、色彩配置与图层树：
   ```powershell
   node "D:\ps\retouch-workbench\scripts\inspect-image.mjs" inspect "<原图路径>"
   ```
4. **先修美颜底图与原生分辨率硬门禁（Beauty-First & Resolution Gate）**：人像正片、美颜精修或 COS 二次元厚涂前，必须严格按 `LightCraft 暖润安全底 → 骨相液化+重跑蒙版 → 瑕疵/超分黑线与眼区穿帮杂发清理 → 校准高低频与D&B → 3步奶油瓷白肌（全臂连续蒙版） → 自然非贴纸眼唇` 先产出并 **独立单张真 `1:1`（`<=512px`）** 验收 **`01_beauty_base.png` + `.pcraft`**，再对服装与下肢依次运行 **`garment-retouch --out-dark-mask`**（纯明度空间去雾+外缘半透明去白边，并把含鞋袜/兜底黑鞋的完整深色服装罩透传给 `harmonize --garment-dark-mask`）与 **`skin-unify`**（下肢褪色膝盖对齐手臂晴光肤色、左缘去白边并叠黑裙/交叠腿遮挡影），才允许继续换背景或合成。**防糊斑红线**：严禁在背景抠图前用 `LaMa` 大面积抹除主体轮廓旁的衣物飘带/发丝交界区（`LaMa` 会把前景与旧背景涂成不透明灰白糊斑并被 `BiRefNet` 锁进主体前景）；如需清理边缘杂物应先保留原物或在换背景后于独立非破坏层处理，复核可疑灰边时必须直接对照最原始输入图（`00_input`）。**原生分辨率门禁**：背景或大型贴片素材相对目标区域的线性原生比必须 `>= 0.6`；严禁把 `<= 1K` 生图盲目 `realesrgan 4x` 直接当成 4K 背景/发片使用，低清底图必须走 **`tile-split` → 分块高清精绘 → `tile-stitch`（低频锁光影 + 亚像素对齐）** 路线消除油画伪影。
5. **按真实画风选头发路线 + 零间隙遮挡铁律（Style-Adaptive Hair Gate）**：
   - **写实人像/实拍风 COS** 默认走 **`hair-comp --mode real-first --groom`** 保留 100% 原生 4K 高清真发（头部、刘海、胸前与跨手臂交界发丝），同时顺发流轻理顺外侧炸毛并回暖漂白发白斑；需要增加飘逸毛流感时，走 **[分层画发丝指南](references/hair-painting-guide.md)** + **`hair-paint`（PhotoCraft 原生极细双端渐细单丝束）** 在原发上拆成 **后景组（`02b_HAIR_PAINT_BEHIND_4K`）+ 前景组（`03b_HAIR_PAINT_FRONT_4K`：`Normal` 飘丝 + `Linear Dodge (Add)` 逆光飞丝）** 非破坏叠加（按起笔局部自吸真发色、严格扩张避开眼区与面部、并与 `0px` 零扩边前景遮挡互不干扰）；**严禁用低清超分油画假发片、平宽扁色带笔刷或 `PIL` 生硬直线划伤实拍质感**；
   - 如确需大范围飘发重塑或二次元厚涂，走 `hybrid-flow` / `full-replace`，必须以引导滤波对齐真实紧贴遮挡物（`0px` 外扩遮罩，`0.8px` 抗锯齿），严禁糊状扩边把跨臂/跨腕发丝切断或在外围留半透明灰白残影条。
6. **关键修改前必留快照**：PhotoCraft 的 `command_batch` **不是原子事务**（中途失败时已完成的前序步骤会保留，不会自动回滚）。每轮批量修改前先调用 `history.newSnapshot` 留底；失败或跑偏时用 `history.selectSnapshot` 或 `edit.undo` 回退。
7. **真 `1:1` 独立单切图 + 自动掩码体检 + 真实看图铁律**：
   - 整图光影与大形：`doc_render_preview`（最大边 2048）。
   - **严禁把多张放大细节缩拼进一张千像素拼图当作 `1:1` 验收**，更**严禁未亲眼看到图片画面就声称视觉通过（`--visual-reviewed` 仅在确实看图审阅后才能启用，否则 `verdict` 停在 `NUMERIC_PASS_VISUAL_UNVERIFIED`）**。导出前必跑 **`retouch-ai.py mask-audit`**（零破洞、零半透明残影、手臂遮挡零断口、中性黑衣零偏棕、织纹留存、人景融合、热点 `1:1` 切片），并逐张审阅 `<=512px` 独立 `1:1` 切片 + `retouch-ai.py qa`。
   - 严禁把“命令返回成功”“建了空白修复层/中性灰层/分频组”或“仅靠数值达标而没真修好视觉瑕疵”说成修图完成。
8. **先存可编辑工程，再导出发布副本**：先通过 `doc_save` 保存可编辑 `.pcraft` / `.psd`，再视交付要求用 `edit.convertToProfile` 转换目标配置（如 `srgb`）并导出 JPEG/PNG，导出后重开复核尺寸、图层与观感。

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
| `matte` / `cutout-refine` | `BiRefNet_lite` + **高分边界分块重构 + 无依赖多层 F/B 光学前景解算 + 发丝断口桥接** | `cutout-refine` 专门根治高倍缩放导致细发丝啃食断口与深灰/紫边溢色，产出纯净直通 RGB + 连续发丝透明遮罩及黑白底校验图 |
| `skin-fs` | OpenCV Contrib `rollingGuidanceFilter` CIE Lab 双尺度三频 + MediaPipe 478 点 + PhotoCraft D&B | 纯手修非生成：低频匀影、中频去斑、高频 100% 锁毛孔，全臂/颈/腿连贯肤色并自动生成 PhotoCraft 50% 灰柔光 `skin_db_neutral_softlight.png` |
| `makeup-paint` | **MediaPipe 478 点 + PhotoCraft 原生 `paint.stroke` 压感手绘妆容** | 闭眼/睁眼自适应（闭眼自动关闭瞳孔高光），导出 5 张独立透明手绘层：睫毛眼线、卧蚕珍珠点睛、眸光、雾面腮红、水润唇釉 |
| `body-liquify` | **MediaPipe 33 骨骼 + 轮廓曲率袜口检测 + PhotoCraft `filter.liquify` 智能滤镜** | 冻住五官后自动压顺斜方肌、手臂、腰线及大腿袜口勒痕并输出可编辑 `.pcraft` 与照镜子对比图 |
| `vfx-overlay` | **能量守恒 Adaptive Black-Floor + Optical Unmult 去黑底引擎** | 专接用户抽卡的黑底 AI 粒子/星芒/光斑：严格按 `a_behind + a_front <= alpha` 分拆“主体后/主体前”，配 `--density-gamma` / `--haze-suppress` / `--max-alpha` 抑制雾斑抢戏、`--subject-front-keep` 保护人物核心且零眼区/片名遮挡 |
| `scene-whitemodel` | **Blender 5.x GPU 6 大场景白模工厂** | 一键出纯净白模（供用户 AI 抽空镜）+ 16-bit 物理 Z-Depth + `scene_meta.json`（真太阳方向）+ 双语背景/黑底特效抽卡提示词 |
| `body-parse` | 默认 **`SegFormer-B2 Clothes` 18 类 ONNX CUDA + 引导滤波**（MIT），回退支持 `MediaPipe`（`--backend`） | 精准拆分 `mask_hair` / `mask_upper_clothes` / `mask_skirt_dress` / `mask_shoes` / `mask_legs` / `mask_arms` 与受保护整体掩码 `mask_protected_body`，并产出 `preview_parse_overlay.png` |
| `mask-audit` | 工程级掩码、**边缘完整度**、**光感区分服饰明暗阈值**与 **`cinema_tone_check`** 自动体检 | 检测破洞、残影、断发微粒、边缘色偏与画面影调黑位/反差（`p02_L` / `dyn_range`）；高调服饰（`p50 >= 120`）自动区分自然侧光梯度与黑衣断层，支持 `--scene-meta` 校验物理太阳光向 |
| `hair-comp` | 自适应风格头发合成与零间隙遮挡（`SegFormer` + `BiRefNet` + CIE Lab） | 写实正片默认 `real-first` 保原生 4K 真发，支持 `--groom` 顺发流切向轻量理顺（仅驯服炸毛、保大形与刘海/眼周）；`hybrid-flow`/`full-replace` 支持零扩边紧贴遮挡与降油画频段匹配 |
| `hair-paint` | PhotoCraft 原生极细单丝束前后分层画发（`brush.presets.importImage` + `paint.stroke` 方向压感） | 永不删原发、局部自吸真发根色融合深中浅调，输出后景柔流层 `hair_paint_01_soft_flow_behind.png` + 前景飘逸 `Normal` 层 + `Linear Dodge (Add)` 逆光飞丝毛流感层，带扩张脸/眼避让与 `0px` 零扩边前景遮挡擦除 |
| `garment-retouch` | `SegFormer` 语义分服 + CIE Lab 纯明度三频 + 浅色质感配方 | 新增 **`pastel-lace` / `translucent-organza` / `satin-ribbon` / `knit-stocking`** 纯物理配方，浅色衣装绝不误压暗，兼保黑衣不偏棕与 `--out-dark-mask` 透传 |
| `skin-unify` | 下肢褪色炫光肤色对齐 + 外缘去白边 + 裙/腿物理遮挡影 | 以手臂晴光肤色为参考，用归一化卷积在不破坏皮肤纹理的前提下恢复下肢膝盖血色与反光压制，清掉左缘 cutout 白边并叠黑裙与交叠腿物理投影 |
| `harmonize` | Apache-2.0 [`PhotoBlaze-PIH`](https://huggingface.co/JeremySD99/PhotoBlaze-PIH) ONNX CUDA（`curves.onnx` + `gain.onnx`） | 全分辨率 32 点调和曲线 + 空间光照增益 + 肤色与 `--garment-dark-mask` 全黑衣防回灰保护 + 双半径光学 `Light Wrap` + 颗粒匹配层 |
| `relight-sculpt` | Apache-2.0 `Depth-Anything-V2-Small` ONNX 2.5D 法线重打光（**零人物像素重绘**） | 输出 4 张独立非破坏 PhotoCraft 剪贴层（背光体积暗面 `Multiply` / 迎光键光 `Screen` / 菲涅耳逆光包裹 `Screen` / 地面漫反射 `Soft Light`），重塑 4:1–6:1 体积光影 |
| `cinema-grade` | Blender 5.2 `AgX` / 2383 标准 `.cube` 3D LUT + PhotoCraft 原生 `filter.cameraRaw` / `colorLookup`（`tetrahedral`） | `bg-anchor` 压实背景非光区黑位并产出近景暗框景深遮挡；`grade` 执行电影冷暖分色阶、四面体 3D LUT 与严格来自实景高光的微光晕 |
| `fg-vfx-3d` | **Blender 5.x GPU 3D 物理前景散景 + 贴身流光音律丝带** | 用真实 `f/1.18` 大光圈渲染近景虚化铁艺/花叶/暖金折射晶滴（实体遮挡底边并拉出纵深），并产出“后穿前绕”贴身丝带及受 2.5D 法线调制的局部因果反光与拱门背光边缘光 |
| `tile-split` / `tile-stitch` | 重叠分块与低频锁光影 + 亚像素 ECC 无缝拼合 | 把低分建筑/场景底片按 2×2 分块高清重绘后再按低频锁定回拼，突破 `0.6` 原生分辨率门禁 |
| `inpaint` | `LaMa`（`big-lama.pt`，Apache-2.0）仅裁剪 Mask ROI 推理，**掩码外零改动** | 加 `--patch-only` 输出透明局部补丁 PNG，直接用 `doc_place` 叠进 PhotoCraft 当非破坏修复层 |
| `upscale` | `Real-ESRGAN-ncnn-vulkan`（MIT，Vulkan GPU），支持 `--scale 2\|4` 与 `--style photo\|anime` | `--style photo` 压制蜡质油画锐化脊线；对 `<1024px` 跑 `4x` 自动警告并提示走 `tile-split/stitch` |
| `relight` | （可选离线扩展）隔离 `venv-gen` 运行 `IC-Light`，精简环境默认跳过 | 常规重打光与溶图改走 PhotoCraft 反曲线 / Light Wrap 或内置图片模型 |
| `painterly` | （可选离线扩展）隔离 `venv-gen` 运行 `Counterfeit + Tile`，精简环境默认跳过 | 常规厚涂/飘发与材质重绘改走 PhotoCraft + 内置图片模型 + 锁罩回贴 |
| `qa` | 高光溢出 / 死黑 / 高饱和比例 / 清晰度 + 可选 `--skin-mask` Lab 均值 | 快速定位脸/臂肤色偏冷发灰（查 `b*`）与曝光溢出 |
| `compare` | 生成带全图 + `1:1` 原像素裁切对比的拼图 PNG | 视觉自检前后变化与细节 |

> **如何把侧车蒙版/补丁导入 PhotoCraft**：
> 1. 蒙版：用 `doc_place` 把 `mask_*.png` 临时置入、或用 `file.open` 打开后 `select.all` → `edit.copy` / `select.loadSelection` 载入为目标图层/组蒙版，确保磨皮或重绘严格避开眼眉唇；
> 2. 液化冻区：直接读取 `faces.json` 中的 `liquify_freeze_polygons` 点位传入 `filter.liquify` 的 `lassoMask` 笔画，开推前冻住五官；
> 3. 局部消除：优先 `inpaint --patch-only` 生成透明补丁层并用 `doc_place` 置入，保留可开关和蒙版擦回余地。

### 2.4 3D 物理占位、白模建景与真投影侧车（Blender）
- **脚本套件**：`scripts\blender_scene_kit.py`（`retouch-ai.py scene-whitemodel`）、`scripts\blender_fg_vfx_kit.py`（`retouch-ai.py fg-vfx-3d`）与 `scripts\blender-shadow-catcher.py`。
- **职责边界（Blender 白模与物理前景/环绕光效 → 用户 AI 抽背景与黑底特效 → PhotoCraft 纯手修总装）**：
  - `scene-whitemodel` 一键导出 6 类高精架构白模空镜 + 16-bit Z 深度 + `scene_meta.json` + 配套空镜/黑底特效抽卡提示词；
  - `fg-vfx-3d` 用 Cycles GPU 真实大光圈景深（`f/1.18`）渲染实景大虚化前景暗框（雕花铁艺、暖金透明折射晶滴消除空背景暗环、背光虚化玫瑰花叶实体将主体夹入空间纵深并遮挡底边截断），并生成围绕人物身姿“后穿前绕”的三维流光丝带与对人物邻近皮肤/衣袖的因果实光反射（`vfx_wrap_causal_spill_screen.png` + `subject_arch_backrim_screen.png`）；
  - 用户用白模在外部高清图模抽回空镜背景、并用纯黑底（`#000000`）提示词抽回特效叠层；侧车经 `cutout-refine`（完整发丝精抠与边缘色去污）、`vfx-overlay`（能量守恒 Unmult + `--density-gamma` / `--haze-suppress` 克制雾斑与主体前景核心避让）、`relight-sculpt` 与 `cinema-grade` 进 PhotoCraft 完成纯非破坏总装。

## 3. 常用修图动作 → PhotoCraft 命令速查

> 调用任何命令前，不确定的可选参数以 `command_list` 或 `photocraft-cli commands --filter <id> --json` 现场输出为准。注意不同命令的透明度量纲差异：
> - `layer.new.layer`、`layer.newAdjustmentLayer.*`、`layer.setProps`、`paint.stroke` 使用 **`0..1`**；
> - `layer.new.group` 使用 **`0..100`**；
> - `paint.dodge`、`paint.burn`、`paint.smudge`、`paint.spotHealing`、`paint.cloneStamp`、`paint.remove` 使用 **`1..100`**。

| 修图环节 | 常用 PhotoCraft 命令 ID | 关键注意点 |
| --- | --- | --- |
| 显影与调色 | `filter.cameraRaw`、`layer.newAdjustmentLayer.curves` / `hueSaturation` / `selectiveColor` / `colorBalance` / `levels` / `colorLookup` / `gradientMap`、`edit.assignProfile`、`edit.convertToProfile` | `curves` 支持 `points` 及通道分别调；区分“指定”与“转换”配置。 |
| 分频与图层管理 | `image.applyImage`、`filter.blur.gaussianBlur`、`filter.other.highPass`、`layer.new.layer`（`fillNeutral: true` 可建中性灰柔光层）、`layer.new.group`、`layer.moveTo`、`layer.setProps` | `layer.new.layer` 的 `opacity` 为 `0..1`，`layer.new.group` 为 `0..100`。 |
| 选区与蒙版 | `select.subject`、`select.refineEdge`（支持 `decontaminate: true`）、`select.loadLuminosity`、`select.rect`、`select.lasso`、`select.colorRange`、`select.modify.feather`/`expand`/`contract`、`layer.layerMask.revealSelection`/`hideSelection`/`revealAll`/`hideAll`、`layer.matting.colorDecontaminate`/`defringe`/`removeBlackMatte`/`removeWhiteMatte`、`layer.vectorMask.fromPath` | `select.subject` 只是初始蒙版，可先跑 `matte` + `body-parse` 补强；去杂边优先 `colorDecontaminate`/`defringe`，勿猛缩边缘。 |
| 绘画与厚涂 | `paint.stroke`（支持 `target: "pixels"\|"mask"` 与 `fill: true`）、`paint.mixerBrush`、`paint.dodge`、`paint.burn`、`paint.smudge`、`path.stroke`（支持 `simulatePressure: true` 模拟压感描发丝）、`brush.presets.importAbr` | 混色涂抹前确认 `sampleAllLayers`，防皮肤/衣色串入头发。 |
| 修复与清理 | `edit.contentAwareFill`、`paint.healingBrush`、`paint.spotHealing`、`paint.patch`、`paint.cloneStamp`、`paint.remove` | 跨材质或大块路人优先 `retouch-ai.py inpaint --patch-only` + `doc_place`。 |
| 塑形与液化 | `layer.smartObjects.convertToSmartObject`、`filter.liquify`（`forwardWarp`/`reconstruct`/`smooth`/`freeze`/`thaw`/`lassoMask`）、`edit.puppetWarp`、`edit.perspectiveWarp`、`edit.transform.warp`、`filter.distort.displace` | 先转智能对象再跑 `filter.liquify`，可用 `faces.json` 的 `lassoMask` 冻住五官；四点透视与水波倒影用 `perspectiveWarp` / `displace`。 |
| 合成、溶图与光效 | `doc_place`、`layer.createClippingMask`、`layer.layerStyle.blendingOptions`（`blendIf` / `fillOpacity` / `channels`）、`image.adjustments.matchColor`、`filter.blur.average`、`filter.blur.lensBlur`（`depthMap`）、`filter.blurGallery.fieldBlur`/`tiltShift`、`filter.render.relight`/`lightingEffects`、`filter.noise.addNoise` | 建 `CHECK LAYERS` 查黑位/饱和度/色偏；反曲线打光保质感；`linearDodge` 发光降 `fillOpacity`；颗粒加在最顶层。 |
| 锐化与快照 | `filter.sharpen.unsharpMask`、`filter.sharpen.smartSharpen`、`history.newSnapshot`、`history.selectSnapshot` | 批量改动前先 `history.newSnapshot`；超分图不默认再叠锐化。 |

## 4. 按需查阅参考手册

| 场景 | 读取文件 |
| --- | --- |
| 场照（`field`）/ 正片（`formal`）标准流程、美颜前置门禁与常见翻车排查 | [工作流程](references/workflows.md) |
| **人像与 COS 骨相液化实操（五大原则、冻区、照镜子验收、重跑蒙版）** | [液化指南](references/liquify-guide.md) |
| **商业自然精修与奶油瓷白肌调色（硬门禁、高低频校准、3 步不偏灰调色）** | [自然精修与奶油肌](references/natural-beauty-guide.md) |
| **服饰分层精修与去雾保色（纯明度三频分离、防黑衣发棕、分材质配方与 `garment-retouch`）** | [服饰精修指南](references/garment-retouch-guide.md) |
| **COS 分层画发丝实战（3 笔刷、7 步原发延展、`Linear Dodge (Add)` 飞丝毛流感与 `hair-paint`）** | [分层画发丝指南](references/hair-painting-guide.md) |
| **COS 与人像专业合成与溶图（检查层、黑位、Blender 极简白模+Shadow Catcher、双轮生图接地、反曲线光与颗粒）** | [合成与溶图指南](references/compositing-guide.md) |
| **调色、光影重塑与同层溶图（去泥巴黄、冷暖分离、4:1 重打光与锐度匹配）** | [调色与光影指南](references/color-lighting-guide.md) |
| 二次元厚涂 COS（反贴纸眼唇、藏色发束与 PhotoCraft+强图模+锁罩回贴 SOP） | [厚涂方法卡](references/painterly-methods.md) |
| 显影色彩、高低频、选区修边、液化冻区、合成透视与输出避坑细则 | [技术与避坑细则](references/techniques-and-pitfalls.md) |
| COS 12 类技能学习地图、低风险对照实验与公开教程索引 | [学习路线与索引](references/learning-roadmap.md) |
