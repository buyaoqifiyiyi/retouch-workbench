---
name: retouch-workbench
description: 基于开源引擎 PhotoCraft + LightCraft（原生 MCP / CLI）及本地开源 AI 侧车（uniface / BiRefNet / SegFormer / Depth-Anything-V2 / MediaPipe / LaMa / PhotoBlaze-PIH / Real-ESRGAN）与 Blender 3D/OCIO .cube 电影 LUT 的本机人像与 COS 后期修图、2.5D 重打光、3D 前景环绕特效与方法学习工作台技能；适用于场照返图、角色正片、二次元厚涂、局部修复与影视级合成。
---

# Retouch Workbench · COS 与人像后期工作台（纯开源双引擎 + AI 侧车栈）

在当前会话接收修图需求与本机素材路径，优先通过已接入的 **LightCraft MCP**（批量 RAW 暗房/组图统一）与 **PhotoCraft MCP**（单张分层精修/厚涂重绘）配合本地宽松开源 AI 侧车 `retouch-ai.py` 执行非破坏编辑。

## 1. 核心原则与防呆护栏

1. **人像零 AI 重绘 · 协同边界铁律（Pure-PC Person Retouch × AI Empty BG & Black-BG VFX）**：
   - **人物像素严禁 AI 重绘**：所有面部、五官、皮肤、妆容、发丝与服饰精修，必须 100% 使用本机 **LightCraft + PhotoCraft** 原生工具/画笔/液化/三频/D&B 与纯确定性数学算法完成（`retouch-ai.py skin-fs` / `makeup-paint` / `body-liquify` / `garment-retouch` / `hair-paint`）。本地神经模型**严格仅用于生成黑白选区蒙版、透明通道、去杂边光解算与关键点引导**（如 `face-parse` / `body-parse` / `cutout-refine`），绝不重绘或涂抹改变人物原生皮肤与材质像素。
   - **外部 AI 生图（严格归用户抽卡 · 硬暂停门禁）**：① 按 `scene-whitemodel` + `handoff-pack` 输出的无人白模底图与提示词抽出的**严格无人高清空镜背景**；② 按纯粹黑底（`#000000`）提示词抽出的**复杂梦幻粒子/光斑/流光特效叠层**。**硬性边界：除非用户在当前明确指令“你帮我直接生成/抽卡”，严禁私自调用 `image-generator` 替用户生成背景或特效！**生成白模与 `HANDOFF_FOR_USER.md` 后必须硬暂停交给用户，详见 [人机协同与白模抽卡规范](references/human-ai-collaboration-guide.md)。
2. **原图绝对只读**：绝不覆盖原图或用户已有交付文件。所有工程文件（`.pcraft` / `.psd`）、侧车输出与交付副本统一写入 `D:\ps\<项目子目录>\`。
3. **接图必先验真**：开修前运行检查脚本核对文件头真实编码（防范 JPEG 改名 `.png`）、像素尺寸、位深、色彩配置与图层树：
   ```powershell
   node "D:\ps\retouch-workbench\scripts\inspect-image.mjs" inspect "<原图路径>"
   ```
4. **先修美颜底图与原生分辨率硬门禁（Beauty-First & Resolution Gate）**：人像正片、美颜精修或 COS 二次元厚涂前，必须严格按 `LightCraft 暖润安全底 → 骨相液化+重跑蒙版 → 瑕疵/超分黑线与眼区穿帮杂发清理 → 校准高低频与D&B → 3步奶油瓷白肌（全臂连续蒙版） → 自然非贴纸眼唇` 先产出并 **独立单张真 `1:1`（`<=512px`）** 验收 **`01_beauty_base.png` + `.pcraft`**，再对服装与下肢依次运行 **`garment-retouch --out-dark-mask`**（纯明度空间去雾+外缘半透明去白边，并把含鞋袜/兜底黑鞋的完整深色服装罩透传给 `harmonize --garment-dark-mask`）与 **`skin-unify`**（下肢褪色膝盖对齐手臂晴光肤色、左缘去白边并叠黑裙/交叠腿遮挡影），才允许继续换背景或合成。**防糊斑红线**：严禁在背景抠图前用 `LaMa` 大面积抹除主体轮廓旁的衣物飘带/发丝交界区（`LaMa` 会把前景与旧背景涂成不透明灰白糊斑并被 `BiRefNet` 锁进主体前景）；如需清理边缘杂物应先保留原物或在换背景后于独立非破坏层处理，复核可疑灰边时必须直接对照最原始输入图（`00_input`）。**原生分辨率门禁**：背景或大型贴片素材相对目标区域的线性原生比必须 `>= 0.6`；严禁把 `<= 1K` 生图盲目 `realesrgan 4x` 直接当成 4K 背景/发片使用，低清底图必须走 **`tile-split` → 分块高清精绘 → `tile-stitch`（低频锁光影 + 亚像素对齐）** 路线消除油画伪影。
5. **按真实画风选头发路线 + 零间隙遮挡铁律（Style-Adaptive Hair Gate）**：
   - **写实人像/实拍风 COS 轻量保原形**走 **`hair-comp --mode real-first --groom`**（结构张量切向轻量理顺外侧炸毛、保留 `82%` 原生微发丝并回暖漂白发斑）；需要轻量补充空气毛流感走 **`hair-paint`**（PhotoCraft 原生渐细单丝束 `02b_HAIR_PAINT_BEHIND_4K` + `03b_HAIR_PAINT_FRONT_4K`）。
   - **兼顾真实“一根一根”1:1 发丝质感与动漫整头融合美学（推荐）**直接走 **[头发后期指南](references/hair-painting-guide.md)** + **`hair-sculpt --engine anime-sculpt`（默认 v11 · 全头无折叠 `Rigid MLS` 微托冠 + 亚像素解析逐丝微簇聚散引擎 + 发根锚定实拍采样色 + 外部镂空散光包裹与自然 `85mm` 景深）**，彻底取消生硬宽塑料带，改为 `0.8–2.2px` 独立微丝按“2–3 股有聚有散小撮 + 斜向真透气缝隙 + 长短错落丝梢 + 高密交叠细丝中轴”构造（保留 `--engine tutorial-rake` 与 `--engine blender` 作可选回退）；按发型（`long-cascade` / `twintails` / `short-braids-wispy`）输出深埋发根、避让腕饰/衣领纯白反光/黑裙的零侵入分层 `.pcraft`；**严禁用整块宽色带（一坨/塑料管）、等距平行梳齿线（蜘蛛网/面条）、点状锯齿描边或非预乘白边破坏实拍质感**；
   - 无论哪种路线，必须保留原图真实五官、眼区 `0px` 零侵入，并以引导滤波对齐紧贴遮挡物，严禁糊状扩边把跨臂/跨腕发丝切断或在深色衣装留下灰雾。
6. **关键修改前必留快照**：PhotoCraft 的 `command_batch` **不是原子事务**（中途失败时已完成的前序步骤会保留，不会自动回滚）。每轮批量修改前先调用 `history.newSnapshot` 留底；失败或跑偏时用 `history.selectSnapshot` 或 `edit.undo` 回退。
7. **真 `1:1` 独立单切图 + 自动掩码体检 + 真实看图铁律**：
   - 整图光影与大形：`doc_render_preview`（最大边 2048）。
   - **严禁把多张放大细节缩拼进一张千像素拼图当作 `1:1` 验收**，更**严禁未亲眼看到图片画面就声称视觉通过（`--visual-reviewed` 仅在确实看图审阅后才能启用，否则 `verdict` 停在 `NUMERIC_PASS_VISUAL_UNVERIFIED`）**。**严禁只盯脸部漏审身体**：每次验收必须按**全身四区**（① 面部眼唇与手腕/发际交界、② 躯干手臂与手指肤色清晰度、③ 胯部深色服饰内腔与接触边缘、④ 下肢靴鞋与前景特效交界）逐区审阅 `<=512px` 独立 `1:1` 切片 + `retouch-ai.py qa`。导出前必跑 **`retouch-ai.py mask-audit`**（零破洞、零半透明残影、手臂遮挡零断口、中性黑衣零偏棕、织纹与全身皮肤微反差留存、`mask_sanity_check` 零溢出与未认领蜂窝洞、人景融合、热点 `1:1` 切片；传入 `--skin-mask` 及可选 `--face-skin-mask`）。
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
| `makeup-paint` | **MediaPipe 478 点 + PhotoCraft 多混合模式（`Multiply`/`Soft Light`/`Color`/`Screen`）+ 原生 `Blend-If` 光学补妆引擎 v2** | 支持 `natural` / `editorial` / `cos-anime` / `painterly` 四档通用画风、闭眼自适应与原色保留，导出 9 层零贴纸光学妆层（泪沟鼻D&B/眼白/眼影深度与发色/虹膜上睑压影与折射/贴根眼线与真睫强化/腮红/PiXimperfect 三层护纹唇妆） |
| `body-liquify` | **MediaPipe 33 骨骼 + 轮廓曲率袜口检测 + PhotoCraft `filter.liquify` 智能滤镜** | 冻住五官后自动压顺斜方肌、手臂、腰线及大腿袜口勒痕并输出可编辑 `.pcraft` 与照镜子对比图 |
| `vfx-overlay` | **能量守恒 Unmult + 2.5D 法线因果重打光引擎** | 专接用户抽卡的黑底 AI 特效：`a_behind + a_front <= alpha` 前后拆层、`--density-gamma` / `--haze-suppress` 控雾；细肢体前景仅放行锐利细星点（拦截大虚化光斑糊脚踝），菲涅尔边缘光严格锁死外轮廓（零内部白雾）；接入 `--normal-map` / `--fg-rgba` / `--garment-mask` / `--bg` 自动算 **2.5D 因果漫射 (`Screen`) + 近距锐高光 (`Linear Dodge`) + 周边建筑环境溢光 (`Screen`)** 且零眼区/片名遮挡 |
| `scene-whitemodel` | **Blender 5.x GPU 6 大场景白模工厂 + 自动出交接包** | 一键出纯净白模（供用户 AI 抽空镜）+ 16-bit 物理 Z-Depth + `scene_meta.json` + 自动触发 `handoff-pack` 产出交接卡片并锁定 `AWAITING_USER_ASSETS` |
| `handoff-pack` | **白模抽卡与构图硬暂停交接打包器** | 输出可复制双语空镜/纯黑底特效提示词（`HANDOFF_FOR_USER.md`）、状态锁 `handoff_manifest.json`（禁止擅自自生成）与含支撑线/主光箭头的构图参考 `03_layout_and_light_guide.png` |
| `body-parse` | 默认 **`SegFormer-B2 Clothes` 18 类 ONNX CUDA + 引导滤波**（MIT），回退支持 `MediaPipe`（`--backend`） | 精准拆分 `mask_hair` / `mask_upper_clothes` / `mask_skirt_dress` / `mask_shoes` / `mask_legs` / `mask_arms` 与受保护整体掩码 `mask_protected_body`，并产出 `preview_parse_overlay.png` |
| `mask-audit` | 工程级掩码、**全身皮肤清晰度**、**参考比对高调/漆皮梯度**与 **`cinema_tone_check`** 自动体检 | 检测破洞、残影、断发微粒、边缘色偏、**全身皮肤微反差塌陷（`body_skin_clarity_check` 接入 `--skin-mask`/`--face-skin-mask`）**、**服饰溢出主 Alpha 与内部蜂窝破洞（`mask_sanity_check`）**及黑位动态跨度；支持 `--reference` 比对漆皮真实反光梯度防误杀，支持 `--scene-meta` 校验太阳光向 |
| `hair-comp` | 自适应风格头发合成与切向防糊理顺（`SegFormer` + `BiRefNet` + 结构张量门控 + CIE Lab） | 写实正片 `real-first --groom` 保原生 4K 编发与真发丝高频微理顺；兼顾紧贴遮挡与防漂白发白斑 |
| `hair-paint` | PhotoCraft 原生极细单丝束前后分层轻量飞丝（`brush.presets.importImage` + `paint.stroke` 方向压感） | 永不删原发、局部自吸真发色，输出后景顺流层 + 前景 `Normal` / `Linear Dodge (Add)` 毛流感层 |
| `hair-sculpt` | **Whole-Head Analytical Strand Engine（默认 `--engine anime-sculpt` v11：无折叠 `Rigid MLS` 整头提廓 + `0.8–2.2px` 亚像素解析微簇发丝聚散 + 外部透空光晕包裹 × `85mm` 景深）**；可选回退 `tutorial-rake` / `blender` | 告别“一坨”硬塑料片与平行面条，以真正可数的一根一根微丝聚成兼顾发量实感与斜向透气的动漫飘发（`long-cascade`/`twintails`/`short-braids-wispy`），一键装配零扩边遮挡与安全避让的 `.pcraft` 分层堆栈 |
| `garment-retouch` | `SegFormer` 语义分服 + CIE Lab 纯明度三频 + 浅色质感配方 | 新增 **`pastel-lace` / `translucent-organza` / `satin-ribbon` / `knit-stocking`** 纯物理配方，浅色衣装绝不误压暗，兼保黑衣不偏棕与 `--out-dark-mask` 透传 |
| `skin-unify` | 下肢褪色炫光肤色对齐 + 外缘去白边 + 裙/腿物理遮挡影 | 以手臂晴光肤色为参考，用归一化卷积在不破坏皮肤纹理的前提下恢复下肢膝盖血色与反光压制，清掉左缘 cutout 白边并叠黑裙与交叠腿物理投影 |
| `harmonize` | Apache-2.0 [`PhotoBlaze-PIH`](https://huggingface.co/JeremySD99/PhotoBlaze-PIH) ONNX CUDA（`curves.onnx` + `gain.onnx`） | 全分辨率 32 点调和曲线 + 空间光照增益 + 肤色与 `--garment-dark-mask` 全黑衣防回灰保护 + 双半径光学 `Light Wrap` + 颗粒匹配层 |
| `relight-sculpt` | Apache-2.0 `Depth-Anything-V2-Small` 2.5D **去场馆平光 × 多光源场景解析重打光**（零像素重绘） | **去室内平光反光**（保高频毛孔、`relight_00`）+ 4 张方向重塑层 + **自动解析背景彩色霓虹/灯源（或 `--accent-light`）出漫射/高光/深黑护色 3 层**，并提供 PhotoCraft 原生装组与鼻斜坡法线保护 |
| `cinema-grade` | Blender 5.2 `AgX` / 2383 标准 `.cube` 3D LUT + PhotoCraft 原生 `filter.cameraRaw` / `colorLookup`（`tetrahedral`） | `bg-anchor` 压实背景非光区黑位并产出近景暗框景深遮挡；`grade` 执行电影冷暖分色阶、四面体 3D LUT 与严格来自实景高光的微光晕 |
| `output-sharpen` | **线性亮度 Richardson-Lucy 反卷积 + 非对称防白边 + 四级语义焦点蒙版**（纯本机零 AI 重绘） | 置于顶层颗粒正下方：只提眼瞳/唇线/发丝/衣料与前景结构锐度，严格避开奶油光滑肤质、扣像边缘环与远景散景 |
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
| **人机协同分工、四道硬门禁、白模提示词打包与回图交接规范（必读防越权）** | [人机协同与交接规范](references/human-ai-collaboration-guide.md) |
| **人像与 COS 骨相液化实操（五大原则、冻区、照镜子验收、重跑蒙版）** | [液化指南](references/liquify-guide.md) |
| **商业自然精修与奶油瓷白肌调色（硬门禁、高低频校准、3 步不偏灰调色）** | [自然精修与奶油肌](references/natural-beauty-guide.md) |
| **商用与 COS 光学补妆精修圣经（PiXimperfect 三层唇妆、`Blend-If` 护纹、真眼光学压影与 4 档画风）** | [光学补妆精修圣经](references/makeup-retouch-guide.md) |
| **服饰分层精修与去雾保色（纯明度三频分离、防黑衣发棕、分材质配方与 `garment-retouch`）** | [服饰精修指南](references/garment-retouch-guide.md) |
| **COS 头发后期指南（真实发丝质感 × 大胆动漫飘逸感：`hair-sculpt` / `hair-paint` / `hair-comp` 完整工作流与避坑实证）** | [头发后期指南](references/hair-painting-guide.md) |
| **COS 与人像专业合成与溶图（检查层、黑位、Blender 极简白模+Shadow Catcher、双轮生图接地、反曲线光与颗粒）** | [合成与溶图指南](references/compositing-guide.md) |
| **调色、光影重塑与同层溶图（去泥巴黄、冷暖分离、4:1 重打光与锐度匹配）** | [调色与光影指南](references/color-lighting-guide.md) |
| 二次元厚涂 COS（反贴纸眼唇、藏色发束与 PhotoCraft+强图模+锁罩回贴 SOP） | [厚涂方法卡](references/painterly-methods.md) |
| 显影色彩、高低频、选区修边、液化冻区、合成透视与输出避坑细则 | [技术与避坑细则](references/techniques-and-pitfalls.md) |
| COS 12 类技能学习地图、低风险对照实验与公开教程索引 | [学习路线与索引](references/learning-roadmap.md) |
