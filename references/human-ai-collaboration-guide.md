# 人机协同与白模抽卡硬门禁规范（Human-AI Collaboration & Hard-Stop Handoff）

面向需要替换场景、重塑氛围光影或融入复杂粒子特效的 COS 正片后期与 2.5D 厚涂流程，明确 **AI 与用户各自的职责红线、四道硬门禁以及白模/黑底抽卡标准交接格式**。

---

## 一、核心分工铁律（严禁私自代抽）

1. **人物像素纯本机精修（100% Pure-PC）**：人物肤色、五官妆容、假发主形与服饰材质均通过本机 LightCraft + PhotoCraft + 确定性数学算法与语义蒙版（`skin-fs` / `makeup-paint` / `body-liquify` / `garment-retouch` / `hair-paint` / `relight-sculpt`）完成，**绝对禁止对人物像素做 AI 重绘**。
2. **背景空镜与纯黑底特效归用户外部抽卡**：
   - **默认硬禁令**：除非用户在当前明确对话里写出“你帮我直接生成 / 抽卡 / 先出图”等明确授权，**严禁私自调用 `image-generator` 或内置生图能力越权替用户生成背景图或特效图！**
   - Agent 的职责是**先造好白模与规范提示词并彻底停下来交给用户**，待用户选定并发回自己满意的高清背景图与黑底特效图后，再执行溶图、多光源重打光与因果特效合成。

---

## 二、四道强制人机协同门禁（Gates 0–3）

### Gate 0｜立项一轮对齐（Pre-Flight Alignment）
开工前确认且不擅自脑补三件事（用户未说明时用简短直白选项集中确认）：
1. **成品画风**：摄影级写实正片 vs 2.5D 半厚涂板绘风；
2. **妆容取向**：自然清透保真（保原瞳原睫） vs 角色动漫 COS 妆（指定瞳色/唇色/眼尾层次）；
3. **场景与抽卡边界**：由工作台产出透视白模/构图线与提示词，交由用户外部高清图模抽卡后回传（或用户已有现成参考/底图）。

### Gate 1｜Stage-1 美颜与妆容 1:1 验收门禁（Beauty-First）
完成 `skin-fs` + `makeup-paint`（[光学补妆精修圣经](makeup-retouch-guide.md)）+ `garment-retouch` 后，先导出 `01_beauty_base.png` + `.pcraft` 及 **「原图 vs 妆后」独立 `<=512px` `1:1` 对照图**。在用户视角或目视核验过关前，**绝不拿未定妆或有瑕疵的底图往下跑 4K 合成**。

### Gate 2｜Stage-2 硬暂停交接门禁（Hard-Stop Handoff）
跑完 `retouch-ai.py scene-whitemodel`（或定制 Blender 道具/护栏透视白模）及 **`retouch-ai.py handoff-pack`** 后，**必须立刻暂停工具循环、停止继续合成**，向用户完整交付以下 4 项后再等待回图：
1. **白模底图与构图参考路径**（必须带可点击 `file:///` 链接 + 纯本地路径）：
   - 严格无人的建筑/地形透视白模 `01_blender_whitemodel_empty.png`（说明：抽空镜背景时**只上传这一张白模作为参考垫图**）；
   - 含人物站位、栏杆/地平线与主光箭头的构图参考 `03_layout_and_light_guide.png`（明确注明：**仅供对照机位和光向，千万不要把带有半透明人物的布局图上传给生图模型**）。
2. **可直接复制的中英双语《空镜背景抽卡提示词》**：明确要求锁定白模机位/视平线/透视/主光方向、输出长宽比与最低原生分辨率（16:9 建议 `>= 2560×1440`，避免低清超分模糊）、**严格无人、无剪影、无文字、无大片死白糊雾**。
3. **可直接复制的中英双语《纯黑底特效抽卡提示词》**：**醒目注明“切勿垫入场景白模图”**，严格要求 `#000000` 纯黑底色、围绕主体四周留出充足黑底负空间、面部与中心主体区克制稀疏（不挡脸眼）、无建筑地面与文字。
4. **回传指引清单**：请用户回传 ① 选定的高清空镜路径、②（可选）黑底特效路径、③ 偏好的写实/厚涂定调。

### Gate 3｜回图验真 → 接触对齐 → 因果光影收束
收到用户回图后，依次执行：
1. 用 `inspect-image.mjs inspect` 核验真实分辨率、文件格式与原景光向；若用户在场景里画出了新的栏杆/桌面/台阶斜率，用最小仿射微调将人物手/脚接触点精确落位到实际物体支撑面上；
2. 先跑 **`retouch-ai.py relight-sculpt --delight-strength 0.65 --garment-mask ... --skin-mask ...`**（去掉原场馆平闪杂黄 + 从用户背景图自动或手动解析多光源重塑 4:1 立体感与彩色副光）；
3. 再跑 **`retouch-ai.py vfx-overlay --normal-map ... --fg-rgba ... --garment-mask ... --bg ...`**（按 `Density-Gamma` 与 `Haze-Suppress` 能量守恒拆前/后景，并用 2.5D 法线点乘特效能量场自动算出落在人物邻近衣摆/发丝/肢体与周边建筑上的真实因果反光）；
4. 最后进 PhotoCraft 总装调色、统一颗粒并跑 `mask-audit --visual-reviewed` 与独立 `1:1` 切片终检。

---

## 三、`handoff-pack` 工具用法与状态锁定文件

`retouch-ai.py scene-whitemodel` 结束时会自动调用 `handoff-pack`，也可在自定义白模后单独调用：

```powershell
& "$env:LOCALAPPDATA\RetouchTools\venv\Scripts\python.exe" "D:\ps\retouch-workbench\scripts\retouch-ai.py" handoff-pack `
  --out-dir "work\handoff" --project-name "Project_4K" --character "角色名" `
  --whitemodel "work\whitemodel\01_blender_whitemodel_empty.png" `
  --subject-rgba "work\cutout\subject_rgba.png" `
  --key-light-side top-left --prop-line 0.0,0.80,1.0,0.44 --overwrite
```

- 生成文件：
  - `HANDOFF_FOR_USER.md`：排版整洁的中文提示词 + 英文提示词 + 上传避坑须知卡片；
  - `handoff_manifest.json`：结构化交接锁文件（内含 `"status": "AWAITING_USER_ASSETS"` 与 `"allow_agent_self_gen": false`）；
  - `03_layout_and_light_guide.png`：含人物透视位置、支撑线标注与清晰主光箭头的可视化参考图。
