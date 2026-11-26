# COS 分层画发丝实战指南（PhotoCraft 原生 3 笔刷工作流）

基于实操 COS 后期画发丝教程逐帧拆解（Procreate 3 笔刷分层画法迁移至本机开源 **PhotoCraft** 原生画笔引擎 + `retouch-ai.py hair-paint`），彻底解决“用低清 AI 假发片贴出油画假壳”和“用硬直线描出铁丝感”的两大顽疾。

---

## 一、核心心法（4 条铁律）

1. **永不删原发，只在原发基础上“原生延展 + 分层叠加”**：100% 保留原片真实的头顶发根、刘海、肩颈与跨手臂真发，绝不用整片 AI 假发直接覆盖替换；新增飘发必须从真实发区内部（向内缩进 `20–45px`）顺势长出，杜绝悬空浮丝。
2. **只吸取原发自身已有的深/中/浅色（Eyedropper-Only 和谐律）**：严禁凭空捏造高饱和荧光色或偏色；底色吸取原发暗部（Lab `L*` 约 `p14–p26`），铺实用中调（`p42–p56`），卡亮面吸原发亮部（`p72–p84`），发光飞丝沿原发亮部色相克制提亮。
3. **三支渐细笔刷各司其职，且走向遵循统一风向又允许“不那么听话”**：先宽束铺暗底走向，再中束穿插并补刘海飘逸碎丝，最后单根极细笔开 `Linear Dodge (Add)` 补逆光飞丝与头顶/刘海**毛流感**；所有飘发统一顺应场景风向，但次级曲线带适度弧度差异与穿插，绝不画成机械平行的梳齿。
4. **软橡皮淡出过渡 + 零扩边紧贴擦出前景遮挡**：高光与飞丝两端必须随压感自然尖细淡出（必要时软橡皮轻擦减淡）；最后用 `0px` 零扩边紧贴蒙版把压在头发前面的手、手腕、手臂、肩膀与衣服干净擦出，绝不留糊状白边或断发槽。

---

## 二、3 支标准发丝笔刷与 PhotoCraft 引擎映射

永久笔刷笔尖资产位于 `D:\ps\retouch-workbench\resources\brushes\`（灰度黑墨白透，可直接经 `brush.presets.importImage` 载入 PhotoCraft）：

| 笔刷角色 | 本机笔尖文件 | PhotoCraft 动态参数与 4K 实拍经验 | 负责环节 |
| --- | --- | --- | --- |
| **1. 多股概念笔刷（可选底形）** | `hair_01_multi_clump.png`（6 颗独立细圆点） | `shapeDynamics.angle="direction"`、`size="penPressure"`、`spacing=0.035`；小间距连续拖动易融合，适合手绘铺大走向 | 手绘暗底大形与大块发束起底 |
| **2. 细发束穿插笔（可选次级）** | `hair_02_fine_clump.png`（3 颗不对称细圆点） | 同上，尺寸约主笔 `45–55%`；4K 下慎防平扫出扁带 | 人工手绘次级弧线穿插与碎丝 |
| **3. 极细单丝/聚簇主力笔（`hair-paint` 默认核心）** | `hair_03_single_strand.png`（单根柔边双端尖细笔） | `size = 1.8–4.5px`、`size_min = 0.02`；自动化按一锚点发射 2–4 根微分散渐细 Bezier 单丝束（贴合原发下落角度 + 局部外扬 S 弧），丝丝分明绝不成丝带 | 后景顺流飘发、前景空气碎丝、发梢与发冠 `Linear Dodge (Add)` 逆光飞丝 |

### Procreate ↔ PhotoCraft 混合模式与避坑对照
- `N`（正常） → **`Normal`**（主体后景柔流层、前景飘逸细丝层）
- `A`（添加） → **`Linear Dodge (Add)`**（极细发光飞丝与逆光毛流感层，透明度 `0.35–0.55` 防死白）
- `Cd`（颜色减淡） → **`Color Dodge`**（局部暖/冷逆光透射加强）
- `M`（正片叠底） → **`Multiply`**（发束交叠暗侧阴影）
- 橡皮擦擦淡 → `paint.stroke` 设 `mode: "clear"` 配合低不透明度软圆笔，或非破坏图层蒙版柔边轻擦。
- **4K 实拍铁律**：严禁跨胸/近颊乱拉横向铁丝线（必须剔除横穿脸颊与黑衣的侧向乱丝、并对面部做扩展防侵入保护）；取色改为**起笔周围真发中位色 65% + 全局暗/中/亮调 35%**，杜绝在浅金高光发上画出突兀深棕条。

---

## 三、标准 7 步流程（手绘逻辑 & `hair-paint` 自动实现映射）

1. **实拍真发先轻理顺（`hair-comp --mode real-first --groom`）**：不改原发大形与真实发丝细节，仅用结构张量切向轻量理顺外侧炸毛（`--groom-strength 0.35`），并把漂白发高光灰白反光柔和暖回丝绸金色（需精准区分白丝带/头饰与暖色漂白发，防止后者被漏跳过）。
2. **外轮廓+发梢+发冠动态起锚，拆入主体后景层（`02b_HAIR_PAINT_BEHIND_4K` · `Normal`）**：沿迎风外轮廓、底部发梢与发冠动态扫描锚点（绝不扎堆挤在腰带旁被衣服蒙版擦光），使柔和飞丝自然从黑腰带、肩颈与服装背面长出。
3. **前景飘逸细丝与微风 S 弧穿插（`03b_HAIR_PAINT_FRONT_4K` · `Normal`）**：用聚簇极细单丝束按原发下落切向衔接、在下半段与外缘轻柔外扬 `35°` 左右 S 形飞出轮廓，保持全脸扩展避让与眼区 `0px` 零侵入。
4. **新建添加层加极细发光飞丝与发冠毛流感（`03b_HAIR_PAINT_FRONT_4K` · `Linear Dodge (Add)`）**：吸取局部亮调并克制混暖光，从最外缘 `8–16px` 起笔向外甩出逆光单丝与发冠微风碎丝，在亮背景下清晰呈现丝绸毛流感。
5. **高光轻补 + 软橡皮淡出**：高光丝过亮或两端生硬时，用软橡皮轻擦淡出两端，与原发自然融为一体。
6. **轻量光学匹配（`0.35–0.42px`）**：仅用极轻光学柔化对齐 85mm 镜头焦平面（切忌 `>0.6px` 把细单丝糊软），并与后景空气透视和顶层颗粒同层融合。
7. **零扩边擦出前景遮挡物**：沿手部、手腕、手臂、肩颈与服饰边界以 `0px` 紧贴抗锯齿擦出前景遮挡，仅后景流层可流畅穿过腰带背面。

---

## 四、一键侧车调用（`retouch-ai.py hair-paint`）

```powershell
& "$env:LOCALAPPDATA\RetouchTools\venv\Scripts\python.exe" D:\ps\retouch-workbench\scripts\retouch-ai.py hair-paint `
  --subject-rgba "harmonize\harmonized_fg_rgba.png" `
  --hair-mask "body_parse_warped\mask_hair.png" `
  --occluder-mask "body_parse_warped\mask_occluder_arms_face.png" `
  --face-mask "body_parse_warped\mask_face_skin.png" `
  --clothes-mask "body_parse_warped\mask_clothes_all.png" `
  --wind-dir auto --intensity medium --out-dir "hair_paint" --overwrite
```

- `--intensity`：`subtle`（极轻点缀）、`medium`（默认推荐：清晰可见且自然融入实拍的飘逸毛流感与飞丝）、`dramatic`（大风长发厚涂风）。
- 默认沿迎风外轮廓、发梢底缘、发冠及外侧鬓角动态取锚（绝不扎堆挤在腰带旁被衣服遮罩擦光），输出 **前后分层** 的 3 张独立透明 4K PNG（后景顺流底层 `hair_paint_01_soft_flow_behind.png` 放于主体层之下，自然穿出腰带与肩颈背面；前景飘丝层 `hair_paint_02_airy_wisps_front.png` 与 `hair_paint_03_add_backlit_flow.png` 分别以 `Normal` 与 `Linear Dodge (Add)` 叠于主体上方 `03b_HAIR_PAINT_FRONT_4K`），配合 `hair-comp --groom` 轻理顺实拍真发，立出清晰但不改原发大形的飘逸毛流感。
