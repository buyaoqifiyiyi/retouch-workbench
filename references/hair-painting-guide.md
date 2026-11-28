# COS 头发后期指南（v11 逐根微簇真发丝 × 动漫整头融合美学）

本指南基于两段实操 COS 后期教程与多张高质感 COS 成片标杆，并在 3 组 4K COS 实拍基准案例（`case1_rose` 长金发、`case2_miku` 双马尾、`case3_nazuna` 夜景短发编发）历经 **多轮 1:1 五区独立微距审片与双零迎合盲审定分** 后定版（v11）。

在**零 AI 重绘人物像素**铁律下，主引擎采用 **`hair-sculpt --engine anime-sculpt`（默认 v11 Analytical Strand-by-Strand Micro-Clustered Engine）**：彻底废除早期的实心宽塑料带（避免“像一坨”与生硬边缘），改为以 **亚像素解析胶囊微丝 + 有聚有散的 2–3 股微簇（小撮） + 高密交叠细丝中轴 + 外部透空散光包裹** 来构造“一根一根真发丝”的动漫飘逸发型。

---

## 一、三档路线速选矩阵

| 场景需求 | 推荐命令 / 模式 | 核心定位 |
| --- | --- | --- |
| **写实人像轻微炸毛理顺** | `hair-comp --mode real-first --groom` | 保留 100% 原发与大形，仅以结构张量切向平滑外缘炸毛 |
| **轻中度纯手绘空气飞丝** | `hair-paint` | 沿迎风缘/发梢用 PhotoCraft 渐细单丝束加 `Normal` 飘丝与 `Linear Dodge (Add)` 逆光飞丝 |
| **真发丝感 × 动漫整头融合（默认主力）** | `hair-sculpt --engine anime-sculpt` | 无折叠 `Rigid MLS` 整头提廓 + 逐根解析微丝聚散 + 原假发发根锚定吸色 + 零碰撞分层（另有 `tutorial-rake` / `blender` 可选回退） |

---

## 二、v11 逐根解析发丝引擎核心技法（`hair_anime_engine.py`）

1. **全头协同塑形（`sculpt_anime_head_and_flow`）**：
   - 使用矢量化 `Rigid MLS` 对头冠与迎风侧发轻度微托（保持最小雅可比行列式 `min_det_J >= 0.65`），脸、眼、手指、服饰与高光饰品严格锁定 `0.00px` 位移。
   - 对白色漆皮衣领等高亮反光边界加入暗部/距离双门禁（`(d_clo_n > 10.0) & (luma_n < 150.0)`），严防冷暖调色带脏漆皮高光。
2. **告别“一坨”的亚像素解析发丝（`_rasterize_analytical_strand` & `_strand_color_profile`）**：
   - 沿 2D Catmull-Rom 样条精确计算局部亚像素有向距离与能量守恒反走样，单根渲染 `0.8–2.2px` 真实发丝，从根源消灭宽丝带锯齿与糊斑。
   - **发根无缝溶入**：前段 `ts` 经 `_smoothstep(0.18, 0.44, ts)` 锚定原假发局部实拍吸色，确保与原片根部色差自然贴合；内部采用明暗错层着色（Shadow / Mid / Bright / Rim），保留真实单反交错暗沟与高亮丝。
3. **聚散美学与“厚实但不成坨”（`_emit_strand_wisp` & `_emit_layered_hair_stream`）**：
   - **禁止等距平行排线**：将每簇拆分成 `2–3` 股中心偏移与收束度不同的不对称微簇（小撮），让相邻细丝既有亲密合股又有斜向透气暗缝与 `1–1.5px` 游离散丝，尾部通过错落提前收束实现丝梢渐消。
   - **长悬空大流瀑不漏底策略（Miku 双马尾经验）**：不能靠实心粗绿管堵漏，而应在主弯中轴使用高密交叠的多重 `1–2px` 细丝簇（并将强化段延过易褪色分段直至末端收口区间前），既能让强逆光窗柱/栏杆零穿透，又在 `1:1` 微距下保持根根分明的丝绸肌理。
4. **外部散光包裹与真实 85mm 光学景深（`_apply_backlight_lightwrap` & `_apply_optical_dof`）**：
   - 逆光包裹仅施加于不与原发重迭的外部透空区域（`ext_only`），避免把原发根部漂白；再按离焦距离平滑叠加轻柔光学虚化并由 `hair_sculpt_ops.py` 匹配原底片传感器颗粒，消除 AI/手绘干涩感。

---

## 三、三案 4K 实战基准与双独立盲审验收实绩（v11 定版）

| 案例 | 关键造型与避让要点 | 双独立 1:1 严审得分（均 `>= 8.5`） |
| --- | --- | --- |
| **Case 1 · Rose (`v11.2`)** | 顺黑裙弧度柔和 S/J 弯金色流瀑，发根色差 `dE=2.38`，前缀桥接丝自然跨越衔接；眼部 `0px`、皇冠白纱零污染 | **9.1 / 10** · **8.6 / 10** |
| **Case 2 · Miku (`v11.7`)** | 右双马尾安全穿过颈臂间隙（臂隙漏条 `0`、耳饰 `0`、腕花 `0`、右窗 `0`），高密细丝延伸至 `y=1565` 彻底闭合中下段 S 弯漏底并保留飘逸发梢 | **9.3 / 10** · **8.7 / 10** |
| **Case 3 · Nazuna (`v11.6`)** | 16 丝聚拢新月呆毛 + 鬓角/丸子头外缘 `1px` 银紫微丝 + 强月光逆光感，并严格锁定白色漆皮衣领纯白高光 `delta=0.0` | **9.1 / 10** · **8.8 / 10** |

---

## 四、PhotoCraft 交付图层栈与常用命令行

`hair-sculpt` 自动生成无损 `.pcraft` 分层工程（原图只读不修改）：
- **整头微调底版**：`hair_sculpt_subject_reshaped_rgba.png`（面部/衣物零位移）
- **后景多层发丝**：`02b_HAIR_SCULPT_BEHIND`（含 `hair_sculpt_01_behind.png` 与遮挡投影 `hair_sculpt_02_cast_shadow_mul.png`）
- **根部跨接桥丝**：`hair_sculpt_02_mid_connect.png`（跨越衔接不遮眼与白纱）
- **前景微丝与逆光缘**：`03b_HAIR_SCULPT_FRONT`（`hair_sculpt_03_front.png`、`hair_sculpt_04_backlit_rim_add.png`、`hair_sculpt_05_real_polish_softlight.png`）

```powershell
# 查看默认引擎与完整参数
$py = "$env:LOCALAPPDATA\RetouchTools\venv\Scripts\python.exe"
& $py scripts\retouch-ai.py hair-sculpt --help

# 运行默认 v11 逐根解析发丝重塑（生成分层 PNG 与 PhotoCraft 工程参数）
& $py scripts\retouch-ai.py hair-sculpt --subject-rgba work\harmonize\harmonized_fg_rgba.png --hair-mask work\body\mask_hair.png --bg work\bg.png --occluder-mask work\body\mask_occluder_arms_face.png --face-mask work\body\mask_face_skin.png --eyes-brows-mask work\body\mask_eyes_brows.png --clothes-mask work\body\mask_clothes_all.png --engine anime-sculpt --style auto --out-dir work\hair_sculpt --overwrite
```
