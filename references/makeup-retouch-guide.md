# 商用人像与 COS 光学补妆精修圣经（Anti-Sticker Digital Makeup）

本手册融合商用美妆精修标准（**PiXimperfect** 唇/眼多混合模式拆层、`Blend-If` 混合颜色带保护纹理高光、**Retouching Academy / Julia Kuzmenko** 非破坏曲线塑形与 50% 不透明度控制）、日韩清透妆/2.5D 半厚涂五官解剖法（**CLIP STUDIO TIPS** 肤色与光影边虚实律），以及开源色度分离原理（CIE Lab 锁明度上色），制定 `retouch-workbench` 的通用电子补妆铁律与工程参数。

---

## 一、五条反“廉价贴纸脸”核心铁律

1. **先审原片、去瑕保真（Zero Pave-Over）**：动手前先 1:1 对照原图。真人 COS/人像本身已有紧贴睫毛根的眼线、自然唇纹转折、干净眼神光与协调妆色时，**数字补妆只做光学润色与微塑形，绝不用大块不透明像素直接“盖死重写”**。严禁在无明显斑点区域硬画“局部圆斑遮瑕”（极易反手造出原图不存在的灰紫假淤斑）。
2. **禁止 `Normal` 实心盖色，全面改用光学混合模式 + `Blend-If`**：
   - **压深与结构（眼窝、外眼尾、睫毛根 tightline、上睑落于虹膜阴影、唇缝、嘴角收尖）** → **`Multiply（正片叠底）`**；
   - **泛色、眼影晕染、腮红气血、唇釉底色、中性灰修容与虹膜折射光泽** → **`Soft Light（柔光）` / `Color（颜色）` + `Blend-If（混合颜色带）`**（PhotoCraft `layer.layerStyle.blendingOptions` 调节 `underlying` 拆分滑块），令原始皮肤毛孔、唇纹纵褶与真实高亮金属/水光穿透妆层；
   - **真高光提纯（湿润唇珠、下唇微光泽、眼神光点睛）** → **`Screen（滤色）`**，且**必须以原图自身亮度微脊（`L > local_p78`）为门控**，严禁往哑光皮肤/嘴唇上硬盖几何纯白圆点。
3. **尊重五官解剖与光学物理**：
   - **眼白（Sclera）**：只做柔和去血丝（降 `Lab a*`/`b*` 杂红黄）与极克制球面中段提亮（`<= +6.5 L*`，软封顶 `<= 88 L*`），内/外眼角与上睑投影必须保留球体暗转折，**严禁漂成死白石膏**。
   - **瞳孔与虹膜（Iris Optics）**：上眼睑在虹膜顶部 `25%–35%` **必然投下真实阴影**（无压影就会变成惊悚“死鱼眼/贴纸美瞳”）；外缘虹膜环使用 `2.5–3.5px` 柔渐变深棕/深酒红（禁死黑硬圈）；改瞳色优先用 `Color` + CIE Lab 锁明度转移保留真实虹膜纤维，仅在下半弧微注折射清透感；**优先提纯原图已有角膜眼神光**，仅当原图完全无眼神光时才点 1 颗小方向柔光，严禁硬圆点或夸张星芒贴片。
   - **眼线与睫毛**：内眼线（tightline）用深浓缩咖啡/紫棕（如 `#24161a`）沿上睫毛根贴画、加 `0.7–1.0px` 高斯柔化匹配相机焦外光学 PSF，眼尾短挑顺应下睑真实外延切线。`natural` / `editorial` 默认**只提取增强原片真睫毛微反差（`enhance`）**；`cos-anime` / `painterly` 允许在前者基础上仅在外眼睑 45% 区域补画最多 3–4 簇弯曲渐细弧线（`Multiply` + PSF 柔化），**严禁沿眼眶等距拉出笔直“苍蝇腿/蜘蛛腿”竖刺**。
   - **眼影与卧蚕**：眼窝暗部（`Multiply`）走眶骨与外 V 转折，下睑点缀严格收在紧贴外 1/3 下睫毛根（宽 `<= 0.045 × 眼宽`），杜绝满下眼脸发炎泛红或暗黄脏泥；卧蚕高光必须带肤色暖度（柔光微提 `+4%–7%`）且阴影线仅存于正中下方并向内外眼角渐隐为 0，**禁画整条粗白虫与灰暗泪沟袋**。
   - **嘴唇（PiXimperfect 三层拆光法）**：选区严格锁定为 `MediaPipe 外唇多边形 ∩ 膨胀后 BiSeNet mask_lips` 并对唇缘做 `1.8–2.5px` 羽化（禁生硬涂出下唇外缘）：① 柔光色层（`Soft Light` + `Blend-If` `[20,66,200,246]`）匹配夜景或场景主色调；② 立体深度层（`Multiply`）专压咬唇内缝、微笑嘴角尖端与下唇底轻凹影；③ 真高光丝带层（`Screen`）随真实唇纹断续反射，禁荧光珊瑚粉大面积糊平唇纹。
   - **鼻影与修容**：用 `50% #808080` 中性灰在 `Soft Light` 以大羽化柔和过渡，鼻侧影止于鼻翼上方不抹黑鼻孔外侧；**明确禁用贯穿鼻梁的死白矩形长条（“阿凡达通天感叹号白杠”）与鼻底纯白圆点**——即便在 2.5D 厚涂下也只能保留大面柔光体积转折。
   - **腮红**：限缩在皮肤安全蒙版内并主动避让眼下泪沟、嘴唇与刘海，扫在颧骨高位至外眼下弧线之间，色相融合自身面颊血色，与场景主光同温。
4. **镜头清晰度与传感器颗粒统一**：一切手绘眼线、睫毛、唇釉高光必须轻微匹配原机位光学软度（`0.5–1.0px` PSF 模糊）并接受顶层 `0.8%–1.5%` 单色颗粒统一。
5. **50% 复核原则与原图 1:1 并排必检铁律**：完成所有妆容组后默认从适屏远观检视收敛总不透明度，**必须在 `<=512px` 独立 `1:1` 视图里直接并排对比「原图 vs 妆后」**；一旦任何局部显得比原图更假、更脏或更浮，宁可回退削减也不保留坏修改。

---

## 二、4 档通用画风预设与命令用法

工作台 `retouch-ai.py makeup-paint`（底层实现见 [makeup_retouch_ops.py](../scripts/makeup_retouch_ops.py)）已内建全套 9 层非破坏光学拆层管线，支持跨角色、跨光线通用调参：

| 画风 (`--style`) | 适用场景 | 睫毛 (`--lash-mode`) | 瞳色与唇妆策略 | 推荐不透明度 |
| --- | --- | --- | --- | --- |
| `natural` | 日常人像、清透日系场照 | `enhance`（纯提纯原真睫毛，零加笔画） | 默认保原瞳色（`keep`），MLBB 低饱和裸粉润唇 | 妆容组整体 `0.55–0.75` |
| `editorial` | 高级棚拍商业肖像、时尚大片 | `enhance` | 保原瞳深邃立面或微调瞳色，高级哑光/微缎面唇与明确骨相微 D&B | `0.70–0.85` |
| `cos-anime` | 写实风 COS 正片、角色还原向 | `subtle-wisp`（真睫增强 + 外眼尾少量弧形柔和簇） | 支持 `--iris-color` 锁明度染宝石色/深酒红并留上睑实影，精致咬唇与适度下至微醺 | `0.70–0.90` |
| `painterly` | 2.5D COS 半厚涂 / 板绘风融合 | `subtle-wisp` | 更明确的上睑压影与下半弧通透折射、高反差唇缝与水光脊、柔和 T 区微高光 | `0.75–0.95` |

### 标准 CLI 调用示例

```powershell
# 1) 自然清透人像（保留原瞳色与真睫毛）
& "$env:LOCALAPPDATA\RetouchTools\venv\Scripts\python.exe" "D:\ps\retouch-workbench\scripts\retouch-ai.py" makeup-paint `
  "work\skin_base.png" --style natural --face-parse-dir "work\face_parse" --faces-json "work\face_parse\faces.json" `
  --out-dir "work\makeup_v2" --build-pcraft --overwrite

# 2) 角色向 COS 精细补妆（指定宝石瞳色与暗夜浆果调，同步导出 PhotoCraft Blend-If 分层工程）
& "$env:LOCALAPPDATA\RetouchTools\venv\Scripts\python.exe" "D:\ps\retouch-workbench\scripts\retouch-ai.py" makeup-paint `
  "work\skin_base.png" --style cos-anime --face-parse-dir "work\face_parse" --faces-json "work\face_parse\faces.json" `
  --iris-color "#8b1d35" --lip-tone "#a84c60" --eyeshadow-tone "#6d3b52" --blush-tone "#c86d84" `
  --intensity 0.72 --out-dir "work\makeup_v2" --build-pcraft --overwrite
```

> 所有输出图层的 `blend`、`opacity`、`blendIf`（如 `{"channel": "gray", "underlying": [20, 66, 200, 246]}`）都会写入 `makeup_meta.json`；使用 `build_makeup_pcraft_cmds(meta)` 即可一键把全部带原生 `Blend-If` 保护的图层组无损拼进总装 `.pcraft` 工程。

---

## 三、9 层标准图层职责与 PhotoCraft `Blend-If` 规范

1. **`makeup_01_conceal_db_softlight.png`（`Soft Light` · 中性灰 `#808080` 底）**：数据驱动泪沟/嘴角暗沉柔提 + 止于鼻翼上方的柔和鼻梁微塑形（绝不生造局部假遮瑕斑块与硬感叹号白条）。
2. **`makeup_02_sclera_clean_normal.png`（`Normal`）**：去红丝并保留眼角暗转折的球形克制提亮（`L* <= 88`）。
3. **`makeup_03_eyeshadow_depth_multiply.png`（`Multiply` · `blendIf [18,65,220,255]`）**：眼眶深邃度、外 V 晕染、紧贴下外睫毛根下至与淡薄卧蚕中段暗弧。
4. **`makeup_04_eyeshadow_tint_softlight.png`（`Soft Light` · `blendIf [22,68,210,248]`）**：轻薄眼影发色与偏暖肤调的卧蚕中段提亮。
5. **`makeup_05_iris_depth_multiply.png`（`Multiply`）**：上眼睑投下的 `30%` 瞳孔顶影、柔边虹膜外环（`2.5–3.5px`）与避开真眼神光的深瞳仁。
6. **`makeup_06a_iris_tint_color.png`（`Color` · `blendIf [24,68,195,242]`）+ `makeup_06b_iris_caustic_softlight.png`（`Soft Light`）**：锁明度虹膜染瞳 + 下半弧通透折射反光 + 原生角膜眼神光局部提清。
7. **`makeup_07_tightline_lashes_multiply.png`（`Multiply`）**：贴根浓棕眼线 + 全风格真睫毛对比强化 +（仅风格化模式）外眼尾弧形柔和微簇。
8. **`makeup_08_blush_softlight.png`（`Soft Light` · `blendIf [28,78,208,248]`）**：严锁皮肤、避开泪沟与唇区的高位透肤微醺腮红。
9. **`makeup_09a_lips_glaze_softlight.png`（`Soft Light` · `blendIf [20,66,200,246]`）+ `09b_depth`（`Multiply`）+ `09c_specular`（`Screen`）**：PiXimperfect 三层拆光唇妆（保真唇纹底色 + 咬唇唇缝深度 + 真实唇脊微高光）。

---

## 四、1:1 自检失效对照表

| 失败视觉表现 | 根因定位 | 修正手段 |
| --- | --- | --- |
| 美瞳发亮像死鱼眼/贴纸 | 缺上睑压影、虹膜未保真原亮度、或用了实心 `Normal` | 启用/加浓 `05_iris_depth_multiply` 顶影，确保 `06a` 用 `Color`+`Blend-If` |
| 睫毛像直竖火柴棍/苍蝇腿 | 等距几何直线段且未随眼尾弧线或过锐 | `natural`/`editorial` 切 `enhance`；风格化只画外 45% 弯弧并做 PSF 柔化 |
| 嘴巴像塑料粉块/外溢 | 用了过亮饱和粉或 `Normal` 实心填色糊平唇纹 | 切 `Soft Light`+`Blend-If`、调暗哑协调色相（如豆沙/浆果），用唇纹调制 `09c` 高光 |
| 鼻梁像涂了白油漆或脏灰条 | 画了硬边 `!` 白杠，或鼻侧影直拖到鼻翼旁 | 彻底删除感叹号白杠；侧影仅留在山根至中段且大半径羽化 |
| 眼下像过敏哭肿/泥巴眼袋 | 下至刷太宽太红，或卧蚕暗线画穿整只眼不衰减 | 收窄下至至外 1/3 紧贴根部，卧蚕阴影两端彻底渐隐到 0 |
| 脸上多出原图没有的紫灰圆点 | 人为硬编码局部“遮瑕提亮”拉动邻域偏色 | 移除一切硬编码坐标暗斑，改用数据比对式低频局部匀色 |

---

## 五、参考方法与开源依据

- **商业摄影精修理论**：PiXimperfect《Create Highly Realistic Lipstick in Photoshop》（Multiply 唇纹暗部 + Soft Light 底色 + Screen 高光三层拆光）与《How to Apply Beautiful Eye Makeup in Photoshop》（Blend-If 保护高光与纹理）；Retouching Academy / Julia Kuzmenko（`Curves` + `Underlying Layer` 拆分防糊、`50%` 叠层克制律与 1:1 复核）。
- **日韩清透与 2.5D 五官塑形**：商业清透妆与半厚涂五官质感规范（眼窝压影先于眼球、真高光微脊、拒绝死黑线条与通天白鼻梁）；CLIP STUDIO TIPS `articles/10434`（肤色环境融入）与 `articles/16821`（边线虚实对比）。
- **开源算法思路核查**：评估 GitHub `Jayanths9/Virtual_Makeup`、`MYlab10/Virtual_makeup_try_on_tool` 等非 GAN 虚拟上妆项目后，吸收其 **MediaPipe 478 点定位 + CIE Lab 锁明度上色（保护毛孔/唇纹）** 核心思路，并在本工作台补齐商业分层混合模式、`Blend-If` 混合颜色带与刘海/眼裂遮挡保护。
