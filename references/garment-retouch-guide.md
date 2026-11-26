# 服饰精修与去雾保色指南（PhotoCraft 版）

面向 COS 与人像后期中的深色裙装、水手服白衬衣、袜靴与皮革服饰精修。核心解决三个高频问题：**逆光眩光冲灰（黑白发蒙）、RGB 硬去雾导致黑衣变红褐色/巧克力棕、以及粗糙全局磨皮或单件衣服断裂色阶**。

---

## 一、为什么“黑衣服越去雾越发棕”

1. **RGB 空气光除法放大偏色的数学根源**：
   - 在 RGB 空间套用 `J = (I - A) / t + A` 时，若场景高光空气光 `A` 略微偏暖（例如 `R > B` 仅差 `7` 个色阶），对发灰暗部除以透射率 `t ≈ 0.50` 会把极微弱的暖色偏差放大约 **2 倍**，直接把原本的中性黑裙/黑袜/黑鞋推成显眼的**红褐色或巧克力棕**。
2. **铁律：暗部去雾必须只在亮度 `L*`（Luminosity）执行，禁止 RGB 除以暖大气光**。

---

## 二、SegFormer-B2 精准语义分服 + CIE Lab 纯明度三频精修

使用 `retouch-ai.py body-parse`（默认 `SegFormer-B2 Clothes` 18 类 ONNX CUDA）先精准拆分：
- 上装（`mask_upper_clothes`）与连衣裙/半身裙（`mask_skirt_dress`）；
- 鞋履（`mask_shoes`）；
- 腿部语义内的深色袜子（只在 `mask_legs` 内部结合低色度 `C* < 16` 与引导滤波边缘吸附提取，绝不伤及膝盖/大腿皮肤）；
- 注意：`SegFormer` 偶发将强眩光发白的站立腿膝盖边沿并入 `mask_pants`，因此 `garment-retouch` 在处理 `mask_pants` 时会自动排除暖色高亮肤区（`L* > 50 & a* > 4.2`），避免误染膝盖；
- 调用专用侧车命令 **`retouch-ai.py garment-retouch`**（推荐同步加 `--out-dark-mask` 导出完整深色服装保护罩，透传给 `harmonize --garment-dark-mask`）：

```powershell
& "$env:LOCALAPPDATA\RetouchTools\venv\Scripts\python.exe" "D:\ps\retouch-workbench\scripts\retouch-ai.py" garment-retouch "<主图路径>" --body-parse-dir "<body_parse目录>" --out "<输出图>" --out-patch "<非破坏补丁PNG>" --out-dark-mask "<完整深色保护罩PNG>" --overwrite
```

### 2.1 核心三频分治结构（CIE Lab 边缘保持引导滤波）
| 频段 | 处理目标 | 具体策略 |
| --- | --- | --- |
| **低频（大块眩光与均匀度）** | 消除逆光大片灰雾、左右裙摆明暗断层与不规则脏斑 | 48px 分块暗底估算 + 仅针对 `L*` 通道的空间透射率回拉（暗底锚定 `L* ≈ 12–16`，下限卡在 `L* ≥ 4.5` 确保零死黑） + 引导滤波温和压平高亮眩光斑 |
| **中性黑防偏色守卫** | 消灭黑裙/黑袜/黑鞋发棕或泛彩噪 | 对低明度低彩度区域（`L* < 52, C* < 20`）把 `a*/b*` 平滑拉回纯净中性微冷炭黑，并只对 `a*/b*` 通道跑轻量双边滤波消彩噪；同时完整放行高彩度彩色刺绣与滚边 |
| **中频（立体褶皱塑形）** | 保住裙褶、衣领、袖笼的真实 3D 体积起伏 | 适度增强中频明暗对比（`drape-contrast`），严禁整片涂抹成死板黑块或塑料布 |
| **高频（织纹与高光质感）** | 留住真实斜纹布纹/纱织与皮鞋高光反射 | 100% 保留细碎布纹 `L_hi`，绝不叠高强度高斯降噪 |

---

## 三、分材质参数配方

| 材质类别（`--material`） | 典型服装部件 | 处理要点 |
| --- | --- | --- |
| **`dark-twill`**（深色制服/斜纹裙） | 百褶裙、水手服深色背带裙、西装外套 | 开强纯明度空间去雾（`0.78–0.85`）与中性黑守卫（`0.80`），拉开中频褶皱体积并锁死织纹 |
| **`dark-matte`**（深色哑光针织/袜类） | 黑丝/中筒黑袜、针织开衫、暗绒布 | 清理下肢逆光眩光泛白，对齐同件纯度，保留柔和哑光微织感 |
| **`glossy-leather`**（漆皮/鞋靴配饰） | 玛丽珍小皮鞋、后踢鞋靴、皮带、金属扣 | 放宽黑底候选到 `L* < 86`、强压缩 `L* > 28` 过曝白斑并保留高光轮廓；`residual_dark_garment` 以 `MORPH_CLOSE(7)` 兜底未被分割到的踢出鞋靴并隔离暖色膝盖皮肤 |
| **`light-cotton`**（浅色棉布/衬衫/飘带） | 奶白水手服衬衫、百褶白领、雪纺飘带 | 关掉暗部去雾（`0.0`），对 `L* > 86` 做柔和高光滚降防死白溢出，微提衣领折痕中频清晰度 |

### 3.1 纯半透边缘去白边铁律（Alpha-Only Fringe Decontamination）
- 清理黑裙、黑袜、后踢鞋边缘的白雾抠图边时，**只允许覆盖半透明羽化边缘（`0.02 < alpha < 0.88` 及其紧邻 `1px` 羽化缘）**，并就近从本地实心主体边缘（`alpha >= 0.90`）外推真实明暗与色彩；**严禁向实心内部硬扩或锁定 `L* < 38` 远距离强行填纯黑**，否则会在亮鞋面与鞋底边沿画出难看的黑锯齿描边。

### 3.2 下肢肤色与遮挡光影协同（`retouch-ai.py skin-unify`）
- 深色服饰、黑袜与鞋靴去雾并完成 `--out-dark-mask` 导出后，对下肢紧接运行 `skin-unify`：
```powershell
& "$env:LOCALAPPDATA\RetouchTools\venv\Scripts\python.exe" "D:\ps\retouch-workbench\scripts\retouch-ai.py" skin-unify "<主图>" --body-parse-dir "<body_parse目录>" --warmth-pull 0.72 --glare-compress 0.62 --skirt-shadow 0.42 --defringe --out "<输出图>" --out-patch "<下肢非破坏补丁PNG>" --overwrite
```
- 工作原理：自动以 `mask_legs` 为种子外扩 `165px` 并穿透 `mask_pants` 误标暖膝区域（完整涵盖 `SegFormer` 漏标的大腿上段/膝盖） → 以手臂晴光肤色 `mask_arms` 为参考在归一化卷积低频拉回暖桃血色并保住 `L_fine` 真实皮肤纹理 → 清理腿部外缘抠图白边 → 按黑裙下摆与上方交叠腿/黑袜口叠加方向性接触遮挡影。

---

## 四、PhotoCraft 非破坏装配与自动体检验收

1. **非破坏图层置入**：优先使用 `--out-patch` 生成透明局部补丁 PNG，在 PhotoCraft 里置入为主体组剪贴智能对象（方便随时微调不透明度或用黑蒙版擦回特定细节）。
2. **体检硬门禁（`mask-audit --neutral-garment`）**：
   - 运行 `retouch-ai.py mask-audit` 时传入 `--garment-mask`、`--neutral-garment` 与 `--reference`：
   - **色偏红线**：暗部核心 `|mean_a| > 5.0`、`mean_c > 8.5` 或相对中性参考偏色 `ΔC > 5.5` 直接判败（直接抓出黑衣发棕）；
   - **织纹红线**：织纹保留比 `< 0.80`（被糊成塑料布）或高频噪点 `> 1.35x` 直接判败；
   - **同件服饰明暗断层**：带间亮度跳变 `> 38` 直接判败。
