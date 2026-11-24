# COS 后期学习路线与公开参考索引

用于系统学习 COS 后期技法、查漏补缺或按专题做对比实验；日常修单张图按需挑选，不要求每张图走完所有专题。

## 1. COS 后期 12 类学习地图与 PhotoCraft 对应能力

| 学习类别 | 学习重点与练习方式 | PhotoCraft 对应工具与注意点 |
| --- | --- | --- |
| 1. RAW / 基础显影 | 曝光、白平衡、高光阴影、色彩混合、镜头校正；先全局后局部 | `filter.cameraRaw`；JPEG 已被相机渲染，不可把滤镜调色当成真实 RAW 传感器恢复 |
| 2. 混合光与颜色分区 | 分开处理闪光肤色、现场环境光、服饰主色 | 曲线/HSL/可选颜色调整层 + 空间蒙版；避免全图去某色洗掉氛围轮廓光 |
| 3. 光影塑形与双曲线 | 大范围体积与局部斑块明暗分开，明暗双曲线配合局部蒙版 | 两条 `layer.newAdjustmentLayer.curves` 或 `paint.dodge`/`paint.burn`；保护眼妆与五官转折 |
| 4. 高低频（频率分离） | 先验证未编辑时重构还原度，低频匀色调、高频保纹理 | `image.applyImage` + `filter.blur.gaussianBlur` + `linearLight`；低频只从低频取样 |
| 5. 液化塑形 | 前推、重建、冻结保护锚点；检查五官、衣纹与背景直线 | 复制转智能对象 → `filter.liquify`（`freeze`/`lassoMask`/`reconstruct`）；大形用 `edit.puppetWarp` |
| 6. 电子补妆 | 眼线、睫毛、眼影色区、唇缝与高光独立层控制 | `layer.new.layer` + `paint.stroke` + `Color`/`Screen` 混合模式 |
| 7. 假发与补发丝 | 区分杂丝清理、断丝补接、大束概括与压感细丝 | `paint.mixerBrush` / `paint.smudge` + `path.stroke`（`simulatePressure: true`） |
| 8. 抠图与边缘细化 | 主体选区、发丝边缘、颜色净化、深浅底对照检查 | `select.subject` → `select.refineEdge`（`decontaminate`）+ `paint.stroke(target:"mask")` |
| 9. 服装修整 | 污点、浮线、非结构皱纹与局部材质连续 | 高低频低频修暗斑、`paint.cloneStamp`/`paint.patch`/`edit.contentAwareFill` |
| 10. 场景透视与合成 | 尺度、视平线、接触影、遮挡、环境色、景深与颗粒对齐 | `doc_place` + 自由/透视变形 + 局部调色 + 模糊滤镜 |
| 11. 发光与氛围特效 | 发光核心、外扩辉光、邻近物体受光染色与前后遮挡 | 独立组 + `Screen`/`Linear Dodge` + 空间蒙版 |
| 12. 锐化与交付导出 | 按输出尺寸看 100% 边缘，防范白边/噪点，转换 sRGB 嵌入配置 | `filter.sharpen.unsharpMask`/`smartSharpen` + `doc_save` + 重开核对 |

---

## 2. 三个低风险对照实验设计

适合练习新引擎或校准参数时在独立副本中使用（每次只改一个变量，并保留基线开关对比）：

1. **实验 A｜混合光肤色与主色衣物分区**：为皮肤与高饱和服饰（如红/蓝礼服）建独立局部调整组，从皮肤蒙版中扣除眼唇发丝与衣物，比较通道曲线与低强度局部 HSL，核对是否既消除了皮肤局部斑驳又不连带抽走口红/腮红或洗掉现场轮廓光。
2. **实验 B｜保护眼妆的双曲线对比柔光刷**：挑远离眼妆的一小块脸颊做对照，对比“独立提亮/压暗曲线+扣孔蒙版”与“柔光层 `paint.stroke`”，观察 1:1 皮肤纹理保留度、色彩脏污程度及回调灵活性。
3. **实验 C｜非结构布皱低频修整 vs 贴片克隆**：对衣料/背景布一段平坦非结构褶皱分别对比“低频局部匀色/修补（高频不动）”与“同材质顺纹局部贴片”，检查织纹方向、接缝连续与重复采样伪影。

---

## 3. 已定位公开资料与访问层级

以下资料保留了清晰的来源边界（区分“读到正文”“仅自动文稿线索”“仅标题/元数据索引，未观看完整视频”），后续若需深挖某专题可据此补看原视频：

### 机制与图文正文（可直接核对原理）
- **ふぉーかす（COS 实战图文）**：[人物篇（飞丝/胶带/衣装/假耳）](https://fo-cus.jp/?p=5437)、[背景篇](https://fo-cus.jp/archives/5486)、[皮肤质感保留](https://fo-cus.jp/archives/5167)
- **Frekki（COS 合成流程文稿）**：[Cosplay Photo Edits](https://frekki.wordpress.com/2017/03/04/cosplay-photo-edit-walkthrough/)（环境色、轮廓光与空间整合参考）
- **PHLEARN（人像精修知识结构）**：[Retouching Learning Path](https://phlearn.com/playlist/retouching-learning-path/)
- **CLIP STUDIO TIPS（厚涂绘画方法正文）**：[掌握肤色](https://tips.clip-studio.com/zh-cn/articles/10434)、[厚涂绘制头发](https://tips.clip-studio.com/zh-cn/articles/10848)、[皮肤高光与阴影](https://tips.clip-studio.com/zh-cn/articles/16821)、[初学者厚涂讲座](https://tips.clip-studio.com/zh-cn/articles/16840)
- **Adobe 官方图像机制说明**（用于核对通用非破坏修图概念）：[Camera Raw 与滤镜区别](https://helpx.adobe.com/camera-raw/desktop/get-started/overview-and-setup/differences-camera-raw-camera-raw.html)、[Refine Hair](https://helpx.adobe.com/photoshop/desktop/make-selections/automatic-color-based-selections/make-improved-hair-selections.html)、[混合模式](https://helpx.adobe.com/photoshop/desktop/repair-retouch/adjust-light-tone/blending-mode-descriptions.html)、[Apply Image 与通道计算](https://helpx.adobe.com/photoshop/desktop/adjust-color/hdr-and-channel-manipulation/blend-channels-using-calculations.html)、[液化冻结与解冻](https://helpx.adobe.com/photoshop/desktop/effects-filters/artistic-stylize-filters/freeze-or-thaw-areas.html)、[色彩配置指定与转换](https://helpx.adobe.com/photoshop/desktop/adjust-color/color-profiles/change-a-documents-color-profile.html)

### 公开视频索引（仅定位到入口/简介或自动文稿，未观看原视频）
- RAW 场照全流程：[寄生鱼（带年）](https://www.bilibili.com/video/BV1CRmEB5Eir/)
- 灰调/暗调场照与合成专题：[阡夜陌（灰调）](https://www.bilibili.com/video/BV1XC411z7Ub/)、[我心_摄影版（暗调）](https://www.bilibili.com/video/BV1ED421p7RR/)、[带年（合成全流程）](https://www.bilibili.com/video/BV1R8oRYcEk6/)
- 分频与双曲线：[摄影师洪峰（高低频）](https://www.bilibili.com/video/BV1A4411J7jm/)、[摄影师泰罗（中性灰/双曲线）](https://www.bilibili.com/video/BV1a4411G7R4/)、[设计原来如此（服装高低频去皱）](https://www.bilibili.com/video/BV16E42137Xq/)
- 补妆与画发丝：[龙吉_脑洞稀缺患者（补妆 / 假双）](https://www.bilibili.com/video/BV1KK411y7cV/)、[龙吉_脑洞稀缺患者（假双专门）](https://www.bilibili.com/video/BV1NW41197Pu/)、[天一不咕（PS 画头发教学）](https://www.bilibili.com/video/BV18W4y1C74G/)、[粥佳雯（鼠标补发丝，含自动文稿线索）](https://www.douyin.com/video/7116083153655090468)
- 通道抠图与光效：[小黑（三种抠图，含自动文稿线索）](https://www.douyin.com/video/7022812631157329164)、[-莽S-（发光武器特效）](https://www.bilibili.com/video/BV1PF411m7Et/)、[萌系农村人（COS 一站式自动文稿线索）](https://www.douyin.com/video/7630430375730838437)
