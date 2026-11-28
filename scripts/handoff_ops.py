"""Human-AI Hard-Stop Handoff Packager for white-model empty-BG & pure-#000000 VFX card-rolling."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Pure-Python preset prompts mirrored without importing Blender's `bpy`
PRESET_PROMPTS: dict[str, dict[str, str]] = {
    "crystal-conservatory": {
        "bg_cn": "严格保留参考白模图中的建筑透视、机位视平线、穹顶弧柱、前景栏杆、左侧钢琴舞台与右上向左下投射的日光阴影方向。将其渲染为日系剧场版实拍感的高清空镜：挑高纯白烤漆钢架与水晶玻璃穹顶音乐厅花房，搭配精致香槟金雕花饰边与大理石反光地面；右侧玻璃拱门外夕阳斜射出通透温暖的香槟金色光束，左侧玻璃穹顶外的黄昏天空与玻璃反光柔和融入初音薄荷青蓝（#39C5BB）与晚霞樱粉渐层；白色三角钢琴旁点缀浅粉奶白玫瑰花瀑与水晶吊灯。大光圈电影实拍质感，严格无人空镜，严禁任何人物、剪影或文字。",
        "bg_en": "Strictly preserve the exact camera perspective, horizon, arched columns, dome ribs, left grand piano dais, balustrade, and sun shadow direction from the reference white-model image. Render as a high-resolution photorealistic live-action Japanese cinema still of an empty luxury crystal glass conservatory music hall at golden-hour sunset: white-painted steel arches, champagne-gold filigree, crystal chandeliers, blush-pink & cream rose cascades around a white grand piano, and ivory marble floor. Warm champagne-gold sunbeams stream from upper-right, balanced by pastel mint-cyan (#39C5BB) and sakura twilight dome reflections on the upper-left. STRICTLY EMPTY STAGE with NO people, NO silhouettes, NO text.",
        "vfx_en": "16:9 cinema VFX overlay plate on a strictly pitch-black background (#000000) for Screen/Add compositing. Ethereal dreamy crystal starlight 4-point and 6-point glints, fine champagne-gold & mint-cyan (#39C5BB) light motes, and sparse softly glowing pastel sakura petals drifting diagonally along the sunbeam direction. Generous negative black space, NO people, NO scenery, NO text.",
    },
    "european-arcade": {
        "bg_cn": "严格保留参考白模图中的欧式拱廊透视、机位视平线与日光斜射阴影。将其渲染为实拍电影感黄金时刻欧式石砌拱廊花庭空镜：温润米白石灰岩立柱与拱券、攀援浅粉与奶油色藤本玫瑰、天然石板地面的温柔光斑与长投影，远景柔和虚化的通透黄金暮空。电影定焦实拍质感，严格无人空镜，严禁任何人物、剪影或文字。",
        "bg_en": "Strictly preserve camera perspective, eye-level horizon, vaulted colonnade ribs, pavers, and sunlight shadow angle from the reference white-model. Render as a photorealistic cinema film still of an empty sunlit European limestone rose arcade at golden hour, warm carved stone arches, climbing pastel pink roses, long golden light shafts across textured flagstones, creamy sunlit bokeh depth. STRICTLY EMPTY STAGE with NO people, NO text.",
        "vfx_en": "16:9 cinema VFX overlay plate on strictly pitch-black background (#000000). Floating warm champagne-gold sun dust motes, subtle drifting backlit rose petals, delicate optical lens flare glints drifting along the light direction, generous pure-black air space, NO people, NO ground, NO text.",
    },
    "seaside-stage": {
        "bg_cn": "严格保留参考白模图中的海滨舞台透视、玻璃栏杆与主光阴影方向。将其渲染为日系剧场版实拍感的黄昏海滨玻璃露天音乐舞台空镜：轻盈纯白与香槟金舞台地面、高透玻璃护栏、海天交界处染上薄荷青绿（#39C5BB）与樱粉晚霞，远处柔和虚化的港湾摩天轮与点点海面碎金反光。实拍电影质感，严格无人空镜，严禁人物或文字。",
        "bg_en": "Strictly preserve perspective, stage deck, glass balustrade, and sunlight angle from the white-model reference. Render as a photorealistic live-action Japanese cinema still of an empty coastal open-air concert stage at sunset: sleek white-and-champagne deck, clear glass railing, ocean horizon glowing with pastel mint-cyan (#39C5BB) and sakura sunset clouds, distant softly defocused harbor ferris wheel. STRICTLY EMPTY STAGE, NO people, NO text.",
        "vfx_en": "16:9 cinema VFX overlay plate on pitch-black (#000000) background: luminous sea-breeze starlight sparkles, soft mint-cyan & warm amber bokeh orbs, translucent shimmer dust drifting across the frame, generous pure-black spacing, NO people, NO text.",
    },
    "rooftop-twilight": {
        "bg_cn": "严格保留参考白模图中的天台透视、护栏、灯架结构与斜射日光方向。将其渲染为实拍电影感都市黄昏天台音乐广场空镜：洁净浅色天台地砖、极简金属护栏与白色灯架，远处黄昏东京湾风格城市天际线在暖金逆光中柔和虚化，晚风感空气透视。严格无人空镜，严禁人物或文字。",
        "bg_en": "Strictly preserve rooftop deck perspective, guardrail, lighting trusses, and sun direction from the white-model. Render as a photorealistic cinema film still of an empty twilight urban rooftop stage overlooking a softly defocused golden-hour city skyline and bay, clean architectural concrete tiles, warm amber backlight glow. STRICTLY EMPTY STAGE, NO people, NO text.",
        "vfx_en": "16:9 cinema VFX overlay plate on pitch-black (#000000) background: fine golden twilight dust specks, subtle anamorphic bokeh light orbs, soft breeze light streaks, generous pure-black areas, NO people, NO text.",
    },
    "eastern-pavilion": {
        "bg_cn": "严格保留参考白模图中的中式亭台柱廊、曲桥与水面透视、主光投射方向。将其渲染为实拍电影感古风临水游廊亭台空镜：雅致素木白墙飞檐亭台、雕花栏杆、青碧澄澈带倒影的莲池水面，薄雾斜阳穿过树影与廊柱洒下金色暖光。写实摄影质感，严格无人空镜，严禁人物或文字。",
        "bg_en": "Strictly preserve architectural pavilion columns, curved bridge, water plane, and sun direction from the white-model. Render as a photorealistic live-action cinema still of an empty serene Chinese waterside pavilion & curved veranda at late afternoon sun: refined pale timber, carved stone balustrade, reflective celadon pond water, soft golden sunbeams through mist. STRICTLY EMPTY STAGE, NO people, NO text.",
        "vfx_en": "16:9 cinema VFX overlay plate on pitch-black (#000000) background: dreamy celadon-jade and warm gold firefly light motes, delicate water-shimmer glints, sparse translucent magnolia/peach petals on pure black, NO people, NO text.",
    },
    "studio-cyclorama": {
        "bg_cn": "严格保留参考白模图中的弧形影棚背景与百叶窗光影条纹结构。将其渲染为高级时尚人像棚拍电影感空镜：极简高档哑光奶油灰白弧形无影墙，一侧大窗户透过格栅洒入柔和清晰的黄金时刻斜射暖光斑，真实微细灰泥质感。严格无人空镜，严禁人物或文字。",
        "bg_en": "Strictly preserve curved cyclorama sweep and window gobo shadow geometry from the white-model. Render as a high-end fashion studio film still: architectural warm matte plaster cyclorama wall illuminated by crisp soft-edged golden window light slats. STRICTLY EMPTY STUDIO, NO people, NO text.",
        "vfx_en": "16:9 cinema VFX overlay plate on pitch-black (#000000) background: delicate sunbeam micro-dust particles, soft optical prism bokeh gleams, clean generous black negative space, NO people, NO text.",
    },
}


def _uri(p: Path | None) -> str:
    if p is None:
        return ""
    return "file:///" + str(p.resolve()).replace("\\", "/")


def _draw_light_arrow(draw: ImageDraw.ImageDraw, w: int, h: int, side: str, font: Any) -> None:
    s = side.strip().lower()
    if s in ("top-left", "left"):
        start = (int(w * 0.10), int(h * (0.14 if "top" in s else 0.45)))
        end = (int(w * 0.30), int(h * (0.34 if "top" in s else 0.48)))
    elif s in ("top-right", "right"):
        start = (int(w * 0.90), int(h * (0.14 if "top" in s else 0.45)))
        end = (int(w * 0.70), int(h * (0.34 if "top" in s else 0.48)))
    elif s == "back":
        start = (int(w * 0.50), int(h * 0.08))
        end = (int(w * 0.50), int(h * 0.28))
    else:
        return

    col = (255, 224, 110, 235)
    draw.line([start, end], fill=col, width=max(4, int(w * 0.004)))
    ang = math.atan2(end[1] - start[1], end[0] - start[0])
    head_len = max(18.0, w * 0.018)
    for delta in (2.55, -2.55):
        hx = end[0] + head_len * math.cos(ang + delta)
        hy = end[1] + head_len * math.sin(ang + delta)
        draw.line([end, (int(hx), int(hy))], fill=col, width=max(4, int(w * 0.004)))

    lbl = f"KEY LIGHT: {s.upper()}"
    tx = max(16, min(w - 300, start[0] - 80 if start[0] > w * 0.5 else start[0]))
    ty = max(12, start[1] - 32)
    draw.rectangle([tx - 6, ty - 4, tx + 240, ty + 24], fill=(16, 20, 28, 195))
    draw.text((tx, ty), lbl, fill=(255, 230, 130, 255), font=font)


def run_handoff_pack(
    *,
    out_dir: Path,
    project_name: str = "COS_Project",
    character: str = "",
    whitemodel_path: Path | None = None,
    subject_rgba_path: Path | None = None,
    scene_meta_path: Path | None = None,
    preset: str | None = None,
    bg_prompt_cn: str = "",
    bg_prompt_en: str = "",
    vfx_prompt_cn: str = "",
    vfx_prompt_en: str = "",
    aspect: str = "16:9",
    target_width: int = 3840,
    target_height: int = 2160,
    key_light_side: str = "auto",
    subject_x_frac: float = 0.50,
    prop_line_norm: tuple[float, float, float, float] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate layout guide, copy-paste prompts (HANDOFF_FOR_USER.md), and AWAITING_USER_ASSETS lock."""
    out_d = out_dir.resolve()
    out_d.mkdir(parents=True, exist_ok=True)

    meta_scene: dict[str, Any] = {}
    if scene_meta_path is not None and scene_meta_path.exists():
        meta_scene = json.loads(scene_meta_path.read_text(encoding="utf-8"))

    resolved_preset = preset or meta_scene.get("preset")
    resolved_side = key_light_side.strip().lower()
    if resolved_side == "auto":
        resolved_side = str(meta_scene.get("sun_side") or meta_scene.get("lighting", {}).get("sun_side") or "top-left").lower()
    s_x_frac = float(meta_scene.get("subject_x_frac", subject_x_frac))

    wm_p = whitemodel_path.resolve() if (whitemodel_path is not None and whitemodel_path.exists()) else None
    if wm_p is None and meta_scene.get("outputs", {}).get("whitemodel_empty"):
        cand = Path(str(meta_scene["outputs"]["whitemodel_empty"]))
        if cand.exists():
            wm_p = cand.resolve()

    pr = PRESET_PROMPTS.get(str(resolved_preset or ""), {})
    b_cn = bg_prompt_cn.strip() or str(pr.get("bg_cn") or "").strip()
    b_en = bg_prompt_en.strip() or str(pr.get("bg_en") or "").strip()
    v_en = vfx_prompt_en.strip() or str(pr.get("vfx_en") or "").strip()
    v_cn = vfx_prompt_cn.strip() or (
        f"严格使用纯黑底色（#000000），{aspect} 宽画幅电影合成特效叠层；仅围绕边缘点缀梦幻光斑、流光丝线与细碎微光星尘，画面中央及人脸位置保持克制稀疏与纯黑负空间；严禁上传白模参考图，严禁建筑、地面、人物或文字。"
    )

    if not b_cn and not b_en:
        b_en = (
            f"Strictly preserve exact camera perspective, horizon line, contact geometry, and {resolved_side} key-light direction from the attached reference image. "
            f"Render as an empty, ultra-detailed high-resolution {aspect} cinema film background still matching the lighting cues. STRICTLY EMPTY STAGE with NO people, NO silhouettes, NO text."
        )
        b_cn = (
            f"严格保留上传白模参考图的机位透视、地平线、支撑接触面结构与 {resolved_side} 主光方向，渲染为高清晰 {aspect} 电影实拍感高清无人空镜；严禁出现任何人物、剪影、路人或文字。"
        )
    if not v_en:
        v_en = (
            f"{aspect} cinema VFX particle & light-trail overlay plate on a strictly pitch-black background (#000000) for Screen/Add compositing. "
            "Ethereal floating glow motes, soft volumetric lens sparkles and flowing light ribbons framed around the edges with generous pure-black negative space in the center. NO people, NO scenery, NO ground, NO text."
        )

    min_roll_w = max(1280, int(round(target_width * 0.6667)))
    min_roll_h = max(720, int(round(target_height * 0.6667)))

    # Optional visual composition & lighting guide PNG (NEVER uploaded to AI BG generator!)
    layout_p: Path | None = None
    sub_p = subject_rgba_path.resolve() if (subject_rgba_path is not None and subject_rgba_path.exists()) else None
    if wm_p is not None and (sub_p is not None or prop_line_norm is not None or resolved_side):
        layout_p = out_d / "03_layout_and_light_guide.png"
        if layout_p.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite without --overwrite: {layout_p}")
        base_im = Image.open(wm_p).convert("RGBA")
        if base_im.width > 1920:
            nh = max(64, int(round(base_im.height * 1920.0 / base_im.width)))
            base_im = base_im.resize((1920, nh), Image.Resampling.LANCZOS)
        lw, lh = base_im.size
        overlay = Image.new("RGBA", (lw, lh), (0, 0, 0, 0))
        dr = ImageDraw.Draw(overlay)
        try:
            font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", size=max(16, int(lw * 0.012)))
        except Exception:
            font = ImageFont.load_default()

        if sub_p is not None:
            sub_im = Image.open(sub_p).convert("RGBA")
            # Resize preserving aspect to fit naturally into ~86% height or full width
            scale = min((lw * 0.92) / max(1, sub_im.width), (lh * 0.88) / max(1, sub_im.height))
            sw, sh = max(16, int(round(sub_im.width * scale))), max(16, int(round(sub_im.height * scale)))
            sub_rs = sub_im.resize((sw, sh), Image.Resampling.LANCZOS)
            r_ch, g_ch, b_ch, a_ch = sub_rs.split()
            a_ghost = a_ch.point(lambda v: int(v * 0.82))
            sub_ghost = Image.merge("RGBA", (r_ch, g_ch, b_ch, a_ghost))
            px = int(np.clip(lw * s_x_frac - sw * 0.5, 0, max(0, lw - sw)))
            py = lh - sh
            overlay.paste(sub_ghost, (px, py), sub_ghost)

        if prop_line_norm is not None:
            x0, y0, x1, y1 = [float(v) for v in prop_line_norm]
            p0 = (int(x0 * lw), int(y0 * lh))
            p1 = (int(x1 * lw), int(y1 * lh))
            dr.line([p0, p1], fill=(80, 235, 255, 235), width=max(4, int(lw * 0.003)))
            dr.text((min(lw - 280, max(16, p0[0] + 18)), max(16, int(0.5 * (p0[1] + p1[1]) - 28))), "CONTACT / SUPPORT LINE", fill=(110, 245, 255, 255), font=font)

        _draw_light_arrow(dr, lw, lh, resolved_side, font)
        banner_h = max(36, int(lh * 0.04))
        dr.rectangle([0, lh - banner_h, lw, lh], fill=(24, 14, 18, 220))
        dr.text((16, lh - banner_h + 6), "LAYOUT / LIGHTING REFERENCE ONLY — DO NOT UPLOAD TO AI BG GEN (UPLOAD EMPTY WHITE-MODEL ONLY)", fill=(255, 210, 120, 255), font=font)

        comp_im = Image.alpha_composite(base_im, overlay).convert("RGB")
        comp_im.save(layout_p, "PNG")

    md_p = out_d / "HANDOFF_FOR_USER.md"
    man_p = out_d / "handoff_manifest.json"
    if (md_p.exists() or man_p.exists()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite without --overwrite: {md_p}")

    char_note = f"（角色：{character}）" if character.strip() else ""
    wm_line = f"- **无人白模参考图（抽背景必传这一张）**：[{wm_p.name}]({_uri(wm_p)}) (`{wm_p}`)\n" if wm_p else "- **无人白模参考图**：未指定（若按纯文案生图请锁定相同机位与主光方向）\n"
    lay_line = f"- **构图与主光对照图（仅自查，切勿上传给 AI 生图）**：[{layout_p.name}]({_uri(layout_p)}) (`{layout_p}`)\n" if layout_p else ""

    md_text = (
        f"# 阶段二白模与抽卡交接卡 · {project_name}{char_note}\n\n"
        f"> **人机协同状态：`AWAITING_USER_ASSETS`（工作流已按规范硬暂停，本机绝不私自代抽）**\n\n"
        f"## 1. 本地参考文件与上传须知\n\n"
        f"{wm_line}"
        f"{lay_line}"
        f"- **目标合成规格**：`{target_width}×{target_height}`（`{aspect}`），建议外部生图原片分辨率 **`>= {min_roll_w}×{min_roll_h}`**，保证原生清晰度 `>= 0.6×`。\n"
        f"- **主光预设方向**：`{resolved_side}`\n\n"
        f"## 2. 空镜背景抽卡提示词（请仅垫入「无人白模图」）\n\n"
        f"### 中文提示词\n"
        f"```text\n{b_cn}\n```\n\n"
        f"### English Prompt\n"
        f"```text\n{b_en}\n```\n\n"
        f"## 3. 纯黑底（`#000000`）特效叠层提示词（切勿垫入建筑白模图）\n\n"
        f"### 中文提示词\n"
        f"```text\n{v_cn}\n```\n\n"
        f"### English Prompt\n"
        f"```text\n{v_en}\n```\n\n"
        f"## 4. 请回传给工作台的 3 项信息\n\n"
        f"1. **高清空镜背景图路径**（必填）；\n"
        f"2. **纯黑底特效图路径**（选填；回传后侧车会自动按 `2.5D` 法线算出它落在人物衣裙、皮肤与周边地面上的因果光影）；\n"
        f"3. **终稿风格确认**（如：`写实电影后期` / `2.5D 厚涂风格` / `双版都做`）。\n"
    )
    md_p.write_text(md_text, encoding="utf-8")

    manifest: dict[str, Any] = {
        "ok": True,
        "status": "AWAITING_USER_ASSETS",
        "allow_agent_self_gen": False,
        "project_name": project_name,
        "character": character,
        "preset": resolved_preset,
        "aspect": aspect,
        "target_size": [target_width, target_height],
        "min_recommended_roll": [min_roll_w, min_roll_h],
        "key_light_side": resolved_side,
        "outputs": {
            "handoff_md": str(md_p),
            "handoff_manifest": str(man_p),
            "whitemodel_empty": str(wm_p) if wm_p else None,
            "layout_guide": str(layout_p) if layout_p else None,
        },
        "prompts": {
            "bg_cn": b_cn,
            "bg_en": b_en,
            "vfx_cn": v_cn,
            "vfx_en": v_en,
        },
        "expected_user_returns": [
            "user_bg_path",
            "user_vfx_path (optional, #000000 black background)",
            "style_preference (photorealistic | painterly | both)",
        ],
    }
    man_p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
