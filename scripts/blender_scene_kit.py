"""Parameterized Blender 5.x GPU white-model scene generator for user AI background card-rolling + 16-bit Z-depth & scene_meta.json."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from mathutils import Euler, Vector


PRESET_PROMPTS: dict[str, dict[str, str]] = {
    "crystal-conservatory": {
        "bg_cn": "严格保留参考白模图中的建筑透视、机位视平线、穹顶弧柱、前景栏杆、左侧钢琴舞台与右上向左下投射的日光阴影方向。将其渲染为日系剧场版实拍感的高清空镜：挑高纯白烤漆钢架与水晶玻璃穹顶音乐厅花房，搭配精致香槟金雕花饰边与大理石反光地面；右侧玻璃拱门外夕阳斜射出通透温暖的香槟金色光束，左侧玻璃穹顶外的黄昏天空与玻璃反光柔和融入初音薄荷青蓝（#39C5BB）与晚霞樱粉渐层；白色三角钢琴旁点缀浅粉奶白玫瑰花瀑与水晶吊灯。大光圈电影实拍质感，严格无人空镜，严禁任何人物、剪影或文字。",
        "bg_en": "Strictly preserve the exact camera perspective, horizon, arched columns, dome ribs, left grand piano dais, balustrade, and sun shadow direction from the reference white-model image. Render as a high-resolution photorealistic live-action Japanese cinema still of an empty luxury crystal glass conservatory music hall at golden-hour sunset: white-painted steel arches, champagne-gold filigree, crystal chandeliers, blush-pink & cream rose cascades around a white grand piano, and ivory marble floor. Warm champagne-gold sunbeams stream from upper-right, balanced by pastel mint-cyan (#39C5BB) and sakura twilight dome reflections on the upper-left. STRICTLY EMPTY STAGE with NO people, NO silhouettes, NO text.",
        "vfx_en": "16:9 cinema VFX overlay plate on a strictly pitch-black background (#000000) for Screen/Add compositing. Ethereal dreamy crystal starlight 4-point and 6-point glints, fine champagne-gold & mint-cyan (#39C5BB) light motes, and sparse softly glowing pastel sakura petals drifting diagonally along the sunbeam direction. Generous negative black space, NO people, NO scenery, NO text.",
    },
    "european-arcade": {
        "bg_cn": "严格保留参考白模图中的欧式拱廊透视、机位视平线与日光斜射阴影。将其渲染为实拍电影感黄金时刻欧式石砌拱廊花庭空镜：温润米白石灰岩立柱与拱券、攀援浅粉与奶油色藤本玫瑰、 sunlit 天然石板地面的温柔光斑与长投影，远景柔和虚化的通透黄金暮空。电影定焦实拍质感，严格无人空镜，严禁任何人物、剪影或文字。",
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


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(description="Blender GPU White-Model Scene Kit")
    ap.add_argument("--preset", default="crystal-conservatory", choices=sorted(PRESET_PROMPTS.keys()))
    ap.add_argument("--aspect", default="16:9", choices=["16:9", "9:16", "2.39:1"])
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--lens", type=float, default=38.0)
    ap.add_argument("--sun-azimuth", type=float, default=42.0, help="Degrees: +right, -left")
    ap.add_argument("--sun-elevation", type=float, default=18.0, help="Degrees above horizon")
    ap.add_argument("--subject-x-frac", type=float, default=0.70, help="Suggested subject anchor fraction")
    ap.add_argument("--subject-rgba", default="", help="Optional subject RGBA cutout for layout reference render")
    ap.add_argument("--samples", type=int, default=28)
    ap.add_argument("--out-dir", required=True)
    return ap.parse_args(argv)


def setup_cycles(w: int, h: int, samples: int) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for btype in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = btype
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type == btype]
            if gpus:
                for d in prefs.devices:
                    d.use = (d.type == btype)
                scene.cycles.device = "GPU"
                print(f"SCENE_KIT_GPU={btype}")
                break
        except Exception:
            pass
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.03
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 6
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 3
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "AgX" if "AgX" in [i.identifier for i in scene.view_settings.bl_rna.properties["view_transform"].enum_items] else "Filmic"
    scene.view_settings.exposure = 0.25


def make_mat(name: str, rgb: tuple[float, float, float], rough: float = 0.5, metal: float = 0.0) -> Any:
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    return m


def make_glass_mat(name: str = "ClearGlass") -> Any:
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    tr.inputs["Color"].default_value = (0.94, 0.97, 0.98, 1.0)
    gl = nt.nodes.new("ShaderNodeBsdfGlossy")
    gl.inputs["Color"].default_value = (0.92, 0.96, 1.0, 1.0)
    gl.inputs["Roughness"].default_value = 0.08
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.14
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lw.outputs["Fresnel"], mix.inputs["Fac"])
    nt.links.new(tr.outputs["BSDF"], mix.inputs[1])
    nt.links.new(gl.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return m


def add_cylinder(name: str, loc: tuple[float, float, float], r: float, depth: float, mat: Any, segs: int = 24) -> Any:
    bpy.ops.mesh.primitive_cylinder_add(vertices=segs, radius=r, depth=depth, location=loc)
    o = bpy.context.active_object
    o.name = name
    o.data.materials.append(mat)
    return o


def add_box(name: str, loc: tuple[float, float, float], scale: tuple[float, float, float], mat: Any, rot_z: float = 0.0) -> Any:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc, rotation=(0.0, 0.0, rot_z))
    o = bpy.context.active_object
    o.name = name
    o.scale = scale
    o.data.materials.append(mat)
    return o


def build_preset(preset: str, sun_az: float, sun_el: float) -> Vector:
    white = make_mat("WhitePlaster", (0.92, 0.92, 0.93), 0.45)
    ivory = make_mat("IvoryFloor", (0.88, 0.87, 0.86), 0.16)
    gold = make_mat("ChampagneTrim", (0.82, 0.72, 0.56), 0.28, 0.75)
    glass = make_glass_mat()
    water = make_mat("Water", (0.35, 0.55, 0.58), 0.04)

    # Sky & Sun
    world = bpy.data.worlds.new("SceneWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out_w = nt.nodes.new("ShaderNodeOutputWorld")
    bg_w = nt.nodes.new("ShaderNodeBackground")
    bg_w.inputs["Strength"].default_value = 1.1
    bg_w.inputs["Color"].default_value = (0.76, 0.86, 0.94, 1.0)
    nt.links.new(bg_w.outputs["Background"], out_w.inputs["Surface"])

    az_r = math.radians(sun_az)
    el_r = math.radians(sun_el)
    sun_from = Vector((math.sin(az_r) * math.cos(el_r), math.cos(az_r) * math.cos(el_r), math.sin(el_r))).normalized()
    sun_ray = -sun_from
    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 8.0))
    sun_obj = bpy.context.active_object
    sun_obj.name = "Hero_Sun"
    sun_obj.rotation_euler = sun_ray.to_track_quat("-Z", "Y").to_euler()
    sun_obj.data.energy = 5.5
    sun_obj.data.angle = math.radians(2.2)
    sun_obj.data.color = (1.0, 0.92, 0.78)

    if preset == "crystal-conservatory":
        bpy.ops.mesh.primitive_plane_add(size=32.0, location=(0.0, 9.0, 0.0))
        bpy.context.active_object.data.materials.append(ivory)
        # Curved rotunda columns, connecting entablature lintel & ribs
        cx, cy, rad = 0.0, 10.5, 9.2
        cols = []
        for k in range(13):
            ang = math.radians(-78.0 + k * 13.0)
            px, py = cx + rad * math.sin(ang), cy + rad * math.cos(ang) * 0.72
            cols.append((px, py))
            add_cylinder(f"Col_{k}", (px, py, 2.8), 0.16, 5.6, white)
            add_cylinder(f"RibTop_{k}", (px * 0.5, cy + (py - cy) * 0.5, 6.8), 0.08, 4.0, white)
        for k in range(len(cols) - 1):
            x0, y0 = cols[k]
            x1, y1 = cols[k + 1]
            span = math.hypot(x1 - x0, y1 - y0)
            ang_b = math.atan2(y1 - y0, x1 - x0)
            add_box(f"Lintel_{k}", (0.5 * (x0 + x1), 0.5 * (y0 + y1), 5.65), (span + 0.2, 0.28, 0.34), gold, rot_z=ang_b)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=36, ring_count=18, radius=9.4, location=(cx, cy, 3.2))
        dome = bpy.context.active_object
        dome.scale = (1.0, 0.72, 0.65)
        dome.data.materials.append(glass)
        # Left dais + white grand piano blockout
        add_cylinder("PianoDais", (-2.6, 8.8, 0.14), 1.85, 0.28, ivory)
        add_box("PianoBody", (-2.65, 8.8, 0.95), (1.45, 1.25, 0.65), white, rot_z=0.25)
        add_box("PianoLid", (-2.65, 8.7, 1.48), (1.35, 1.15, 0.06), white, rot_z=0.25)
        # Foreground low balustrade left
        add_box("RailTop", (-2.8, 4.1, 0.82), (4.6, 0.14, 0.10), white)
        for p_i in range(6):
            add_cylinder(f"Bal_{p_i}", (-4.8 + p_i * 0.85, 4.1, 0.41), 0.06, 0.82, white)
    elif preset == "european-arcade":
        bpy.ops.mesh.primitive_plane_add(size=32.0, location=(0.0, 9.0, 0.0))
        bpy.context.active_object.data.materials.append(ivory)
        for k in range(7):
            x = -6.0 + k * 2.0
            add_box(f"Pier_{k}", (x, 10.2, 2.4), (0.55, 0.65, 4.8), white)
            add_box(f"Lint_{k}", (x + 1.0, 10.2, 4.9), (2.1, 0.65, 0.6), white)
        add_box("BackWall", (0.0, 14.5, 2.8), (18.0, 0.5, 5.6), ivory)
    elif preset == "seaside-stage":
        bpy.ops.mesh.primitive_plane_add(size=40.0, location=(0.0, 18.0, -0.4))
        bpy.context.active_object.data.materials.append(water)
        add_box("StageDeck", (0.0, 6.5, 0.05), (14.0, 11.0, 0.12), ivory)
        add_box("GlassRail", (0.0, 11.4, 0.58), (12.0, 0.06, 0.95), glass)
        add_cylinder("TrussL", (-5.2, 10.8, 3.2), 0.18, 6.4, gold)
        add_cylinder("TrussR", (5.2, 10.8, 3.2), 0.18, 6.4, gold)
        bpy.ops.mesh.primitive_torus_add(major_radius=4.5, minor_radius=0.14, location=(-3.6, 26.0, 5.4), rotation=(math.radians(90), 0, 0))
        bpy.context.active_object.data.materials.append(white)
    elif preset == "rooftop-twilight":
        add_box("RoofDeck", (0.0, 7.0, 0.0), (16.0, 13.0, 0.2), ivory)
        add_box("GuardRail", (0.0, 12.2, 0.55), (14.5, 0.08, 0.9), white)
        for k, bx in enumerate((-7.5, -4.2, -1.2, 2.6, 6.8)):
            bh = 4.0 + (k % 3) * 2.4
            add_box(f"SkyBldg_{k}", (bx, 25.0 + (k % 2) * 4.0, bh * 0.5 - 2.0), (2.2, 2.2, bh), white)
    elif preset == "eastern-pavilion":
        bpy.ops.mesh.primitive_plane_add(size=36.0, location=(0.0, 12.0, -0.18))
        bpy.context.active_object.data.materials.append(water)
        add_box("Deck", (0.0, 5.5, 0.05), (12.0, 9.0, 0.12), ivory)
        for k in (-3.6, -1.2, 1.2, 3.6):
            add_cylinder(f"PavCol_{k}", (k, 10.4, 2.4), 0.15, 4.8, gold)
        add_box("PavRoof", (0.0, 10.6, 5.1), (9.4, 3.2, 0.45), white)
        add_box("MoonBridge", (-3.2, 14.5, 0.55), (5.5, 1.2, 0.18), white, rot_z=0.18)
    else:  # studio-cyclorama
        add_box("FloorSweep", (0.0, 8.0, 0.0), (18.0, 14.0, 0.1), ivory)
        add_box("CycWall", (0.0, 13.8, 3.5), (18.0, 0.4, 7.0), white)
        for k in range(5):
            add_box(f"Gobo_{k}", (4.5, 7.5 + k * 0.9, 2.4), (0.08, 0.35, 3.8), white)

    return sun_ray


def main() -> None:
    args = parse_args()
    out_d = Path(args.out_dir).resolve()
    out_d.mkdir(parents=True, exist_ok=True)
    w, h = args.width, args.height

    setup_cycles(w, h, args.samples)
    scene = bpy.context.scene
    sun_ray = build_preset(args.preset, args.sun_azimuth, args.sun_elevation)

    cam_d = bpy.data.cameras.new("MainCam")
    cam_d.lens = args.lens
    cam_d.sensor_fit = "HORIZONTAL"
    cam_d.sensor_width = 36.0
    cam_o = bpy.data.objects.new("MainCam", cam_d)
    bpy.context.collection.objects.link(cam_o)
    cam_o.location = (0.0, 0.0, 1.48)
    cam_o.rotation_euler = (math.radians(92.0), 0.0, 0.0)
    scene.camera = cam_o

    p_wm = out_d / "01_blender_whitemodel_empty.png"
    scene.render.filepath = str(p_wm)
    bpy.ops.render.render(write_still=True)

    # Optional subject layout reference preview
    p_prev = out_d / "03_layout_preview_with_subject.png"
    if args.subject_rgba and Path(args.subject_rgba).exists():
        from PIL import Image

        bg_im = Image.open(p_wm).convert("RGB")
        sub_im = Image.open(Path(args.subject_rgba)).convert("RGBA")
        target_h = int(h * 0.88)
        target_w = max(16, int(round(sub_im.width * target_h / sub_im.height)))
        sub_rs = sub_im.resize((target_w, target_h), Image.Resampling.LANCZOS)
        px = int(np.clip(w * args.subject_x_frac - target_w * 0.5, 0, max(0, w - target_w)))
        py = h - target_h
        bg_im.paste(sub_rs.convert("RGB"), (px, py), sub_rs.split()[-1])
        bg_im.save(p_prev, "PNG")

    # 16-bit normalized linear Z-depth map
    d_mat = bpy.data.materials.new("ZDepth")
    d_mat.use_nodes = True
    nt = d_mat.node_tree
    nt.nodes.clear()
    out_d_n = nt.nodes.new("ShaderNodeOutputMaterial")
    cam_n = nt.nodes.new("ShaderNodeCameraData")
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.clamp = True
    mr.inputs["From Min"].default_value = 1.5
    mr.inputs["From Max"].default_value = 28.0
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(cam_n.outputs["View Z Depth"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], em.inputs["Color"])
    nt.links.new(em.outputs["Emission"], out_d_n.inputs["Surface"])

    bpy.context.view_layer.material_override = d_mat
    scene.view_settings.view_transform = "Raw"
    scene.render.image_settings.color_mode = "BW"
    scene.render.image_settings.color_depth = "16"
    scene.cycles.samples = 12
    scene.cycles.use_denoising = False
    p_zd = out_d / "02_blender_zdepth_16bit.png"
    scene.render.filepath = str(p_zd)
    bpy.ops.render.render(write_still=True)

    sun_side = "right" if args.sun_azimuth > 12 else ("left" if args.sun_azimuth < -12 else "back")
    meta = {
        "ok": True,
        "preset": args.preset,
        "width": w,
        "height": h,
        "lens_mm": args.lens,
        "sun_azimuth_deg": args.sun_azimuth,
        "sun_elevation_deg": args.sun_elevation,
        "sun_ray_vec": [round(float(v), 4) for v in sun_ray],
        "sun_side": sun_side,
        "subject_x_frac": args.subject_x_frac,
        "zdepth_range_m": [1.5, 28.0],
        "outputs": {
            "whitemodel_empty": str(p_wm),
            "zdepth_16bit": str(p_zd),
            "layout_preview": str(p_prev) if p_prev.exists() else None,
        },
    }
    (out_d / "scene_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    pr = PRESET_PROMPTS[args.preset]
    md = (
        f"# {args.preset} 抽卡提示词\n\n"
        f"## 1. 高清空镜背景（仅上传 `01_blender_whitemodel_empty.png` 作为垫底参考）\n"
        f"- 中文：{pr['bg_cn']}\n"
        f"- English：{pr['bg_en']}\n\n"
        f"## 2. 纯黑底复杂粒子特效叠层（`#000000` 纯黑底色，勿垫建筑参考图）\n"
        f"- English：{pr['vfx_en']}\n"
    )
    (out_d / "prompts_bg_and_vfx.md").write_text(md, encoding="utf-8")
    print("SCENE_WHITEMODEL_DONE")


if __name__ == "__main__":
    main()
