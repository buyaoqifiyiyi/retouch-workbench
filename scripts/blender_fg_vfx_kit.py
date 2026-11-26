"""Blender 5.x Cycles GPU 3D optical-bokeh foreground framing props & pose-wrapped interactive light filaments."""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Any

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(description="Blender GPU 3D Foreground Props & Interactive Pose-Wrap VFX Kit")
    ap.add_argument("--preset", default="crystal-conservatory", choices=["crystal-conservatory", "european-arcade", "seaside-stage", "rooftop-twilight", "eastern-pavilion"])
    ap.add_argument("--width", type=int, default=2560)
    ap.add_argument("--height", type=int, default=1440)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--subject-x-frac", type=float, default=0.70)
    ap.add_argument("--out-dir", required=True)
    return ap.parse_args(argv)


def uv_to_world(u: float, v_top: float, dist: float) -> Vector:
    """Convert normalized 16:9 image UV (origin top-left) + camera distance into exact world XYZ for lens=50mm/36mm sensor."""
    x = (2.0 * u - 1.0) * 0.36 * dist
    y = -5.0 + dist
    z = 1.55 + (1.0 - 2.0 * v_top) * 0.2025 * dist
    return Vector((x, y, z))


def setup_scene(w: int, h: int, samples: int) -> Any:
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
                print(f"FG_VFX_3D_GPU={btype}")
                break
        except Exception:
            pass
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.035
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 6
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 3
    scene.cycles.transmission_bounces = 6
    scene.cycles.transparent_max_bounces = 8

    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    if "AgX" in [i.identifier for i in scene.view_settings.bl_rna.properties["view_transform"].enum_items]:
        scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = 0.15

    # Camera matching cinema framing
    cam_data = bpy.data.cameras.new("MainCam")
    cam_data.lens = 50.0
    cam_data.sensor_width = 36.0
    cam_obj = bpy.data.objects.new("MainCam", cam_data)
    scene.collection.objects.link(cam_obj)
    cam_obj.location = (0.0, -5.0, 1.55)
    cam_obj.rotation_euler = (math.pi / 2.0, 0.0, 0.0)
    scene.camera = cam_obj
    return cam_obj


def set_input(bsdf: Any, name: str, val: Any) -> None:
    if name in bsdf.inputs:
        bsdf.inputs[name].default_value = val


def make_botanical_mat(name: str, base_rgb: tuple[float, float, float], subsurf: float = 0.35, rough: float = 0.38, coat: float = 0.25) -> Any:
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    set_input(bsdf, "Base Color", (*base_rgb, 1.0))
    set_input(bsdf, "Roughness", rough)
    set_input(bsdf, "Subsurface Weight", subsurf)
    set_input(bsdf, "Subsurface Radius", (0.4, 0.2, 0.2))
    set_input(bsdf, "Coat Weight", coat)
    return m


def make_iron_mat(name: str, rgb: tuple[float, float, float], metal: float = 0.85, rough: float = 0.36) -> Any:
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    set_input(bsdf, "Base Color", (*rgb, 1.0))
    set_input(bsdf, "Metallic", metal)
    set_input(bsdf, "Roughness", rough)
    return m


def make_crystal_mat(name: str = "FG_CrystalGlass") -> Any:
    """Warm champagne-translucent crystal with inner sunset glow (eliminates black refraction ring)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    mix = nt.nodes.new("ShaderNodeMixShader")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    tr.inputs["Color"].default_value = (0.98, 0.90, 0.78, 1.0)
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (1.0, 0.86, 0.64, 1.0)
    em.inputs["Strength"].default_value = 1.65
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.35
    nt.links.new(lw.outputs["Facing"], mix.inputs["Fac"])
    nt.links.new(tr.outputs["BSDF"], mix.inputs[1])
    nt.links.new(em.outputs["Emission"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return m


def make_silk_ribbon_mat(name: str, rgb: tuple[float, float, float], strength: float = 3.2) -> Any:
    """Silky soft-feathered emission shader (bright core, glassy transparent edges via LayerWeight)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    mix = nt.nodes.new("ShaderNodeMixShader")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*rgb, 1.0)
    em.inputs["Strength"].default_value = strength
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.45
    inv = nt.nodes.new("ShaderNodeMath")
    inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1.0
    pw = nt.nodes.new("ShaderNodeMath")
    pw.operation = "POWER"
    pw.inputs[1].default_value = 1.45
    nt.links.new(lw.outputs["Facing"], inv.inputs[1])
    nt.links.new(inv.outputs["Value"], pw.inputs[0])
    nt.links.new(pw.outputs["Value"], mix.inputs["Fac"])
    nt.links.new(tr.outputs["BSDF"], mix.inputs[1])
    nt.links.new(em.outputs["Emission"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return m


def ensure_collection(name: str) -> Any:
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def link_to_col(obj: Any, col: Any) -> None:
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    col.objects.link(obj)


def add_curve_tube(
    name: str,
    col: Any,
    pts_with_radius: list[tuple[Vector, float]],
    mat: Any,
    res_u: int = 24,
) -> Any:
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = res_u
    curve.bevel_depth = 1.0
    curve.bevel_resolution = 4
    curve.use_fill_caps = True
    spline = curve.splines.new("NURBS")
    spline.points.add(len(pts_with_radius) - 1)
    for idx, (pt, rad) in enumerate(pts_with_radius):
        spline.points[idx].co = (pt.x, pt.y, pt.z, 1.0)
        spline.points[idx].radius = rad
    spline.use_endpoint_u = True
    spline.order_u = min(4, len(pts_with_radius))
    obj = bpy.data.objects.new(name, curve)
    obj.data.materials.append(mat)
    col.objects.link(obj)
    return obj


def add_rose_blossom(name: str, col: Any, center: Vector, scale: float, rot_euler: tuple[float, float, float], petal_mat: Any, core_mat: Any) -> None:
    rng = random.Random(hash(name) & 0xFFFF)
    parent = bpy.data.objects.new(name, None)
    parent.location = center
    parent.rotation_euler = rot_euler
    col.objects.link(parent)

    # Inner dark bud sphere
    bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=8, radius=0.018 * scale, location=(0, 0, 0))
    bud = bpy.context.active_object
    bud.name = f"{name}_bud"
    bud.parent = parent
    bud.data.materials.append(core_mat)
    link_to_col(bud, col)

    # 3 concentric rings of sculpted cupped petals
    rings = [(4, 0.022, 0.028, 0.55), (6, 0.036, 0.036, 0.88), (7, 0.052, 0.044, 1.18)]
    for r_i, (count, r_dist, p_sz, tilt) in enumerate(rings):
        for k in range(count):
            ang = (2.0 * math.pi * k / count) + r_i * 0.42 + rng.uniform(-0.15, 0.15)
            bpy.ops.mesh.primitive_grid_add(x_subdivisions=4, y_subdivisions=4, size=p_sz * scale)
            pet = bpy.context.active_object
            pet.name = f"{name}_p_{r_i}_{k}"
            for v in pet.data.vertices:
                v.co.z += 0.012 * scale * (1.0 - (v.co.x / max(1e-4, 0.5 * p_sz * scale)) ** 2)
                if v.co.y < 0:
                    v.co.x *= 0.55
            pet.scale = (1.15, 1.0, 1.0)
            pet.location = (math.cos(ang) * r_dist * scale, math.sin(ang) * r_dist * scale, (0.008 - 0.004 * r_i) * scale)
            pet.rotation_euler = (tilt, rng.uniform(-0.12, 0.12), ang + math.pi / 2.0)
            pet.parent = parent
            pet.data.materials.append(petal_mat)
            link_to_col(pet, col)


def add_leaf(name: str, col: Any, pos: Vector, scale: float, rot: tuple[float, float, float], leaf_mat: Any) -> None:
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=4, y_subdivisions=6, size=0.065 * scale)
    lf = bpy.context.active_object
    lf.name = name
    for v in lf.data.vertices:
        t = v.co.y / max(1e-4, 0.0325 * scale)
        taper = max(0.08, 1.0 - abs(t) ** 1.45)
        v.co.x *= taper * 0.55
        v.co.z -= 0.012 * scale * abs(t) ** 1.6
    lf.location = pos
    lf.rotation_euler = rot
    lf.data.materials.append(leaf_mat)
    link_to_col(lf, col)


def add_crystal_drop(name: str, col: Any, top_uv: tuple[float, float], dist: float, length_u: float, radius: float, crystal_mat: Any, brass_mat: Any) -> None:
    top_p = uv_to_world(top_uv[0], top_uv[1], dist)
    bot_p = uv_to_world(top_uv[0], top_uv[1] + length_u, dist)
    add_curve_tube(f"{name}_chain", col, [(top_p + Vector((0, 0, 0.04)), 0.003), (top_p, 0.003)], brass_mat, res_u=8)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=radius, location=bot_p)
    drp = bpy.context.active_object
    drp.name = name
    drp.scale = (0.72, 0.72, 1.65)
    drp.data.materials.append(crystal_mat)
    link_to_col(drp, col)


def build_foreground_props(col_fg: Any, preset: str) -> None:
    iron_mat = make_iron_mat("FG_DarkIron", (0.045, 0.075, 0.095), metal=0.82, rough=0.36)
    gold_mat = make_iron_mat("FG_AntiqueGold", (0.68, 0.50, 0.25), metal=0.92, rough=0.28)
    leaf_mat = make_botanical_mat("FG_DarkIvyLeaf", (0.035, 0.095, 0.098), subsurf=0.35, rough=0.30, coat=0.35)
    rose_mat = make_botanical_mat("FG_BlushRosePetal", (0.68, 0.34, 0.39), subsurf=0.70, rough=0.40, coat=0.10)
    rose_cream = make_botanical_mat("FG_CreamRosePetal", (0.78, 0.64, 0.58), subsurf=0.65, rough=0.42, coat=0.10)
    rose_core = make_botanical_mat("FG_RoseCore", (0.28, 0.09, 0.13), subsurf=0.40, rough=0.48, coat=0.05)
    crystal_mat = make_crystal_mat("FG_CrystalPrism")

    # Warm golden sunset backlight + subtle amber glint light on foreground rims (dark silhouettes + bright rim bokeh!)
    bpy.ops.object.light_add(type="AREA", location=uv_to_world(0.45, 0.42, 6.5))
    sun_back = bpy.context.active_object
    sun_back.name = "FG_SunsetBacklight"
    sun_back.data.energy = 850.0
    sun_back.data.color = (1.0, 0.74, 0.44)
    sun_back.data.size = 3.6
    sun_back.rotation_euler = (-math.pi / 2.0, 0.0, 0.0)
    link_to_col(sun_back, col_fg)

    bpy.ops.object.light_add(type="AREA", location=uv_to_world(0.20, 0.15, 0.9))
    fill_cool = bpy.context.active_object
    fill_cool.name = "FG_CoolTealFill"
    fill_cool.data.energy = 14.0
    fill_cool.data.color = (0.28, 0.68, 0.75)
    fill_cool.data.size = 2.8
    fill_cool.rotation_euler = (math.pi / 2.0, 0.0, 0.0)
    link_to_col(fill_cool, col_fg)

    # 1. Right foreground framing arch rib & antique gold filigree scroll (dist=2.35, crisp painterly DoF, outside Miku u=0.865)
    if preset in ("crystal-conservatory", "european-arcade", "eastern-pavilion"):
        arch_pts = [
            (uv_to_world(0.925, -0.05, 2.38), 0.042),
            (uv_to_world(0.920, 0.18, 2.38), 0.036),
            (uv_to_world(0.945, 0.48, 2.38), 0.034),
            (uv_to_world(0.975, 0.82, 2.38), 0.040),
            (uv_to_world(1.010, 1.06, 2.38), 0.048),
        ]
        add_curve_tube("FG_RightArchPillar", col_fg, arch_pts, iron_mat)
        scroll_pts = [
            (uv_to_world(0.925, 0.06, 2.35), 0.014),
            (uv_to_world(0.875, 0.13, 2.35), 0.011),
            (uv_to_world(0.865, 0.22, 2.35), 0.010),
            (uv_to_world(0.915, 0.25, 2.35), 0.008),
        ]
        add_curve_tube("FG_RightArchScroll", col_fg, scroll_pts, gold_mat)

    # 2. Hanging warm champagne crystal chandelier drops (top-right arch only -> creamy golden optical bokeh discs!)
    crystals = [
        ((0.892, 0.0), 2.15, 0.11, 0.020),
        ((0.924, 0.0), 2.18, 0.17, 0.024),
        ((0.958, 0.0), 2.16, 0.14, 0.021),
        ((0.982, 0.02), 2.20, 0.20, 0.026),
    ]
    for idx, (uv, d_c, lu, rad) in enumerate(crystals):
        add_crystal_drop(f"FG_CrystalDrop_{idx}", col_fg, uv, d_c, lu, rad, crystal_mat, gold_mat)

    # 3. Bottom-right & bottom-center-left & bottom-left rose + dark-teal foliage clusters (dist=2.12..2.28)
    sprays = [
        ((1.03, 1.07), (0.89, 0.94), (0.775, 0.935), 2.12, rose_mat, 1.35),
        ((0.98, 1.09), (0.86, 0.99), (0.835, 0.955), 2.18, rose_cream, 1.18),
        ((0.95, 1.06), (0.915, 0.90), (0.900, 0.870), 2.22, rose_mat, 1.08),
        ((0.49, 1.09), (0.545, 0.98), (0.585, 0.942), 2.16, rose_cream, 1.12),
        ((0.47, 1.07), (0.505, 0.96), (0.538, 0.915), 2.22, rose_mat, 1.00),
        ((-0.03, 1.06), (0.06, 0.94), (0.115, 0.905), 2.10, rose_mat, 1.25),
        ((0.01, 1.08), (0.10, 1.00), (0.165, 0.972), 2.16, rose_cream, 1.08),
    ]
    rng = random.Random(20261126)
    for idx, (uv0, uv1, uv2, d_s, r_mat, sc) in enumerate(sprays):
        p0, p1, p2 = uv_to_world(uv0[0], uv0[1], d_s), uv_to_world(uv1[0], uv1[1], d_s), uv_to_world(uv2[0], uv2[1], d_s)
        add_curve_tube(f"FG_Stem_{idx}", col_fg, [(p0, 0.008 * sc), (p1, 0.006 * sc), (p2, 0.004 * sc)], leaf_mat)
        add_rose_blossom(f"FG_Rose_{idx}", col_fg, p2, sc * 1.15, (0.35 + rng.uniform(-0.2, 0.2), rng.uniform(-0.25, 0.25), rng.uniform(0, 6.28)), r_mat, rose_core)
        for li in range(3):
            t_l = 0.28 + 0.25 * li
            lp = p0.lerp(p2, t_l) + Vector((rng.uniform(-0.04, 0.04), rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.03)))
            add_leaf(f"FG_Leaf_{idx}_{li}", col_fg, lp, sc * rng.uniform(1.15, 1.55), (0.5 + rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(0, 6.28)), leaf_mat)

    # Extra framing dark leaves along far-left lower margin & far-right mid-lower frame
    extra_leaves = [
        (0.01, 0.78, 2.15, 2.1), (0.04, 0.84, 2.12, 2.2), (0.08, 0.98, 2.08, 2.0),
        (0.755, 0.985, 2.14, 1.5), (0.93, 0.76, 2.22, 1.9), (0.97, 0.83, 2.18, 2.0),
    ]
    for ei, (lu, lv, ld, lsc) in enumerate(extra_leaves):
        add_leaf(f"FG_EdgeLeaf_{ei}", col_fg, uv_to_world(lu, lv, ld), lsc, (0.42 + rng.uniform(-0.3, 0.3), rng.uniform(-0.5, 0.5), rng.uniform(0, 6.28)), leaf_mat)


def _sample_catmull_path(pts_uvd: list[tuple[float, float, float]], samples_per_seg: int = 16) -> list[tuple[float, float, float]]:
    out: list[tuple[float, float, float]] = []
    n = len(pts_uvd)
    for i in range(n - 1):
        p0 = pts_uvd[max(0, i - 1)]
        p1 = pts_uvd[i]
        p2 = pts_uvd[i + 1]
        p3 = pts_uvd[min(n - 1, i + 2)]
        for s in range(samples_per_seg + (1 if i == n - 2 else 0)):
            t = s / float(samples_per_seg)
            t2, t3 = t * t, t * t * t
            res = []
            for k in range(3):
                val = 0.5 * (
                    (2.0 * p1[k])
                    + (-p0[k] + p2[k]) * t
                    + (2.0 * p0[k] - 5.0 * p1[k] + 4.0 * p2[k] - p3[k]) * t2
                    + (-p0[k] + 3.0 * p1[k] - 3.0 * p2[k] + p3[k]) * t3
                )
                res.append(val)
            out.append((res[0], res[1], res[2]))
    return out


def build_pose_wrapped_filaments(col_behind: Any, col_front: Any, subject_x_frac: float) -> None:
    """Whisper-thin 3D tapered melody silk ribbons looping behind Miku's shoulder -> front across skirt -> piano lid."""
    dx = subject_x_frac - 0.70  # adaptable anchor shift
    gold_mat = make_silk_ribbon_mat("VFX_SilkGold", (1.0, 0.78, 0.44), strength=1.55)
    mint_mat = make_silk_ribbon_mat("VFX_SilkMint", (0.36, 0.92, 0.85), strength=1.40)
    mote_gold = make_silk_ribbon_mat("VFX_MoteGold", (1.0, 0.82, 0.52), strength=3.4)

    # Master trajectory from Miku's mic (right) -> looping BEHIND hair/shoulder -> sweeping IN FRONT of skirt -> across hall to piano
    ctrl_behind_top = [
        (0.805 + dx, 0.345, 4.92),
        (0.845 + dx, 0.380, 5.25),
        (0.820 + dx, 0.435, 5.65),
        (0.715 + dx, 0.465, 5.85),
        (0.605 + dx, 0.515, 5.55),
        (0.550 + dx, 0.595, 5.02),
    ]
    ctrl_front_wrap = [
        (0.550 + dx, 0.595, 4.92),
        (0.578 + dx, 0.648, 4.55),
        (0.665 + dx, 0.682, 4.35),
        (0.768 + dx, 0.715, 4.42),
        (0.742 + dx, 0.765, 4.68),
        (0.625 + dx, 0.780, 4.96),
    ]
    ctrl_behind_tail = [
        (0.625 + dx, 0.780, 5.04),
        (0.510 + dx * 0.6, 0.755, 5.25),
        (0.395, 0.705, 5.45),
        (0.295, 0.650, 5.65),
        (0.215, 0.600, 5.85),
    ]

    def build_strand(tag: str, col: Any, pts: list[tuple[float, float, float]], r_peak: float, mat: Any, offset_uv: tuple[float, float] = (0.0, 0.0), start_tap: bool = True, end_tap: bool = True) -> None:
        dense = _sample_catmull_path(pts, samples_per_seg=8)
        n_d = len(dense)
        tube_pts: list[tuple[Vector, float]] = []
        for idx, (u, v, d) in enumerate(dense):
            frac = idx / max(1, n_d - 1)
            tap = 1.0
            if start_tap and frac < 0.18:
                tap *= math.sin((frac / 0.18) * (math.pi / 2.0))
            if end_tap and frac > 0.82:
                tap *= math.sin(((1.0 - frac) / 0.18) * (math.pi / 2.0))
            rad = max(0.0012, r_peak * (0.28 + 0.72 * tap))
            wave = 0.006 * math.sin(frac * math.pi * 3.0)
            tube_pts.append((uv_to_world(u + offset_uv[0], v + offset_uv[1] + wave, d), rad))
        add_curve_tube(tag, col, tube_pts, mat, res_u=16)

    # Primary golden melody strand + airy mint counter-strand + delicate specular pearls
    build_strand("VFX_BehindTop_Gold", col_behind, ctrl_behind_top, 0.0082, gold_mat, (0.0, 0.0), True, True)
    build_strand("VFX_BehindTop_Mint", col_behind, ctrl_behind_top, 0.0046, mint_mat, (0.006, 0.011), True, True)

    build_strand("VFX_FrontWrap_Gold", col_front, ctrl_front_wrap, 0.0092, gold_mat, (0.0, 0.0), True, True)
    build_strand("VFX_FrontWrap_Mint", col_front, ctrl_front_wrap, 0.0050, mint_mat, (-0.005, 0.012), True, True)

    build_strand("VFX_BehindTail_Gold", col_behind, ctrl_behind_tail, 0.0072, gold_mat, (0.0, 0.0), True, True)
    build_strand("VFX_BehindTail_Mint", col_behind, ctrl_behind_tail, 0.0040, mint_mat, (0.004, -0.009), True, True)

    # Subtle glowing champagne-pearl motes drifting strictly beside the curved filaments (small, airy, tasteful)
    rng = random.Random(7788)
    for tag, col, path_pts, count in (
        ("BehindTop", col_behind, _sample_catmull_path(ctrl_behind_top, 6), 10),
        ("FrontWrap", col_front, _sample_catmull_path(ctrl_front_wrap, 6), 10),
        ("BehindTail", col_behind, _sample_catmull_path(ctrl_behind_tail, 6), 14),
    ):
        for mi in range(count):
            u, v, d = path_pts[int((mi + 0.5) / count * (len(path_pts) - 1))]
            jitter_u = rng.uniform(-0.022, 0.022)
            jitter_v = rng.uniform(-0.022, 0.022)
            pos = uv_to_world(u + jitter_u, v + jitter_v, d + rng.uniform(-0.10, 0.10))
            rad = rng.uniform(0.0032, 0.0072) * (d / 5.0)
            bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=rad, location=pos)
            sp = bpy.context.active_object
            sp.name = f"VFX_Mote_{tag}_{mi}"
            sp.data.materials.append(mote_gold if mi % 3 != 0 else mint_mat)
            link_to_col(sp, col)


def main() -> None:
    args = parse_args()
    out_d = Path(args.out_dir).resolve()
    out_d.mkdir(parents=True, exist_ok=True)
    cam = setup_scene(args.width, args.height, args.samples)
    scene = bpy.context.scene

    col_fg = ensure_collection("FG_Bokeh_Props")
    col_beh = ensure_collection("VFX_Wrap_Behind")
    col_frt = ensure_collection("VFX_Wrap_Front")

    build_foreground_props(col_fg, args.preset)
    build_pose_wrapped_filaments(col_beh, col_frt, float(args.subject_x_frac))

    # Render Pass A: Real Physical Optical Bokeh Foreground Props (f/0.85, focus=5.0m)
    col_fg.hide_render = False
    col_beh.hide_render = True
    col_frt.hide_render = True
    cam.data.dof.use_dof = True
    cam.data.dof.focus_distance = 5.0
    cam.data.dof.aperture_fstop = 1.18
    cam.data.dof.aperture_blades = 9
    world = bpy.data.worlds.new("FG_World")
    scene.world = world
    world.use_nodes = True
    bg_n = world.node_tree.nodes.get("Background")
    if bg_n:
        bg_n.inputs["Color"].default_value = (0.05, 0.11, 0.14, 1.0)
        bg_n.inputs["Strength"].default_value = 0.40
    scene.render.filepath = str(out_d / "fg_3d_props_raw.png")
    bpy.ops.render.render(write_still=True)

    # Render Pass B: Behind-Subject Wrapped Filaments (pitch-black world, gentle f/4.5 depth softness)
    col_fg.hide_render = True
    col_beh.hide_render = False
    col_frt.hide_render = True
    cam.data.dof.aperture_fstop = 4.5
    if bg_n:
        bg_n.inputs["Strength"].default_value = 0.0
    scene.render.filepath = str(out_d / "vfx_wrap_behind_raw.png")
    bpy.ops.render.render(write_still=True)

    # Render Pass C: Front-of-Subject Wrapped Filaments
    col_beh.hide_render = True
    col_frt.hide_render = False
    scene.render.filepath = str(out_d / "vfx_wrap_front_raw.png")
    bpy.ops.render.render(write_still=True)
    print("FG_VFX_3D_RENDER_DONE")


if __name__ == "__main__":
    main()
