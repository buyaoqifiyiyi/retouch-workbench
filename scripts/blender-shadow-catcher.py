"""
Render physically accurate 3D cast shadows and ground reflections from a 2D foreground cutout PNG
by re-projecting the cutout onto an upright 3D quad grounded along the standing heel/forefoot contact
line in Blender Cycles. Zero third-party Python deps required outside Blender's bundled numpy.
"""

import argparse
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Euler, Vector


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    p = argparse.ArgumentParser(description="Render 3D Shadow Catcher & Floor Reflection from a 2D cutout PNG")
    p.add_argument("--cutout", required=True, help="Path to transparent RGBA subject cutout PNG")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--width", type=int, default=2304)
    p.add_argument("--height", type=int, default=4128)
    p.add_argument("--lens", type=float, default=39.0)
    p.add_argument("--sensor-height", type=float, default=36.0)
    p.add_argument("--cam-loc", default="0.04,0.0,1.18", help="x,y,z in meters")
    p.add_argument("--cam-rot-deg", default="87.2,0.0,-3.2", help="rx,ry,rz in degrees")
    p.add_argument("--sun-dir", default="1.0,-0.34,-0.60", help="dx,dy,dz ray vector")
    p.add_argument("--sun-angle-deg", type=float, default=1.9)
    p.add_argument("--sun-energy", type=float, default=6.2)
    p.add_argument("--sky-rgb", default="0.58,0.72,0.94")
    p.add_argument("--sky-strength", type=float, default=0.85)
    p.add_argument("--floor-z", type=float, default=0.0)
    p.add_argument("--floor-roughness", type=float, default=0.22)
    p.add_argument("--heel-px", default="", help="Optional x,y top-down heel contact pixel")
    p.add_argument("--fore-px", default="", help="Optional x,y top-down forefoot contact pixel")
    p.add_argument("--samples", type=int, default=48)
    return p.parse_args(argv)


def parse_vec3(s: str) -> Vector:
    vals = [float(v.strip()) for v in s.split(",")]
    return Vector((vals[0], vals[1], vals[2]))


def parse_vec2(s: str):
    vals = [float(v.strip()) for v in s.split(",")]
    return vals[0], vals[1]


def detect_contact_pixels(img_rgba_topdown: np.ndarray):
    alpha = img_rgba_topdown[:, :, 3]
    ys, xs = np.where(alpha > 0.50)
    if len(xs) == 0:
        raise RuntimeError("Cutout alpha mask has no opaque foreground pixels.")
    y_bot = int(ys.max())
    row_slice = alpha[max(0, y_bot - 160):y_bot + 2, :]
    _, bx = np.where(row_slice > 0.45)
    x0, x1 = int(bx.min()), int(bx.max())
    heel_end = x0 + int(0.32 * (x1 - x0))
    fore_start = x0 + int(0.52 * (x1 - x0))
    heel_pts, fore_pts = [], []
    y_start = max(0, y_bot - 180)
    for x in range(x0, x1 + 1):
        col = np.where(alpha[y_start:y_bot + 2, x] > 0.45)[0]
        if len(col) == 0:
            continue
        yb = y_start + int(col.max())
        if x <= heel_end:
            heel_pts.append((float(x), float(yb)))
        elif x >= fore_start:
            fore_pts.append((float(x), float(yb)))
    if not heel_pts or not fore_pts:
        return (float(x0 + 0.25 * (x1 - x0)), float(y_bot)), (float(x0 + 0.75 * (x1 - x0)), float(y_bot))
    hx = float(np.median([p[0] for p in heel_pts]))
    hy = float(np.median([p[1] for p in heel_pts]))
    fx = float(np.median([p[0] for p in fore_pts]))
    fy = float(np.median([p[1] for p in fore_pts]))
    return (hx, hy), (fx, fy)


def setup_cycles(samples: int, w: int, h: int):
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
                print(f"BLENDER_SHADOW_CATCHER backend={btype}")
                break
        except Exception:
            pass
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.02
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 6
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 3
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"


def camera_ray(cam_obj, scene, u: float, v_topdown: float):
    """Return world-space (origin, unit_dir) for normalized top-left coords u in [0,1], v in [0,1]."""
    frame = cam_obj.data.view_frame(scene=scene)
    mw = cam_obj.matrix_world
    # view_frame corners: 0=TR, 1=BR, 2=BL, 3=TL
    tl = mw @ frame[3]
    tr = mw @ frame[0]
    br = mw @ frame[1]
    bl = mw @ frame[2]
    p_uv = (
        tl * ((1.0 - u) * (1.0 - v_topdown))
        + tr * (u * (1.0 - v_topdown))
        + bl * ((1.0 - u) * v_topdown)
        + br * (u * v_topdown)
    )
    origin = mw.translation.copy()
    return origin, (p_uv - origin).normalized()


def intersect_plane(origin: Vector, d: Vector, p0: Vector, normal: Vector) -> Vector:
    denom = d.dot(normal)
    if abs(denom) < 1e-6:
        raise RuntimeError("Ray parallel to plane.")
    t = (p0 - origin).dot(normal) / denom
    return origin + d * t


def build_cutout_material(img, emissive=False):
    mat = bpy.data.materials.new("CutoutCardMat_" + ("Emissive" if emissive else "Shadow"))
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.extension = "CLIP"
    trans = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")

    if emissive:
        main_shader = nt.nodes.new("ShaderNodeEmission")
        main_shader.inputs["Strength"].default_value = 1.0
        nt.links.new(tex.outputs["Color"], main_shader.inputs["Color"])
        nt.links.new(main_shader.outputs["Emission"], mix.inputs[2])
    else:
        main_shader = nt.nodes.new("ShaderNodeBsdfPrincipled")
        main_shader.inputs["Roughness"].default_value = 0.75
        nt.links.new(tex.outputs["Color"], main_shader.inputs["Base Color"])
        nt.links.new(main_shader.outputs["BSDF"], mix.inputs[2])

    nt.links.new(tex.outputs["Alpha"], mix.inputs["Fac"])
    nt.links.new(trans.outputs["BSDF"], mix.inputs[1])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return mat


def save_srgb_multiply_png(with_path: Path, without_path: Path, out_path: Path, w: int, h: int):
    img_w = bpy.data.images.load(str(with_path))
    img_o = bpy.data.images.load(str(without_path))
    arr_w = np.array(img_w.pixels[:], dtype=np.float32).reshape(h, w, 4)
    arr_o = np.array(img_o.pixels[:], dtype=np.float32).reshape(h, w, 4)
    ratio_lin = np.clip(arr_w[:, :, :3] / np.maximum(arr_o[:, :, :3], 0.015), 0.0, 1.0)
    # Convert physical linear shadow ratio into sRGB Multiply factor
    ratio_srgb = np.power(ratio_lin, 1.0 / 2.2)

    out_img = bpy.data.images.new("ShadowMultiply", width=w, height=h, alpha=False)
    out_img.colorspace_settings.name = "Non-Color"
    rgba = np.dstack([ratio_srgb, np.ones((h, w, 1), dtype=np.float32)])
    out_img.pixels.foreach_set(rgba.ravel())
    out_img.filepath_raw = str(out_path)
    out_img.file_format = "PNG"
    out_img.save()
    bpy.data.images.remove(img_w)
    bpy.data.images.remove(img_o)
    bpy.data.images.remove(out_img)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    w, h = args.width, args.height

    setup_cycles(args.samples, w, h)
    scene = bpy.context.scene

    cutout_img = bpy.data.images.load(str(Path(args.cutout)))
    img_w, img_h = cutout_img.size[0], cutout_img.size[1]
    px_rgba = np.array(cutout_img.pixels[:], dtype=np.float32).reshape(img_h, img_w, 4)[::-1]

    if args.heel_px and args.fore_px:
        heel_px = parse_vec2(args.heel_px)
        fore_px = parse_vec2(args.fore_px)
    else:
        heel_px, fore_px = detect_contact_pixels(px_rgba)
    print(f"CONTACT_PX heel={heel_px} fore={fore_px}")

    # World sky fill (drives realistic cool French sky illumination inside cast shadows)
    world = bpy.data.worlds.new("ShadowWorld")
    scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    sky_rgb = parse_vec3(args.sky_rgb)
    bg_node.inputs["Color"].default_value = (sky_rgb.x, sky_rgb.y, sky_rgb.z, 1.0)
    bg_node.inputs["Strength"].default_value = args.sky_strength

    # Sun light
    sun_dir = parse_vec3(args.sun_dir).normalized()
    bpy.ops.object.light_add(type="SUN", location=(-4.0, 6.0, 5.0))
    sun_obj = bpy.context.active_object
    sun_obj.name = "Scene_Sun"
    sun_obj.rotation_euler = sun_dir.to_track_quat("-Z", "Y").to_euler()
    sun_obj.data.energy = args.sun_energy
    sun_obj.data.angle = math.radians(args.sun_angle_deg)
    sun_obj.data.color = (1.0, 0.96, 0.90)

    # Exact camera
    cam_data = bpy.data.cameras.new("MainCamera")
    cam_data.lens = args.lens
    cam_data.sensor_fit = "VERTICAL"
    cam_data.sensor_height = args.sensor_height
    cam_obj = bpy.data.objects.new("MainCamera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.location = parse_vec3(args.cam_loc)
    rot_deg = parse_vec3(args.cam_rot_deg)
    cam_obj.rotation_euler = Euler((math.radians(rot_deg.x), math.radians(rot_deg.y), math.radians(rot_deg.z)), "XYZ")
    scene.camera = cam_obj
    bpy.context.view_layer.update()

    # Ground contact anchors on Z = floor_z
    z_vec = Vector((0.0, 0.0, 1.0))
    p_floor = Vector((0.0, 0.0, args.floor_z))
    o_h, d_h = camera_ray(cam_obj, scene, heel_px[0] / float(img_w), heel_px[1] / float(img_h))
    o_f, d_f = camera_ray(cam_obj, scene, fore_px[0] / float(img_w), fore_px[1] / float(img_h))
    p_heel = intersect_plane(o_h, d_h, p_floor, z_vec)
    p_fore = intersect_plane(o_f, d_f, p_floor, z_vec)

    edge = Vector((p_fore.x - p_heel.x, p_fore.y - p_heel.y, 0.0)).normalized()
    card_normal = Vector((-edge.y, edge.x, 0.0))
    if card_normal.dot(cam_obj.location - p_heel) < 0:
        card_normal = -card_normal

    verts_uv = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    verts = []
    for (u, v_td) in verts_uv:
        o_c, d_c = camera_ray(cam_obj, scene, u, v_td)
        verts.append(tuple(intersect_plane(o_c, d_c, p_heel, card_normal)))

    mesh = bpy.data.meshes.new("CutoutCardMesh")
    mesh.from_pydata(verts, [], [(0, 1, 2, 3)])
    uv_layer = mesh.uv_layers.new(name="UVMap")
    uv_coords = [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
    for i, loop in enumerate(mesh.loops):
        uv_layer.data[i].uv = uv_coords[loop.vertex_index]
    mesh.update()

    card_obj = bpy.data.objects.new("Subject_Cutout_Card", mesh)
    bpy.context.collection.objects.link(card_obj)
    mat_shadow = build_cutout_material(cutout_img, emissive=False)
    mat_emissive = build_cutout_material(cutout_img, emissive=True)
    card_obj.data.materials.append(mat_shadow)
    card_obj.visible_camera = False
    card_obj.visible_shadow = True
    card_obj.visible_diffuse = True
    card_obj.visible_glossy = True

    # Orthogonal sun-facing cross-silhouette card + shoe-sole volume so side sunlight never reduces a flat card to a razor-thin line!
    p_mid = 0.5 * (p_heel + p_fore)
    sun_horiz = Vector((-sun_dir.x, -sun_dir.y, 0.0))
    cross_normal = sun_horiz.normalized() if sun_horiz.length > 1e-4 else Vector((1.0, 0.0, 0.0))
    if cross_normal.dot(card_normal) < 0:
        cross_normal = -cross_normal
    # Rotate original card vertices 70 deg around vertical Z axis through p_mid (compressed 0.55x across cross plane for human torso/skirt girth)
    cam_right = Vector((card_normal.y, -card_normal.x, 0.0)).normalized()
    cross_right = Vector((-cross_normal.y, cross_normal.x, 0.0)).normalized()
    cross_verts = []
    for vx, vy, vz in verts:
        rel = Vector((vx, vy, vz)) - p_mid
        horiz_coord = rel.dot(cam_right) * 0.58
        nv = p_mid + cross_right * horiz_coord + Vector((0.0, 0.0, rel.z))
        cross_verts.append((nv.x, nv.y, max(args.floor_z - 0.01, nv.z)))
    cross_mesh = bpy.data.meshes.new("CutoutCrossMesh")
    cross_mesh.from_pydata(cross_verts, [], [(0, 1, 2, 3)])
    cross_uv = cross_mesh.uv_layers.new(name="UVMap")
    for i, loop in enumerate(cross_mesh.loops):
        cross_uv.data[i].uv = uv_coords[loop.vertex_index]
    cross_mesh.update()
    cross_obj = bpy.data.objects.new("Subject_Cross_Volume_Card", cross_mesh)
    bpy.context.collection.objects.link(cross_obj)
    cross_obj.data.materials.append(mat_shadow)
    cross_obj.visible_camera = False
    cross_obj.visible_shadow = True
    cross_obj.visible_diffuse = True
    cross_obj.visible_glossy = False

    # Solid shoe sole thickness prism bridging P_heel -> P_fore so standing heel & forefoot anchor crisply
    sole_dir = p_fore - p_heel
    sole_len = max(0.08, sole_dir.length)
    sole_yaw = math.atan2(sole_dir.y, sole_dir.x)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(p_mid.x, p_mid.y, args.floor_z + 0.065), rotation=(0.0, 0.0, sole_yaw))
    shoe_obj = bpy.context.active_object
    shoe_obj.name = "Standing_Shoe_Volume"
    shoe_obj.scale = (sole_len * 0.92, 0.095, 0.13)
    shoe_obj.visible_camera = False
    shoe_obj.visible_shadow = True
    shoe_obj.visible_diffuse = False
    shoe_obj.visible_glossy = False

    # Large horizontal floor catcher plane
    bpy.ops.mesh.primitive_plane_add(size=40.0, location=(0.0, 12.0, args.floor_z))
    floor_obj = bpy.context.active_object
    floor_obj.name = "Ground_Shadow_Catcher"
    floor_mat = bpy.data.materials.new("GroundStone")
    floor_mat.use_nodes = True
    bsdf = floor_mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.85, 0.85, 0.86, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.86
    floor_obj.data.materials.append(floor_mat)

    # Pass 1: Native RGBA Shadow Catcher
    floor_obj.is_shadow_catcher = True
    scene.render.film_transparent = True
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "AgX" if "AgX" in [i.identifier for i in scene.view_settings.bl_rna.properties["view_transform"].enum_items] else "Filmic"
    scene.render.filepath = str(out_dir / "01_blender_shadow_catcher_rgba.png")
    bpy.ops.render.render(write_still=True)

    # Pass 2: Linear with/without ratio -> Color-accurate Multiply map for PhotoCraft
    floor_obj.is_shadow_catcher = False
    scene.render.film_transparent = False
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "16"
    scene.view_settings.view_transform = "Raw"
    tmp_with = out_dir / "_tmp_floor_with_shadow.png"
    tmp_without = out_dir / "_tmp_floor_no_shadow.png"

    for o in (card_obj, cross_obj, shoe_obj):
        o.visible_shadow = True
    card_obj.visible_diffuse = True
    cross_obj.visible_diffuse = True
    scene.render.filepath = str(tmp_with)
    bpy.ops.render.render(write_still=True)

    for o in (card_obj, cross_obj, shoe_obj):
        o.visible_shadow = False
        o.visible_diffuse = False
    scene.render.filepath = str(tmp_without)
    bpy.ops.render.render(write_still=True)

    mult_out = out_dir / "02_blender_shadow_multiply.png"
    save_srgb_multiply_png(tmp_with, tmp_without, mult_out, w, h)
    tmp_with.unlink(missing_ok=True)
    tmp_without.unlink(missing_ok=True)

    # Pass 3: Soft Floor Stone Reflection pass (uses clean front Subject_Cutout_Card only!)
    bg_node.inputs["Strength"].default_value = 0.0
    sun_obj.data.energy = 0.0
    card_obj.data.materials[0] = mat_emissive
    for o in (card_obj, cross_obj, shoe_obj):
        o.visible_shadow = False
        o.visible_diffuse = False
    cross_obj.visible_glossy = False
    shoe_obj.visible_glossy = False
    card_obj.visible_glossy = True

    bsdf.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    bsdf.inputs["Roughness"].default_value = args.floor_roughness
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.85
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 0.45

    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "AgX" if "AgX" in [i.identifier for i in scene.view_settings.bl_rna.properties["view_transform"].enum_items] else "Filmic"
    scene.render.filepath = str(out_dir / "03_blender_floor_reflection.png")
    bpy.ops.render.render(write_still=True)
    print("BLENDER_SHADOW_CATCHER_DONE")


if __name__ == "__main__":
    main()
