# blender -b -P turntable.py -- <in.glb> <out_dir> [n=8] [shading=TEXTURE|SOLID]
import bpy, sys, math, mathutils
argv = sys.argv[sys.argv.index("--") + 1:]
src, out = argv[0], argv[1]
n = int(argv[2]) if len(argv) > 2 else 8
shade = argv[3] if len(argv) > 3 else "TEXTURE"
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
pts = [o.matrix_world @ mathutils.Vector(c) for o in objs for c in o.bound_box]
lo = mathutils.Vector([min(p[i] for p in pts) for i in range(3)])
hi = mathutils.Vector([max(p[i] for p in pts) for i in range(3)])
ctr = (lo + hi) / 2
size = max(hi - lo)
print("BBOX", tuple(round(v, 4) for v in lo), tuple(round(v, 4) for v in hi))
scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.light = "STUDIO"
scene.display.shading.color_type = shade
scene.render.resolution_x = scene.render.resolution_y = 768
scene.render.film_transparent = False
scene.world = bpy.data.worlds.new("w")
scene.display.shading.background_type = "VIEWPORT"
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
scene.collection.objects.link(cam)
scene.camera = cam
cam.data.type = "ORTHO"
cam.data.ortho_scale = size * 1.15
for i in range(n):
    a = 2 * math.pi * i / n
    # glTF imports Y-up as Blender Z-up: orbit around Blender Z
    cam.location = ctr + mathutils.Vector((math.sin(a) * size * 3, -math.cos(a) * size * 3, 0))
    d = ctr - cam.location
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = f"{out}/v{i:02d}.png"
    bpy.ops.render.render(write_still=True)
