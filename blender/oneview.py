# blender -b -P oneview.py -- <in.glb> <out.png> <azimuth_deg> <shading SOLID|TEXTURE> [res=1024] [elev_deg=0]
import bpy, sys, math, mathutils
argv = sys.argv[sys.argv.index("--") + 1:]
src, out, az, shade = argv[0], argv[1], float(argv[2]), argv[3]
res = int(argv[4]) if len(argv) > 4 else 1024
el = math.radians(float(argv[5])) if len(argv) > 5 else 0.0
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
pts = [o.matrix_world @ mathutils.Vector(c) for o in objs for c in o.bound_box]
lo = mathutils.Vector([min(p[i] for p in pts) for i in range(3)]); hi = mathutils.Vector([max(p[i] for p in pts) for i in range(3)])
ctr = (lo + hi) / 2; size = max(hi - lo)
s = bpy.context.scene
s.render.engine = "BLENDER_WORKBENCH"; s.display.shading.light = "STUDIO"; s.display.shading.color_type = "SINGLE" if shade == "SOLID" else shade
s.display.shading.show_cavity = shade == "SOLID"
s.render.resolution_x = s.render.resolution_y = res
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); s.collection.objects.link(cam); s.camera = cam
cam.data.type = "ORTHO"; cam.data.ortho_scale = size * 1.05
a = math.radians(az)
cam.location = ctr + mathutils.Vector((math.sin(a) * math.cos(el), -math.cos(a) * math.cos(el), math.sin(el))) * size * 3
cam.rotation_euler = (ctr - cam.location).to_track_quat("-Z", "Y").to_euler()
s.render.filepath = out
bpy.ops.render.render(write_still=True)
