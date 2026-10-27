# blender -b -P sil.py -- <in.glb> <out.png> <fov_rad> <distance> <res> [axis=+Y]
# silhouette (white on black) from a perspective camera at distance*axis looking at the origin, Blender Z up
import bpy, sys, math, mathutils
argv = sys.argv[sys.argv.index("--") + 1:]
src, out, fov, dist, res = argv[0], argv[1], float(argv[2]), float(argv[3]), int(argv[4])
axis = argv[5] if len(argv) > 5 else "+Y"
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
s = bpy.context.scene
s.render.engine = "BLENDER_WORKBENCH"; s.display.shading.light = "FLAT"; s.display.shading.color_type = "SINGLE"
s.display.shading.single_color = (1, 1, 1); s.display.shading.background_type = "VIEWPORT"; s.display.shading.background_color = (0, 0, 0)
s.view_settings.view_transform = "Standard"
s.render.resolution_x = s.render.resolution_y = res
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); s.collection.objects.link(cam); s.camera = cam
cam.data.type = "PERSP"; cam.data.sensor_fit = "HORIZONTAL"; cam.data.angle = fov
sgn = -1 if axis[0] == "-" else 1
v = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[axis[1]]
cam.location = mathutils.Vector(v) * sgn * dist
cam.rotation_euler = (-cam.location).to_track_quat("-Z", "Y").to_euler()
s.render.filepath = out
bpy.ops.render.render(write_still=True)
