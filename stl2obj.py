'''

Credits to Anna Sirota.

'''

import bpy
import sys

argv = sys.argv
argv = argv[argv.index("--") + 1:] # get all args after "--"

stl_in = argv[0]
obj_out = argv[1]


bpy.ops.import_mesh.stl(filepath=stl_in, axis_forward='-Z', axis_up='Y')

for obj in bpy.data.objects:
    obj.select = True

bpy.data.objects['Torus'].select = False
bpy.data.objects['Lamp'].select = False
bpy.data.objects['Camera'].select = False

bpy.ops.export_scene.obj(filepath=obj_out, axis_forward='-Z', axis_up='Y', use_selection=True)