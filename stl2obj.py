'''

Credits to Anna Sirota.

'''

import bpy
import sys

argv = sys.argv
argv = argv[argv.index("--") + 1:] # get all args after "--"

stl_in = argv[0]
obj_out = argv[1]

r = float(argv[2])
g = float(argv[3])
b = float(argv[4])


bpy.ops.import_mesh.stl(filepath=stl_in, axis_forward='-Z', axis_up='Y')

for obj in bpy.data.objects:
    obj.select = True

bpy.data.objects['Torus'].select = False
bpy.data.objects['Lamp'].select = False
bpy.data.objects['Camera'].select = False

cur_obj = bpy.context.selected_objects[0]
cur_obj.scale = (0.03,0.03,0.03)

mat = bpy.data.materials.new(name="Mat") #set new material to variable
cur_obj.data.materials.append(mat) #add the material to the object
bpy.context.object.active_material.diffuse_color = (r, g, b) #change color

bpy.ops.export_scene.obj(filepath=obj_out, axis_forward='-Z', axis_up='Y', use_selection=True, use_materials=True)