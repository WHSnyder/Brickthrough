import bpy
import random
import math
import time
import numpy as np
import json
import mathutils as mu
import os
from math import degrees

write_path = "/home/will/projects/legoproj/cvscripts/calib_data/"

objdata = {}

scene = bpy.context.scene
scene_objs = bpy.data.objects

camera = bpy.data.objects['Camera']
calib = scene_objs["Calib"]


scene.render.resolution_x = 512
scene.render.resolution_y = 512
scene.render.resolution_percentage = 100

bpy.context.scene.update()

projection_matrix = camera.calc_matrix_camera(
        bpy.context.scene.render.resolution_x,
        bpy.context.scene.render.resolution_y)

bpy.context.scene.update()


scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = write_path + "calib.png"
bpy.ops.render.render(write_still = 1)

cammat = camera.matrix_world.copy()

objdata["View"] = str(cammat.inverted())
objdata["Projection"] = str(projection_matrix)
objdata["Model"] = str(calib.matrix_world)


verts = []
for vertex in calib.data.vertices:
    vert = vertex.co
    verts.append([vert[0], vert[1], vert[2]])


objdata["ObjCoords"] = verts


with open(write_path +  "calibdata.txt", 'w') as fp:
    json.dump(objdata, fp)