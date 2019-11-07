import bpy
import random
import math
import time
import numpy as np
import json
import mathutils as mu
import os
from math import degrees





random.seed()

def hasNumbers(instr):
    return any(char.isdigit() for char in instr)

mode = "test"
num = 0

write_path = "/home/will/projects/legoproj/data/{}_combodset_{}/".format(mode,num)

while os.path.exists(write_path):
    num += 1
    write_path = "/home/will/projects/legoproj/data/{}_combodset_{}/".format(mode,num)

os.mkdir(write_path)





PI = 3.1415

millis = lambda: int(round(time.time() * 1000))
timestart = millis()

scene = bpy.context.scene
scene_objs = bpy.data.objects

imgsdir = "/home/will/projects/legoproj/downloads/table/"


imgpaths = os.listdir(imgsdir)
imgs = []

for img in bpy.data.images:
    bpy.data.images.remove(img)

for path in imgpaths:
    img = bpy.data.images.load(filepath=imgsdir+path)
    imgs.append(img)

tablemat = bpy.data.materials["Table"]
#tablemat = mat.copy()


nodes = tablemat.node_tree.nodes

# get some specific node:
# returns None if the node does not exist
imgnode = nodes.get("Image Texture")
imgnode.image = imgs[random.randint(3,80)]

table = scene_objs["Table"]

#table.data.materials[0] = tablemat

"""

for mat in bpy.data.materials:
    if hasNumbers(mat.name):
        bpy.data.materials.remove(mat)
"""

objs = bpy.context.selected_objects

obj = objs[0]
obj.active_material_index = 0

if obj.name not in bpy.data.materials:
    objmat = bpy.data.materials["WhiteShadeless"].copy()
else:
	objmat = bpy.data.materials[obj.name]

objmat.use_nodes = True
objmat.name = obj.name
objmat.node_tree.nodes["Emission"].inputs["Color"].default_value=[1.0,1.0,0.5,1.0]

obj.data.materials[0] = objmat 
bpy.context.scene.update()









'''


def getMat(name):
    for mat in bpy.data.materials:
        if mat.name == name:
            return mat

black_shadeless = getMat("BlackShadeless")
white_shadeless = getMat("WhiteShadeless")
black = getMat("Black")
gray = getMat("Gray")
lgray = getMat("LightGray")
blue = getMat("Blue")

bck = bpy.data.objects['Background']



scene.render.resolution_x = 512
scene.render.resolution_y = 512
scene.render.resolution_percentage = 100

bpy.context.scene.update()

projection_matrix = camera.calc_matrix_camera(
        bpy.context.scene.render.resolution_x,
        bpy.context.scene.render.resolution_y)

bpy.context.scene.update()



'''





#Rendering/masking methods

def shadeMasks(objects, x, objdata):

    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100                
    scene.render.image_settings.file_format = 'PNG'

    bck.data.materials[0] = black_shadeless

    for key in objects:
        for obj in objects[key]:
            obj.data.materials[0] = black_shadeless

    for key in objects:
        count = 0
        for obj in objects[key]:
            obj.data.materials[0] = white_shadeless
            

            scene.render.filepath = write_path + str(x) + "_" + mode + "_" + obj.name.replace(".","_") + ".png"
            bpy.ops.render.render(write_still = 1)
            count+=1

            obj.data.materials[0] = black_shadeless

            objdata[obj.name] = str(obj.matrix_world)


'''
def subset(objs,objdeg,matdeg):
    numobjs = objs


#Wing generation and children placement


def gimme():
    return False if random.randint(0,2) == 0 else True


num = 100


os.system("rm " + write_path + "*.png")

if not os.path.exists(write_path + "mats/"):
    os.mkdir(write_path + "mats/")
else:
    os.system("rm " + write_path + "mats/*")


for x in range(num):

    objdata = {}

    

    camera.location = (random.randint(5,7) * -1 if random.randint(0,1) < 1 else 1, random.randint(5,7) * -1 if random.randint(0,1) < 1 else 1, random.randint(6,7))

    bpy.context.scene.update()

    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100

    bck.data.materials[0] = white_shadeless

    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = write_path + str(x) + "_" + mode + "_a" + ".png"
    bpy.ops.render.render(write_still = 1)

    cammat = camera.matrix_world.copy()

    objdata["Camera"] = str(cammat.inverted())
    objdata["Projection"] = str(projection_matrix)

    shadeMasks(objs,x, objdata)

    with open(write_path + "mats/" + str(x) + ".txt", 'w') as fp:
        json.dump(objdata, fp)


print("Generated " + str(x+1) + " images in " + str(float(millis() - timestart)/1000.0) + " seconds")
'''