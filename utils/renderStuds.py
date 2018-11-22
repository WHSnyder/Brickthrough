import bpy
import random
#import math
import time
import numpy as np

import mathutils as mu



#from math import degrees


mode = "val"

write_path = "/Users/will/projects/legoproj/data_oneofeach/{}_oneofeach/".format(mode)

PI = 3.1415

millis = lambda: int(round(time.time() * 1000))
timestart = millis()

scene = bpy.context.scene
scene_objs = bpy.data.objects

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
pole = scene_objs['Pole']
brick = scene_objs['Brick']
wing = scene_objs['Wing']
camera = bpy.data.objects['Camera']

objs = {'Pole':[], 'Wing':[], 'Brick':[]}

random.seed()


def mltup(tup, num):
    return tuple(num * x for x in tup)

def add2tup(tup, num):
    return tuple(num + x for x in tup)

def addtups(tup1, tup2):
    return tuple(x + y for x,y in zip(tup1,tup2))

def objcopy(obj):
    newObj = obj.copy()
    newObj.data = obj.data.copy()
    scene.objects.link(newObj)

    return newObj


wing = scene_objs['Wing']
wingStudVerts = []


#bpy.context.scene.objects.active = wing
#bpy.ops.object.mode_set(mode='EDIT', toggle=False)
#bpy.ops.mesh.select_all(action='DESELECT')

#sel_mode = bpy.context.tool_settings.mesh_select_mode
#bpy.context.tool_settings.mesh_select_mode = [True, False, False]
#bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

mesh = wing.data
#mesh.vertices[1].select = True

offset = mu.Vector((4,4,0)) 

verts = []        

#print('Boutta')

selected_verts = [v for v in mesh.vertices if v.select]

for vert in selected_verts:
    verts.append(vert.co)

'''
for vert in mesh.vertices:
    coord = 100*vert.co
    if abs(coord[2] - 6.4) < .00001:
        if abs(coord[0] % 8) < .00001 and abs(coord[1] % 8) < .00001:
            verts.append(vert.co)
'''

writestring = 'Wing\n'

for vert in verts:
    writestring += "{},{},{}\n".format(vert[0], vert[1], vert[2])

file = open("/Users/will/Desktop/wing.txt","w")
file.write(writestring)
file.close()
    
bpy.ops.object.mode_set(mode='EDIT', toggle=False)
bpy.context.tool_settings.mesh_select_mode = sel_mode