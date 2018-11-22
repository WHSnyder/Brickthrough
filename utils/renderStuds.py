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


bpy.context.scene.objects.active = wing
bpy.ops.object.mode_set(mode='EDIT', toggle=False)
bpy.ops.mesh.select_all(action='DESELECT')

sel_mode = bpy.context.tool_settings.mesh_select_mode
bpy.context.tool_settings.mesh_select_mode = [True, False, False]
bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

mesh = wing.data
#mesh.vertices[1].select = True

offset = mu.Vector((4,4,0)) 

verts = []        

#print('Boutta')

for vert in mesh.vertices:
    coord = 100*vert.co
    if abs(coord[2] - 6.4) < .00001:
        if abs(coord[0] % 4) < .00001 and abs(coord[1] % 4) < .00001:
            verts.append(vert.co)

writestring = 'Brick\n'

for vert in verts:
    writestring += "{},{},{}\n".format(vert[0], vert[1], vert[2])

file = open("/Users/will/Desktop/brick.txt","w")
file.write(writestring)
file.close()



    
bpy.ops.object.mode_set(mode='EDIT', toggle=False)
bpy.context.tool_settings.mesh_select_mode = sel_mode



'''
Rendering/masking methods
'''
'''
def shadeMasks(objects, x):

    bck.data.materials[0] = black_shadeless

    for key in objects:
        for obj in objects[key]:
            obj.data.materials[0] = black_shadeless

    for key in objects:
        count = 0
        for obj in objects[key]:
            obj.data.materials[0] = white_shadeless
            
            scene.render.resolution_x = 64
            scene.render.resolution_y = 64
            scene.render.resolution_percentage = 100
                    
            scene.render.image_settings.file_format = 'PNG'
            scene.render.filepath = write_path + str(x) + "_" + mode + "_" + key + str(count) + ".png"
            bpy.ops.render.render(write_still = 1)
            count+=1

            obj.data.materials[0] = black_shadeless
'''
'''
Wing generation and children placement
'''
'''
def gimme():
    return False if random.randint(0,2) == 0 else True

def genPiece(center):

    switch = random.randint(0,2)
    posm = (.7, .2, 0)
    obj = None

    if gimme():
        obj = objcopy(pole)
        mult = random.randint(-1,1)
        obj.location = addtups( center , tuple(mult * x for x in posm) )
        pt = 90 if mult <= 0 else -90
        pt = pt + .7 * mult * 50
        obj.rotation_euler = (0,0, math.radians(pt))
        print("Generating pole")
        return 'Pole', obj

    else:
        obj = objcopy(brick)
        pt = random.randint(0,20)/20
        obj.rotation_euler = (0,0,pt * PI)
        obj.location = addtups( center , mltup(posm,.6) )
        print("Generating brick")
        return 'Brick', obj



def genWing(center):

    print("Generating wing")
    
    if gimme() or gimme():
        newWing = objcopy(wing)
        newWing.location = (0,0,0)
        newWing.rotation_euler = (0,0,0)
        objs["Wing"].append(newWing)
        newWing.parent = center  
        newWing.matrix_parent_inverse = center.matrix_world.inverted()
     
    ###region 1
    if gimme():
        l, o = genPiece((0,1.6,.7))
        objs[l].append(o)
        o.parent = center
        o.matrix_parent_inverse = center.matrix_world.inverted()

    ###region 2
    if gimme():
        l, o = genPiece((0,-.7,.7))
        objs[l].append(o)
        o.parent = center
        o.matrix_parent_inverse = center.matrix_world.inverted()

    ###region 3
    if gimme():
        l, o = genPiece((-.6,-1.6,.7))
        objs[l].append(o)
        o.parent = center
        o.matrix_parent_inverse = center.matrix_world.inverted()



c1 = bpy.data.objects.new("empty", None)
bpy.context.scene.objects.link(c1)

c2 = bpy.data.objects.new("empty", None)
bpy.context.scene.objects.link(c2)

print(bck)

num = 300

for x in range(num):

    c1.location = (0,0,0)
    c2.location = (0,0,0)
    
    if gimme():
        w1 = genWing(c1)
        c1.rotation_euler = (0,0,PI/2*random.randint(-18,18)/18)
        c1.location = (-3, 2 * random.randint(-1,1), 0)

        w2 = genWing(c2)
        c2.rotation_euler = (0,0,PI/2*random.randint(-18,18)/18)
        c2.location = (3, 2 * random.randint(-1,1), .7)

        camera.location = (random.randint(6,11) * -1 if random.randint(0,1) < 1 else 1, random.randint(6,11) * -1 if random.randint(0,1) < 1 else 1, random.randint(12,13))

    else:
        w2 = genWing(c2)
        c2.location = (.3 * random.randint(-1,1), .3 * random.randint(-1,1), 0)
        c2.rotation_euler = (0,0,PI/2*random.randint(-18,18)/18)

        camera.location = (random.randint(5,7) * -1 if random.randint(0,1) < 1 else 1, random.randint(5,7) * -1 if random.randint(0,1) < 1 else 1, random.randint(6,7))


    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100

    bck.data.materials[0] = white_shadeless

    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = write_path + str(x) + "_" + mode + "_a" + ".png"
    bpy.ops.render.render(write_still = 1)

    shadeMasks(objs,x)

    for key in objs:
        for obj in objs[key]:
            print("wiping")
            scene_objs.remove(obj, do_unlink=True)
        objs[key].clear()


print("Generated " + str(x+1) + " images in " + str(float(millis() - timestart)/1000.0) + " seconds")
scene_objs.remove(c1, do_unlink=True)
scene_objs.remove(c2, do_unlink=True)
'''