import bpy
import random
import math
import time
import numpy as np

from math import degrees



def shadeMasks(objects, mask_name):





PI = 3.1415



images = 4

millis = lambda: int(round(time.time() * 1000))

timestart = millis()

scene = bpy.context.scene


pole = bpy.data.objects['Pole']
brick = bpy.data.objects['Brick']
wing = bpy.data.objects['Wing'] 

camera = bpy.data.objects['Camera']

objects = [pole, wing, brick]#[pole, brick, wing]


mat = bpy.data.materials.new(name="Black")
mat.diffuse_color = (.1,.1,.1)

gray = bpy.data.materials.new(name="Gray")
gray.diffuse_color = (.5,.5,.5)

lgray = bpy.data.materials.new(name="LightGray")
lgray.diffuse_color = (.7,.7,.7)

blue = bpy.data.materials.new(name="Blue")
blue.diffuse_color = (0.0,0.0,1.0)


wingRange = (1.5, .5, 0.0)
wingStart = (2.4, .1, 0.0)





brickRange = ()



labels = ""



random.seed()




wings = random.randint(0,2)

if wings == 0:

if wings == 1:

if wings == 2:



wings = [wing, ]


def gimme():
    return False if random.randint(0,1) == 0 else True



def genPiece(center):

    switch = random.randint(0,2)
    posm = (.7, .2, 0)
    obj = None

    if (switch == 0):
        return '',None
    elif (switch == 1):
        obj =  pole.copy()
        mult = random.randint(-1,1)
        obj.location = center + tuple(mult * x for x in posm)
        pt = 90 if mult <= 0 else -90
        pt = pt + .7 * mult * 50
        obj.rotation_euler = (0,0, math.radians(pt))
        return 'pole',obj
    else:
        obj = brick.copy()
        pt = random.randint(0,20)/20
        obj.rotation_euler = (0,0,pt * PI)
        obj.location = center + .6 * posm
        return 'brick',obj







def genWing(center):
    
    newWing = o1 = o2 = o3 = None
    l0 = l1 = l2 = l3 = ''

    if gimme():
        newWing = wing.copy()
        newWing.location = (0,0,0)
        newWing.rotation_euler = (0,0,0)
        l0 = "wing"        

    ###region 1
    if gimme():
        l1, o1 = genPiece((0,2.6,.7))

    ###region 2
    if gimme():
        l2, o2 = genPiece(0,-1,.7)

    ###region 3
    if gimme():
        l3, o3 = genPiece(-.9,-2.3,.7)

    labels = [l0,l1,l2,l3]
    objs = [newWing,o1,o2,o3]

    for o in objs:
        if o is not None:
            o.parent = center

    return labels,objs 













for piece in objects:
    
    piece.data.materials.append(mat)
    piece.data.materials[0] = mat 
    
    

for x in range(images):

    c1 = bpy.data.objects.new("empty", None)
    c1.location = (0,0,0)
    bpy.context.scene.objects.link(c1)

    c2 = bpy.data.objects.new("empty", None)
    c2.location = (0,0,0)
    bpy.context.scene.objects.link(c2)

    w1 = w2 = (None,None)

    if gimme():
        w1 = genWing(c1)
        if gimme():
            w2 = genWing(c2)
            c2.location = (2 * random.randint(-1,1), 3 + random.randint(-1,1), 0)
            c2.rotation_euler = (0,0,PI/2*random.randint(-18,18)/18)

        c1.location = (2 * random.randint(-1,1), -3 + random.randint(-1,1), 0)
        c1.rotation_euler = (0,0,PI/2*random.randint(-18,18)/18)
    else:
        w2 = genWing(c2)
        c2.rotation_euler = (0,0,PI/2*random.randint(-18,18)/18)









    
    i = random.randint(0,1)
    
    piece = objects[i]
    
    for y in range(2):
        
        if (y == i):
            objects[y].location = (0,0,0)
        else:
            objects[y].location = (1000000,1000000,1000000)
    
    angle = random.randint(0, 180)
    piece.rotation_euler = (0.0,0.0,math.radians(angle))
    
    objects[i].location = (random.randint(0,3), random.randint(0,3), random.randint(0,2)) 
    
    
    camera.location = (random.randint(4,11), random.randint(4,11), random.randint(4,10))
    
    
    
    
    scene.render.resolution_x = 256
    scene.render.resolution_y = 256
    scene.render.resolution_percentage = 100
            
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = "/Users/will/projects/legoproj/regtest/test" + str(x) + ".png"
    bpy.ops.render.render(write_still = 1)
    
    labels += (str(x) + " " + str(degrees(piece.rotation_euler[2])) + "\n")


print("Generated " + str(x+1) + " images in " + str(float(millis() - timestart)/1000.0) + " seconds") 

text_file = open("/Users/will/projects/legoproj/regtest/labels.txt", "w")
text_file.write(labels)
text_file.close()    