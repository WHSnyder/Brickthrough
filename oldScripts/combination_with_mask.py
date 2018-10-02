import bpy
import random
import math
import time
import numpy as np


from math import degrees




write_path = "/Users/will/projects/legoproj/combo_renders/test"


PI = 3.1415



images = 4

millis = lambda: int(round(time.time() * 1000))
timestart = millis()

scene = bpy.context.scene

black = bpy.data.materials.new(name="Black")
black.diffuse_color = (.1,.1,.1)

gray = bpy.data.materials.new(name="Gray")
gray.diffuse_color = (.5,.5,.5)

lgray = bpy.data.materials.new(name="LightGray")
lgray.diffuse_color = (.7,.7,.7)

blue = bpy.data.materials.new(name="Blue")
blue.diffuse_color = (0.0,0.0,1.0)

maskmats = []
for x in range(0,10):
    m = (10-x)/20 + .5
    mask = bpy.data.materials.new(name="White" + str(10 - x))
    mask.diffuse_color = (m,m,m)
    mask.use_shadeless = True
    maskmats.append(mask)


black_shadeless = bpy.data.materials.new(name="BlackShadeless")
black_shadeless.diffuse_color = (0.0,0.0,0.0)
black_shadeless.use_shadeless = True

pole = bpy.data.objects['Pole']
pole.data.materials.append(black)
#pole.data.materials[0] = black 
pole.active_material = black

brick = bpy.data.objects['Brick']
brick.data.materials.append(gray)
#brick.data.materials[0] = gray
pole.active_material = gray 

wing = bpy.data.objects['Wing'] 
wing.data.materials.append(lgray)
#wing.data.materials[0] = lgray
wing.active_material = lgray

camera = bpy.data.objects['Camera']

objs = {'Pole':[], 'Wing':[], 'Brick':[]}

random.seed()

bck = bpy.data.objects['Background']
bck.data.materials.append(black_shadeless)
bck.data.materials.append(maskmats[0])


def mltup(tup, num):
    return tuple(num * x for x in tup)

def add2tup(tup, num):
    return tuple(num + x for x in tup)

def addtups(tup1, tup2):
    return tuple(x + y for x,y in zip(tup1,tup2))
'''
Rendering/masking methods
'''

def shadeMasks(objects, mask_name, x):
    count = 0
    if len(objects[mask_name]) > 0:
        for key in objects:
            for obj in objects[key]:
                if key != mask_name:
                    obj.hide = True
                else:
                    curmask = maskmats[count]
                    obj.hide = False
                    obj.data.materials.append(curmask)
                    obj.active_material = curmask
                    count = count + 1

        #scene.render.setBackgroundColor(0.0,0.0,0.0)
        bck.active_material = black_shadeless

        scene.render.resolution_x = 64
        scene.render.resolution_y = 64
        scene.render.resolution_percentage = 100
                
        scene.render.image_settings.file_format = 'PNG'
        scene.render.filepath = write_path + mask_name + str(x) + ".png"
        bpy.ops.render.render(write_still = 1)




'''
Wing generation and children placement
'''

def gimme():
    return False if random.randint(0,1) == 0 else True

def genPiece(center):

    switch = random.randint(0,2)
    posm = (.7, .2, 0)
    obj = None

    #if (switch == 0):
    #    return '',None
    if gimme():
        obj =  pole.copy()
        mult = random.randint(-1,1)
        obj.location = addtups( center , tuple(mult * x for x in posm) )
        pt = 90 if mult <= 0 else -90
        pt = pt + .7 * mult * 50
        obj.rotation_euler = (0,0, math.radians(pt))
        return 'Pole',obj
    else:
        obj = brick.copy()
        pt = random.randint(0,20)/20
        obj.rotation_euler = (0,0,pt * PI)
        obj.location = addtups( center , mltup(posm,.6) )
        return 'Brick',obj



def genWing(center):
    
    if gimme():
        newWing = wing.copy()
        newWing.location = (0,0,0)
        newWing.rotation_euler = (0,0,0)
        objs["Wing"].append(newWing)
        newWing.parent = center       
    ###region 1
    if gimme():
        l, o = genPiece((0,2.6,.7))
        objs[l].append(o)
        o.parent = center
    ###region 2
    if gimme():
        l, o = genPiece((0,-1,.7))
        objs[l].append(o)
        o.parent = center
    ###region 3
    if gimme():
        l, o = genPiece((-.9,-2.3,.7))
        objs[l].append(o)
        o.parent = center

c1 = bpy.data.objects[].new("empty", None)
bpy.context.scene.objects.link(c1)

c2 = bpy.data.objects[]new("empty", None)
bpy.context.scene.objects.link(c2)

for x in range(images):

    c1.location = (0,0,0)
    c2.location = (0,0,0)

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
        c2.location = (3 * random.randint(-1,1), 3 + random.randint(-1,1), 0)
        c2.rotation_euler = (0,0,PI/2*random.randint(-18,18)/18)

    camera.location = (random.randint(4,11), random.randint(4,11), random.randint(4,10))

    scene.render.resolution_x = 256
    scene.render.resolution_y = 256
    scene.render.resolution_percentage = 100

    #scene.render.setBackgroundColor(1.0,1.0,1.0)
    bck.active_material = maskmats[0]
            
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = write_path + str(x) + ".png"
    bpy.ops.render.render(write_still = 1)

    for key in objs:
        if key != "":
            shadeMasks(objs, key, x)
            objs[key].clear()


print("Generated " + str(x+1) + " images in " + str(float(millis() - timestart)/1000.0) + " seconds") 

'''
text_file = open("/Users/will/projects/legoproj/regtest/labels.txt", "w")
text_file.write(labels)
text_file.close() 
'''   