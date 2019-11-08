import bpy
import random
import math
import time
import numpy as np
import json
import mathutils as mu
import os
from math import degrees
import colorsys

print("Begining.....\n\n\n")

millis = lambda: int(round(time.time() * 1000))
timestart = millis()
random.seed()

HSV = True

#def hasNumbers(instr):
#    return any(char.isdigit() for char in instr)

mode = "test"
num = 0
write_path = "/home/will/projects/legoproj/data/{}_combodset_{}/".format(mode,num)
while os.path.exists(write_path):
    num += 1
    write_path = "/home/will/projects/legoproj/data/{}_combodset_{}/".format(mode,num)
os.mkdir(write_path)


bpy.context.scene.render.engine = 'CYCLES'




scene = bpy.context.scene
scene_objs = bpy.data.objects

camera = bpy.data.objects['Camera']



imgsdir = "/home/will/projects/legoproj/downloads/table/"
imgpaths = os.listdir(imgsdir)
imgs = []

for img in bpy.data.images:
    bpy.data.images.remove(img)
for path in imgpaths:
    img = bpy.data.images.load(filepath=imgsdir+path)
    imgs.append(img)

tablemat = bpy.data.materials["Table"]
nodes = tablemat.node_tree.nodes
imgnode = nodes.get("Image Texture")

def changeTable():
    imgnode.image = imgs[random.randint(0,70)]

#table = scene_objs["Table"]

gray = [.5,.5,.5,1.0]
lgray = [.8,.8,.8,1.0]
red = [1.0,1.0,1.0,1.0]
blue = [.1,.1,1.0,1.0]
black = [.2,.2,.2,1.0]
dirty = [.2,.3,.2,1.0]

matnames = ["gray","lgray","red","blue","black","dirty"]
matcolors = [gray, lgray, red, blue, black, dirty]
mats = []

for i in range(0,len(matnames)):
    matname = matnames[i]
    if matname not in bpy.data.materials:
        mat = bpy.data.materials["Gray"].copy()
        mat.name = matname
    else:
        mat = bpy.data.materials[matname]

    color = matcolors[i]
    mat.use_nodes = True
    mat.node_tree.nodes["Diffuse BSDF"].inputs["Color"].default_value = color

    mats.append(mat)




objmasks = {}
objs = []

for obj in bpy.context.selected_objects:
    if obj.name !="Table":
        objs.append(obj)


scenedata = {}
scenedata["objects"] = {}
scenedata["dataroot"] = write_path

for obj in objs:
    objdata = {}
    objdata["modelmat"] = str(obj.matrix_world)
    scenedata["objects"][obj.name] = objdata

    
bpy.context.scene.update()


#incs = list(map(lambda x: round(x,1), list(np.arange(.2,1.2,.2))))
incs0 = [0.0, .6, .3, 1.0]
incs1 = [.3, .8, .5, 0.1]
incs2 = [1.0, .0, .5, 0.4]
endlist = len(objs)


unit = 255/endlist




i = 0

if not HSV:
    for r in incs0:
        if i == endlist:
            break;
        for g in incs1:
            if i == endlist:
                break;
            for b in incs2:
                if i == endlist:
                    break;

                obj = objs[i]
                obj.active_material_index = 0

                if obj.name not in bpy.data.materials:
                    objmat = bpy.data.materials["WhiteShadeless"].copy()
                else:
                    objmat = bpy.data.materials[obj.name]

                objmat.use_nodes = True
                objmat.name = obj.name

                if objmat.name not in obj.data.materials:
                    obj.data.materials.append(objmat)

                color = [r,g,b,1.0]
                objmat.node_tree.nodes["Emission"].inputs["Color"].default_value = color

                objmasks[obj.name] = objmat
                scenedata["objects"][obj.name]["maskcolor"] = color[0:3]

                i+=1

else:
    for i,obj in enumerate(objs):

        obj.active_material_index = 0

        if obj.name not in bpy.data.materials:
            objmat = bpy.data.materials["WhiteShadeless"].copy()
        else:
            objmat = bpy.data.materials[obj.name]

        objmat.use_nodes = True
        objmat.name = obj.name

        if objmat.name not in obj.data.materials:
            obj.data.materials.append(objmat)

        color = colorsys.hsv_to_rgb(unit*i/255,.8,.7)
        color = [color[0],color[1],color[2],1.0]

        objmat.node_tree.nodes["Emission"].inputs["Color"].default_value = color

        objmasks[obj.name] = objmat
        scenedata["objects"][obj.name]["maskhue"] = unit*i/255

scenedata["renders"] = []




black_shadeless = bpy.data.materials["BlackShadeless"]
bck = bpy.data.objects['Background']
bck.data.materials[0] = black_shadeless
bck.active_material_index = 0




scene.render.resolution_x = 512
scene.render.resolution_y = 512
scene.render.resolution_percentage = 100
bpy.context.scene.update()
projection_matrix = camera.calc_matrix_camera(
        bpy.context.scene.render.resolution_x,
        bpy.context.scene.render.resolution_y)
bpy.context.scene.update()
scenedata["projection"] = str(projection_matrix)



#Masking 
def shadeMasks(objects, path):

    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100                

    bck.hide_render = False
    bck.hide = False

    for obj in objects:
        obj.data.materials[0] = objmasks[obj.name]

    bpy.context.scene.update()
            
    scene.render.image_settings.file_format = 'PNG'        
    scene.render.filepath = path
    bpy.ops.render.render(write_still = 1)


#Main render
def shade(x,subset):

    changeTable()

    renderfile = "{}_a.png".format(x)
    maskfile = "mask_{}.png".format(x)

    render_path = write_path + renderfile
    mask_path = write_path + maskfile

    scenedata["renders"].append({"x":x, "r":renderfile, "m":maskfile, "c": str(camera.matrix_world.copy().inverted())})

    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100

    bck.hide_render = True
    bck.hide = True

    bpy.context.scene.update()

    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = render_path
    bpy.ops.render.render(write_still = 1)

    shadeMasks(subset,mask_path)






def getMatSubset(percent):

    nummats = len(mats)
    choices = int(round(percent*nummats))
 
    if choices == 0:
    	return [mats[random.randint(0,nummats-1)]]
 
    res = [bpy.data.materials["dirty"]]
 
    for mat in mats:
        des = True if random.randint(0,nummats) <= choices else False
        if des:
            res.append(mat)

    return res


def getObjSubset(percent,matchoices):

    numobjs = len(objs)
    choices = int(round(percent*numobjs))

    nummatchoices = len(matchoices)

    if choices == 0:
        choices = 1

    res = []

    for obj in objs:

        des = True if random.randint(0,numobjs) <= choices else False

        if des:
            res.append(obj)
            obj.hide_render = False
            obj.data.materials[0] = matchoices[random.randint(0,nummatchoices-1)] 
        else:
            obj.hide_render = True

    return res




world = bpy.data.worlds["World.001"]
world.use_nodes = True
bg = world.node_tree.nodes["Background"]


num = 50

for x in range(num):

    strength = random.randint(1,6)*.2
    bg.inputs[1].default_value = strength

    #select subset
    objslice = random.randint(1,8)*.05
    matslice = random.randint(1,8)*.1

    matz = getMatSubset(matslice)
    objectz = getObjSubset(objslice,matz)

    camera.location = (random.randint(3,8) * -1 if random.randint(0,1) < 1 else 1, random.randint(3,8) * -1 if random.randint(0,1) < 1 else 1, random.randint(3,8))

    bpy.context.scene.update()

    shade(x,objectz)
    
    #select material subset    camera.location = (random.randint(5,7) * -1 if random.randint(0,1) < 1 else 1, random.randint(5,7) * -1 if random.randint(0,1) < 1 else 1, random.randint(6,7))


    #change light and lighting

    #change camera views









with open(write_path + "data.json", 'w') as fp:
    json.dump(scenedata,fp)

print("Generated " + str(x+1) + " images in " + str(float(millis() - timestart)/1000.0) + " seconds")

for obj in objs:
    obj.hide = False
    obj.hide_render = False