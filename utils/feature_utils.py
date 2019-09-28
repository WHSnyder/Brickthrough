import numpy as np
import json
import re
import math

expr = re.compile("([-]?[0-9]*\.[0-9]{4})")
dim = 512

def matrix_from_string(matstring):

    matches = expr.findall(matstring)

    nums = np.asarray(list(map(lambda x: float(x), matches)), dtype=np.float32)
    nums = np.reshape(nums, (4,4))

    return nums
    


def get_object_matrices(filename):

    data = {}

    with open(filename) as json_file:
        data = json.load(json_file)

    for key in data:
        data[key] = matrix_from_string(data[key])

    return data



def get_object_studs(objname):

    filename = "/Users/will/Desktop/{}.txt".format(objname)
    
    with open(filename, "r") as fp:
        verts = fp.read()
    lines = verts.split("\n")[1:]
    verts = []

    for line in lines:

        if line == "":
            break

        parts = line.split(",")

        nums = list(map(lambda x: float(x), parts))
        vert = np.ones(4, dtype=np.float32)
        vert[0:3] = nums[0:3]

        verts.append(vert)

    return verts



def verts_to_screen(model, view, frust, verts):
    
    #mvp = np.matmul( frust, np.matmul(view, model) )
    screenverts = []
    worldverts = []
    camverts = []

    #print("Model: \n{}".format(str(model)))
    #print("View: \n{}".format(str(view)))
    #print("Frust: \n{}".format(str(frust)))
    #print("--------------------------------------")
    #print("Verts local coordinates: \n{}\n".format(str(verts)))

    for vert in verts:
        #print("Shape: " + str(vert.shape))
        worldvert = np.matmul(model, vert)
        camvert = np.matmul(view, worldvert)
        screenvert = np.matmul(frust, camvert)
        screenvert = screenvert/screenvert[3]

        if abs(screenvert[0]) < 1 and abs(screenvert[1]) < 1:
            screenvert[0:2] = (screenvert[0:2] + 1)/2
            screenverts.append(screenvert)
        
        worldverts.append(worldvert)
        camverts.append(camvert)

    #print("Verts world coordinates: \n{}\n".format(worldverts))
    #print("Verts camera coordinates: \n{}\n".format(camverts))
    #print("Verts screen coordinates: \n{}\n".format(screenverts))
    #print("--------------------------------------")

    return screenverts


brickstuds = get_object_studs("brick")
wingstuds = get_object_studs("wing")


def getStudMask(i):

    modelmats = get_object_matrices(datadir + "mats/{}.txt".format(i))
    cammat = modelmats["Camera"]
    projmat = modelmats["Projection"]

    maskdim = int(dim/2)
    scenestuds = np.zeros((maskdim,maskdim))
    screenverts = []

    for key in modelmats:

        if "Brick" in key:
            studs = brickstuds
        elif "Wing" in key:
            studs = wingstuds
        else:
            continue
        screenverts += verts_to_screen(modelmats[key], cammat, projmat, studs) 

    for vert in screenverts:
        npcoord = tuple([math.floor((1 - vert[1]) * maskdim), math.floor(vert[0] * maskdim)])
        scenestuds[npcoord[0], npcoord[1]] = 1

    scenestuds = np.reshape(scenestuds, (maskdim,maskdim,1))

    return scenestuds


def toNDC(verts, dims):
    newverts = []
    for vert in verts:
        npcoord = tuple([math.floor(vert[0] * dims[0]), math.floor((1 - vert[1]) * dims[1])])
        newverts.append(npcoord)
    return np.asarray(newverts)




def getCalibCorrs():
    path = "/Users/will/projects/legoproj/utils/calib_data/calibdata.txt"

    with open(path) as json_file:
        data = json.load(json_file)

    view = matrix_from_string(data["View"])
    model = matrix_from_string(data["Model"])
    proj = matrix_from_string(data["Projection"])

    verts = []
    for vert in np.asarray(data["ObjCoords"]):
        newvert = np.ones(4, dtype=np.float32)
        newvert[0:3] = vert[0:3]
        verts.append(newvert)

    verts = np.asarray(verts)

    screenverts = verts_to_screen(model, view, proj, verts)

    return np.asarray(verts), np.asarray(screenverts)