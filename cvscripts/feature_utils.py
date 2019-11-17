import numpy as np
import json
import re
import math

import matplotlib.pyplot as plt

hf="/Users"

expr = re.compile("([-]?[0-9]*\.[0-9]{4})")
dim = 512



def dictFromJson(filename):

    with open(filename) as json_file:
        data = json.load(json_file)

    return data




def matrix_from_string(matstring):

    matches = expr.findall(matstring)

    nums = np.asarray(list(map(lambda x: float(x), matches)), dtype=np.float32)
    nums = np.reshape(nums, (4,4))

    return nums
    



def get_object_matrices(filename):

    data = dictFromJson(filename)

    for key in data:
        data[key] = matrix_from_string(data[key])

    return data




def get_object_studs(piece):

    #piece = piece.replace(".", "_")
    #piece = piece.split("_")[0]
    file = hf+"/will/projects/legoproj/pieces/{}.json".format(piece)

    r = dictFromJson(file)["studs"]
    #print(r)
    return r




def verts_to_screen(model, view, frust, verts):
    
    screenverts = []
    mvp = np.matmul(frust,np.matmul(view,model))

    for vert in verts:

        screenvert = np.matmul(mvp, vert)
        screenvert = screenvert/screenvert[3]

        if abs(screenvert[0]) < 1 and abs(screenvert[1]) < 1:
            screenvert[0:2] = (screenvert[0:2] + 1)/2
            screenverts.append(screenvert)

    return screenverts




brickstuds = get_object_studs("Brick")
wingrstuds = get_object_studs("WingR")




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
    path = hf+"/will/projects/legoproj/utils/calib_data/calibdata.txt"

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

    screenverts = toNDC(verts_to_screen(model, view, proj, verts), (512,512))

    return np.delete(verts, 3, axis=1), screenverts




def getFeatureBoxes(width, height, centers):

    out = []

    for center in centers:
        x = center[1]
        y = center[0]

        x -= width/2
        y -= height/2

        out.append(tuple(np.int(np.asarray([x,y,width,height]))))

    return out



def toCV2bbox(points):

    out = []

    for point in points:
        [x,y,w,h] = point
        p1 = tuple([x,y])
        p2 = tuple([x + w, y + h]) 
        out.append([p1,p2])

    return out



def getTemplate(piece, num, plot=True):
    
    temppath = hf+"/will/projects/legoproj/data/{}_single/{}_{}_a.png".format(piece.lower(), num, piece.lower())
    tempjson = temppath.replace(".png", ".json")

    data = dictFromJson(tempjson)
    ostuds = get_object_studs(piece)

    img = cv2.imread(temppath)

    model = matrix_from_string(data["objects"][piece+".001"]["modelmat"])
    view = matrix_from_string(data["Camera"])
    proj = matrix_from_string(data["Projection"])

    screenverts = toNDC(verts_to_screen(model, view, proj, ostuds), (512,512))
    
    if plot:

        w = h = 20
        l = w/2

        imgboxes = cv2.copy(img)

        for vert in screenverts:

            x,y = screenverts[0], screenverts[1]
            x1,y1 = x - l, y - l
            x2,y2 = x + l, y + l

            cv2.rectangle(imgboxes, (x1,y1), (x2,y2), (0,0,0), 2)

        plt.imshow(imgboxes, cmap="rgb")
        plt.show()

    return img, np.delete(ostuds, 3, axis=1), screenverts