import numpy as np
import json
import re
import math
import cv2
from matplotlib import pyplot as plt



def getContourMaxArea(contours):

    maxi, maxa = 0,0

    for i,e in enumerate(contours):
        a = cv2.contourArea(e)
        if maxa <= a:
            maxa = a
            maxi = i

    return maxi,maxa



def listContourChildren(index, hierarchy, contours, minArea = 4):

    hierarchy = hierarchy[0]
    nexti = hierarchy[index][2]
    #print(hierarchy)
    #print(nexti)
    holes = []
    
    while nexti != -1:
        cur = contours[nexti]

        if cv2.contourArea(cur) >= 4:
            holes.append(cur)

        nexti = hierarchy[nexti][0]

    return holes



def showComboMask(img, data, objname, mode="obj"):
    
    if mode == "allbut":
        mask = 255 * np.ones((64,64))
        for key in data["objects"]:
            if key != objname:
                m = cv2.imread(data["objects"][key]["maskpath"], 0)
                print(mask.shape)
                mask = cv2.bitwise_or(mask,mask,mask=m)#,mask=None)

    else: 
        file = data["objects"][objname]["maskpath"]
        mask = cv2.imread(file, 0)
    
    mask = cv2.resize(mask, (512,512), interpolation=cv2.INTER_LINEAR)
    masked = cv2.bitwise_and(img,img,mask=mask)

    return masked





def sortHist(hist):

    histlist = list(enumerate(hist))
    sortedhist = sorted(histlist, key=lambda s: 254 - s[1][0])

    return sortedhist


def getRange(hist, index):

    lower = 0
    higher = 0

    for i in range(1,4):
        nextone = index - i
        lower = nextone + 1

        if hist[nextone] >= hist[lower]:
            break

    for i in range(1,4):
        nextone = index + i
        higher = nextone - 1

        if hist[nextone] >= hist[higher]:
            break

    return lower, higher


def drawStuds(img):

    contours, hierarchy = cv2.findContours(img.astype(np.dtype('uint8')), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    index, cnt = getContourMaxArea(contours)

    children = listContourChildren(index, hierarchy, contours)

    rgb = cv2.cvtColor(img.astype(np.dtype("float32")), cv2.COLOR_GRAY2RGB)

    for child in children:
        M = cv2.moments(child)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        cv2.circle(rgb, (cX,cY), 8, (0, 100, 160), 2)

    plt.imshow(rgb)
    plt.show()

    return None














def showSurface(img,rank,buckets=35,show=False):

    img1 = cv2.bilateralFilter(img, 2, 900, 100)

    hist = cv2.calcHist([img1],[0],None,[buckets],[3,254])
    sortedhist = sortHist(hist)

    index = sortedhist[rank][0]

    lower, higher = getRange(hist, index)
    lower = int(lower/buckets * 254)
    higher = int(higher/buckets * 254)

    print("Low:  {}   High:  {}".format(lower,higher))

    kernel = np.ones((2,2),np.uint8)

    threshed = cv2.inRange(img1, lower, higher)

    num, output, stats, centroids = cv2.connectedComponentsWithStats(threshed, connectivity=8)

    num -= 1
    sizes = stats[1:, -1];

    min_size = 20

    img2 = np.zeros((output.shape))
    print(img2.shape)

    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, num):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255  

    if show:
        plt.imshow(img2, cmap="gray")
        plt.show()

    return threshed





def testForHoles(surf, minholes=2, show=False):

    num, output, stats, centroids = cv2.connectedComponentsWithStats(surf, connectivity=8)

    sizes = stats[1:, -1];
    num -= 1


    maxsize = 0
    maxone = 0

    img2 = np.zeros((output.shape))

    for i in range(0, num):
        if sizes[i] >= maxsize:
            maxone = i
            maxsize = sizes[i]

    img2[output == maxone + 1] = 255 

    contours, hierarchy = cv2.findContours(img2.astype(np.dtype('uint8')), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    length = 0
    theone = None
    ind = 0
    for i,cnt in enumerate(contours):
        cur = cv2.arcLength(cnt,True)
        if cur > length:
            length = cur
            theone = cnt
            ind = i

    hull = cv2.convexHull(theone)

    #img3 = cv2.cvtColor(img2.astype(np.dtype("float32")), cv2.COLOR_GRAY2RGB)
    #cv2.drawContours(img2, [hull], 0, (255,0,0), 2)


    if show:
        plt.imshow(img2,cmap="gray")
        plt.show()

    return img2