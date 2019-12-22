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

    if (hierarchy is None):
        return []

    hierarchy = hierarchy[0]
    nexti = hierarchy[index][2]
    
    holes = []
    
    while nexti != -1:
        cur = contours[nexti]

        if cv2.contourArea(cur) >= 4:
            holes.append(cur)

        nexti = hierarchy[nexti][0]

    return holes



def getComboMask(img, data, objname, mode="obj", show=False):
    
    if mode == "allbut":
        mask = 255 * np.ones((64,64))
        for key in data["objects"]:
            if key != objname:
                m = cv2.imread(data["objects"][key]["maskpath"], 0)
                print(mask.shape)
                mask = cv2.bitwise_or(mask,mask,mask=m)#,mask=None)

    else: 
        file = data["objects"][objname]["maskpath"]
        file=file.replace("/Users","/home")
        mask = cv2.imread(file, 0)
    
    mask = cv2.resize(mask, (512,512), interpolation=cv2.INTER_LINEAR)
    masked = cv2.bitwise_and(img,img,mask=mask)

    if show:
        plt.imshow(masked)
        plt.show()

    return masked


def getCentroids(cnts):

    r = []
    for cnt in cnts:
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        r.append((cX,cY))

    return r


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



def getStuds(img, show=False):

    contours, hierarchy = cv2.findContours(img.astype(np.dtype('uint8')), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    index, cnt = getContourMaxArea(contours)

    children = listContourChildren(index, hierarchy, contours)
    roids = getCentroids(children)

    if show: 
        rgb = cv2.cvtColor(img.astype(np.dtype("float32")), cv2.COLOR_GRAY2RGB)

        for roid in roids:
            cv2.circle(rgb, roid, 8, (0, 100, 160), 2)
   
        plt.imshow(rgb/255)
        plt.show()

    return roids






def getSurface(img,rank,buckets=35,show=False):

    img1 = cv2.bilateralFilter(img, 2, 900, 100)

    hist = cv2.calcHist([img1],[0],None,[buckets],[3,254])
    sortedhist = sortHist(hist)

    index = sortedhist[rank][0]

    lower, higher = getRange(hist, index)
    lower = int(lower/buckets * 254)
    higher = int(higher/buckets * 254)

    threshed = cv2.inRange(img1, lower, higher)
    num, output, stats, centroids = cv2.connectedComponentsWithStats(threshed, connectivity=8)

    num -= 1
    sizes = stats[1:, -1];
    min_size = 20

    img2 = np.zeros((output.shape))


    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, num):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255 

    img2 = cv2.medianBlur(img2.astype(np.uint8), 3)

    if show:

        plt.imshow(img2, cmap="gray")
        plt.show()

    return img2.astype(np.dtype("uint8"))




def testForHoles(surf, minholes=2, show=False):
    #surf = cv2.bilateralFilter(surf, 2, 900, 100)

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

    #contours, hierarchy = cv2.findContours(img2.astype(np.dtype('uint8')), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #maxi, maxa = getContourMaxArea(contours)

    studs = getStuds(img2)


    '''
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

    img3 = cv2.cvtColor(img2.astype(np.dtype("float32")), cv2.COLOR_GRAY2RGB)
    cv2.drawContours(img3, [theone], 0, (0,100,255), 2)
    '''

    '''
    if show:
        plt.imshow(img2/255)#,cmap="gray")
        plt.show()
    '''


    return img2, studs




def getStuddedSurface(img, show=False):

    count = 0
    outimg, outlist = None,[]

    for i in range(0,3):
        surf = getSurface(img, i)
        studsimg, studslist = testForHoles(surf)

        if len(studslist) > count:
            outimg = studsimg
            outlist = studslist
            count = len(studslist)

    if show:
        plt.imshow(outimg,cmap="gray")
        plt.show()

    kernel = np.ones((1,1), np.uint8) 
    outimg = cv2.dilate(outimg, kernel, iterations=1) 


    return outimg,outlist


def separate(mask):
    
    kernel = np.ones((2,2), np.uint8) 
    maskdict = {}

    hsvmask = cv2.cvtColor(mask,cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsvmask],[0],None,[180],[0,179])
    
    hues=[]
    for j,e in enumerate(hist):
        if e[0] > 100:
            hues.append(j)

    for hue in hues:

        threshed = cv2.inRange(hsvmask, (hue-1,2,100), (hue+1,255,255))
        threshed = cv2.medianBlur(threshed.astype(np.uint8), 3)
        threshed = cv2.dilate(threshed, kernel, iterations=1)

        if np.sum(threshed) <= 255*100:
            continue;

        maskdict[hue] = threshed

    return maskdict