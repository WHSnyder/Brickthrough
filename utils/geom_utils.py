from __future__ import division



import numpy as np
import numpy.linalg as linalg
import json
import re
import math
import cv2
from matplotlib import pyplot as plt


brickStudsMat = 255 * np.array([
                          [1,1,1,1],
                          [1,1,1,1]], dtype=np.uint8)

wingStudsMat = 255 * np.array([
                        [1,1,1,1,1,1],
                        [1,1,1,1,1,1],
                        [1,1,1,1,1,1],
                        [1,1,1,1,1,0],
                        [1,1,1,1,1,0],
                        [1,1,1,1,1,0],
                        [1,1,1,1,0,0],
                        [1,1,1,1,0,0],
                        [1,1,1,1,0,0],
                        [1,1,1,0,0,0],
                        [1,1,1,0,0,0],
                        [1,1,1,0,0,0]], dtype=np.uint8)

def angleBetween(s, neighbors):

    normalize = lambda n: (n-s)/linalg.norm(n-s) 

    #neighbors = neighbors - cur
    neighs = list(map(normalize, neighbors))

    return np.dot(neighs[0], neighs[1])

def normDot(v1,v2):
    v1 = v1.astype(np.float32)
    #v1[1] = 512-v1[1]
    v2 = v2.astype(np.float32)
    return np.dot(v1/linalg.norm(v1), v2/linalg.norm(v2))




def getNearest(cur,centroids):

    centroids = np.asarray(centroids)
    cur = np.asarray(cur)

    roids = sorted(centroids, key=lambda t: linalg.norm(t-cur))

    return roids[0:3]




def march(start, neigh, cnts):

    jump = 2 * neigh - start

    nexthole = getNearest(jump,cnts)[0]

    if angleBetween(neigh, [start,nexthole]) == 180:
        return nexthole
    else:
        return None



def marchToEnd(start,neigh,cnts):

    s = start
    n = neigh

    nexthole = march(s,n,cnts)

    line = []

    while nexthole is not None:

        line.append(nexthole)
        s = n
        n = nexthole
        nexthole = march(s, n, cnts)

    return line



def marchStudLine(s,n,cnts):

    lineup = marchToEnd(s,n,cnts)
    linedown = marchToEnd(n,s,cnts)
    linedown.reverse()

    stub = [s,n]

    return linedown + stub + lineup


def maxStudLine(s,n,cnts):

    vec = n-s
    vec = vec/linalg.norm(vec)

    parts = []

    for cnt in cnts:

        candid = cnt - s
        candid = candid/linalg.norm(candid)
        dot = abs(np.dot(candid, vec))

        if dot - 1 < .1:
            parts.append(cnt)

    return parts





def getCornerLines(center, neighbors, cnts, img, show=False):

    center = np.array(center)
    neighbors = np.array(neighbors)
    cnts = np.array(cnts)

    #print("Neighbors " + str(neighbors))
    #print("Center " + str(center))

    line1 = marchStudLine(center,neighbors[0],cnts)
    line2 = marchStudLine(center,neighbors[1],cnts)

    if show:

        rgb = cv2.cvtColor(img.astype(np.dtype("float32")), cv2.COLOR_GRAY2RGB)

        cv2.line(rgb, tuple(line1[0]), tuple(line1[-1]), (20,255,20), 2)
        cv2.line(rgb, tuple(line2[0]), tuple(line2[-1]), (20,20,255), 2)

        plt.imshow(rgb/255)
        plt.show()

    return line1, line2



def findCorner(locs):

    locs = np.array(locs)

    for loc in locs:

        corner = getNearest(loc,locs)[1:3]
        dot = abs(angleBetween(loc,corner))

        if dot < .06:
            print(dot)
            return loc, corner

    return None,None




def getMostCommonVec(locs,img=None,show=False):

    locs=np.array(locs)    
    candids =[]

    for loc in locs:

        nearest = getNearest(loc,locs)[1]
        #print("Nearest = " + str(nearest))
        candids.append(np.array(nearest - loc))

    if show and img is not None:
        rgb = cv2.cvtColor(img.astype(np.dtype("float32")), cv2.COLOR_GRAY2RGB)

        for i,candid in enumerate(candids):
            cv2.line(rgb, tuple(locs[i]), tuple(locs[i] + candid), (20,255,20), 2)

        plt.imshow(rgb/255)
        plt.show()

    return np.array(candids)




def getBasis(locs):

    l = getMostCommonVec(locs)
    s = l[0]

    for v in l[1:]:

        #dot=abs(angleBetween(np.array([0,0]), [s,v]))
        #print(dot)

        ndot = abs(normDot(s,v))

        print(ndot)

        if ndot < .99:
            return s,v

    return None,None



def getAllOnBasis(st,b1,b2,locs,img=None,show=False):
    r=[]
    st=np.asarray(st)
    locs=np.asarray(locs)
    b1=np.asarray(b1)
    b2=np.asarray(b2)

    for loc in locs:
        if not (loc == st).all():
        	loc[1] = 512 - loc[1]
            dot1 = abs(normDot(loc - st, b1))
            dot2 = abs(normDot(loc - st, b2))
            
            if dot1 > .9: 
                print(dot1)
                r.append(loc)
            if dot2 > .9:
                print(dot2)
                r.append(loc)
        else:
            print("lol")

    if show and img is not None:
        rgb = cv2.cvtColor(img.astype(np.dtype("float32")), cv2.COLOR_GRAY2RGB)

        for loc in r:
            #cv2.circle(rgb, tuple(loc), 3, (20,255,20), 2)
            cv2.line(rgb, tuple(st),tuple(loc),(200,30,200),1)

        plt.imshow(rgb/255)
        plt.show()

    return r





def getExtremes(img,cnts,show=False,num=5):

    contours, hierarchy = cv2.findContours(img.astype(np.dtype('uint8')), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
    curve = []
    prec = .01
    while (len(curve) != num):
        prec += .01
        curve = cv2.approxPolyDP(hull,prec,True)
    

    pts=[]
    for pt in curve:
        closest = getNearest(pt, cnts)[0]
        pts.append(closest)

    if show:
        rgb = cv2.cvtColor(img.astype(np.dtype("float32")), cv2.COLOR_GRAY2RGB)
        cv2.drawContours(rgb, [curve], 0, (150,30,255), 2)

        for pt in pts:
            cv2.circle(rgb, tuple(pt), 3, (100,100,200),2)

        plt.imshow(rgb/255)
        plt.show()


    return curve



def getBasisCorrs(locs):






