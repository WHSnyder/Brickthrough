import numpy as np
import numpy.linalg as linalg
import json
import re
import math
import cv2
from matplotlib import pyplot as plt





def getTwoNearest(cur,centroids):

    #slow but easier...

    centroids = np.asarray(centroids)
    cur = np.asarray(cur)

    #centroidslist = list(enumerate(centroids))
    #roids = sorted(centroidslist, key=lambda t: linalg.norm(t[1]-cur))

    roids = sorted(centroids, key=lambda t: linalg.norm(t-cur))

    return roids[0:3]




def angleBetween(cur, neighbors):

    normalize = lambda n: (n-cur)/linalg.norm(n-cur) 

    #neighbors = neighbors - cur
    neighs = list(map(normalize, neighbors))

    dot = np.dot(neighs[0], neighs[1])

    if -.1 < dot < .1:
        return 90

    elif -1.1 < dot < -.9:
        return 180
    else:
        return None




def march(start, neigh, cnts):

    jump = 2 * neigh - start

    nexthole = getTwoNearest(jump,cnts)[0]

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




def getStudLine(s,n,cnts):

    lineup = marchToEnd(s,n,cnts)
    linedown = marchToEnd(n,s,cnts)
    linedown.reverse()

    stub = [s,n]

    return linedown + stub + lineup



def drawCornerLines(center, neighbors, cnts, img):

    center = np.array(center)
    neighbors = np.array(neighbors)
    cnts = np.array(cnts)

    line1 = getStudLine(center,neighbors[0],cnts)
    line2 = getStudLine(center,neighbors[1],cnts)

    rgb = cv2.cvtColor(img.astype(np.dtype("float32")), cv2.COLOR_GRAY2RGB)

    cv2.line(rgb, tuple(line1[0]), tuple(line1[-1]), (20,255,20), 3)
    cv2.line(rgb, tuple(line2[0]), tuple(line2[-1]), (20,20,255), 3)

    plt.imshow(rgb/255)
    plt.show()







