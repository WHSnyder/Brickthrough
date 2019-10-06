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





