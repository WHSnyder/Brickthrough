import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon

import argparse
import cv2

lgray = (.8,.8,.8)
gray = (.5,.5,.5)
black = (.1,.1,.1)


random.seed(0)






masks = []

#def bitMask(masks):








while input("Go: ") != "q":

    fig=plt.figure(figsize=(4, 4))

    #num = random.randint(0,2)
    img = cv2.imread("combo.png",0)

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    
    hist = cv2.calcHist([img],[0], None, [15], [2, 240])
    maxnum = 0
    index = 0  
    for i, bucket in enumerate(hist):
        if bucket[0] > maxnum:
            maxnum = bucket[0]
            index = i

    print((index,maxnum))



    
    '''
    fig.add_subplot(2, 1, 1)
    plt.hist(hist)

    fig.add_subplot(2, 1, 2)
    plt.imshow(gray)
   
    plt.show()
    '''

    fig.add_subplot(2, 1, 1)
    #plt.hist(img.ravel(),256,[1,240])
    plt.plot(hist)

    fig.add_subplot(2, 1, 2)
    plt.imshow(img, cmap='gray')

    plt.show()




    '''
    hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)

    hist = cv2.calcHist([hsv],[0, 2], None, [10, 10], [0, 180, 0, 256] )

    fig.add_subplot(2, 1, 1)
    plt.plot(hist)

    fig.add_subplot(2, 1, 2)
    plt.imshow(img)
   
    plt.show()
    '''





















