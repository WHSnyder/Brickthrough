import numpy as np
import cv2
from matplotlib import pyplot as plt

img0 = cv2.imread('2by4_renders/000008.png',0)
img1 = cv2.imread('2by4_renders/000016.png',0)

# Initiate STAR detector
orb = cv2.ORB_create()#ORB()

# find the keypoints with ORB
kp = orb.detect(img,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

# draw only keypoints location,not size and orientation
img0_pts = cv2.drawKeypoints(img0,kp,outImage=np.array([]),color=(0,255,0), flags=0)
plt.imshow(img0_pts),plt.show()