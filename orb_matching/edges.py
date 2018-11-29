import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys


img0 = cv2.imread('wing2.png',0)
#img = cv2.imread('wing.png')
img = cv2.imread('/Users/will/projects/legoproj/data_oneofeach/val_oneofeach/3_val_a.png')


#orb = cv2.ORB_create()#ORB()
#kp0, des0 = orb.detectAndCompute(img0,None)
#kp1, des1 = orb.detectAndCompute(img1,None)


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#gray = np.float32(gray)


gray = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 30, 200)

cv2.imshow('dst',edged)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()


sys.exit()



dst = cv2.cornerHarris(gray,2,3,0.01)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()