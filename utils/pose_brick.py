import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import math


img0 = cv2.imread('wing2.png',0)
img = cv2.imread('/Users/will/projects/legoproj/data_oneofeach/studs_oneofeach/4_studs_a.png')


#orb = cv2.ORB_create()#ORB()
#kp0, des0 = orb.detectAndCompute(img0,None)
#kp1, des1 = orb.detectAndCompute(img1,None)


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#gray = np.float32(gray)


gray = cv2.bilateralFilter(gray, 11, 17, 17)


edged = cv2.Canny(gray, 30, 200)

cv2.imshow('Edges', edged)
if cv2.waitKey(0) & 0xff == 27:
   cv2.destroyAllWindows()


kernel = np.ones((2,2), np.uint8) 
edged = cv2.dilate(edged, kernel, iterations=1) 

cv2.imshow('Dilated', edged)
if cv2.waitKey(0) & 0xff == 27:
   cv2.destroyAllWindows()


contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)



img1 = cv2.cvtColor(edged,cv2.COLOR_GRAY2RGB)

num = len(contours)

for i in range(num):
	one = math.floor(i/num * 255)
	other = 255 - one
	b = 150 if i % 2 else 20
	cv2.drawContours(img1, contours, i, (one, other, b), 1)
	'''
	if cv2.arcLength(contours[i],False) > 690:
		cnt = contours[i]
		cv2.circle(img1, (x, y), i, (one, other, 0), 3)
	'''



#cv2.drawContours(edged, contours, -1, (255, 0, 0), 3) 
  
cv2.imshow('Contours', img1) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 


for contour in contours:
	perimeter = cv2.arcLength(contour,True)
	print("Found contour of length: " + str(perimeter))


