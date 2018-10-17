import numpy as np
import cv2
from matplotlib import pyplot as plt


img0 = cv2.imread('wing_real.png',0)
img1 = cv2.imread('wing.png',0)


orb = cv2.ORB_create()#ORB()
kp0, des0 = orb.detectAndCompute(img0,None)
kp1, des1 = orb.detectAndCompute(img1,None)



#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
#matches = bf.match(des0,des1)


FLANN_INDEX_LSH = 6
FLANN_INDEX_KDTREE = 0
index_params = index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2

search_params = dict(checks=100)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des0,des1,k=2)

# Sort them in the order of their distance.
#matches = sorted(matches, key = lambda x:x.distance)

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = None,
                   flags = 0)

# Draw first 10 matches.
img3 = cv2.drawMatches(img0,kp0,img1,kp1,matches,None,**draw_params)
plt.imshow(img3,),plt.show()



'''

FLANN_INDEX_LSH = 6
FLANN_INDEX_KDTREE = 0
index_params = index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2

search_params = dict(checks=100)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des,desc1,k=2)

print("# matches: " + str(len(matches)))

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
c = 0
for i,tup in enumerate(matches):
    
    if len(tup) < 2: 
        c += 1
        continue

    m = tup[0]
    n = tup[1]
    if m.distance < 100 * n.distance:
        matchesMask[i]=[1,0]

print("masked: " + str(c))

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img3 = cv2.drawMatchesKnn(img0,kp,img1,kp1,matches,None,**draw_params)

plt.imshow(img3,),plt.show()'''