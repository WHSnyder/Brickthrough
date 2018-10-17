import numpy as np
import cv2
from matplotlib import pyplot as plt
import pickle




#From https://isotope11.com/blog/storing-surf-sift-orb-keypoints-using-opencv-in-python
def pickle_keypoints(keypoints, descriptors):
    i = 0
    temp_array = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
        point.class_id, descriptors[i])     
        ++i
        temp_array.append(temp)
    return temp_array

def unpickle_keypoints(array):
    keypoints = []
    descriptors = []
    for point in array:
        temp_feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors)
#end not mine




img0 = cv2.imread('wing_real.png',0)

orb = cv2.ORB_create()#ORB()
kp = orb.detect(img0,None)
kp, des = orb.compute(img0, kp)



img1 = cv2.imread('wing.png',0)

keypoints_database = pickle.load( open( "keypoints_database.p", "rb" ) )
kp1, desc1 = unpickle_keypoints(keypoints_database[0])


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

plt.imshow(img3,),plt.show()