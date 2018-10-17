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




img0 = cv2.imread('wing.png',0)
#img1 = cv2.imread('2by4_renders/000016.png',0)

# Initiate STAR detector
orb = cv2.ORB_create()#ORB()

# find the keypoints with ORB
kp = orb.detect(img0,None)

# compute the descriptors with ORB
kp, des = orb.compute(img0, kp)

# draw only keypoints location,not size and orientation



temp_array = []
temp = pickle_keypoints(kp, des)
temp_array.append(temp)
pickle.dump(temp_array, open("keypoints_database.p", "wb"))





keypoints_database = pickle.load( open( "keypoints_database.p", "rb" ) )
kp, desc = unpickle_keypoints(keypoints_database[0])

img0_pts = cv2.drawKeypoints(img0,kp,outImage=np.array([]),color=(0,255,0), flags=0)









plt.imshow(img0_pts),plt.show()







