import numpy as np
import json
import re
import math
import cv2
from matplotlib import pyplot as plt




def showComboMask(img, data, objname, mode="obj"):
    
    if mode == "allbut":
        mask = 255 * np.ones((64,64))
        for key in data["objects"]:
            if key != objname:
                m = cv2.imread(data["objects"][key]["maskpath"], 0)
                print(mask.shape)
                mask = cv2.bitwise_or(mask,mask,mask=m)#,mask=None)

    else: 
        file = data["objects"][objname]["maskpath"]
        mask = cv2.imread(file, 0)
    
    mask = cv2.resize(mask, (512,512), interpolation=cv2.INTER_LINEAR)
    masked = cv2.bitwise_and(img,img,mask=mask)

    return masked