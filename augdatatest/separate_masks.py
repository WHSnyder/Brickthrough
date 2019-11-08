import os
import cv2
import argparse
import json
import numpy as np


parser = argparse.ArgumentParser()


parser.add_argument('-p', '--path', dest='path',
                  required=True,
                  help='BAse data path?')

parser.add_argument('-n','--num',dest='num',required=False,type=int)

args = parser.parse_args()

jsonpath = os.path.join(args.path,"data.json")
with open(jsonpath) as json_file:
        data = json.load(json_file)

write_path = os.path.join(args.path,"separations")

if not os.path.exists(write_path):
    os.mkdir(write_path)


huedict = {}

for obj in data["objects"]:
    hue = round(data["objects"][obj]["maskhue"],2)
    huedict[int(round(hue*179))] = obj

#print(huedict)

#print("\n\n")



def separate(imgpath,maskpath):

    img = cv2.imread(imgpath)
    cv2.imwrite(os.path.join(write_path,entry["r"]),cv2.bilateralFilter(img, 2, 900, 100))

    mask = cv2.imread(maskpath)
    hsvmask = cv2.cvtColor(mask,cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([hsvmask],[0],None,[180],[0,179])
    #print([(i,e[0]) for (i,e) in list(enumerate(hist)) if e[0] >= 1000])
    #print("\n\n")

    hues=[]
    for j,e in enumerate(hist):
        if e[0] > 1000:
            hues.append(j)

    for hue in hues:

        threshed = cv2.inRange(hsvmask, (hue-1,0,100), (hue+1,255,255))

        if np.sum(threshed) <= 255*1500:
            continue;

        hu = hue
        m = -1
        n = 1
        while hu not in huedict:
            hu+=n*m
            n+=1
            m*=-1 

        threshpath = os.path.join(write_path,"{}_mask_{}_diff_{}.png".format(entry["x"],huedict[hu],hu-hue))
        cv2.imwrite(threshpath,threshed)




if args.num:

    entry = data["renders"][args.num]
    imgpath = os.path.join(args.path,entry["r"]) 
    maskpath = os.path.join(args.path,entry["m"])
    separate(imgpath,maskpath)

else:

    for i,entry in enumerate(data["renders"]):
        imgpath = os.path.join(args.path,entry["r"]) 
        maskpath = os.path.join(args.path,entry["m"])
        separate(imgpath,maskpath)

    