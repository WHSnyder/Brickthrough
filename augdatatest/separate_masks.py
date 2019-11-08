import os
import cv2
import argparse
import json



parser = argparse.ArgumentParser()


parser.add_argument('-p', '--path', dest='path',
                  required=True,
                  help='BAse data path?')

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

print(huedict)

print("\n\n")
for i,entry in enumerate(data["renders"]):

    #if i != 10:
    #    continue;

    imgpath = os.path.join(args.path,entry["r"]) 
    maskpath = os.path.join(args.path,entry["m"])

    img = cv2.imread(imgpath)
    cv2.imwrite(os.path.join(write_path,entry["r"]),cv2.bilateralFilter(img, 2, 900, 100))

    mask = cv2.imread(maskpath)
    hsvmask = cv2.cvtColor(mask,cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([hsvmask],[0],None,[180],[0,180])
    print(list(enumerate(hist)))
    print("\n\n")

    hues=[]
    for j,e in enumerate(hist[1:]):
        if e[0] > 100:
            hues.append(j+1)

    for hue in hues:
        threshed = cv2.inRange(hsvmask, (hue,0,0), (hue,255,255))
        print(hue)

        hu = hue
        m = -1
        n = 1
        while hu not in huedict:
        	hu+=n*m
        	n+=1
        	m*=-1 


        threshpath = os.path.join(write_path,"{}_mask_{}_diff_{}.png".format(entry["x"],huedict[hu],hu-hue))
        cv2.imwrite(threshpath,threshed)