import os
import cv2
import argparse
import json



parser = argparse.ArgumentParser()


parser.add_argument('-p', '--path', dest='path',
                  required=True,
                  help='Base data path?')
parser.add_argument('-n','--name',dest="name",required=True)

args = parser.parse_args()

jsonpath = os.path.join(args.path,"data.json")
with open(jsonpath) as json_file:
        data = json.load(json_file)

write_path = os.path.join(args.path,"separations")

if not os.path.exists(write_path):
    os.mkdir(write_path)

img =np.ones((512,512,3))



huedict = {}

for obj in data["objects"]:
    hue = round(data["objects"][obj]["maskhue"],2)
    huedict[int(round(hue*179))] = obj

print(huedict)
