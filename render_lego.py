import argparse
import itertools
import json
import os
import random
import sys
import time
import copy
import json


cats_dir = '/Users/will/projects/legoproj/piecetypes/'
blend = '/Applications/Blender/blender.app/Contents/MacOS/blender'
render_script = '/Users/will/projects/legoproj/keypointnet/tools/render.py'


def getCategoryColor(category):

    attr_path = cats_dir + category + "/attrs.json"
    data = (0.1,0.1,0.1)

    with open(attr_path) as f:
        data = json.load(f)

    return data["color"]




command = '{} -b --python {} -- -m {} -o {} -s 200 -n {} -fov 5 -c {} {} {}'


parser = argparse.ArgumentParser()

parser.add_argument('-f', '--file', dest='file', nargs='*',
                  required=False, 
                  help='Obj file path?')

parser.add_argument('-p', '--piece', dest='piece', nargs='*',
				  required=False,
				  help='Piece code?')

parser.add_argument('-n', '--num', dest='num', required=False, default=1200, type=int)

parser.add_argument('-t', '--tag', dest='tag', required=False, default="0")


args = parser.parse_args()

obj_path = args.file[0]


if (not os.path.exists(obj_path)):
    print("Obj file not found")
    sys.exit()


parts = obj_path.split("/")

piece_dir = obj_path.replace(parts[-1], "")
category = parts[-4]

count = 0

for content in os.listdir(piece_dir):
    if ("renders" in content):
        tag = int(content[-1])
        if (tag >= count):
            count = tag + 1

output_dir = piece_dir + "renders" + str(count)

color = getCategoryColor(category)
print(color)


os.mkdir(output_dir)
os.system(command.format(blend, render_script, obj_path, output_dir, args.num, color[0], color[1], color[2]))
