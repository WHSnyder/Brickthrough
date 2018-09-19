import argparse
import itertools
import json
import os
import random
import sys
import time
import copy


cats_dir = '/Users/will/projects/legoproj/piecetypes/'
blend = '/Applications/Blender/blender.app/Contents/MacOS/blender'
render_script = '/Users/will/projects/legoproj/keypointnet/tools/render.py'

command = '{} -b --python {} -- -m {} -o {} -s 200 -n 1200 -fov 5 --rotate'


parser = argparse.ArgumentParser()

parser.add_argument('-f', '--file', dest='file', nargs='*',
				  required=True, 
				  help='Obj file path?')


args = parser.parse_args()

obj_path = args.file[0]


if (not os.path.exists(obj_path)):
	print("Obj file not found")
	sys.exit()


parts = obj_path.split("/")

piece_dir = obj_path.replace(parts[-1], "")

count = 0

for content in os.listdir(piece_dir):
	if ("renders" in content):
		if (int(content[-1]) >= count):
			count = count + 1

output_dir = piece_dir + "renders" + str(count)

os.mkdir(output_dir)
os.system(command.format(blend, render_script, obj_path, output_dir))








