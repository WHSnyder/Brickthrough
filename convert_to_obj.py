from __future__ import print_function

import argparse
import itertools
import json
from math import pi
import os
import random
import sys
import time
import copy

cats_dir = '/Users/will/projects/legoproj/piecetypes/'


parser = argparse.ArgumentParser()

parser.add_argument('-f', '--file', dest='file',
				  required=True, 
				  help='Zip file path?')

parser.add_argument('-c', '--cat', dest='cat',
				  required=True,
				  help='Piece category?')


args = parser.parse_args()

category = args.cat + "/"
zip_path = args.file

filename = zip_path.split('/')[-1]
piecename = filename.split('.')[0]


dest_path = cats_dir + category + "pieces/"
piece_path = dest_path + piecename
new_zip_path = piece_path + "/" + filename;


if (not os.path.exists(dest_path)):
	print("Category does not exist yet..")
	sys.exit()	

os.system("unzip " + zip_path + " -d " + dest_path)











