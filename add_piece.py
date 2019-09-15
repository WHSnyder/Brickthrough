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




parser = argparse.ArgumentParser()
'''
parser.add_argument('-n', '--name', dest='name',
                  required=True,
                  help='Piece name?')
'''
parser.add_argument('-c', '--cat', dest='cat',
                  required=True,
                  help='Category?')

parser.add_argument('-f', '--file', dest='file',
				  required=True, 
				  help='File path?')

args = parser.parse_args()


pieces_dir = "./piecetypes/" + args.cat + "/pieces/"

filename = args.file.split('/')[-1]
name = filename.split('.')[0] + "/"

path = pieces_dir + name

if (os.path.exists(pieces_dir)):
	if (not os.path.exists(path)):
		os.mkdir(path)
	os.rename(args.file, path + filename)
else:
	print("Category does not exist yet..")	

