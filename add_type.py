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
import json


root = "./piecetypes/"


parser = argparse.ArgumentParser()

parser.add_argument('-n', '--name', dest='name',
                  required=True,
                  help='Category name')

parser.add_argument('-s', '--symmetry', dest='sym',
                  required=False,
                  help='Is this category symmetric?')

parser.add_argument('-c', '--color', type=float, dest="color", nargs=3, required=False, help="Type color?")

'''
argv = sys.argv[sys.argv.index('--') + 1:]
args, _ = parser.parse_known_args(argv)

'''
args = parser.parse_args()


name = args.name + "/"

path = root + name
records_path = path + "records/"
model_path = path + "ckpts/"
output_path = path + "results/"
pieces_path = path + "pieces/"

os.mkdir(path)
os.mkdir(records_path)
os.mkdir(model_path)
os.mkdir(output_path)
os.mkdir(pieces_path)



# a Python object (dict):
attrs = {
    "color": (random.randrange(0,1,.05), random.randrange(0,1,.05), random.randrange(0,1,.05))
}

if (args.color):
	attrs["color"] = (color[0],color[1],color[2])

attrs_json = json.dumps(x)

os.system("echo \'{}\' > {}attrs.json".format(attrs_json, path))



