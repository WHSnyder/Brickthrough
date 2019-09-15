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


root = "./piecetypes/"


parser = argparse.ArgumentParser()

parser.add_argument('-n', '--name', dest='name',
                  required=True,
                  help='Category name')

parser.add_argument('-s', '--symmetry', dest='sym',
                  required=False,
                  help='Is this category symmetric?')

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