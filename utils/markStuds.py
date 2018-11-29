#!/usr/bin/env python
# coding: utf-8

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy
import scipy.misc
import imageio
from PIL import Image

import os
import sys
import argparse
import random

import tensorflow as tf

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.models import load_model
from keras.applications import mobilenet_v2

import json
import re
import cv2
import math


os.environ['KMP_DUPLICATE_LIB_OK']='True'
datadir = '/Users/will/projects/legoproj/data_oneofeach/studs_oneofeach/'
modeldir = '/Users/will/projects/legoproj/utils/'

random.seed(0)
expr = re.compile("([-]?[0-9]*\.[0-9]{4})")
iters = 1000
dim = 512



parser = argparse.ArgumentParser()
parser.add_argument('-p', '--predict', dest='predict', action='store_true', help='Predict the data or not?', required=False)
args = parser.parse_args()



def matrix_from_string(matstring):

    matches = expr.findall(matstring)

    nums = np.asarray(list(map(lambda x: float(x), matches)), dtype=np.float32)
    nums = np.reshape(nums, (4,4))

    return nums
    
def get_object_matrices(filename):

    data = {}

    with open(filename) as json_file:
        data = json.load(json_file)

    for key in data:
        data[key] = matrix_from_string(data[key])

    return data



def get_object_studs(objname):

    filename = "/Users/will/Desktop/{}.txt".format(objname)
    
    with open(filename, "r") as fp:
        verts = fp.read()
    lines = verts.split("\n")[1:]
    verts = []

    for line in lines:

        if line == "":
            break

        parts = line.split(",")

        nums = list(map(lambda x: float(x), parts))
        vert = np.ones(4, dtype=np.float32)
        vert[0:3] = nums[0:3]

        verts.append(vert)

    return verts



def verts_to_screen(model, view, frust, verts):
    
    #mvp = np.matmul( frust, np.matmul(view, model) )
    screenverts = []
    worldverts = []
    camverts = []

    #print("Model: \n{}".format(str(model)))
    #print("View: \n{}".format(str(view)))
    #print("Frust: \n{}".format(str(frust)))
    #print("--------------------------------------")
    #print("Verts local coordinates: \n{}\n".format(str(verts)))

    for vert in verts:
        #print("Shape: " + str(vert.shape))
        worldvert = np.matmul(model, vert)
        camvert = np.matmul(view, worldvert)
        screenvert = np.matmul(frust, camvert)
        screenvert = screenvert/screenvert[3]

        if abs(screenvert[0]) < 1 and abs(screenvert[1]) < 1:
            screenvert[0:2] = (screenvert[0:2] + 1)/2
            screenverts.append(screenvert)
        
        worldverts.append(worldvert)
        camverts.append(camvert)

    #print("Verts world coordinates: \n{}\n".format(worldverts))
    #print("Verts camera coordinates: \n{}\n".format(camverts))
    #print("Verts screen coordinates: \n{}\n".format(screenverts))
    #print("--------------------------------------")

    return screenverts


brickstuds = get_object_studs("brick")
wingstuds = get_object_studs("wing")


def getStudMask(i):

    modelmats = get_object_matrices(datadir + "mats/{}.txt".format(i))
    cammat = modelmats["Camera"]
    projmat = modelmats["Projection"]

    maskdim = int(dim/2)
    scenestuds = np.zeros((maskdim,maskdim))
    screenverts = []

    for key in modelmats:

        if "Brick" in key:
            studs = brickstuds
        elif "Wing" in key:
            studs = wingstuds
        else:
            continue
        screenverts += verts_to_screen(modelmats[key], cammat, projmat, studs) 

    for vert in screenverts:
        npcoord = tuple([math.floor(1 - vert[1] * maskdim), math.floor(vert[0] * maskdim)])
        scenestuds[npcoord[0], npcoord[1]] = 1

    scenestuds = np.reshape(scenestuds, (maskdim,maskdim,1))

    return scenestuds





if args.predict:

    model = load_model(modeldir + 'studTracer2.h5')

    while input("Predict?: ") != 'q':

        index = random.randint(0, iters-1)
    
        fig = plt.figure(figsize=(4, 4))

        img = np.array(Image.open(datadir + str(index) +"_studs_a.png").convert(mode="L"))/255
        #img = np.array(Image.open("wing.png").convert(mode="L"))/255

        gt = np.reshape(getStudMask(index), (256,256))

        print("Gt sum: {}".format(np.sum(gt)))

        pred = model.predict(np.reshape(img, (1,512,512,1)))
        pred = np.reshape(pred, (256,256))

        print("Pred sum: {}".format(np.sum(pred)))

        fig.add_subplot(2, 2, 1)
        plt.imshow(img, interpolation='nearest', cmap='gray')

        fig.add_subplot(2, 2, 2)
        plt.imshow(gt, interpolation='nearest', cmap='gray')

        fig.add_subplot(2, 2, 3)
        plt.imshow(pred, interpolation='nearest')

        plt.show()
        

        #verts = get_object_studs

        #cv2.imshow('image',pred)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    sys.exit()



'''
Read object transform and generate stud masks.
'''
print("Calculating stud masks...")

studimgs = []

for i in range(iters):

    print("Getting stud data for render {}".format(i))
    
    scenestuds = np.reshape(getStudMask(i), (256,256,1))
    studimgs.append(scenestuds)


studimgs = np.array(studimgs)




'''
Read rendered images...
'''
print("Reading image files...")

imgsarr = []

for i in range(iters):
    print("Reading render {}".format(i))

    img = np.array(Image.open(datadir + str(i) +"_studs_a.png").convert(mode="L"))
    img = np.reshape(img, (512,512,1))/255
    
    imgsarr.append(img)


imgsarr = np.array(imgsarr)



print("Preproc done...")


model = Sequential()
model.add(Conv2D(64, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=(512,512,1),
                 padding='same'))

#model.add(Conv2D(20, (4, 4), activation='relu', padding='same'))


model.add(Conv2D(10, (3, 3), activation='relu', padding='same', dilation_rate=(3,3)))
model.add(MaxPooling2D((2,2), padding = 'same'))

#model.add(Conv2D(15, (3, 3), activation='relu', padding='same', dilation_rate=(2,2) ))
#model.add(MaxPooling2D((2,2), padding = 'same'))

#model.add(keras.layers.Conv2DTranspose(10, (2,2), strides=(2,2), padding='same', activation='relu'))
#model.add(keras.layers.Conv2DTranspose(8, (2,2), strides=(2,2), padding='same', activation='relu'))
#model.add(keras.layers.Conv2DTranspose(5, (2,2), strides=(2,2), padding='same', activation='relu'))
model.add(Conv2D(1, (4,4), activation='relu', padding='same'))

model.compile(optimizer='adam', loss='mse', metrics=['mse','mae'])


print(model.summary())

history = model.fit(imgsarr[0:999], studimgs[0:999], epochs=2, batch_size=15,  verbose=1, validation_split=0.6)

model.save(modeldir + "studTracer2.h5")


# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()