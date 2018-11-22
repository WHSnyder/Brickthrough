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
datadir = '/Users/will/projects/legoproj/augdatatest/kps/'
'''
random.seed(0)

parser = argparse.ArgumentParser()

parser.add_argument('-p', '--predict', dest='predict', action='store_true', help='Predict the data or not?', required=False)

args = parser.parse_args()
'''

#expr = re.compile("^\((([-]?[0-9]*\.[0-9]{4})\,){3}([-]?[0-9]*\.[0-9]{4})\)$")

expr = re.compile("([-]?[0-9]*\.[0-9]{4})")




def get_matrix(lines):
    return np.array([[float(y) for y in x.strip().split(" ")] for x in lines])

def read_projection_matrix():
    filename = "/Users/will/Desktop/projection.txt"
    with open(filename, "r") as f:
        lines = f.readlines()
    
    return get_matrix(lines)




def matrix_from_string(matstring):

    #matstring = re.sub(r"[\n\t\s]*", "", matstring)
    #print(matstring)

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

    if objname == "wing":
        print("Num verts in a wing: " + str(len(lines)))

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
        print("Shape: " + str(vert.shape))
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



modelmats = get_object_matrices("/Users/will/projects/legoproj/data_oneofeach/studs_oneofeach/mats/0.txt")

cammat = modelmats["Camera"]
projmat = modelmats["Projection"]

brickstuds = get_object_studs("brick")
wingstuds = get_object_studs("wing")

#sys.exit()

scenestuds = np.zeros((512,512))

screenverts = []

for key in modelmats:

    if "Brick" in key:
        studs = brickstuds
    elif "Wing" in key:
        studs = wingstuds
    else:
        continue

    #print("====================================\n{} :".format(key))
    screenverts += verts_to_screen(modelmats[key], cammat, projmat, studs) 
    #print("====================================")


for vert in screenverts:
    npcoord = tuple([math.floor((1 - vert[1]) * 512), math.floor(vert[0] * 512)])
    scenestuds[npcoord[0], npcoord[1]] = 1

plt.imshow(scenestuds, interpolation='nearest')        
plt.show()




sys.exit()


training_images = []
points_images = []

for i in range(0,2000):
    img = np.array(Image.open(datadir + str(i) +"pole.png").convert(mode="L"))
    img = np.reshape(img, (256,256,1))
    img = img/255

    pts = np.array(Image.open(datadir + str(i) +"pts.png").convert(mode="L"))
    pts = np.reshape(pts, (256,256,1)) 
    pts = pts/255 
    
    training_images.append(img)
    points_images.append(pts)

imgsarr = np.array(training_images)
ptsarr = np.array(points_images)




if args.predict:

    model = load_model(datadir + 'reg4.h5')
    n = 'g'

    while n != 'q':

        n = input("Predict?: ")

        index = random.randint(0, 2000)
    
        fig=plt.figure(figsize=(4, 4))

        img = np.array(Image.open(datadir + str(index) +"pole.png").convert(mode="L"))

        if n == 'r':
            img = np.array(Image.open(datadir + "idk.png").convert(mode="L"))

        gt = np.array(Image.open(datadir + str(index) +"pts.png").convert(mode="L"))

        pred = model.predict(np.reshape(img, (1,256,256,1)))
        pred = np.reshape(pred, (256,256))

        fig.add_subplot(2, 1, 1)
        plt.imshow(img, interpolation='nearest')

        fig.add_subplot(2, 1, 2)
        plt.imshow(pred, interpolation='nearest')

        fig.add_subplot(2, 2, 2)
        plt.imshow(gt, interpolation='nearest')        

        plt.show()

    sys.exit()

#mobile = mobilenet_v2.MobileNetV2(input_shape=(256,256,3), include_top=False, alpha=1.0, weights='imagenet', input_tensor=None, pooling=None)


#print("Mobiling")
#mobile_results = mobile.predict(imgsarr[0:2])


#sys.exit()

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(256,256,1),
                 padding='same'))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
#model.add(MaxPooling2D((2,2), padding = 'same'))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same', dilation_rate=(4,4) ))
model.add(Dropout(.1))

#model.add(MaxPooling2D((2,2), padding = 'same'))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same', dilation_rate=(16,16) ))
model.add(Dropout(.2))
#model.add(MaxPooling2D((2,2), padding = 'same'))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same', dilation_rate=(4,4) ))
#model.add(MaxPooling2D((2,2), padding = 'same'))
model.add(Dropout(.2))


#model.add(keras.layers.Conv2DTranspose(10, (2,2), strides=(2,2), padding='same', activation='relu'))
#model.add(keras.layers.Conv2DTranspose(8, (2,2), strides=(2,2), padding='same', activation='relu'))
#model.add(keras.layers.Conv2DTranspose(5, (2,2), strides=(2,2), padding='same', activation='relu'))
model.add(Conv2D(1, (4,4), activation='linear', padding='same'))

model.compile(optimizer='adam', loss='mse', metrics=['mse','mae'])


print(model.summary())

history = model.fit(imgsarr[0:1000], ptsarr[0:1000], epochs=2, batch_size=20,  verbose=1, validation_split=0.6)

model.save(datadir + "reg4.h5")


# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()