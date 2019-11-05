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
import feature_utils as fu


os.environ['KMP_DUPLICATE_LIB_OK']='True'
datadir = '/home/will/projects/legoproj/data/studs_oneofeach/'
modeldir = '/home/will/projects/legoproj/utils/'

random.seed(0)
iters = 1000



parser = argparse.ArgumentParser()
parser.add_argument('-p', '--predict', dest='predict', action='store_true', help='Predict the data or not?', required=False)
args = parser.parse_args()





if args.predict:

    model = load_model(modeldir + 'studTracer2.h5')

    while input("Predict?: ") != 'q':

        index = random.randint(0, iters-1)
    
        fig = plt.figure(figsize=(4, 4))

        img = np.array(Image.open(datadir + str(index) +"_studs_a.png").convert(mode="L"))/255
        #img = np.array(Image.open("wing.png").convert(mode="L"))/255

        gt = np.reshape(fu.getStudMask(index), (256,256))

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
    
    scenestuds = np.reshape(fu.getStudMask(i), (256,256,1))
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