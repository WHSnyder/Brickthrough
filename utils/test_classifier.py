#!/usr/bin/env python
# coding: utf-8
import json

import numpy as np
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy
import scipy.misc
import imageio
from PIL import Image

import random

import os
import feature_utils as fu

import cv2

import tensorflow as tf

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, Activation
from keras.utils import to_categorical
from keras.models import load_model
from keras.models import Model


from keras import backend as K

random.seed()


modeldir = '/Users/will/projects/legoproj/utils/'
model = load_model(modeldir + 'classifier_studs2.h5')

print(model.layers)
i = 0
for lay in model.layers:
	print(i)
	print(lay)
	i+=1



'''
laid = K.function([model.layers[0].input], [model.layers[-4].output])

layer_output = laid([x])[0]
'''
int_layer_model = Model(inputs=model.input, outputs=model.get_layer("activation_2").output)# model.layers[-4].output)   model.get_layer(layer_name).output)
#int_output = int_layer_model.predict(data)

os.environ['KMP_DUPLICATE_LIB_OK']='True'



dset = fu.dictFromJson("/Users/will/projects/legoproj/data/studs/dset.json")
dset = dset["list"]

num = 10

training_images = []
labels = np.zeros(num)

print("Setting up...")
for i in range(0,num):
    print("Reading image {}".format(i))

    img = cv2.imread(dset[i][0],cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128,128))
    training_images.append(img)
    labels[i] = dset[i][2]
    
arr = np.array(training_images)
labels = np.array(labels)

print("Consumed.")


while input("Predict?: ") != 'q':

    index = random.randint(0, num-1)

    fig = plt.figure(figsize=(4, 4))

    img = arr[index]/255
    label = labels[index]

    print("Numstuds: {}".format(label))

    print(model.predict(np.reshape(img, (1,128,128,1))) )

    pred = int_layer_model.predict(np.reshape(img, (1,128,128,1)))
    print(pred.shape)
    pred = pred[0,:,:,0]
    pred = np.reshape(pred, (55,55))

    print("Pred sum: {}".format(np.sum(pred)))

    fig.add_subplot(2, 2, 1)
    plt.imshow(img, interpolation='nearest', cmap='gray')

    fig.add_subplot(2, 2, 3)
    plt.imshow(pred, interpolation='nearest', cmap='gray')
    #plt.matshow(pred, interpolation='nearest', cmap='viridis')
    plt.show()
    

    #verts = get_object_studs

    #cv2.imshow('image',pred)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

sys.exit()	






