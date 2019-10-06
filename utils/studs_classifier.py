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

import os
import feature_utils as fu

import cv2



os.environ['KMP_DUPLICATE_LIB_OK']='True'

split = float(sys.argv[1])

if split <= 0 or split >= 1:
    print("invalid split")
    sys.exit()





dset = fu.dictFromJson("/Users/will/projects/legoproj/data/studs/dset.json")
dset = dset["list"]

num = int(len(dset)*.8)
split = int(num * split)


training_images = []
labels = np.zeros(num)

print("Setting up...")
for i in range(0,num):
    print("Reading image {}".format(i))

    img = cv2.imread(dset[i][0],cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128,128))
    training_images.append(img)
    labels[i] = int(dset[i][2])
    
arr = np.array(training_images)
labels = np.array(labels)


print("Consumed.")



import tensorflow as tf

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, Activation
from keras.utils import to_categorical


trainset = arr[:split]
trainset = np.reshape(trainset,(split,128,128,1))

testset = arr[split:]
testset = np.reshape(testset,(num - split,128,128,1))

testset = testset/255.0
trainset = trainset/255.0

trainlabels = labels[:split]
testlabels = labels[split:]

trainlabels = to_categorical(trainlabels, num_classes=37)
testlabels = to_categorical(testlabels, num_classes=37)

print(trainlabels[0])
#sys.exit()


print(trainset.shape)


print("Beginning...")


model = Sequential()

model.add(Conv2D(64, kernel_size=(4, 4),
                 activation='relu',
                 input_shape=(128,128,1)))

model.add(Conv2D(32, (5, 5), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(15, (4, 4)))
model.add(BatchNormalization())
model.add(Activation('relu'))


model.add(Conv2D(1, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dropout(0.25))

model.add(Flatten())


model.add(Dense(80)) 
model.add(BatchNormalization())
model.add(Activation('relu')) 

model.add(Dropout(0.5)) 

model.add(Dense(60)) 
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(37))
model.add(Activation('softmax'))



#model.compile(optimizer='adam', loss='mse', metrics=['mse'])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())


history = model.fit(trainset, trainlabels, batch_size=60, verbose=1, epochs=3, validation_split=0.2)

#history = model.fit(imgsarr[0:999], studimgs[0:999], epochs=2, batch_size=15,  verbose=1, validation_split=0.6)


test_loss, test_acc = model.evaluate(testset[0:1], testlabels[0:1])

print('Test accuracy:', test_acc)


model.save("classifier_studs2.h5")


# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()