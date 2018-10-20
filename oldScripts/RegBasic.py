#!/usr/bin/env python
# coding: utf-8



import numpy as np
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy
import scipy.misc
import imageio
from PIL import Image

import os
import sys



os.environ['KMP_DUPLICATE_LIB_OK']='True'

#split = int(sys.argv[1])

datasize = 1500

training_images = []
points_images = []

for i in range(0,datasize):
    img = np.array(Image.open("./kps/" + str(i) +"pole.png").convert(mode="L"))
    pts = np.array(Image.open("./kps/" + str(i) +"pts.png").convert(mode="L"))
    pts = np.reshape(pts, (4096,)) 
    
    training_images.append(img)
    points_images.append(pts)
    
print(training_images[100].shape)
print(points_images[100].shape)



imgsarr = np.array(training_images)
ptsarr = np.array(points_images)


import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical


'''
trainset = arr[:split]
trainset = np.reshape(trainset,(split,128,128,1))


testset = arr[split:]
testset = np.reshape(testset,(datasize - split,128,128,1))

print(trainset.shape)


testset = testset/255.0
trainset = trainset/255.0

class_dict = {"Wing": 0, "Brick": 1, "Pole": 2}

labels = np.zeros(datasize)

i = 0

with open("./pose_basic_train/labels.txt") as fp:
    
    line = fp.readline()

    while line:
        
        parts = line.split()
        labels[i] = float(parts[1])
        line = fp.readline()
        i+=1


trainlabels = labels[:split]
testlabels = labels[split:]
'''


model = Sequential()
model.add(Conv2D(32, kernel_size=(7, 7),
                 activation='relu',
                 input_shape=(512,512,1)))

model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.25))

model.add(Conv2D(8, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(4, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))

model.compile(optimizer='adam', loss='mse', metrics=['mse','mae'])

history = model.fit(imgsarr, ptsarr, epochs=3, batch_size=20,  verbose=1, validation_split=0.1)

model.save("reg.h5")


# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
