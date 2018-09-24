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



os.environ['KMP_DUPLICATE_LIB_OK']='True'

split = int(sys.argv[1])

datasize = 10000


training_images = []
for i in range(0,datasize):
    img = np.array(Image.open("./pose_basic_train/test" + str(i) +".png").convert(mode="L"))
    
    training_images.append(img)
    
print(training_images[100].shape)
arr = np.array(training_images)




import tensorflow as tf

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical



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



model = Sequential()
model.add(Conv2D(10, kernel_size=(6, 6),
                 activation='relu',
                 input_shape=(128,128,1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.25))



model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))


model.compile(optimizer='adam', loss='mse', metrics=['mse','mae'])
history = model.fit(trainset, trainlabels, epochs=3, batch_size=1000,  verbose=1, validation_split=0.1)


model.save("reg.h5")

print(history.history.keys())

# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
