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

datasize = 25000

training_images = []
for i in range(0,datasize):
    img = np.array(Image.open("./regtest/test" + str(i) +".png").convert(mode="L"))
    
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


def freeze(layer):
    layer.trainable = False


from keras.models import load_model
model = load_model('classifier.h5')

model.layers[0:5] = map(freeze, model.layers[0:5])

dic = model.layers[-1].get_config()
dic['units'] = 64
dic['activation'] = 'linear'

model.pop()
model.pop()

model.add( Dense.from_config(dic) )

dic['units'] = 1
dic['name'] = 'dense_3'
model.add( Dense.from_config(dic) )




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

with open("./regtest/labels.txt") as fp:
    
    line = fp.readline()

    while line:
        
        parts = line.split()
        labels[i] = float(parts[1])
        line = fp.readline()
        i+=1


trainlabels = labels[:split]
testlabels = labels[split:]

#trainlabels = to_categorical(trainlabels)
#testlabels = to_categorical(testlabels)


model.compile(optimizer='adam', loss='mse', metrics=['mse','mae'])

history = model.fit(trainset, trainlabels, epochs=125, batch_size=200,  verbose=1, validation_split=0.1)

#model.fit(trainset, trainlabels, verbose=1, epochs=2)

#test_loss, test_acc = model.evaluate(testset, testlabels)

print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


model.save("classifier_reg.h5")







