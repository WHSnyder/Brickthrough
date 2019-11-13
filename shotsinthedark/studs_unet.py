import cv2
import tensorflow as tf
import sys

import numpy as np

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from PIL import Image

import matplotlib as mpl
import matplotlib.pyplot as plt


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from keras.models import Model
from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, UpSampling2D, Input, concatenate, AveragePooling2D

from keras.models import load_model


datapath = "/home/will/projects/legoproj/data/normz/"



def upsample_conv(filters, kernel_size, strides, padding):
    return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)

def upsample_simple(filters, kernel_size, strides, padding):
    return UpSampling2D(strides)

def conv2d_block(
    inputs, 
    use_batch_norm=True, 
    dropout=0.3, 
    filters=32, 
    kernel_size=(3,3), 
    activation='relu', 
    kernel_initializer='he_normal', 
    padding='same'):
    
    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding) (inputs)
    if use_batch_norm:
        c = BatchNormalization()(c)
    if dropout > 0.0:
        c = Dropout(dropout)(c)
    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding) (c)
    if use_batch_norm:
        c = BatchNormalization()(c)
    return c

def custom_unet(
    input_shape,
    num_classes=3,
    use_batch_norm=True, 
    upsample_mode='deconv', # 'deconv' or 'simple' 
    use_dropout_on_upsampling=False, 
    dropout=0.3, 
    dropout_change_per_layer=0.0,
    filters=40,
    num_layers=4,
    output_activation='relu'): # 'sigmoid' or 'softmax'
    
    p="same"

    if upsample_mode=='deconv':
        upsample=upsample_conv
    else:
        upsample=upsample_simple

    # Build U-Net model
    inputs = Input(input_shape)
    x = inputs   

    down_layers = []
    for l in range(num_layers):
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout, padding=p)
        down_layers.append(x)
        x = AveragePooling2D((2, 2),padding=p) (x)
        dropout += dropout_change_per_layer
        filters = filters*2 # double the number of filters with each layer

    x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout,padding=p)

    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0

    for conv in reversed(down_layers):        
        filters //= 2 # decreasing number of filters with each layer 
        dropout -= dropout_change_per_layer
        x = upsample(filters, (2, 2), strides=(2, 2),padding=p) (x)
        x = concatenate([x, conv])
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout, padding=p)
    
    outputs = Conv2D(3, (3,3), activation='relu', padding=p) (x)    
    
    model = Model(inputs=[inputs], outputs=[outputs])

    return model


def normal_cost(ytrue,ypred):

    #ypred = tf.image.resize(ypred,size=[256,256],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    #ytrue_normed = tf.math.l2_normalize(ytrue,axis=-1,epsilon=1e-12)
    #ypred_normed = tf.math.l2_normalize(ypred,axis=-1,epsilon=1e-12)

    #dots = tf.tensordot(ytrue_normed,ypred_normed,3,name="dotlol")
    cost = tf.reduce_sum(tf.abs(ytrue - ypred))#tf.abs(dots))

    return cost


#import keras.losses
#keras.losses.normal_loss = normal_loss


mynet = custom_unet((512,512,1))
mynet.compile(optimizer='adam', loss=normal_cost)
'''
if input("Enter g to test saved model...") == "g":

    model = load_model(datapath + 'tst.h5',compile=False)

    while input("Predict?: ") != 'q':

        num = input("Num? ")

        fig = plt.figure(figsize=(4, 4))

        img = cv2.imread(datapath + "0_{}.png".format(num),0)
        normals = cv2.imread(datapath + "0_{}_normz.png".format(num))

        pred = model.predict(np.reshape(img, (1,256,256,1)).astype('float32')/255.0)
        pred = np.reshape(pred, (256,256,3))

        fig.add_subplot(2, 2, 1)
        plt.imshow(img.astype('uint8'), interpolation='nearest', cmap='gray')

        fig.add_subplot(2, 2, 2)
        plt.imshow(normals.astype('uint8'), interpolation='nearest')

        fig.add_subplot(2, 2, 3)
        plt.imshow(pred.astype('uint8'), interpolation='nearest')

        plt.show()

    sys.exit()
'''



ximgs = []
yimgs = []
datapath = "/home/will/projects/legoproj/data/normalz/"

for i in range(0,2000):

    imgpath = datapath + "0_{}.png".format(i)
    normpath = datapath + "0_{}_normz.png".format(i)

    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #img = cv2.resize(img,(256,256),interpolation=cv2.INTER_LINEAR)

    normimg = cv2.imread(normpath)
    #normimg = cv2.resize(normimg,(256,256),interpolation=cv2.INTER_LINEAR)

    ximgs.append(img)
    yimgs.append(normimg)


for i in range(0,2000):

    imgpath = datapath + "1_{}.png".format(i)
    normpath = datapath + "1_{}_normz.png".format(i)

    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #img = cv2.resize(img,(256,256),interpolation=cv2.INTER_LINEAR)

    normimg = cv2.imread(normpath)
    #normimg = cv2.resize(normimg,(256,256),interpolation=cv2.INTER_LINEAR)

    ximgs.append(img)
    yimgs.append(normimg)

ximgs = np.array(ximgs,dtype=np.float32)/255.0
ximgs = np.reshape(ximgs,(-1,512,512,1))
yimgs = np.array(yimgs,dtype=np.float32)/255.0


history = mynet.fit(ximgs, yimgs, epochs=4, batch_size=5,  verbose=1, validation_split=0.4)

mynet.save(datapath + "tst.h5")


# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

