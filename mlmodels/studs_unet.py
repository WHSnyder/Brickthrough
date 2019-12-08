import cv2
import tensorflow as tf

import sys

sys.path.append("/home/will/projects/legoproj/dataprep/")

from unet_generator import UnetGenerator

import numpy as np

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session

from PIL import Image

import matplotlib as mpl
import matplotlib.pyplot as plt

config = ConfigProto()
config.gpu_options.allow_growth = True
session = Session(config=config)

from keras.models import Model
from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, UpSampling2D, Input, concatenate, AveragePooling2D
from keras.models import load_model
from keras.optimizers import RMSprop,Adamax


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p','--predict',dest='predict',action='store_true',required=False,help='Predict?')
args = parser.parse_args()






datapath = "/home/will/projects/legoproj/data/exr_dset_0/"

def upsample_conv(filters, kernel_size, strides, padding):
    return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)

def upsample_simple(filters, kernel_size, strides, padding):
    return UpSampling2D(strides)

def conv2d_block(
    inputs, 
    use_batch_norm=True, 
    dropout=0.3, 
    filters=32, 
    kernel_size=(4,4), 
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
    filters=32,
    num_layers=2,
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
        x = MaxPooling2D((2, 2),padding=p) (x)
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
    
    outputs = Conv2D(1, (5,5), activation='sigmoid', padding=p) (x)    
    
    model = Model(inputs=[inputs], outputs=[outputs])

    return model

###################################################################################################
###################################################################################################
###################################################################################################



def iou_cost(ytrue,ypred):

    intersection = tf.math.multiply(ytrue,ypred)
    union = tf.math.subtract( tf.math.add(ytrue,ypred), intersection)

    i = tf.reduce_sum(intersection)
    u = tf.reduce_sum(union)

    return tf.math.abs(1 - (i/u))





if args.predict:

    model = load_model("/home/will/projects/legoproj/nets/tstwing.h5",compile=False)

    while input("Predict?: ") != 'q':

        num = input("Num? ")

        #fig = plt.figure(figsize=(4, 4))

        #img = cv2.imread(datapath + "{}.png".format(num),0)
        img = cv2.imread("/home/will/projects/legoproj/data/kpts_dset_{}/kpts/{}_masked.png".format(0,num),0)
        img = cv2.resize(img,(256,256),interpolation=cv2.INTER_LINEAR)
        
        #mask = cv2.imread(datapath + "studs_{}.png".format(num))

        pred = model.predict( np.reshape(img, (1,256,256,1)).astype('float32')/255.0 )
        pred = 255 * np.reshape(pred, (256,256))

        cv2.imshow("pred",pred)
        cv2.waitKey(0)

        '''fig.add_subplot(2, 2, 1)
        plt.imshow(img.astype('uint8'), interpolation='nearest', cmap='gray')

        fig.add_subplot(2, 2, 2)
        plt.imshow(normals.astype('uint8'), interpolation='nearest')

        fig.add_subplot(2, 2, 3)
        plt.imshow(pred.astype('uint8'), interpolation='nearest')

        plt.show()
        '''

    sys.exit()


mynet = custom_unet((256,256,1))
mynet.compile(optimizer="adam", loss=iou_cost)
train_gen = UnetGenerator(False)
val_gen = UnetGenerator(True)


history = mynet.fit_generator(generator=train_gen,
                    steps_per_epoch=100,
                    validation_data=val_gen,
                    validation_steps=10,
                    use_multiprocessing=True,
                    workers=6,
                    epochs=25)

mynet.save("/home/will/projects/legoproj/nets/tstwing.h5")


# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()