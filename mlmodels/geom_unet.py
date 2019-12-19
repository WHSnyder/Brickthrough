import cv2
import tensorflow as tf

import sys

sys.path.append("/home/will/projects/legoproj/dataprep/")

from geom_generator import GeomGenerator

import numpy as np

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session

from PIL import Image

import matplotlib as mpl
import matplotlib.pyplot as plt

config = ConfigProto()
config.gpu_options.allow_growth = True
session = Session(config=config)

import keras
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
    kernel_size=(3,3), 
    activation='relu', 
    kernel_initializer='he_normal', 
    padding='same',
    dila=1):
    
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
    use_dropout_on_upsampling=True, 
    dropout=0.3, 
    dropout_change_per_layer=0.03,
    filters=64,
    num_layers=5,
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
        #filters*2 # double the number of filters with each layer

    x = conv2d_block(inputs=x, dila=1, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout,padding=p)

    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0

    for conv in reversed(down_layers):        
        #//= 2  decreasing number of filters with each layer 
        dropout -= dropout_change_per_layer
        x = upsample(filters, (2, 2), strides=(2, 2),padding=p) (x)
        x = concatenate([x, conv])
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout, padding=p,activation="relu")

    outputs = Conv2D(3, (3,3), activation="hard_sigmoid", dilation_rate=2, padding=p) (x) 
    
    model = Model(inputs=[inputs], outputs=[outputs])

    return model



###################################################################################################



def geom_loss(y_true, y_pred):

    posmask = tf.cast(y_true > .0001,tf.float32)

    #diffs = tf.math.square((1+y_pred)/2 - y_true)
    diffs = tf.math.square(y_pred - y_true)

    diffsmasked = posmask * diffs

    return tf.reduce_sum(diffsmasked)



def l2_loss_weighted(y_true,y_pred):

    posmask = y_true#tf.cast(y_true > .05,tf.float32)
    negmask = 1 - y_true #tf.cast(y_true < .05,tf.float32)

    diffs = tf.math.square((1+y_pred)/2 - y_true)

    pos = posmask * diffs
    negs = negmask * diffs

    pos = tf.reshape(pos,[10,-1])
    negs = tf.reshape(negs,[10,-1])

    posmask = tf.reshape(posmask,[10,-1])
    negmask = tf.reshape(negmask,[10,-1])

    posweights = tf.reduce_sum(posmask,axis=1) + 1
    negweights = tf.reduce_sum(negmask,axis=1) + 1

    pos = tf.reduce_sum(pos,axis=1)
    negs = tf.reduce_sum(negs,axis=1)
    
    #pos = pos / posweights
    #negs = negs / negweights

    pos = tf.reduce_sum(pos)
    negs = tf.reduce_sum(negs)

    return 3*pos + negs / 25




if args.predict:

    model = load_model("/home/will/projects/legoproj/nets/tstgeom_pole.h5",compile=False)

    while input("Predict?: ") != 'q':

        num = input("Num? ")

        #fig = plt.figure(figsize=(4, 4))

        #img = cv2.imread("/home/will/Downloads/ontable.jpeg",0)
        tag = "{:0>4}".format(num)
        img = cv2.imread("/home/will/projects/legoproj/data/kpts_dset_{}/{}_a.png".format(3,tag),0)
        img = cv2.resize(img,(256,256),interpolation=cv2.INTER_LINEAR)
        
        #mask = cv2.imread(datapath + "studs_{}.png".format(num))

        pred = model.predict( np.reshape(img, (1,256,256,1)).astype('float32')/255.0 )
        #pred = (255.0 * np.reshape(pred, (256,256))).astype(np.uint8)
        #pred = (255.0 * np.reshape((1.0+pred)/2.0, (256,256,3))).astype(np.uint8)
        pred = (255.0 * np.reshape(pred, (256,256,3))).astype(np.uint8)

        outimg = cv2.resize(pred,(512,512),interpolation=cv2.INTER_LINEAR)

        cv2.imshow("pred",outimg)
        cv2.waitKey(0)



    sys.exit()

from keras import losses

mynet = custom_unet((256,256,1))
mynet.compile(optimizer=RMSprop(lr=4e-4), loss=geom_loss)
train_gen = GeomGenerator(False)
val_gen = GeomGenerator(True)

print(mynet.summary())

#sys.exit()


history = mynet.fit_generator(generator=train_gen,
                    steps_per_epoch=100,
                    validation_data=val_gen,
                    validation_steps=20,
                    use_multiprocessing=False,
                    workers=6,
                    epochs=15)

mynet.save("/home/will/projects/legoproj/nets/tstgeom_pole.h5")

print(history.history['loss'])
# "Loss"
plt.plot(history.history['loss'][1:])
plt.plot(history.history['val_loss'][1:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()