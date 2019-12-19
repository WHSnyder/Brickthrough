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
import keras.backend as K


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
    filters=56,
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

    #outputs = Conv2D(3, (3,3), activation="tanh", padding=p) (x)
    geom_output = Conv2D(3, (3,3), activation="hard_sigmoid", padding=p) (x)
    #error_output = Conv2D(1, (3,3), activation="sigmoid", padding=p)(x)

    #outputs = concatenate([geom_output,error_output])

    #print("Output shape:\n\n")
    #print(outputs)
    #print(K.int_shape(outputs))
    
    model = Model(inputs=[inputs], outputs=[geom_output])

    return model


###################################################################################################


def geom_loss(y_true, y_pred):

    posmask = tf.cast(y_true > .0001,tf.float32)

    #diffs = tf.math.square((1+y_pred)/2 - y_true)
    diffs = tf.math.square(y_pred - y_true)

    diffsmasked = posmask * diffs

    return tf.reduce_sum(diffsmasked)



def geom_loss_bayes(y_true, y_pred):

    y_true = tf.slice(y_true,[0,0,0,0],[-1,-1,-1,3])

    geom_out = tf.slice(y_pred,[0,0,0,0],[-1,-1,-1,3])
    error_out = tf.slice(y_pred,[0,0,0,3],[-1,-1,-1,1])
   
    es = tf.shape(error_out)
    error_out = tf.reshape(error_out,[-1,es[1],es[2]])

    posmask = tf.cast(y_true > .0001,tf.float32)
    posmask = tf.reduce_max(posmask,axis=-1)
    masks_sum = tf.reduce_sum(posmask)

    geom_diffs_raw = tf.math.abs((1+geom_out)/2 - y_true, name="diffs_raw")
    geom_diffs = tf.reduce_sum(geom_diffs_raw,axis=-1)
    geom_diffs_mean = tf.reduce_sum(posmask * geom_diffs) / masks_sum

    geom_diffs_mean = tf.reduce_mean(geom_diffs_raw,axis=-1,name="diffs_mean")
    error_pred_loss = tf.math.abs( geom_diffs_mean - error_out,name="maybe" )
    error_pred_loss = tf.reduce_sum(posmask * error_pred_loss) / masks_sum

    return 10 * geom_diffs_mean + error_pred_loss



def geom_loss_bayes_simpler(y_true, y_pred):

    posmask = tf.cast(y_true > .0001,tf.float32)
    posmask = tf.reduce_max(posmask,axis=-1)

    #diffs = tf.math.abs((1+y_pred)/2 - y_true)
    diffs = tf.math.square(y_pred - y_true)

    diffs = tf.reduce_sum(diffs,axis=-1)
    diffsmasked = posmask * diffs

    masksum = tf.reduce_sum(posmask)

    return tf.reduce_sum(diffsmasked)






if args.predict:

    model = load_model("/home/will/projects/legoproj/nets/tstgeom_pole_bayes.h5",compile=False)

    while input("Predict?: ") != 'q':

        num = input("Num? ")

        #fig = plt.figure(figsize=(4, 4))

        #img = cv2.imread("/home/will/Downloads/ontable.jpeg",0)
        tag = "{:0>4}".format(num)
        img = cv2.imread("/home/will/projects/legoproj/data/kpts_dset_{}/{}_a.png".format(3,tag),0)
        img = cv2.resize(img,(256,256),interpolation=cv2.INTER_LINEAR)

        geomraw = cv2.imread("/home/will/projects/legoproj/data/kpts_dset_{}/geom/{}_geom.png".format(3,tag))
        geom = cv2.cvtColor(geomraw,cv2.COLOR_BGR2GRAY)
        geom = cv2.inRange(geom,2,255)
        geom = cv2.resize(geom,(512,512),interpolation=cv2.INTER_LINEAR)

        geomraw = cv2.resize(geomraw,(512,512),interpolation=cv2.INTER_LINEAR)


        
        #mask = cv2.imread(datapath + "studs_{}.png".format(num))

        pred = model.predict( np.reshape(img, (1,256,256,1)).astype('float32')/255.0 )
        #pred = (255.0 * np.reshape(pred, (256,256,3))).astype(np.uint8)
        pred = (255.0 * np.reshape((1.0+pred)/2.0, (256,256,3))).astype(np.uint8)

        #geom_out = pred[0,:,:,0:3]
        #error_out = np.reshape( pred[0,:,:,3:], (256,256) )

        #geom_pred = (255.0 * geom_out).astype(np.uint8)
        #error_pred = (255.0 * error_out).astype(np.uint8)

        
        #pred = (255.0 * np.reshape(pred, (256,256,3))).astype(np.uint8)

        outimg = cv2.resize(pred,(512,512),interpolation=cv2.INTER_LINEAR)
        outimg = cv2.bitwise_and(outimg,outimg,mask=geom)


        #diffs = (255 * diffs/np.amax(diffs)).astype(np.uint8)


        cv2.imshow("out",outimg)
        cv2.waitKey(0)

        geomraw = geomraw.astype(np.float32)
        outimg = outimg.astype(np.float32)
        diffs = np.absolute( geomraw - outimg ).astype(np.uint8)

        cv2.imshow("diffs",diffs)
        cv2.waitKey(0)

        #cv2.imshow("pred",cv2.resize(pred,(512,512),interpolation=cv2.INTER_LINEAR))
        #cv2.waitKey(0)

        #cv2.imshow("error",cv2.resize(error_pred,(512,512),interpolation=cv2.INTER_LINEAR))
        #cv2.waitKey(0)

    sys.exit()



#from keras import losses

mynet = custom_unet((256,256,1))
#mynet.compile(optimizer=RMSprop(lr=4e-4), loss=geom_loss)
mynet.compile(optimizer=RMSprop(lr=4e-4), loss=geom_loss_bayes_simpler)

train_gen = GeomGenerator(False)
val_gen = GeomGenerator(True)

#print(mynet.summary())

#sys.exit()


history = mynet.fit_generator(generator=train_gen,
                    steps_per_epoch=100,
                    validation_data=val_gen,
                    validation_steps=20,
                    use_multiprocessing=False,
                    workers=6,
                    epochs=15)

mynet.save("/home/will/projects/legoproj/nets/tstgeom_pole_bayes_sig.h5")

print(history.history['loss'])
# "Loss"
plt.plot(history.history['loss'][1:])
plt.plot(history.history['val_loss'][1:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()