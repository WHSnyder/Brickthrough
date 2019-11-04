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
import tensorflow as tf
import tensorflow.contrib.slim as slim


sys.path.append("/Users/will/projects/legoproj")

import utils.geom_utils as gu 
import utils.cv_utils as cvu
import utils.feature_utils as fu 

vw = vh = 128


base="/Users/will/projects/legoproj/data/pole_single/"


os.environ['KMP_DUPLICATE_LIB_OK']='True'

split = float(sys.argv[1])

if split <= 0 or split >= 1:
    print("invalid split")
    sys.exit()



training_images = []
locs = []

num = 1479


print("Setting up...")

for i in range(0,num):

    print("Reading image {}".format(i))

    imgpath = base + "{}_pole_a.png".format(i)
    jsonpath = base + "{}_pole_a.json".format(i)

    img = cv2.imread(dset[i][0],cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128,128))
    training_images.append(img)

    mats = fu.getObjectData(jsonpath)
    screencoord = fu.projectPoint([0,-2.08,0],mats)
    locs.append(screencoord)
    

arr = np.array(training_images)
locs = np.array(locs)

print("Consumed.")



def dilated_cnn(images, num_filters, is_training):
  """Constructs a base dilated convolutional network.

  Args:
    images: [batch, h, w, 3] Input RGB images.
    num_filters: The number of filters for all layers.
    is_training: True if this function is called during training.

  Returns:
    Output of this dilated CNN.
  """

  net = images

  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      normalizer_fn=slim.batch_norm,
      activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
      normalizer_params={"is_training": is_training}):
    for i, r in enumerate([1, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1]):
      net = slim.conv2d(net, num_filters, [3, 3], rate=r, scope="dconv%d" % i)

  return net


def prob_network(images, num_filters, is_training):
    with tf.variable_scope("ProbMap"):
        net = dilated_cnn(images, num_filters, is_training)

        modules = 1
        prob = slim.conv2d(net, 1, [3, 3], rate=1, activation_fn=None)
        prob = tf.transpose(prob, [0, 3, 1, 2])

        prob = tf.reshape(prob, [-1, vh * vw])
        prob = tf.nn.softmax(prob)

        return tf.argmax(prob)

























