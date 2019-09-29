from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import matplotlib.pyplot as plt
import numpy as np

import os
from scipy import misc
import sys

import tensorflow as tf
import tensorflow.contrib.slim as slim

import feature_utils as fu








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



def define_graph():

	net = dilated_cnn()
	prob = slim.conv2d(net, 2, [3, 3], rate=1, activation_fn=None)

	



def main():






