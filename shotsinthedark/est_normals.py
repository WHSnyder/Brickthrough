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

import traceback




class TrainingHook(tf.train.SessionRunHook):
  """A utility for displaying training information such as the loss, percent
  completed, estimated finish date and time."""

  def __init__(self, steps):
    self.steps = steps

    self.last_time = time.time()
    self.last_est = self.last_time

    self.eta_interval = int(math.ceil(0.1 * self.steps))
    self.current_interval = 0

  def before_run(self, run_context):
    graph = tf.get_default_graph()
    return tf.train.SessionRunArgs(
        {"loss": graph.get_collection("total_loss")[0]})

  def after_run(self, run_context, run_values):
    step = run_context.session.run(tf.train.get_global_step())
    now = time.time()

    if self.current_interval < self.eta_interval:
      self.duration = now - self.last_est
      self.current_interval += 1
    if step % self.eta_interval == 0:
      self.duration = now - self.last_est
      self.last_est = now

    eta_time = float(self.steps - step) / self.current_interval * \
        self.duration
    m, s = divmod(eta_time, 60)
    h, m = divmod(m, 60)
    eta = "%d:%02d:%02d" % (h, m, s)

    print("%.2f%% (%d/%d): %.3e t %.3f  @ %s (%s)" % (
        step * 100.0 / self.steps,
        step,
        self.steps,
        run_values.results["loss"],
        now - self.last_time,
        time.strftime("%a %d %H:%M:%S", time.localtime(time.time() + eta_time)),
        eta))

    self.last_time = now


def standard_model_fn(
    func, steps, run_config=None, sync_replicas=0, optimizer_fn=None):
  """Creates model_fn for tf.Estimator.

  Args:
    func: A model_fn with prototype model_fn(features, labels, mode, hparams).
    steps: Training steps.
    run_config: tf.estimatorRunConfig (usually passed in from TF_CONFIG).
    sync_replicas: The number of replicas used to compute gradient for
        synchronous training.
    optimizer_fn: The type of the optimizer. Default to Adam.

  Returns:
    model_fn for tf.estimator.Estimator.
  """

  def fn(features, labels, mode, params):
    """Returns model_fn for tf.estimator.Estimator."""

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    ret = func(features, labels, mode, params)

    tf.add_to_collection("total_loss", ret["loss"])
    train_op = None

    training_hooks = []
    if is_training:
      training_hooks.append(TrainingHook(steps))

      if optimizer_fn is None:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
      else:
        optimizer = optimizer_fn

      if run_config is not None and run_config.num_worker_replicas > 1:
        sr = sync_replicas
        if sr <= 0:
          sr = run_config.num_worker_replicas

        optimizer = tf.train.SyncReplicasOptimizer(
            optimizer,
            replicas_to_aggregate=sr,
            total_num_replicas=run_config.num_worker_replicas)

        training_hooks.append(
            optimizer.make_session_run_hook(
                run_config.is_chief, num_tokens=run_config.num_worker_replicas))

      optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5)
      train_op = slim.learning.create_train_op(ret["loss"], optimizer)

    if "eval_metric_ops" not in ret:
      ret["eval_metric_ops"] = {}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=ret["predictions"],
        loss=ret["loss"],
        train_op=train_op,
        eval_metric_ops=ret["eval_metric_ops"],
        training_hooks=training_hooks)
  return fn


def train_and_eval(
    model_dir,
    steps,
    batch_size,
    model_fn,
    input_fn,
    hparams,
    keep_checkpoint_every_n_hours=0.5,
    save_checkpoints_secs=180,
    save_summary_steps=50,
    eval_steps=20,
    eval_start_delay_secs=10,
    eval_throttle_secs=300,
    sync_replicas=0):
  """Trains and evaluates our model. Supports local and distributed training.

  Args:
    model_dir: The output directory for trained parameters, checkpoints, etc.
    steps: Training steps.
    batch_size: Batch size.
    model_fn: A func with prototype model_fn(features, labels, mode, hparams).
    input_fn: A input function for the tf.estimator.Estimator.
    hparams: tf.HParams containing a set of hyperparameters.
    keep_checkpoint_every_n_hours: Number of hours between each checkpoint
        to be saved.
    save_checkpoints_secs: Save checkpoints every this many seconds.
    save_summary_steps: Save summaries every this many steps.
    eval_steps: Number of steps to evaluate model.
    eval_start_delay_secs: Start evaluating after waiting for this many seconds.
    eval_throttle_secs: Do not re-evaluate unless the last evaluation was
        started at least this many seconds ago
    sync_replicas: Number of synchronous replicas for distributed training.

  Returns:
    None
  """

  run_config = tf.estimator.RunConfig(
      keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
      save_checkpoints_secs=save_checkpoints_secs,
      save_summary_steps=save_summary_steps)

  estimator = tf.estimator.Estimator(
      model_dir=model_dir,
      model_fn=standard_model_fn(
          model_fn,
          steps,
          run_config,
          sync_replicas=sync_replicas),
      params=hparams, config=run_config)

  train_spec = tf.estimator.TrainSpec(
      input_fn=input_fn(split="train", batch_size=batch_size),
      max_steps=steps)

  eval_spec = tf.estimator.EvalSpec(
      input_fn=input_fn(split="validation", batch_size=batch_size),
      steps=eval_steps,
      start_delay_secs=eval_start_delay_secs,
      throttle_secs=eval_throttle_secs)

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


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

























