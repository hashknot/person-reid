#
# Based on TensorFlow CIFAR10 tutorial code
# https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import constants
import os
import re
import sys

from six.moves import urllib
import tensorflow as tf

import dataset
from cross_diff import cross_difference, cross_difference2

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', constants.BATCH_SIZE,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'data',
                           """Path to the data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', constants.USE_FP16,
                            """Train the model using fp16.""")

NUM_CLASSES = constants.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = constants.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = constants.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY       = constants.MOVING_AVERAGE_DECAY
NUM_EPOCHS_PER_DECAY       = constants.NUM_EPOCHS_PER_DECAY
LEARNING_RATE_DECAY_FACTOR = constants.LEARNING_RATE_DECAY_FACTOR
INITIAL_LEARNING_RATE      = constants.INITIAL_LEARNING_RATE

TOWER_NAME = 'tower'

def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def inputs(eval_data):
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = FLAGS.data_dir
    images1, images2, labels = dataset.inputs(eval_data=eval_data,
                                              data_dir=data_dir,
                                              batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images1 = tf.cast(images1, tf.float16)
        images2 = tf.cast(images2, tf.float16)
        labels = tf.cast(labels, tf.float16)

    return images1, images2, labels


def tied_conv_max_pool(activations, kernel, biases, conv_layer_name, pool_layer_name):
    with tf.variable_scope(conv_layer_name) as scope:
        conv = tf.nn.conv2d(activations, kernel, [1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(pre_activation, name=conv_layer_name)
        _activation_summary(conv)

    # TODO: What sized filter for max-pooling?
    pool = tf.nn.max_pool(conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name=pool_layer_name)
    return pool

def conv(activations, filter_shape, filter_stride, conv_layer_name):
    with tf.variable_scope(conv_layer_name) as scope:
        kernel = _variable_with_weight_decay(conv_layer_name + '_weights',
                                             shape=filter_shape,
                                             stddev=0.1,
                                             wd=0.0)
        biases = _variable_on_cpu(conv_layer_name + 'biases', filter_shape[-1], tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(activations, kernel, filter_stride, padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(pre_activation, name=conv_layer_name)
        _activation_summary(conv)
    return conv


def inference(images1, images2):
    shape = [5, 5, 3, 20]
    l1_kernel = _variable_with_weight_decay('layer1_weights',
                                            shape=shape,
                                            stddev=0.1,
                                            wd=0.0)
    l1_biases = _variable_on_cpu('layer1_biases', [shape[-1]], tf.constant_initializer(0.0))
    l1_a_pool = tied_conv_max_pool(images1, l1_kernel, l1_biases, 'layer1_a_tied_conv', 'layer1_a_maxpool')
    l1_b_pool = tied_conv_max_pool(images2, l1_kernel, l1_biases, 'layer1_b_tied_conv', 'layer1_b_maxpool')


    shape = [5, 5, 20, 25]
    l2_kernel = _variable_with_weight_decay('layer2_weights',
                                            shape=shape,
                                            stddev=0.1,
                                            wd=0.0)
    l2_biases = _variable_on_cpu('layer2_biases', [shape[-1]], tf.constant_initializer(0.1))
    l2_a_pool = tied_conv_max_pool(l1_a_pool, l2_kernel, l2_biases, 'layer2_a_tied_conv', 'layer2_a_maxpool')
    l2_b_pool = tied_conv_max_pool(l1_b_pool, l2_kernel, l2_biases, 'layer2_b_tied_conv', 'layer2_b_maxpool')

    l3_a_cd = tf.nn.relu(cross_difference2(l2_a_pool, l2_b_pool), 'layer3_a_crossdiff')
    l3_b_cd = tf.nn.relu(cross_difference2(l2_b_pool, l2_a_pool), 'layer3_b_crossdiff')

    shape = [5, 5, 25, 25]
    stride = [1, 5, 5, 1]
    l4_a_conv = tf.nn.dropout(conv(l3_a_cd, shape, stride, 'layer4_a_conv'), 0.5, name='layer4_a_dropout')
    l4_b_conv = tf.nn.dropout(conv(l3_b_cd, shape, stride, 'layer4_b_conv'), 0.5, name='layer4_b_dropout')

    shape = [3, 3, 25, 25]
    stride = [1, 1, 1, 1]
    l5_a_conv = tf.nn.dropout(conv(l4_a_conv, shape, stride, 'layer5_a_conv'), 0.5, name='layer5_a_dropout')
    l5_b_conv = tf.nn.dropout(conv(l4_b_conv, shape, stride, 'layer5_b_conv'), 0.5, name='layer5_b_dropout')

    l5 = tf.concat([l5_a_conv, l5_b_conv], 3, name='layer5_concat')

    with tf.variable_scope('layer6_fully_connected') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(l5, [FLAGS.batch_size, -1], name='layer6_flatten')
        dim = reshape.get_shape()[1].value
        shape = [dim, 500]
        weights = _variable_with_weight_decay('layer6_weights',
                                              shape=shape,
                                              stddev=0.04,
                                              wd=0.004)
        biases = _variable_on_cpu('layer6_biases', [shape[-1]], tf.constant_initializer(0.1))
        l6 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(l6)

    with tf.variable_scope('layer7_softmax') as scope:
        shape = [500, 2]
        weights = _variable_with_weight_decay('layer7_weights',
                                              shape,
                                              stddev=1/500.0,
                                              wd=0.0)
        biases = _variable_on_cpu('layer7_biases',
                                  [shape[-1]],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(l6, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear


def loss(logits, labels):
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op
