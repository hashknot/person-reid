"""

Summary of available functions:

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)

"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys

from six.moves import urllib
import tensorflow as tf

import viper_input
from cross_diff import cross_difference, cross_difference2

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 100,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'data',
                           """Path to the data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

NUM_CLASSES = viper_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = viper_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = viper_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.001       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def _activation_summary(x):
    """
    Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
        x: Tensor
    Returns:
        nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):
    """
    Helper to create a Variable stored on CPU memory.

    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable

    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """
    Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.

    Returns:
        Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def inputs(eval_data):
    """
    Construct input for evaluation using the Reader ops.

    Args:
        eval_data: bool, indicating if one should use the train or eval data set.

    Returns:
        images1: Images. 4D tensor of [batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3] size.
        images2: Images. 4D tensor of [batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3] size.
        labels:  Labels. 1D tensor of [batch_size] size.

    Raises:
        ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = FLAGS.data_dir
    images1, images2, labels = viper_input.inputs(eval_data=eval_data,
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

    # pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
    #                       padding='SAME', name=pool_layer_name)
    return conv


def inference(images1, images2):
    """
    Build the VIPeR model.

    Args:
        images1, images2: Images returned from inputs().

    Returns:
        Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().

    l1_tied_conv_kernel = _variable_with_weight_decay('l1_tied_conv_weights',
                                                      shape=[5, 5, 3, 20],
                                                      stddev=0.1,
                                                      wd=0.0)
    l1_tied_conv_biases = _variable_on_cpu('l1_tied_conv_biases', [20], tf.constant_initializer(0.0))
    l1_pool_a = tied_conv_max_pool(images1, l1_tied_conv_kernel, l1_tied_conv_biases, 'l1_tied_conv_a', 'l1_pool_a')
    l1_pool_b = tied_conv_max_pool(images2, l1_tied_conv_kernel, l1_tied_conv_biases, 'l1_tied_conv_b', 'l1_pool_b')


    l2_tied_conv_kernel = _variable_with_weight_decay('l2_tied_conv_weights',
                                                      shape=[5, 5, 20, 25],
                                                      stddev=0.1,
                                                      wd=0.0)
    l2_tied_conv_biases = _variable_on_cpu('l2_tied_conv_biases', [25], tf.constant_initializer(0.1))
    l2_pool_a = tied_conv_max_pool(l1_pool_a, l2_tied_conv_kernel, l2_tied_conv_biases, 'l2_tied_conv_a', 'l2_tied_pool_a')
    l2_pool_b = tied_conv_max_pool(l1_pool_b, l2_tied_conv_kernel, l2_tied_conv_biases, 'l2_tied_conv_b', 'l2_tied_pool_b')

    l3_cd_a = tf.nn.relu(cross_difference2(l2_pool_a, l2_pool_b), 'l3_cd_a')
    l3_cd_b = tf.nn.relu(cross_difference2(l2_pool_b, l2_pool_a), 'l3_cd_b')

    l4_conv_a = conv(l3_cd_a, [5, 5, 25, 25], [1, 5, 5, 1], 'l4_conv_a')
    l4_conv_b = conv(l3_cd_b, [5, 5, 25, 25], [1, 5, 5, 1], 'l4_conv_b')

    l5_conv_a = conv(l4_conv_a, [3, 3, 25, 25], [1, 1, 1, 1], 'l5_conv_a')
    l5_conv_b = conv(l4_conv_b, [3, 3, 25, 25], [1, 1, 1, 1], 'l5_conv_b')

    l5 = tf.concat([l5_conv_a, l5_conv_b], 3)

    with tf.variable_scope('l6_fc') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(l5, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 500],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [500], tf.constant_initializer(0.1))
        l6 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(l6)

    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [500, 2],
                                              stddev=1/500.0, wd=0.0)
        biases = _variable_on_cpu('biases', [2],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(l6, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear


def loss(logits, labels):
    """
    Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Args:
        logits: Logits from inference().
        labels: Labels from distorted_inputs or inputs(). 1-D tensor
                of shape [batch_size]

    Returns:
        Loss tensor of type float.
    """
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
    """
    Add summaries for losses in VIPeR model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
        total_loss: Total loss from loss().
    Returns:
        loss_averages_op: op for generating moving averages of losses.
    """
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
    """
    Train VIPeR model.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
        total_loss: Total loss from loss().
        global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
        train_op: op for training.
    """
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
        opt = tf.train.GradientDescentOptimizer(lr)
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
