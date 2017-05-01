#
# Based on TensorFlow CIFAR10 tutorial code
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

import constants

def read(filename_queue):
    class Record(object):
        pass

    result = Record()

    label_bytes   = constants.LABEL_BYTES
    result.height = constants.IMAGE_HEIGHT
    result.width  = constants.IMAGE_WIDTH
    result.depth  = constants.IMAGE_CHANNELS
    image_bytes = result.height * result.width * result.depth

    # Every record consists of two images followed by label, with a
    # fixed number of bytes for each.
    record_bytes = 2*image_bytes + label_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    base = 0
    offset = image_bytes
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [base], [base + offset]),
                             [result.depth, result.height, result.width])
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image1 = tf.transpose(depth_major, [1, 2, 0])

    base += offset
    offset = image_bytes
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [base], [base + offset]),
                             [result.depth, result.height, result.width])
    result.uint8image2 = tf.transpose(depth_major, [1, 2, 0])

    base += offset
    offset = label_bytes
    result.label = tf.cast(tf.strided_slice(record_bytes, [base], [base+offset]), tf.int32)

    return result

def _generate_batch(image1, image2, label, min_queue_examples,
                    batch_size, shuffle):
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images1, images2, label_batch = tf.train.shuffle_batch(
            [image1, image2, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images1, images2, label_batch = tf.train.batch(
            [image1, image2, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.summary.image('images1', images1)
    tf.summary.image('images2', images2)

    return images1, images2, tf.reshape(label_batch, [batch_size])

def inputs(eval_data, data_dir, batch_size):
    if not eval_data:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                     for i in xrange(1, constants.DATA_BATCH_COUNT+1)]
        num_examples_per_epoch = constants.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, 'data_test.bin')]
        num_examples_per_epoch = constants.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = read(filename_queue)
    image1 = tf.cast(read_input.uint8image1, tf.float32)
    image2 = tf.cast(read_input.uint8image2, tf.float32)

    height   = constants.IMAGE_HEIGHT
    width    = constants.IMAGE_WIDTH
    channels = constants.IMAGE_CHANNELS

    # Subtract off the mean and divide by the variance of the pixels.
    float_image1 = tf.image.per_image_standardization(image1)
    float_image2 = tf.image.per_image_standardization(image2)

    # Set the shapes of tensors.
    float_image1.set_shape([height, width, channels])
    float_image2.set_shape([height, width, channels])
    read_input.label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    return _generate_batch(float_image1, float_image2, read_input.label,
                           min_queue_examples, batch_size,
                           shuffle=False)
