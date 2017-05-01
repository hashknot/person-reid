#!/usr/bin/env python
#
# Based on TensorFlow CIFAR10 tutorial code
#
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import model

import constants

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'out/test',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'out/train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', constants.EVAL_INTERVAL_SECS,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', constants.EVAL_NUM_EXAMPLES,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', constants.EVAL_RUN_ONCE,
                            """Whether to run eval only once.""")


def eval_once(saver, summary_writer, top_k_op, summary_op):
    """
    Run Eval once.

    Args:
        saver: Saver.
        summary_writer: Summary writer.
        top_k_op: Top K op.
        summary_op: Summary op.
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1

            accuracy = true_count / total_sample_count
            print('%s: Test Accuracy = %.3f' % (datetime.now(), accuracy))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='test_accuracy', simple_value=accuracy)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def test():
    with tf.Graph().as_default() as g:
        eval_data = FLAGS.eval_data == 'test'
        images1, images2, labels = model.inputs(eval_data=eval_data)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = model.inference(images1, images2)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            model.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        while True:
            eval_once(saver, summary_writer, top_k_op, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    test()


if __name__ == '__main__':
    tf.app.run()
