#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Based on TensorFlow CIFAR10 tutorial code
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

import model
import constants

import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'out/train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', constants.MAX_STEPS,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', constants.LOG_DEVICE_PLACEMENT,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', constants.LOG_FREQUENCY,
                            """How often to log results to the console.""")


def train():
    with tf.Graph().as_default() as g:
        global_step = tf.contrib.framework.get_or_create_global_step()

        images1, images2, labels = model.inputs(eval_data=False)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = model.inference(images1, images2)

        # Calculate loss.
        loss = model.loss(logits, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_no_accuracy_op = model.train(loss, global_step)

        def train_accuracy_op():
            # Calculate predictions.
            with tf.control_dependencies([train_no_accuracy_op]):
                return (True, tf.nn.in_top_k(logits, labels, 1))

        def cond_train_accuracy():
            return tf.logical_and(
                tf.greater(global_step, 0),
                tf.equal(tf.truncatemod(global_step, constants.TRAIN_ACCURACY_FREQUENCY), 0)
            )

        train_op = tf.cond(cond_train_accuracy(),
                           train_accuracy_op,
                           lambda: (False, train_no_accuracy_op))

        class _LoggerHook(tf.train.SessionRunHook):
            """
            Logs loss and runtime.
            """

            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %4d, loss = %2.2f (%3.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print (format_str % (datetime.now(), self._step, loss_value,
                                         examples_per_sec, sec_per_batch))

        summary_train_acc_writer = tf.summary.FileWriter(FLAGS.train_dir + '/train_accuracy')
        kwargs = {
            'checkpoint_dir': FLAGS.train_dir,
            'hooks': [tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                      tf.train.NanTensorHook(loss),
                      _LoggerHook()],
            'config': tf.ConfigProto(log_device_placement=FLAGS.log_device_placement),
        }
        with tf.train.MonitoredTrainingSession(**kwargs) as mon_sess:
            while not mon_sess.should_stop():
                accuracy, results = mon_sess.run(train_op)
                if accuracy:
                    true_count = np.sum(results)
                    accuracy = true_count / constants.BATCH_SIZE
                    format_str = ('%s: Training Accuracy = %.3f')
                    print (format_str % (datetime.now(), accuracy))
                    summary = tf.Summary()
                    summary.value.add(tag='train_accuracy', simple_value=accuracy)
                    summary_train_acc_writer.add_summary(summary, mon_sess.run(global_step))

def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
