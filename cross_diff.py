import tensorflow as tf


foo = tf.Variable([[1,2], [3,4]])
sess = tf.InteractiveSession()

init_op = tf.global_variables_initializer()
sess.run(init_op)

import pdb; pdb.set_trace()  # XXX BREAKPOINT
h, w = sess.run(tf.shape(foo))
idx = tf.reshape(foo, [-1, 1])
idx = tf.tile(idx, [1, 25])
idx = tf.reshape(idx, [h, w, 5, 5])

sess.run(idx)
