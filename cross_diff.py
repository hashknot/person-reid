# import tensorflow as tf
#
#
# foo = tf.Variable([[1,2], [3,4]])
# sess = tf.InteractiveSession()
#
# init_op = tf.global_variables_initializer()
# sess.run(init_op)
#
# # import pdb; pdb.set_trace()  # XXX BREAKPOINT
# # h, w = sess.run(tf.shape(foo))
# # idx = tf.reshape(foo, [-1, 1])
# # idx = tf.tile(idx, [1, 25])
# # idx = tf.reshape(idx, [h, w, 5, 5])
# #
# # sess.run(idx)

import tensorflow as tf

sess = tf.InteractiveSession()
init_op = tf.global_variables_initializer()
sess.run(init_op)


def cross_difference(shape, a, b):
    (N, H, W, C) = shape
    kernel_size = 5

    b_padded = tf.pad(b, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT")

    i0 = tf.constant(0)
    m0 = tf.ones([5, 5], dtype=a.dtype)
    cond = lambda i, o: tf.less(i, N*H*W*C)
    def body(i, o):
        n = tf.floordiv(i, N)
        h = tf.floordiv(i, H)
        w = tf.floordiv(i, W)
        c = tf.floordiv(i, C)
        a_scalar = a[n, h, w, c]
        b_stride = tf.strided_slice(b_padded,
                                    [n, h, w, c],
                                    [n+1, h+kernel_size, w+kernel_size, c+1],
                                    [1, 1, 1, 1])
        cross_diff = tf.reshape(tf.map_fn(lambda x: a_scalar - x, b_stride), [5, 5])
        r = tf.concat([o, cross_diff], 0)
        i = tf.add(i, 1)
        return i, r

    i, r = tf.while_loop(cond,
                         body,
                         [i0, m0],
                         [i0.get_shape(), tf.TensorShape([None, 5])])

    cross_diff = r[5:,:,]
    cross_diff = tf.reshape(cross_diff, (N, H, W, C, kernel_size, kernel_size))
    cross_diff = tf.transpose(cross_diff, [0, 1, 4, 2, 5, 3])
    cross_diff = tf.reshape(cross_diff, (N, H*kernel_size, W*kernel_size, C))
    return cross_diff

w = tf.Variable([[[[1,2], [2,3]],
            [[3,5], [4,6]],
             [[5,7], [6,8]]],[[[7,9], [8,6]],
            [[9,6], [10,1]],
             [[1,15], [12,45]]]])

w2 = tf.Variable([[[[3,5], [2,3]],
            [[3,3], [4,3]],
             [[5,4], [6,5]]],[[[7,2], [8,0]],
            [[9,5], [10,4]],
             [[11,6], [12,9]]]])

import pdb; pdb.set_trace()  # XXX BREAKPOINT
shape = (2, 3, 2, 2)
r = cross_difference(shape, w, w2)
