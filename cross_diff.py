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

def shape(t):
    s = t.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])

def cross_difference(a, b):
    # shape = a.get_shape()
    shape = (2, 3, 2, 2)
    N = shape[0]
    H = shape[1]
    W = shape[2]
    C = shape[3]
    kernel_size = 5

    b_padded = tf.pad(b, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT")

    i0 = tf.constant(0)
    m0 = tf.TensorArray(dtype=a.dtype, size=N*H*W*C*kernel_size*kernel_size)
    cond = lambda i, o, a, b_padded: tf.less(i, N*H*W*C)

    def body(i, out, a, b_padded):
        c = tf.mod(i, C)
        remainder = tf.floordiv(i, C)

        w = tf.mod(remainder, W)
        remainder = tf.floordiv(remainder, W)

        h = tf.mod(remainder, H)
        remainder = tf.floordiv(remainder, H)

        n = tf.mod(remainder, N)

        a_scalar = a[n, h, w, c]
        b_stride = tf.strided_slice(b_padded,
                                    [n, h, w, c],
                                    [n+1, h+kernel_size, w+kernel_size, c+1],
                                    [1, 1, 1, 1])
        b_stride = tf.reshape(b_stride, [kernel_size, kernel_size])
        # cross_diff = tf.map_fn(lambda x: a_scalar - x, b_stride)
        # cross_diff = tf.reshape(cross_diff, [kernel_size, kernel_size])
        out = out.write(i, b_stride)
        i = tf.add(i, 1)
        return i, out, a, b_padded

    ret = tf.while_loop(cond,
                        body,
                        [i0, m0, a, b_padded],
                        )

    out = ret[1].stack()
    cross_diff = tf.reshape(out, (N, H, W, C, kernel_size, kernel_size))
    cross_diff = tf.transpose(cross_diff, [0, 1, 4, 2, 5, 3])
    cross_diff = tf.reshape(cross_diff, (N, H*kernel_size, W*kernel_size, C))
    return cross_diff

def cross_difference2(a, b):
    shape = a.get_shape().as_list()
    # shape = (2, 3, 2, 2)
    N = shape[0]
    H = shape[1]
    W = shape[2]
    C = shape[3]
    kernel_size = 5

    a_resize = tf.image.resize_images(a, [H*kernel_size, W*kernel_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    b_resize = tf.image.resize_images(b, [H*kernel_size, W*kernel_size])
    return tf.subtract(a_resize, b_resize)

    # a_t = tf.transpose(a, [0, 3, 1, 2])
    # a_upsample = tf.reshape(a_t, [-1, 1])
    #
    # a_upsample = tf.tile(a_upsample, [1, kernel_size])
    # a_upsample = tf.reshape(a_upsample, [N*C*H, W*kernel_size])
    #
    # a_upsample = tf.tile(a_upsample, [1, kernel_size])
    # a_upsample = tf.reshape(a_upsample, [N*C*H*kernel_size, W*kernel_size])
    # a_upsample = tf.reshape(a_upsample, [N, C, H*kernel_size, W*kernel_size])
    # a_upsample = tf.transpose(a_upsample, [0, 2, 3, 1])

    return a_upsample

    idx = tf.reshape(foo, [-1, 1])
    idx = tf.tile(idx, [1, 25])
    idx = tf.reshape(idx, [h, w, 5, 5])

    cross_diff = tf.reshape(out, (N, H, W, C, kernel_size, kernel_size))
    cross_diff = tf.reshape(cross_diff, (N, H*kernel_size, W*kernel_size, C))
    return cross_diff

if __name__ == '__main__':
    w  = tf.Variable([[[[1.,2.], [2.,3.]],
                       [[3.,5.], [4.,6.]],
                       [[5.,7.], [6.,8.]]],[[[7.,9.], [8.,6.]],
                                            [[9.,6.], [1.,1.]],
                                            [[1.,1.], [1.,4.]]]
                      ])

    w2 = tf.Variable([[[[3.,5.], [2.,3.]],
                       [[3.,3.], [4.,3.]],
                       [[5.,4.], [6.,5.]]],[[[7.,2.], [8.,0.]],
                                            [[9.,5.], [1.,4.]],
                                            [[1.,6.], [1.,9.]]]
                      ])

    # w = tf.ones([128, 37, 12, 25])
    # w2 = tf.zeros([128, 37, 12, 25])

    r = cross_difference2(w2, w)


    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    sess.run(r)
