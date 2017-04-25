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

def get_conv_img(t, ksize):
    shape = t.get_shape().as_list()
    N = shape[0]
    H = shape[1]
    W = shape[2]
    C = shape[3]
    patches = tf.extract_image_patches(t,
                                       [1, ksize[0], ksize[1], 1],
                                       [1, 1, 1, 1],
                                       [1, 1, 1, 1],
                                       padding='SAME')
    patches = tf.reshape(patches, [N, H, W, ksize[0], ksize[1], C])
    patches = tf.transpose(patches, [0, 1, 3, 2, 4, 5])
    patches = tf.reshape(patches, [N, H*ksize[0], W*ksize[1], C])
    return patches

def cross_difference(a, b):
    shape = a.get_shape().as_list()
    N = shape[0]
    H = shape[1]
    W = shape[2]
    C = shape[3]
    kernel_size = 5

    a_resize = tf.image.resize_images(a, [H*kernel_size, W*kernel_size],
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    b_resize = get_conv_img(b, [kernel_size, kernel_size])
    return tf.subtract(a_resize, b_resize)

def cross_difference2(a, b):
    shape = a.get_shape().as_list()
    # shape = (2, 3, 2, 2)
    N = shape[0]
    H = shape[1]
    W = shape[2]
    C = shape[3]
    kernel_size = 3

    a_resize = tf.image.resize_images(a, [H*kernel_size, W*kernel_size],
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    b_resize = tf.image.resize_images(b, [H*kernel_size, W*kernel_size])
    return tf.subtract(a_resize, b_resize)

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

    r = cross_difference(w2, w)


    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    sess.run(r)
