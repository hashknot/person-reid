import tensorflow as tf

def get_conv_img(t, ksize):
    shape = t.get_shape().as_list()
    N = shape[0]
    H = shape[1]
    W = shape[2]
    C = shape[3]
    patches = tf.extract_image_patches(t, [1, ksize[0], ksize[1], 1],
                                       [1, 1, 1, 1],
                                       [1, 1, 1, 1],
                                       padding='SAME')
    patches = tf.reshape(patches, [N, H, W, ksize[0], ksize[1], C])
    patches = tf.transpose(patches, [0, 1, 3, 2, 4, 5])
    patches = tf.reshape(patches, [N, H*ksize[0], W*ksize[1], C])
    return patches


if __name__ == '__main__':
    w  = tf.Variable([[[[1.,2.], [2.,3.]],
                        [[3.,5.], [4.,6.]],
                        [[5.,7.], [6.,8.]]],[[[7.,9.], [8.,6.]],
                                            [[9.,6.], [1.,1.]],
                                            [[1.,1.], [1.,4.]]]
                        ])

    init_op = tf.global_variables_initializer()
    sess = tf.Session()

    sess.run(init_op)
    foo = get_conv_img(w, [3, 3])
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    sess.run(foo)
