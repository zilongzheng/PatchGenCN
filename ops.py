try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf

# pad = (k-1) // 2 = SAME !
# output = ( input - k + 1 + 2p ) // s


def snconv2d(x, channels, kernel=4, stride=2, use_bias=True, kernel_initializer=tf.random_normal_initializer(stddev=0.02), scope=None):
    with tf.variable_scope(scope, default_name='snconv2d'):
        w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=kernel_initializer)
        bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                            strides=[1, stride, stride, 1], padding='VALID')
        if use_bias :
            x = tf.nn.bias_add(x, bias)


        return x


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)


    return w_norm

def celu(x, alpha=1.0):
    r"""Tensorflow Implementation of CELU.
    
    .. math::
        \text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))

    """
    return tf.nn.relu(x) + tf.minimum(0, alpha * (tf.exp(x / alpha) - 1))
