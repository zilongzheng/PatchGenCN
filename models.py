try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf
from ops import snconv2d

class PatchGenCN():
    def __init__(self, img_size, opt, name):
        self.img_nc = opt.img_nc
        self.img_size = img_size
        self.name = name

    def __call__(self, img, is_training=None):
        kernel_init = tf.random_normal_initializer(stddev=0.005)
        activation = tf.nn.elu
        norm_layer = None
        if self.img_size < 32:
            return ebm_sm(img, kernel_init=kernel_init, norm_layer=norm_layer, activation=activation, name=self.name)
        elif self.img_size < 64:
            return ebm_sm(img, kernel_init=kernel_init, norm_layer=norm_layer, activation=activation, name=self.name)
        elif self.img_size < 128:
            return ebm_md(img, kernel_init=kernel_init, norm_layer=norm_layer, activation=activation, name=self.name)
        elif self.img_size < 256:
            return ebm_md(img, kernel_init=kernel_init, norm_layer=norm_layer, activation=activation, name=self.name)
        else:
            raise NotImplementedError("Current model does not support image size >= 256.")

def conv_block(inputs, num_filters, kernel_size, strides, padding, kernel_init, sn=True, norm_layer=None, activation=None):
    if padding.lower() == 'same':
        pad_size = (kernel_size - 1) // 2
        inputs = tf.pad(inputs, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], mode='REFLECT')
    else:
        pad_size = 0
    if sn:
        out = snconv2d(inputs, num_filters, kernel_size, stride=strides, kernel_initializer=kernel_init)
    else:
        out = tf.layers.conv2d(inputs, num_filters, kernel_size, strides, padding, kernel_initializer=kernel_init)
    if norm_layer is not None:
        out = norm_layer(out)
    if activation is not None:
        out = activation(out)
    return out


def ebm_sm(img, kernel_init, norm_layer, activation, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        out = conv_block(img, 64, 3, strides=1, padding='valid', kernel_init=kernel_init, norm_layer=norm_layer, activation=activation)
        out = conv_block(out, 32, 3, strides=1, padding='valid', kernel_init=kernel_init, norm_layer=norm_layer, activation=activation)

        out = conv_block(out, 32, 3, strides=1, padding='valid', kernel_init=kernel_init, norm_layer=norm_layer, activation=activation)
        out = conv_block(out, 32, 3, strides=1, padding='valid', kernel_init=kernel_init, norm_layer=norm_layer, activation=activation)

        out = conv_block(out, 1, 3, strides=1, padding='valid', kernel_init=kernel_init, sn=False)

        return out

def ebm_md(img, kernel_init, norm_layer, activation, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # receptive_field = (15 - 1) * 1 + 3 = 17
        out = conv_block(img, 128, 3, strides=1, padding='valid', kernel_init=kernel_init, norm_layer=norm_layer, activation=activation)
        out = conv_block(out, 64, 3, strides=1, padding='valid', kernel_init=kernel_init, norm_layer=norm_layer, activation=activation)

        out = conv_block(out, 64, 3, strides=1, padding='valid', kernel_init=kernel_init, norm_layer=norm_layer, activation=activation)
        out = conv_block(out, 64, 3, strides=1, padding='valid', kernel_init=kernel_init, norm_layer=norm_layer, activation=activation)
        out = conv_block(out, 1, 3, strides=1, padding='valid', kernel_init=kernel_init, sn=False)
        return out


def ebm_lg(img, kernel_init, norm_layer, activation, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        out = conv_block(img, 128, 3, strides=1, padding='valid', kernel_init=kernel_init, norm_layer=norm_layer, activation=activation)
        out = conv_block(out, 128, 3, strides=1, padding='valid', kernel_init=kernel_init, norm_layer=norm_layer, activation=activation)
        
        out = conv_block(out, 64, 3, strides=1, padding='valid', kernel_init=kernel_init, norm_layer=norm_layer, activation=activation)
        out = conv_block(out, 64, 3, strides=1, padding='valid', kernel_init=kernel_init, norm_layer=norm_layer, activation=activation)

        out = conv_block(out, 1, 3, strides=1, padding='valid', kernel_init=kernel_init)
        return out