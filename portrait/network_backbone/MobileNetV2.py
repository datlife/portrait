import tensorflow as tf
from tensorflow.nn import relu6
from tensorflow.layers import Dense, Conv2D, BatchNormalization
from tensorflow.python.keras.layers import DepthwiseConv2D
from tensorflow.python.keras.layers import Activation, GlobalAvgPool2D, Add


def MobilenetV2(inputs, alpha=1.0, num_classes=1000, include_top=True):
    
    with tf.variable_scope('first_block'):
        first_block = _make_divisible(32 * alpha, 8)
        x = Conv2D(first_block, 3, 2, padding='same', use_bias=False)(inputs)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
        x = Activation(relu6)(x)

    x = inverted_residuals(x, 16, 3, stride=1, expansion=1, block_id=0, alpha=alpha, residual=False)

    x = inverted_residuals(x, 24, 3, stride=2, expansion=6, block_id=1, alpha=alpha, residual=False)
    x = inverted_residuals(x, 24, 3, stride=1, expansion=6, block_id=2, alpha=alpha)

    x = inverted_residuals(x, 32, 3, stride=2, expansion=6, block_id=3, alpha=alpha, residual=False)
    x = inverted_residuals(x, 32, 3, stride=1, expansion=6, block_id=4, alpha=alpha)
    x = inverted_residuals(x, 32, 3, stride=1, expansion=6, block_id=5, alpha=alpha)

    x = inverted_residuals(x, 64, 3, stride=2, expansion=6, block_id=6, alpha=alpha, residual=False)
    x = inverted_residuals(x, 64, 3, stride=1, expansion=6, block_id=7, alpha=alpha)
    x = inverted_residuals(x, 64, 3, stride=1, expansion=6, block_id=8, alpha=alpha)
    x = inverted_residuals(x, 64, 3, stride=1, expansion=6, block_id=9, alpha=alpha)

    x = inverted_residuals(x, 96, 3, stride=1, expansion=6, block_id=10, alpha=alpha, residual=False)
    x = inverted_residuals(x, 96, 3, stride=1, expansion=6, block_id=11, alpha=alpha)
    x = inverted_residuals(x, 96, 3, stride=1, expansion=6, block_id=12, alpha=alpha)

    x = inverted_residuals(x, 160, 3, stride=2, expansion=6, block_id=13, alpha=alpha, residual=False)
    x = inverted_residuals(x, 160, 3, stride=1, expansion=6, block_id=14, alpha=alpha)
    x = inverted_residuals(x, 160, 3, stride=1, expansion=6, block_id=15, alpha=alpha)

    x = inverted_residuals(x, 320, 3, stride=1, expansion=6, block_id=16, alpha=alpha, residual=False)

    x = Conv2D(_make_divisible(1280 * alpha, 8), 1, use_bias=False)(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = Activation(relu6)(x)

    if include_top:
        with tf.variable_scope('logits'):
            x = GlobalAvgPool2D()(x)
            x = Dense(num_classes, activation='softmax', use_bias=True)(x)
    return x
def inverted_residuals(inputs, 
                    filters, 
                    kernel,
                    stride, 
                    expansion, 
                    alpha, 
                    atrous_rate=1,
                    residual=True,
                    block_id=None):
  """
  """
  scope = 'expanded_conv_' + str(block_id) if block_id else 'expanded_conv'
  with tf.variable_scope(scope):
    # #######################################################
    # Expand and Pointwise
    # #######################################################
    if block_id:
        with tf.variable_scope('expand'):
            in_channels = inputs.get_shape().as_list()[-1]
            x = Conv2D(filters= expansion * in_channels,
                       kernel_size=1,
                       padding='SAME',
                       use_bias=False,
                       activation=None)(inputs)
            x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
            x = Activation(relu6)(x)
    else:
        x = inputs
    # ########################################################
    # Depthwise
    # ########################################################
    with tf.variable_scope('depthwise'):
        x = DepthwiseConv2D(kernel_size=kernel,
                            strides=stride,
                            activation=None,
                            use_bias=False,
                            dilation_rate=(atrous_rate, atrous_rate),
                            padding='SAME')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
        x = Activation(relu6)(x)

    # ########################################################
    # Linear Projection
    # ########################################################
    with tf.variable_scope('project'):
        pointwise_filters = int(filters * alpha)
        pointwise_filters = _make_divisible(pointwise_filters, 8)  # Why 8???
        x = Conv2D(filters= pointwise_filters,
                   kernel_size=1,
                   padding='SAME',
                   use_bias=False,
                   activation=None)(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
        x = Activation(relu6)(x)
        if residual:
            x = Add()([inputs, x])

        return x


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v