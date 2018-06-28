import functools
import tensorflow as tf


def _mobilenetv2_conv_block(inputs, filters, expansion, stride, alpha, block_id):
  """
  https://arxiv.org/pdf/1801.04381.pdf
  https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet_v2.py
  """
  num_input_channels = inputs.get_shape().as_list()[-1]
  pointwise_conv_filters = int(filters * alpha)
  pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
  x = inputs
  prefix = 'block_{}_'.format(block_id)
      
  if block_id:
    # Expand
    x = tf.layers.Conv2D(
        expansion * num_input_channels,
        kernel_size=1,
        padding='same',
        use_bias=False,
        activation=None,
        name=prefix + 'expand')(x)
    x = tf.layers.BatchNormalization(
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'expand_BN')(x)
    x = tf.nn.relu6(x, name=prefix + 'expand_relu')
  else:
      prefix = 'expanded_conv_'

  # Depthwise
  x = tf.keras.layers.DepthwiseConv2D(
      kernel_size=3,
      strides=stride,
      activation=None,
      use_bias=False,
      padding='same',
      name=prefix + 'depthwise')(x)
  x = tf.layers.BatchNormalization(
      epsilon=1e-3,
      momentum=0.999,
      name=prefix + 'depthwise_BN')(x)
  x = tf.nn.relu6(x, name=prefix + 'depthwise_relu')

  # Project
  x = tf.layers.Conv2D(
      pointwise_filters,
      kernel_size=1,
      padding='same',
      use_bias=False,
      activation=None,
      name=prefix + 'project')(x)

  x = tf.layers.BatchNormalization(
      epsilon=1e-3, 
      momentum=0.999, 
      name=prefix + 'project_BN')(x)

  if num_input_channels == pointwise_filters and stride == 1:
      return tf.keras.layers.Add(name=prefix + 'add')([inputs, x])

  return x 

def _xecpetion_conv_block(inputs):
  raise NotImplementedError

def pad_inputs(inputs, kernel_size, data_format='channel_last'):
  """Pads along spatial dims of input size.

  Args:
    inputs : 4D Tensor - 
      [batch, height, width, channels] or
      [batch, channels, height, width] if 'channel_first'

    kernel_size: the kernel to be convoled over inputs
    data_format: 'channels_last' or 'channels_first'

  Returns:
    padded Tensor
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  
  if data_format == 'channel_first':
    padded_inputs = tf.pad(
        inputs, 
        [[0, 0], 
         [0, 0],
        [pad_beg, pad_end], 
        [pad_beg, pad_end]])
        
  elif data_format == 'channel_last':
    padded_inputs = tf.pad(
        inputs,
        [[0, 0], 
        [pad_beg, pad_end],
        [pad_beg, pad_end],
        [0, 0]])
  else:
    raise ValueError('Unsupported `data_format` value.')

  return padded_inputs


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v