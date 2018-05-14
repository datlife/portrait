import tensorflow as tf
import functools


def _mobilenetv2_conv_block(inputs):
  x = tf.layers.SeparableConv2D()(inputs)
  x = tf.layers.BatchNormalization()(x)
  x = tf.nn.relu6(x)
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



