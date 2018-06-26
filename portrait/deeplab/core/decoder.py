"""Decoder implementation
"""
import tensorflow as tf


def reconstruct(encoded_features, 
                low_level_features,
                decoder_height,
                decoder_width,
                input_size,
                num_classes,
                model_variant='mobilenet_v2'):
  """

  Args:
    encoded_features:
    low_level_features:

  Returns:

  """
  # Extract low-level features from (1x1) Conv
  low_level_features = tf.layers.Conv2D(
      filters=48, 
      kernel_size=(1, 1),
      strides=1)(low_level_features)

  # Upsample encoded features by 4
  upsampled_features = tf.image.resize_bilinear(
      images=encoded_features,
      size=tf.shape(low_level_features)[1:3],
      align_corners=True)

  upsampled_features.set_shape(
      [None, 56,  56, None])

  # Concat upsampled and low-level features
  x = tf.concat(
      axis=3,    # (batch, width, height, depth), use depth
      values=[upsampled_features, low_level_features]) 

  # Extract through 3x3 Conv
  x = tf.layers.Conv2D(
      filters=256,
      kernel_size=(3, 3))(x)

  x = tf.layers.Conv2D(
      filters=256,
      kernel_size=(3, 3))(x)

  x = tf.layers.Conv2D(
      filters=num_classes,
      kernel_size=(1, 1),
      activation=None)(x)

  # Upsample to match input size
  logits = tf.image.resize_bilinear(
      x, input_size)
  
  return logits
