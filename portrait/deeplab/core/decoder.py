"""Decoder implementation
"""
import tensorflow as tf
from portrait.deeplab.core import ops


def reconstruct(encoded_features, 
                low_level_features,
                input_size,
                num_classes,
                model_variant='mobilenet_v2'):
  """
  """
  with tf.variable_scope("decoder"):

    # Extract low-level features from (1x1) Conv
    low_level_features = tf.layers.Conv2D(
        filters=48, 
        kernel_size=(1, 1),
        strides=1,
        padding='same')(low_level_features)

    # Upsample encoded features by 4, 2 if 'mobilnet'
    upsampled_features = tf.image.resize_bilinear(
        images=encoded_features,
        size=tf.shape(low_level_features)[1:3],
        align_corners=True)

    # Concat upsampled and low-level features
    x = tf.concat(
        axis=3,    # (batch, width, height, depth), use depth
        values=[upsampled_features, low_level_features]) 

    # Extract through 3x3 Conv
    if 'mobilenet' in model_variant:
      x = ops._mobilenetv2_conv_block(
         inputs = x, filters=256, stride=1,
         expansion=1, alpha=1.0, block_id=1)
      x = ops._mobilenetv2_conv_block(
         inputs = x, filters=256, stride=1,
         expansion=1, alpha=1.0, block_id=2)

    else:  # Xception
      x = tf.layers.Conv2D(256, (3, 3), 1)(x)
      x = tf.layers.Conv2D(256, (3, 3), 11)(x)

    x = tf.layers.Conv2D(
        filters=num_classes,
        kernel_size=(1, 1),
        activation=None)(x)

    # Upsample to match input size
    x = tf.image.resize_bilinear(x, input_size, align_corners=True)

    x.set_shape([None, input_size[0], input_size[1], num_classes])
    return x
