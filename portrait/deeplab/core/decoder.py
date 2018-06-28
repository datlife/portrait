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
  with tf.variable_scope("Decoder"):
    low_feat_size =tf.shape(low_level_features)[1:3]
    # Extract low-level features from (1x1) Conv
    with tf.variable_scope('1x1Conv'):
      low_level_features = tf.layers.Conv2D(
          filters=48, 
          kernel_size=(1, 1),
          strides=1,
          padding='same')(low_level_features)

    # Upsample encoded features by 4, 2 if 'mobilnet'
    with tf.variable_scope('UpsampleEncodedFeatures'):
      upsampled_features = tf.image.resize_bilinear(
          images=encoded_features,
          size=low_feat_size,
          align_corners=True)

    # Concat upsampled and low-level features
    with tf.variable_scope('Concat'):
      x = tf.concat(
          axis=3,
          values=[upsampled_features, low_level_features]) 
  
    with tf.variable_scope('3x3Conv'):
      if 'mobilenet' in model_variant:
        x = ops._mobilenetv2_conv_block(
            inputs = x, 
            filters=256, 
            stride=1,
            expansion=1, 
            alpha=1.0, block_id=1)
        x = ops._mobilenetv2_conv_block(
            inputs = x, 
            filters=256, 
            stride=1,
            expansion=1, 
            alpha=1.0, block_id=2)
      else:  # Xception
        x = tf.layers.Conv2D(256, (3, 3), 1)(x)
        x = tf.layers.Conv2D(256, (3, 3), 11)(x)

    with tf.variable_scope('UpsampleOutputFeatures'):
      x = tf.image.resize_bilinear(x, input_size, align_corners=True)
      x.set_shape([None, input_size[0], input_size[1], None])

  with tf.variable_scope('Prediction'):    
    x = tf.layers.Conv2D(
        filters=num_classes,
        kernel_size=(1, 1),
        activation=None)(x)
    return x
