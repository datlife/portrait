r"""Model Definition of DeepLab V3+

Diagram: docs/deeplab_v3_architecture_diagram.png

Overview:

1. Atrous Convolution: feature map resolution controller and 
depth-of-field filter adjuster. Important benfits:
    * A generalization of CNN operation
    * Capture multi-scale features
  A formal definition of atrous conv can be described as:

      y[i] = \sum_{k}[i + r.k]w[k]

  whereas,
    * r : atrous rate (standard conv, r = 1.0)
    * x : feature map input
    * y : feature map output
    * i : location of a feature in feature map y (2D signals)
    * k : not mentioned
    * w : a convolution filter

"""
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
from portrait.deeplab.core import encoder, decoder


def deeplab_v3_plus_model(images, 
    is_training, num_classes, network_backbone, output_stride):
  """"Define Encoder-Decoder Atrous Separable Convolutional
  Neural network, a.k.a DeepLabV3+.

  Args:
    images: 4D Tensor - represent image batch size
      [batch, height, width, channels]
  """
  if output_stride not in {8, 16}:
    raise ValueError(
      "`output_stride` should be 8, 16, or 32.")  

  if network_backbone not in ['mobilenet_v2', 'xception']:
    raise ValueError(
        "`network_backbone` should be 'mobilenet_v2' or 'xception'")

  input_size = images.get_shape().as_list()[1:3]
  with tf.variable_scope('deeplabv3_plus', reuse=tf.AUTO_REUSE):
    encoded_features, low_level_features = \
      encoder.extract_features(
          images=images, 
          is_training=True, 
          network_backbone=network_backbone,
          output_stride=output_stride)

    segmentation_mask = decoder.reconstruct(
        encoded_features, 
        low_level_features,
        input_size=input_size,
        num_classes=num_classes,
        model_variant=network_backbone)

    return segmentation_mask
