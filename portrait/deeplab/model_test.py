"""Tests for encoder"""
import numpy as np
import tensorflow as tf
from portrait.deeplab.model import deeplab_v3_plus_model

def test_build_deeplabv3_model():

  num_classes = 20
  inputs = tf.placeholder(
      shape=[None, 224, 224, 3], dtype=tf.float32)

  mask = deeplab_v3_plus_model(
      images=inputs,
      is_training=True,
      num_classes=num_classes,
      network_backbone='mobilenet_v2',
      output_stride=8)
  
  assert mask.get_shape().as_list() == \
      [None, 224, 224, num_classes]

# def test_forwardpass_model():
