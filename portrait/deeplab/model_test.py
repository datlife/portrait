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
  
  # Count total number of parameters
  params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

  # for v in tf.trainable_variables():
  #   print('{:83} {:-14} {}'.format(
  #       v.name, 
  #       np.prod(v.get_shape().as_list()),
  #       v.get_shape().as_list()))

  # with tf.Session() as sess:
  #     writer = tf.summary.FileWriter("output", sess.graph)
  #     print(sess.run(mask))
  #     writer.close()

  assert params == 2949108  # ~ 2.9M Params
  assert mask.get_shape().as_list() == \
      [None, 224, 224, num_classes]
