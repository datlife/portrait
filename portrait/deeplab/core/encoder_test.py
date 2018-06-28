"""Tests for encoder"""
import pytest
import tensorflow as tf
from portrait.deeplab.core.encoder import extract_features


def test_deeplabv3plus_mobilnetv2_encoder():
  """"Encoder Constructor Test"""
  inputs = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32)
  encoded_features, low_level_features = extract_features(
      images=inputs,
      is_training=True,
      network_backbone='mobilenet_v2', 
      output_stride=8)

  assert [None, 7, 7, 256] == \
      encoded_features.get_shape().as_list()
      
  assert [None, 56, 56, 192] == \
      low_level_features.get_shape().as_list()


def test_invalid_network_backbone_encoder():
  """Test Invalid Iput `network_backbone`"""
  inputs = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32)
  with pytest.raises(ValueError, message="Expecting ValueError"): 
    _, _ = extract_features(
        images=inputs,
        is_training=True,
        network_backbone='ICanDoIt', 
        output_stride=8)

def test_invalid_network_output_stride():
  """Test Invalid Iput `output_stride`"""
  inputs = tf.placeholder(
      shape=[None, 224, 224, 3], dtype=tf.float32)
  with pytest.raises(ValueError, message="Expecting ValueError"): 
    _, _ = extract_features(
        images=inputs,
        is_training=True,
        network_backbone='mobilenet_v2', 
        output_stride=12)
