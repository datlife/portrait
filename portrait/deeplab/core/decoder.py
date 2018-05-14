"""Decoder implementation
"""
import tensorflow as tf


def reconstruct(encoded_features, low_level_features,
                model_variant='mobilenet_v2'):
  """

  Args:
    encoded_features:
    low_level_features:

  Returns:

  """
  raise NotImplementedError

def upsample(features, rate=4):
  raise NotImplementedError

