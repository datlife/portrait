import numpy as np
import tensorflow as tf

def create_test_inputs(batch, height, width, channels):
  """Create mock Images """
  if None in [batch, height, width, channels]:
    return tf.placeholder(tf.float32, (batch, height, width, channels))
  else:
    return tf.to_float(
        np.tile(np.reshape(
            np.reshape(np.arange(height), [height, 1]) +
            np.reshape(np.arange(width), [1, width]),
            [1 ,height, width, 1]),
          [batch, 1, 1, channels]))


# TODO: add FeatureExtractorTest