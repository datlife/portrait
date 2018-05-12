"""Tests for encoder"""
import tensorflow as tf
import numpy as np
from encoder import deeplab_v3_encoder

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

class DeepLabV3PlusEncoderTest(tf.test.TestCase):

  def testBuildDeepLabV3PlusEncoder(self):
    """"Encoder Constructor Test"""

    images = create_test_inputs(2, 224, 224, 3)

    encoded_features = deeplab_v3_encoder(
      images=images,
      is_training=True,
      output_stride=8)

    self.assertListEqual(
      encoded_features.get_shape().as_list(), 
      [2, 28, 28, 256])

  

if __name__ == '__main__':
  tf.test.main()
