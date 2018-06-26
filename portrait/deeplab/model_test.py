"""Tests for encoder"""
import numpy as np
import tensorflow as tf
from portrait.deeplab.model import deeplab_v3_plus_model


def create_test_inputs(batch, height, width, channels):
  """Create mock Images """
  if None in [batch, height, width, channels]:
    return tf.placeholder(tf.float32, (batch, height, width, channels))
  else:
    return tf.to_float(
        np.tile(np.reshape(
            np.reshape(np.arange(height), [height, 1]) +
            np.reshape(np.arange(width), [1, width]),
            [1, height, width, 1]),
          [batch, 1, 1, channels]))


class DeepLabV3PlusTest(tf.test.TestCase):
  def testBuildDeepLabV3Plus(self):
    """"Encoder Constructor Test"""
    images = create_test_inputs(2, 224, 224, 3)
    
    segmentation_mask = deeplab_v3_plus_model(
        images=images)
        
    self.assertListEqual(
        segmentation_mask.get_shape().as_list(), 
        [2, 224, 224, 20])

  
if __name__ == '__main__':
  tf.test.main()
