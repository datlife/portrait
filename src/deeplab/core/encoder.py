"""Implementation of DeepLabV3+  as encoder - A deep atrous separable 
convolution neural network. (so wordy!!!)

For object segmentation task:
  * Output stride (input / output res. ratio)  = 16 (or 8) 
  for denser feature extraction.

  * Atrous Conv Rate = 2 , or 4 to the last two blocks/
  (for output stride = 8).

  * Atrous Spatial Pyramid Pooling : https://arxiv.org/pdf/1706.05587.pdf
  (Feature_Map) --> (Global Pooling) ---> (1 x 1) Conv, 256 filters
  ---> (Batch Normalization) --> Atrous Conv Rates =(12, 24, 36)
  ---> Output stride = 8 (reduce atrous rate by factor of two if output_stride = 16)

"""
import tensorflow as tf
def deeplab_v3_encoder(inputs, output_stride=8):
  """

  """
  if output_stride not in {8, 16}:
    raise ValueError("`output_stride` should be 8, 16, or 32.")  

  features_ = tf.
  features = atrous_spatial_pyramid_pooling(
    features_map,
    atrous_rates=[12, 24, 36])

  encoder = tf.concat(features)
  encoder = tf.layers.conv2d(encoder, [1, 1])

  return lower_level_features, endcorer

def atrous_spatial_pyramid_pooling(features_map, atrous_rates=[12, 24, 36]):
  """
  """
  atrous_rates = [6, 12, 16] 
  for rate in atrous_rates:
    atrous_conv = tf.layers.SeparableConv2D()

  raise NotImplementedError

def _mobile_net_v2(inputs, include_top=False):
  """MobileNet feature extractor
  """
  model = tf.keras.applications.MobileNet(
      input_tensor=inputs, include_top=include_top)
  return model

def _aligned_xeception(inputs, include_top=False):
  """Aligned Xception feature extractor

  Args:
    inputs: a Tensor - shape [batch, width, height, channels]
  """
  model = tf.keras.applications.Xception(
    input_tensor=inputs, include_top=include_top)
  return model
  