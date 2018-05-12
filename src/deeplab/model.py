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
import tensorflow as tf
from deeplab.core import encoder
from deeplab.core import decoder

def model_fn(inputs, labels, mode, params):
  """"Define Encoder-Decoder Atrous Separable Convolutional
  Neural network.

  
  """

  encoded_features, low_level_features = \
    encoder.extract_features(inputs)

  outputs = decoder.reconstruct(
    encoded_features, low_level_features)

  return outputs
