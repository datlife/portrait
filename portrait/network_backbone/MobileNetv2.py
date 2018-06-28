import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import BatchNormalization


def _inverted_res_block(inputs, filters, expansion, stride, alpha)