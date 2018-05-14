import tensorflow as tf
from portrait.deeplab import model


def model_fn(inputs, labels, mode, params):
  raise NotImplementedError


if __name__ == '__main__':
  inputs = tf.placeholder(shape=[None, None, None, 3], dtype=tf.float32)
  deeplabv3 = model.deeplab_v3_plus_model(inputs)
