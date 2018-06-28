import tensorflow as tf
import tensorflow_hub as hub

MOBILENET_V2_HUB = \
    "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/1"
XCEPTION_HUB = None


def feature_extractor(images, is_training, model_variant='mobilenet_v2'):
  """Extract sematic features from images using pre-trained
  network backbones: 'MobileNet V2' or 'Xception' model.

  Args:
    images:  - tf.float32 Tensor
    is_training: - boolean
    model_variant: - string -

  Returns:
    feature_map - tf.float32 Tensor
    low_level_features - tf.float32 Tensor

  Raise:
    ValueError: `model_variant` is an invalid string
  """
  if model_variant == 'mobilenet_v2':
    mobilnet = hub.Module(MOBILENET_V2_HUB, is_training)
    endpoints = mobilnet(
        images, 
        signature="image_feature_vector",
        as_dict=True)
    feature_map = endpoints["MobilenetV2/layer_18"]
    low_level_features = endpoints["MobilenetV2/layer_4/depthwise_output"]
    return feature_map, low_level_features

  elif model_variant == 'xception':
    raise NotImplementedError
  
  else:
    raise ValueError(
        '`model_variant` only supports MobileNet-V2 and Xception')
