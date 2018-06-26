import tensorflow_hub as hub

MOBILENET_V2_HUB = \
    "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/1"
XCEPTION_HUB = None


def feature_extractor(images, is_training, model_variant='mobilenet_v2'):
  """Extract sematic features from images using pre-trained
  network backbones: 'MobileNet V2' or 'XCeption'.
  
  Construct feature extractor using Tensorflow Hub
  Ref: https://www.tensorflow.org/hub/

  Args:
    images:
    is_training:
    model_variant:

  Returns:
    feature_map
  """
  if model_variant == 'mobilenet_v2':
    try:
      model = hub.Module(MOBILENET_V2_HUB, trainable=is_training)
    except Exception as e:
      print(e)
      raise ValueError("Are you connected to Internet?")

    outputs = model(
        images, 
        signature="image_feature_vector",
        as_dict=True)

    feature_map = outputs["MobilenetV2/layer_7/output"]

    # https://github.com/tensorflow/models/blob/master/research/deeplab/core/feature_extractor.py#L101
    low_level_features = outputs["MobilenetV2/layer_4/depthwise_output"]
    return feature_map, low_level_features

  elif model_variant == 'xception':
    raise NotImplementedError
  
  else:
    raise ValueError(
        '`model_variant` only supports MobileNet-V2 and Xception')
