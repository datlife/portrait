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
    feature_mao
  """
  if model_variant == 'mobilenet_v2':
    try:
      model = hub.Module(MOBILENET_V2_HUB, trainable=is_training)
    except Exception as e:
      raise ValueError("Are you connected to Internet?")

    outputs = model(
        images, 
        signature="image_feature_vector",
        as_dict=True)

    feature_map = outputs["MobilenetV2/layer_7/output"]
    low_level_features = None
    return feature_map, low_level_features

  else:
    raise NotImplementedError
