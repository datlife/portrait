import tensorflow_hub as hub

MOBILENET_V2_HUB = \
    "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/1"
XCEPTION_HUB = None

def feature_extractor(images, is_training, model='mobilenet_v2'):
  """Extract sematic features from images using pre-trained
  network backbones: 'MobileNet V2' or 'XCeption'.
  
  Construct feature extractor using Tensorflow Hub
  Ref: https://www.tensorflow.org/hub/

  Args:
    images:
    is_training:
    model:

  Returns:
    feature_mao
  """
  if model == 'mobilenet_v2':
    feature_extractor = hub.Module(MOBILENET_V2_HUB, trainable=is_training)
    outputs = feature_extractor(
        images, 
        signature="image_feature_vector",
        as_dict=True)
    feature_map = outputs["MobilenetV2/layer_7/output"]
    low_level_features = None
    return feature_map, low_level_features

  else:
    raise NotImplementedError