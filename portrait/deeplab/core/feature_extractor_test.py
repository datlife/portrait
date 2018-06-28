import re
import tensorflow as tf
from portrait.deeplab.core.feature_extractor import feature_extractor


def test_construct_mobilenet_v2_140():
  """Test MobileNet V2 Constructor"""

  MOBILENET_V2_HUB = \
      "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/1"
      
  inputs = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32)

  feature_map, low_level_feat = feature_extractor(
      inputs, 
      is_training=True, 
      model_variant='mobilenet_v2')

  assert [None, 7, 7, 320] == \
      feature_map.get_shape().as_list()
      
  assert [None, 56, 56, 144] == \
      low_level_feat.get_shape().as_list()
      

# Display network 
# Simple natural sort
# convert = lambda text: int(text) if text.isdigit() else text
# alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key)]
# sorted_keys = sorted(endpoints.keys(), key=alphanum_key)
# print('{')
# for k in sorted_keys:
#     print('"{}": {},'.format(k, endpoints[k].shape))
# print('}')