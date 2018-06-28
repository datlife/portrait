"""Implementation of DeepLabV3+ - A deep atrous separable 
convolution neural network. (so wordy!!!)

For object segmentation task:
  * Output stride (input / output res. ratio)  = 16 (or 8) 
  for denser feature extraction.

  * Atrous Conv Rate = 2 , or 4 to the last two blocks/
  (for output stride = 8).

  * Atrous Spatial Pyramid Pooling : https://arxiv.org/pdf/1706.05587.pdf

  (Feature_Map) --> (Global Pooling) ---> (1 x 1) Conv, 256 filters
  ---> (Batch Normalization) --> Atrous Conv Rates =(12, 24, 36)
  ---> Output stride = 8 (reduce atrous rate by factor of two if output_stride=16)

# TODO:
   * Add suggested hyperparameters for training
   * Removed fixed image size
"""
import tensorflow as tf
from portrait.deeplab.core import ops
from portrait.deeplab.core.feature_extractor import feature_extractor


def extract_features(images,
                     is_training=True,
                     network_backbone='mobilenet_v2', 
                     activation_fn=tf.nn.relu6,
                     normalizer_fn=tf.layers.BatchNormalization(),
                     output_stride=8):
  """Extract sematic  features, given a network
  backbone (e.g. MobileNetv2, Xception). Then, perform
  "atrous spatial pyramid pooling" to get a concatenated 
  encoded features.

  Args:
    images:
    is_training:
    network_backbone:
    output_stride:
  
  Returns:
    encoded_features - 4D Tensor
    low_level_features - 4D Tensor
    
  Raise:
    ValueError: `output_stride` or `network_backbone` input
      is invalid.
  """
  with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
    # Extract features
    feature_map, low_level_features = feature_extractor(
        images=images,
        model_variant=network_backbone,
        is_training=is_training)
        
    # Perform ASSP op
    assp_features = atrous_spatial_pyramid_pooling(
        network_backbone,
        feature_map,
        depth=256,
        activation_fn=activation_fn,
        normalizer_fn=normalizer_fn,
        output_stride=8)

    # Merge all
    encoded_features = tf.layers.Conv2D(256, (1, 1))(assp_features)
    encoded_features = normalizer_fn(encoded_features)
    encoded_features = activation_fn(encoded_features)

    return encoded_features, low_level_features


def atrous_spatial_pyramid_pooling(network_backbone,
                                   feature_map,
                                   depth=256,
                                   activation_fn=tf.nn.relu6,
                                   normalizer_fn=tf.layers.BatchNormalization,                                  
                                   output_stride=8):
  """
  """
  atrous_rates = [12, 24, 36] if output_stride == 8 else [6, 12, 18]
  
  with tf.variable_scope('aspp'):
    logit_branches = []

    # 1x1 Conv
    with tf.variable_scope('1x1_conv_pooling'):
      conv_1x1 = tf.layers.Conv2D(
          filters=depth, 
          kernel_size=(1, 1))(feature_map)

      conv_1x1 = tf.layers.BatchNormalization(
          epsilon=1e-3,
          momentum=0.999)(conv_1x1)
      conv_1x1 = activation_fn(conv_1x1)
      logit_branches.append(conv_1x1)

    # 3x3 Atrous Separable Convs,
    if network_backbone != 'mobilenet_v2':
      for idx, rate in enumerate(atrous_rates):
        scope = 'aspp_%s' % idx
        assp_features = _atrous_separable_conv(
            features=feature_map,
            output_depth=depth,
            kernel_size=3,
            atrous_rate=rate,
            activation_fn=activation_fn,
            normalizer_fn=normalizer_fn,
            scope=scope)
        logit_branches.append(assp_features)


    # Image feature level
    with tf.variable_scope('image_level_pooling'):
      # global average pooling
      image_feature = tf.reduce_mean(
          feature_map, [1, 2], 
          name='global_average_pooling', 
          keepdims=True)

      if 'mobilenet' in network_backbone:
        image_feature = ops._mobilenetv2_conv_block(
            inputs=image_feature,
            filters=depth,
            expansion=1,
            stride=1,
            alpha=1.0,
            block_id=1)
      else: # xception
        image_feature = tf.layers.Conv2D(
            filters=depth, 
            kernel_size=(1, 1))(image_feature)
        image_feature = normalizer_fn(image_feature)
        image_feature = activation_fn(image_feature)

      image_feature = tf.image.resize_bilinear(
          images=image_feature,
          size=tf.shape(feature_map)[1:3],
          align_corners=True,
          name='upsample')

      logit_branches.append(image_feature)

    return tf.concat(logit_branches, 3)


def _atrous_separable_conv(features, 
                           output_depth,
                           kernel_size=3,
                           strides=1,
                           atrous_rate=1, 
                           activation_fn=tf.nn.relu6,
                           normalizer_fn=tf.nn.batch_normalization,
                           scope=None):
  """
  
  Args:
    features: 
    output_depth: 
    kernel_size: 
    strides: 
    atrous_rate: 
    weight_decay: 
    activation_fn: 
    normalizer_fn: 
    scope: 

  Returns:

  """"""
  @TODO: add weight_decay, weight_regularizer, scope
  """
  with tf.variable_scope(scope):
    if strides == 1:
      padding = 'same'
    else:
      padding = 'valid'
      kernel_size_effective = kernel_size + (kernel_size - 1) * (atrous_rate - 1)
      features = ops.pad_inputs(features, kernel_size_effective)

    # depthwise
    depthwise_conv = tf.keras.layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=strides,
        depth_multiplier=1,
        dilation_rate=(atrous_rate, atrous_rate),
        padding=padding)(features)
    depthwise_conv = normalizer_fn(depthwise_conv)
    depthwise_conv = activation_fn(depthwise_conv)

    # pointwise
    pointwise_conv = tf.layers.Conv2D(
        filters=output_depth,
        kernel_size=(1, 1),
        padding='SAME')(depthwise_conv)
    pointwise_conv = normalizer_fn(pointwise_conv)
    pointwise_conv = activation_fn(pointwise_conv)

  return pointwise_conv


def scale_dimension(dim, scale):
  """Scales the input dimension.

  Args:
    dim: Input dimension (a scalar or a scalar Tensor).
    scale: The amount of scaling applied to the input.

  Returns:
    Scaled dimension.
  """
  if isinstance(dim, tf.Tensor):
    return tf.cast((tf.to_float(dim) - 1.0) * scale + 1.0, dtype=tf.int32)
  else:
    return int((float(dim) - 1.0) * scale + 1.0)