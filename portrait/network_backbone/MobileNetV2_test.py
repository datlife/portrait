import numpy as np
import tensorflow as tf
from portrait.network_backbone.MobileNetV2 import MobilenetV2


# def test_mobilenet_v2():
#     inputs = tf.keras.Input(shape=[224, 224, 3], batch_size=1, dtype=tf.float32)
#     with tf.Session() as sess:
#         outputs = MobilenetV2(inputs, alpha=1.0, num_classes=1000, include_top=True)
#         model = tf.keras.Model(inputs, outputs)

#         # Count multi-adds operations
#         flops = tf.profiler.profile(
#             sess.graph,
#             options=tf.profiler.ProfileOptionBuilder.float_operation())

#         # Count trainable parameters
#         params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

#         assert params == 3504872  # 3.5M
#         assert flops.total_float_ops == 608771220  #  ~6.0B

def test_mobilenet_v2_inference():
    import cv2 as cv
    from tensorflow.keras.applications.mobilenet import decode_predictions, preprocess_input

    img = cv.resize(cv.imread('docs/cute_cat.jpg'), (224, 224))
    x = np.expand_dims(img, 0).astype(float)


    with tf.Session() as sess:
        inputs = tf.keras.Input(shape=[224, 224, 3], batch_size=1, dtype=tf.float32)
        outputs = MobilenetV2(inputs, alpha=1.0, num_classes=1000, include_top=True)
        model = tf.keras.Model(inputs, outputs)
        predictions = model.predict(preprocess_input(x), batch_size=1)
        print(decode_predictions(predictions, top=3))
        assert 1 == 2
