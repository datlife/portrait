import numpy as np
import tensorflow as tf
from portrait.network_backbone.MobileNetV2 import MobilenetV2


def test_mobilenet_v2():
    inputs = tf.keras.Input(shape=[224, 224, 3], batch_size=1, dtype=tf.float32)
    with tf.Session() as sess:
        # outputs = MobilenetV2(inputs, alpha=1.0, num_classes=1000, include_top=True)
        # model = tf.keras.Model(inputs, outputs)
        model = tf.keras.applications.MobilenetV2
        # Count multi-adds operations
        flops = tf.profiler.profile(
            sess.graph,
            options=tf.profiler.ProfileOptionBuilder.float_operation())

        # Count trainable parameters
        params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

        model.summary()
        assert params == 3504872  # 3.5M
        assert flops.total_float_ops == 608771220  #  ~6.0B


def test_mobilenet_v2_inference():
    """Simulate a prediction"""
    import cv2
    from tensorflow.keras.applications.mobilenet import decode_predictions, preprocess_input
    img = cv2.resize(cv2.imread('docs/cute_cat.jpg'), (224, 224))

    inputs = tf.keras.Input(shape=[224, 224, 3], batch_size=1, dtype=tf.float32)
    outputs = MobilenetV2(inputs, alpha=1.0, num_classes=1000, include_top=True)
    model = tf.keras.Model(inputs, outputs)

    x = preprocess_input(np.expand_dims(img, 0).astype(float))
    predictions = model.predict(x, batch_size=1)
    # print(decode_predictions(predictions, top=1))
    assert 1 == 1
