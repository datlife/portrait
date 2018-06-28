import tensorflow as tf
from portrait.deeplab.core.decoder import reconstruct

def test_deeplabv3plus_decoder():
    """Test Decoder"""
    encoded_features = tf.placeholder(
        shape=[None, 7, 7, 256], dtype=tf.float32)
    low_level_features = tf.placeholder(
        shape= [None, 56, 56, 144], dtype=tf.float32)

    logits = reconstruct(
        encoded_features,
        low_level_features,
        input_size=(224, 224),
        num_classes=20,
        model_variant='mobilenet_v2'
    )

    assert [None, 224, 224, 20] == \
        logits.get_shape().as_list()