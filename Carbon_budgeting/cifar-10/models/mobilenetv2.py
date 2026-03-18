import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def preprocess(x_train, x_test):
    """
    MobileNetV2 requires inputs in [-1, 1].
    Keras' preprocess_input expects pixel values in [0, 255],
    so we rescale from [0, 1] → [0, 255] before calling it.
    """
    x_train_m = preprocess_input(x_train * 255.0)
    x_test_m  = preprocess_input(x_test  * 255.0)
    return x_train_m, x_test_m


def get_mobilenetv2(num_classes=10):
    """MobileNetV2 (trained from scratch) adapted for CIFAR (32x32x3 input)."""
    base = tf.keras.applications.MobileNetV2(
        weights=None,
        include_top=False,
        input_shape=(32, 32, 3),
    )

    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(base.input, output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model
