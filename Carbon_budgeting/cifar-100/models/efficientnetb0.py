import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input


def preprocess(x_train, x_test):
    """
    EfficientNetB0 expects inputs in [0, 255] range which its internal
    preprocessing rescales. We pass x * 255.0 (from the [0,1] normalized data)
    through Keras' preprocess_input before feeding it to the model.
    """
    x_train_e = preprocess_input(x_train * 255.0)
    x_test_e  = preprocess_input(x_test  * 255.0)
    return x_train_e, x_test_e


def get_efficientnetb0(num_classes=100):
    """EfficientNetB0 (trained from scratch) adapted for CIFAR-100 (32x32x3 input)."""
    base = tf.keras.applications.EfficientNetB0(
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
