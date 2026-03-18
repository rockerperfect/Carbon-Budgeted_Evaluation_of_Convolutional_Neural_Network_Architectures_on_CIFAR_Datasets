import tensorflow as tf


def preprocess(x_train, x_test):
    """ResNet50 uses standard [0, 1] normalized inputs. No extra preprocessing needed."""
    return x_train, x_test


def get_resnet50(num_classes=100):
    """ResNet-50 (trained from scratch) adapted for CIFAR-100 (32x32x3 input)."""
    base = tf.keras.applications.ResNet50(
        weights=None,
        include_top=False,
        input_shape=(32, 32, 3),
    )

    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(base.input, output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model
