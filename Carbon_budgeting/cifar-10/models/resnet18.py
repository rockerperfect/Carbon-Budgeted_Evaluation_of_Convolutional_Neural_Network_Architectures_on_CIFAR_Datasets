import tensorflow as tf


def preprocess(x_train, x_test):
    """ResNet18 uses standard [0, 1] normalized inputs. No extra preprocessing needed."""
    return x_train, x_test


def _resnet_block(x, filters, stride):
    """Basic residual block with optional projection shortcut."""
    shortcut = x

    x = tf.keras.layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, use_bias=False)(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.Add()([x, shortcut])
    return tf.keras.layers.ReLU()(x)


def get_resnet18(num_classes=10):
    """ResNet-18 adapted for CIFAR (32x32x3 input, no maxpool stem)."""
    inputs = tf.keras.Input(shape=(32, 32, 3))

    # CIFAR stem — small conv, no maxpool
    x = tf.keras.layers.Conv2D(64, 3, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Stage 1
    for _ in range(2):
        x = _resnet_block(x, 64, stride=1)

    # Stage 2
    x = _resnet_block(x, 128, stride=2)
    x = _resnet_block(x, 128, stride=1)

    # Stage 3
    x = _resnet_block(x, 256, stride=2)
    x = _resnet_block(x, 256, stride=1)

    # Classification head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model
