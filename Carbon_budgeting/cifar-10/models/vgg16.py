import tensorflow as tf
import numpy as np


def preprocess(x_train, x_test):
    """VGG16 uses standard [0, 1] normalized inputs. No extra preprocessing needed."""
    return x_train, x_test


def get_vgg16(num_classes=10):
    """VGG16 (trained from scratch) adapted for CIFAR (32x32x3 input)."""
    base = tf.keras.applications.VGG16(
        weights=None,
        include_top=False,
        input_shape=(32, 32, 3),
    )

    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(base.input, output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model
