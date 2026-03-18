import tensorflow as tf
import numpy as np


def preprocess(x_train, x_test):
    """LeNet-5 uses standard [0, 1] normalized inputs. No extra preprocessing needed."""
    return x_train, x_test


def get_lenet5(num_classes=100):
    """LeNet-5 adapted for CIFAR-100 (32x32x3 input, 100 output classes)."""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(16, (5, 5), activation='relu'),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model
