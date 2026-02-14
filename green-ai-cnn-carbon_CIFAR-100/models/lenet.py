import tensorflow as tf
from tensorflow.keras import layers, models

def build_lenet(num_classes):
    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),          # <-- fixes warning too
        layers.Conv2D(
        6, kernel_size=5, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.AveragePooling2D(pool_size=(2, 2)),
        layers.Conv2D(16, kernel_size=5, activation="relu"),
        layers.AveragePooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(120, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.Dense(84, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model
