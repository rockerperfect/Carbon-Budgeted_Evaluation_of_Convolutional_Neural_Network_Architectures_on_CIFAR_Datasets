import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_vgg16(num_classes):
    weight_decay = 1e-4

    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),

        # Block 1
        layers.Conv2D(64, 3, padding="same",
                      kernel_regularizer=regularizers.l2(weight_decay)),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.Conv2D(64, 3, padding="same",
                      kernel_regularizer=regularizers.l2(weight_decay)),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.MaxPooling2D(),

        # Block 2
        layers.Conv2D(128, 3, padding="same",
                      kernel_regularizer=regularizers.l2(weight_decay)),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.Conv2D(128, 3, padding="same",
                      kernel_regularizer=regularizers.l2(weight_decay)),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.MaxPooling2D(),

        # Block 3
        layers.Conv2D(256, 3, padding="same",
                      kernel_regularizer=regularizers.l2(weight_decay)),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.Conv2D(256, 3, padding="same",
                      kernel_regularizer=regularizers.l2(weight_decay)),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.Conv2D(256, 3, padding="same",
                      kernel_regularizer=regularizers.l2(weight_decay)),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.MaxPooling2D(),

        # Classifier
        layers.Flatten(),
        layers.Dense(512, activation="relu",
                     kernel_regularizer=regularizers.l2(weight_decay)),
        layers.Dense(512, activation="relu",
                     kernel_regularizer=regularizers.l2(weight_decay)),
        layers.Dense(num_classes, activation="softmax")
    ])

    return model