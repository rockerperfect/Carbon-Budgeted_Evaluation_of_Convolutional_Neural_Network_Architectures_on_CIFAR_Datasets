import tensorflow as tf
from tensorflow.keras import layers, models

def residual_block(x, filters):
    shortcut = x
    # First Conv Layer
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Second Conv Layer
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)

    # Add shortcut before final ReLU
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def build_resnet18(num_classes):
    inputs = layers.Input(shape=(32, 32, 3))
    
    # Initial Layer
    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Residual Stages
    for _ in range(2):
        x = residual_block(x, 64)
    
    # Final Classification Layers (OUTSIDE the loop)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

    return models.Model(inputs, outputs)