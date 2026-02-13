import tensorflow as tf

def build_convnext_tiny(num_classes):
    base = tf.keras.applications.ConvNeXtTiny(
        include_top=False,
        input_shape=(32,32,3),
        weights=None
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(base.input, outputs)
