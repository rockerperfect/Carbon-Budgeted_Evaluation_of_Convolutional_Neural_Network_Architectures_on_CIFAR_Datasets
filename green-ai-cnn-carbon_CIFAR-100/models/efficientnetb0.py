import tensorflow as tf

def build_efficientnetb0(num_classes):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=(32,32,3),
        weights=None
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    outputs = tf.keras.layers.Dense(
    num_classes,
    activation="softmax",
    kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    return tf.keras.Model(base.input, outputs)
