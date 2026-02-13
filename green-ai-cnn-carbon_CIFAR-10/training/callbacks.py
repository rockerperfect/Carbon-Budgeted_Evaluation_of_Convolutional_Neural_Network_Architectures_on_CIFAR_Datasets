import tensorflow as tf

def get_early_stopping(patience=10):
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
