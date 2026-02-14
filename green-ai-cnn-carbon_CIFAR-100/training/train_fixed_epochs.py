import tensorflow as tf
import time

def train_fixed_epochs(
    model,
    x_train, y_train,
    x_val, y_val,
    epochs=50,
    batch_size=64,
    learning_rate=3e-4
):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    start_time = time.time()

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2
    )

    training_time = time.time() - start_time

    return {
        "history": history.history,
        "training_time_sec": training_time,
        "epochs_trained": epochs
    }
