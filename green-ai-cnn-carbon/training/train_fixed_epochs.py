import tensorflow as tf
import time

def train_fixed_epochs(
    model,
    x_train, y_train,
    x_val, y_val,
    epochs=50,
    batch_size=64,
    learning_rate=0.001
):
    # ---- NEW: learning-rate schedule (cosine decay) ----
    steps_per_epoch = len(x_train) // batch_size
    total_steps = epochs * steps_per_epoch

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=total_steps
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule
    )
    # ----------------------------------------------------

    model.compile(
        optimizer=optimizer,
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
