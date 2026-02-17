import tensorflow as tf
import time
from .callbacks import get_early_stopping

def train_until_convergence(
    model,
    x_train, y_train,
    x_val, y_val,
    max_epochs=100,
    batch_size=64,
    learning_rate=0.001,
    patience=10
):
    # ---- NEW: learning-rate schedule (cosine decay) ----
    steps_per_epoch = len(x_train) 
    total_steps = max_epochs * steps_per_epoch

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

    early_stopping = get_early_stopping(patience)

    start_time = time.time()

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=2
    )

    training_time = time.time() - start_time
    epochs_trained = len(history.history["loss"])

    return {
        "history": history.history,
        "training_time_sec": training_time,
        "epochs_trained": epochs_trained
    }
