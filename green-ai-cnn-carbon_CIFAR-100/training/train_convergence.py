import time
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping


def train_until_convergence(
    model,
    x_train, y_train,
    x_val, y_val,
    max_epochs=150,
    min_epochs=40,
    patience=20,
    batch_size=64
):
    """
    Robust convergence training for CIFAR-100.
    Early stopping is DISABLED until min_epochs is reached.
    """

    start_time = time.time()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history_all = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    # ---- PHASE 1: Forced warmup (NO early stopping) ----
    print(f"🚦 Warmup phase: training {min_epochs} epochs without early stopping")

    history_warmup = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=min_epochs,
        batch_size=batch_size,
        verbose=2
    )

    for k in history_all:
        history_all[k] += history_warmup.history.get(k, [])

    # ---- PHASE 2: Convergence with early stopping ----
    remaining_epochs = max_epochs - min_epochs

    print(f"🎯 Convergence phase: up to {remaining_epochs} epochs with early stopping")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history_conv = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=remaining_epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=2
    )

    for k in history_all:
        history_all[k] += history_conv.history.get(k, [])

    epochs_trained = len(history_all["loss"])
    training_time = time.time() - start_time

    return {
        "history": history_all,
        "training_time_sec": training_time,
        "epochs_trained": epochs_trained
    }