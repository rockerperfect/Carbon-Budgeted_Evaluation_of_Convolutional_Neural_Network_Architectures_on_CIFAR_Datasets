import csv
import os

CSV_HEADER = [
    "model",
    "dataset",
    "training_mode",
    "run_id",
    "epochs_trained",
    "final_val_accuracy",
    "best_val_accuracy",
    "training_time_sec",
    "co2_g",
    "co2_kg"
]

def init_csv(csv_path):
    """
    Create CSV file with header if it does not exist.
    """
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)


def log_run(
    csv_path,
    model,
    dataset,
    training_mode,
    run_id,
    epochs_trained,
    history,
    training_time_sec,
    carbon_dict
):
    """
    Append one experiment run to CSV.
    """
    init_csv(csv_path)

    final_val_acc = history["val_accuracy"][-1]
    best_val_acc = max(history["val_accuracy"])

    row = [
        model,
        dataset,
        training_mode,
        run_id,
        epochs_trained,
        round(final_val_acc, 4),
        round(best_val_acc, 4),
        round(training_time_sec, 2),
        round(carbon_dict["co2_g"], 4),
        round(carbon_dict["co2_kg"], 6)
    ]

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
