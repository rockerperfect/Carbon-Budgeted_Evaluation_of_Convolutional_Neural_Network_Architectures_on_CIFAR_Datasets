from codecarbon import EmissionsTracker
import time
import os
import uuid

def start_tracker(
    model_name,
    dataset_name,
    run_id,
    output_dir
):
    """
    Starts CodeCarbon tracker for one training run.
    Compatible with all CodeCarbon versions.
    """

    os.makedirs(output_dir, exist_ok=True)

    experiment_id = f"{model_name}_{dataset_name}_run{run_id}_{uuid.uuid4().hex[:6]}"

    tracker = EmissionsTracker(
        project_name="green_ai_cnn",
        experiment_id=experiment_id,
        output_dir=output_dir,
        measure_power_secs=1,
        log_level="error"
    )

    tracker.start()
    start_time = time.time()

    return tracker, start_time


def stop_tracker(tracker, start_time):
    """
    Stops tracker and returns structured carbon results.
    """
    emissions_kg = tracker.stop()
    runtime_sec = time.time() - start_time

    return {
        "co2_kg": emissions_kg,
        "co2_g": emissions_kg * 1000 if emissions_kg is not None else None,
        "runtime_sec": runtime_sec
    }
