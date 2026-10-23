
from collections import Counter
from datasets import load_dataset
import numpy as np

DATASET_ID = "vzc-research-chapter/VZCrash"
N_SAMPLES = 30

print(f"Inspecting up to {N_SAMPLES} training records...")

label_counts = Counter()
sensor_shapes = Counter()
missing_values = Counter()
records_read = 0

try:
    dataset = load_dataset(
        DATASET_ID,
        split="train",
        streaming=True
    )

    for record in dataset:
        label = int(record["label"])
        label_counts[label] += 1
        records_read += 1

        for column in ["gsensor", "gyro", "gps_speed"]:
            values = np.asarray(record[column], dtype=float)

            if records_read == 1:
                sensor_shapes[column] = values.shape

            if not np.isfinite(values).all():
                missing_values[column] += 1

        if records_read >= N_SAMPLES:
            break

except Exception as e:
    print(f"\nReading stopped: {type(e).__name__}: {e}")

print(f"\nRecords successfully inspected: {records_read}")
print("Label counts:", dict(label_counts))
print("Sensor shapes:", dict(sensor_shapes))
print("Records with non-finite sensor values:", dict(missing_values))
