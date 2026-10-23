
from datasets import load_dataset
import numpy as np

DATASET_ID = "vzc-research-chapter/VZCrash"

print("Opening VZCrash in streaming mode...")

try:
    dataset = load_dataset(
        DATASET_ID,
        split="train",
        streaming=True
    )

    sample = next(iter(dataset))

    print("\nColumns:", list(sample.keys()))
    print("\nEvent ID:", sample["event_id"])
    print("Label:", sample["label"])
    print("Vehicle size:", sample["vehiclesize"])
    print("Gyroscope HD:", sample["gyro_is_hd"])

    for column in ["gsensor", "gyro", "gps_speed"]:
        values = np.asarray(sample[column], dtype=object)
        print(f"\n{column}:")
        print("  Shape:", values.shape)
        print("  First values:", values[:3])

except Exception as e:
    print("\nInspection failed:")
    print(type(e).__name__, e)
