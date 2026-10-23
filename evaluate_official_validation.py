
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from huggingface_hub import hf_hub_download
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

from build_baseline import extract_features


DATASET_ID = "vzc-research-chapter/VZCrash"
MODEL_FILE = Path("random_forest_baseline.joblib")
FEATURE_FILE = Path("baseline_features.csv")


def main():
    if not MODEL_FILE.exists() or not FEATURE_FILE.exists():
        raise FileNotFoundError(
            "Run build_baseline.py first so the saved model "
            "and feature CSV are available."
        )

    model = joblib.load(MODEL_FILE)

    # Preserve exactly the feature columns and order used in training.
    feature_columns = [
        c for c in pd.read_csv(FEATURE_FILE, nrows=0).columns
        if c != "is_crash"
    ]

    all_rows = []
    all_labels = []

    shard_names = [
        f"data/validation-{i:05d}-of-00003.parquet"
        for i in range(3)
    ]

    for shard_name in shard_names:
        print(f"\nGetting {shard_name}...")

        path = hf_hub_download(
            repo_id=DATASET_ID,
            filename=shard_name,
            repo_type="dataset",
        )

        parquet_file = pq.ParquetFile(path)
        shard_count = 0

        for batch in parquet_file.iter_batches(batch_size=64):
            for record in batch.to_pylist():
                label = int(record["label"])

                if label not in (0, 1, 2):
                    continue

                features = extract_features(record)
                all_rows.append(features)
                all_labels.append(1 if label == 0 else 0)
                shard_count += 1

        print(f"Records read from shard: {shard_count:,}")

    X_val = pd.DataFrame(all_rows).reindex(columns=feature_columns)
    X_val = X_val.replace([np.inf, -np.inf], np.nan)
    y_val = np.asarray(all_labels)

    print("\nOfficial validation records:", f"{len(y_val):,}")
    print("Binary label counts:", Counter(y_val))
    print("Evaluating saved Random Forest...")

    predictions = model.predict(X_val)

    print("\nAccuracy:", round(accuracy_score(y_val, predictions), 4))
    print("\nClassification report:")
    print(
        classification_report(
            y_val,
            predictions,
            labels=[0, 1],
            target_names=["Non-crash", "Crash"],
            zero_division=0,
        )
    )

    # Rows = actual; columns = predicted.
    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ).ravel()

    print("Confusion matrix:")
    print([[tn, fp], [fn, tp]])

    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    print(f"\nFalse-positive rate: {fpr:.4%}")
    print(f"False-negative rate: {fnr:.4%}")

    probabilities = model.predict_proba(X_val)[:, 1]
    ap = average_precision_score(y_val, probabilities)
    print(f"Average precision (AP): {ap:.4f}")

    results = X_val.copy()
    results["actual_is_crash"] = y_val
    results["predicted_is_crash"] = predictions
    results["crash_probability"] = probabilities
    results.to_csv("official_validation_predictions.csv", index=False)

    print("\nSaved official_validation_predictions.csv")


if __name__ == "__main__":
    main()
