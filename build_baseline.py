
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


DATA_FILE = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "datasets--vzc-research-chapter--VZCrash"
    / "snapshots"
    / "42e1482d005c0e545379d6cfa4939486c9fe4249"
    / "data"
    / "train-00000-of-00011.parquet"
)

N_SAMPLES = 1000
RANDOM_STATE = 42


def summarize_signal(values, prefix):
    arr = np.asarray(values, dtype=np.float64)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    features = {}

    for axis in range(arr.shape[1]):
        x = arr[:, axis]
        x = x[np.isfinite(x)]

        if len(x) == 0:
            continue

        name = f"{prefix}_axis{axis}"

        features[f"{name}_mean"] = np.mean(x)
        features[f"{name}_std"] = np.std(x)
        features[f"{name}_min"] = np.min(x)
        features[f"{name}_max"] = np.max(x)
        features[f"{name}_median"] = np.median(x)
        features[f"{name}_q25"] = np.percentile(x, 25)
        features[f"{name}_q75"] = np.percentile(x, 75)
        features[f"{name}_rms"] = np.sqrt(np.mean(x ** 2))
        features[f"{name}_mean_abs"] = np.mean(np.abs(x))

    # Combined magnitude captures overall sensor intensity.
    if arr.shape[1] >= 2:
        valid = arr[np.isfinite(arr).all(axis=1)]

        if len(valid):
            magnitude = np.linalg.norm(valid, axis=1)
            features[f"{prefix}_mag_mean"] = np.mean(magnitude)
            features[f"{prefix}_mag_std"] = np.std(magnitude)
            features[f"{prefix}_mag_max"] = np.max(magnitude)
            features[f"{prefix}_mag_rms"] = np.sqrt(
                np.mean(magnitude ** 2)
            )
            features[f"{prefix}_mag_q95"] = np.percentile(
                magnitude, 95
            )

    return features


def extract_features(record):
    features = {}
    features.update(summarize_signal(record["gsensor"], "acc"))
    features.update(summarize_signal(record["gyro"], "gyro"))
    features.update(summarize_signal(record["gps_speed"], "speed"))
    return features


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Parquet file not found: {DATA_FILE}")

    print("Reading the local Parquet shard...")
    parquet_file = pq.ParquetFile(DATA_FILE)

    rows = []
    labels = []
    event_ids = []

    for batch in parquet_file.iter_batches(batch_size=64):
        for record in batch.to_pylist():
            label = int(record["label"])

            if label not in (0, 1, 2):
                continue

            features = extract_features(record)

            if not features:
                continue

            # Assumes the dataset mapping is:
            # 0 = crash, 1 = near-miss, 2 = normal driving.
            # Binary target: crash versus non-crash.
            rows.append(features)
            labels.append(1 if label == 0 else 0)
            event_ids.append(record["event_id"])

            if len(rows) >= N_SAMPLES:
                break

        if len(rows) >= N_SAMPLES:
            break

    X = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    y = pd.Series(labels, name="is_crash")

    print(f"\nRecords collected: {len(X)}")
    print("Binary label counts:", Counter(y))
    print("Feature count:", X.shape[1])

    if y.nunique() < 2 or y.value_counts().min() < 2:
        raise ValueError(
            "Not enough examples of both classes in this sample. "
            "We need to collect a larger or more representative sample."
        )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    )

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

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

    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_val, predictions, labels=[0, 1]))

    X.assign(is_crash=y).to_csv("baseline_features.csv", index=False)
    joblib.dump(model, "random_forest_baseline.joblib")

    print("\nSaved baseline_features.csv")
    print("Saved random_forest_baseline.joblib")
    print("\nThis is a preliminary baseline, not final evaluation.")


if __name__ == "__main__":
    main()
