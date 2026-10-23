# VZCrash Random Forest Baseline

A preliminary Random Forest baseline for binary crash detection using the VZCrash dataset. The scripts extract summary features from sensor signals, train a model, and evaluate it on the official validation shards.

## Scripts

- `build_baseline.py` — reads the first cached training shard, extracts features, trains the model, and reports a stratified holdout evaluation.
- `evaluate_official_validation.py` — evaluates the saved model on the dataset's three official validation shards.
- `inspect_dataset.py`, `inspect_sample.py`, and `inspect_batch.py` — inspect dataset access and sample structure.
- `test_download.py` — checks access to one training shard.

## Setup

Install the dependencies below in a Python environment:

```powershell
python -m pip install -r requirements.txt
```

The training script expects the first training Parquet shard in the Hugging Face cache at the path defined by `DATA_FILE` in `build_baseline.py`. The validation script downloads validation shards from Hugging Face.

Run training before evaluation:

```powershell
python build_baseline.py
python evaluate_official_validation.py
```

Generated feature data, predictions, the serialized model, local credentials, and the virtual environment are excluded from version control.

This is a preliminary baseline and should not be treated as a validated crash detection system.
