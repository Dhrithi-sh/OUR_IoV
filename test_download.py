
from huggingface_hub import hf_hub_download
from pathlib import Path

REPO_ID = "vzc-research-chapter/VZCrash"
FILENAME = "data/train-00000-of-00011.parquet"

print("Checking access to one training shard...")

try:
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        repo_type="dataset",
    )

    size_mb = Path(path).stat().st_size / (1024 * 1024)
    print(f"\nDownload successful: {path}")
    print(f"File size: {size_mb:.2f} MB")

except Exception as exc:
    print(f"\nDownload failed: {type(exc).__name__}: {exc}")
