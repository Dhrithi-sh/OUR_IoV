
from huggingface_hub import HfApi

DATASET_ID = "vzc-research-chapter/VZCrash"

api = HfApi()

try:
    info = api.dataset_info(DATASET_ID)
    print("Dataset access successful!")
    print("Dataset:", info.id)
    print("\nAvailable files:")

    files = api.list_repo_files(
        repo_id=DATASET_ID,
        repo_type="dataset"
    )

    for file in files:
        print(file)

except Exception as e:
    print("Could not access the dataset.")
    print("Error:", e)
