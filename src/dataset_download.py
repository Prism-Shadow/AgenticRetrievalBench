from huggingface_hub import snapshot_download

REPO_ID = "PrismShadow/AgenticSearch"

snapshot_download(repo_id=REPO_ID, local_dir="data", repo_type="dataset")