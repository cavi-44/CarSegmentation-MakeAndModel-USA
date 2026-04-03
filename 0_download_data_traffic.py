import os
from huggingface_hub import snapshot_download

# Configuration
REPO_ID = "LibreYOLO/road-traffic"
LOCAL_DIR = "data/raw_road_traffic"


def download_road_traffic():
    """
    Downloads the LibreYOLO road-traffic dataset from Hugging Face.
    This includes images and YOLO format labels.
    """
    print(f"Starting download of {REPO_ID} to {LOCAL_DIR}...")

    try:
        # snapshot_download downloads the entire repository/dataset
        # ignore_patterns helps to skip unnecessary files like .gitattributes
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False,
            ignore_patterns=[".git*", "README.md"]
        )

        print(f"\nSuccess! Dataset downloaded to: {os.path.abspath(LOCAL_DIR)}")
        print("Structure check:")
        for root, dirs, files in os.walk(LOCAL_DIR):
            level = root.replace(LOCAL_DIR, '').count(os.sep)
            indent = ' ' * 4 * (level)
            print(f"{indent}{os.path.basename(root)}/")

    except Exception as e:
        print(f"An error occurred during download: {e}")


if __name__ == "__main__":
    download_road_traffic()