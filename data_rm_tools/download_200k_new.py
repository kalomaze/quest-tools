#!/usr/bin/env python3
# /// script
# requires-python = ">=3.7"
# dependencies = [
#     "huggingface_hub",
# ]
# ///

from huggingface_hub import snapshot_download
import os

def download_dataset():
    # Repository settings
    repo_id = "Quest-AI/quest-corruption-200k-dataset-v2"
    local_dir = "quest-corruption-repo"
    
    print(f"Downloading dataset from {repo_id} to {local_dir}...")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"Successfully downloaded to {local_dir}")
        
        # List downloaded files
        print("\nDownloaded files:")
        for file in os.listdir(local_dir):
            print(f"- {file}")
            
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    download_dataset()