#!/usr/bin/env python3
# /// script
# requires-python = ">=3.7"
# dependencies = [
#     "huggingface_hub",
# ]
# ///

from huggingface_hub import HfApi, create_repo
import os

def upload_to_hf():
    # Initialize the Hugging Face API
    api = HfApi()
    
    # Repository ID and settings
    repo_id = "Quest-AI/quest-corruption-200k-dataset-v2"
    
    # Create repository if it doesn't exist
    print("Creating repository if it doesn't exist...")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=False,
            exist_ok=True
        )
        print("Repository ready")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return

    # Files to upload
    files = [
        "HuggingFaceFW_fineweb-edu_sample-350BT_train_200000_filtered_corrupted.jsonl",
        "HuggingFaceFW_fineweb-edu_sample-350BT_train_200000_filtered_corrupted_axolotl.jsonl",
        "HuggingFaceFW_fineweb-edu_sample-350BT_train_200000_filtered_corrupted_xml.jsonl"
    ]
    
    # Upload each file
    for file in files:
        if not os.path.exists(file):
            print(f"Warning: {file} not found, skipping...")
            continue
            
        print(f"Uploading {file}...")
        try:
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=repo_id,
                repo_type="dataset",
            )
            print(f"Successfully uploaded {file}")
        except Exception as e:
            print(f"Error uploading {file}: {e}")

if __name__ == "__main__":
    upload_to_hf()