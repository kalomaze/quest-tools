#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets",
#     "pandas",
#     "tqdm",
# ]
# ///
import json
import time
from datasets import load_dataset, get_dataset_split_names, get_dataset_config_names
from tqdm import tqdm

def main():
    dataset_name = "HuggingFaceFW/fineweb-edu"
    trust_remote_code = False

    # Step 1: Check for configurations
    try:
        configs = get_dataset_config_names(dataset_name, trust_remote_code=trust_remote_code)
        has_configs = True
    except ValueError:
        configs = [None]
        has_configs = False

    if has_configs:
        print("Available configurations:")
        for i, config_name in enumerate(configs):
            print(f"{i}: {config_name}")

        try:
            chosen_config_index = int(input("Enter the number corresponding to a configuration: ").strip())
            chosen_config = configs[chosen_config_index]
        except (ValueError, IndexError):
            print("Invalid choice for configuration.")
            return
    else:
        chosen_config = None

    time.sleep(0.1)  # Small pause after configuration selection

    # Step 2: Get available splits
    splits = get_dataset_split_names(dataset_name, chosen_config, trust_remote_code=trust_remote_code)

    print(f"\nAvailable splits:")
    for i, split_name in enumerate(splits):
        print(f"{i}: {split_name}")

    try:
        chosen_split_index = int(input("Enter the number corresponding to a split: ").strip())
        chosen_split = splits[chosen_split_index]
    except (ValueError, IndexError):
        print("Invalid choice for split.")
        return

    time.sleep(0.1)  # Small pause after split selection

    # Step 3: Ask user how many rows to download
    try:
        num_rows = int(input("How many rows do you want to download? ").strip())
        if num_rows <= 0:
            raise ValueError
    except ValueError:
        print("Please enter a valid positive integer for the number of rows.")
        return

    # Step 4: Load the dataset using streaming
    print(f"\nLoading dataset '{dataset_name}'...")
    time.sleep(0.1)  # Small pause before loading

    try:
        dataset = load_dataset(
            dataset_name,
            name=chosen_config,
            split=chosen_split,
            streaming=True,
            trust_remote_code=trust_remote_code
        )

        safe_dataset_name = dataset_name.replace('/', '_')
        config_part = f"_{chosen_config}" if chosen_config else ""
        safe_filename = f"{safe_dataset_name}{config_part}_{chosen_split}_{num_rows}.jsonl"

        with open(safe_filename, 'w', encoding='utf-8') as f:
            for item in tqdm(dataset.take(num_rows)):
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                time.sleep(0.001)  # Tiny pause between writes

        time.sleep(0.1)  # Small pause before final message
        print(f"Saved {num_rows} rows to {safe_filename}")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

if __name__ == '__main__':
    main()