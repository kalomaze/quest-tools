#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "transformers",
#     "huggingface_hub",
#     "pandas",
#     "tqdm",
# ]
# ///
import pandas as pd
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
import os
from tqdm import tqdm

def add_token_boundaries(tokenizer, tokens):
    """Add brackets around token boundaries"""
    text = ""
    for token in tokens:
        decoded = tokenizer.decode([token])
        text += f"[{decoded}] "
    return text.strip()

def load_tokenizer(repo_id="Qwen/Qwen2.5-7B", local_dir="qwen_7b"):
    """Load the tokenizer, downloading if necessary"""
    if not os.path.exists(local_dir):
        print(f"Downloading {repo_id} to {local_dir}...")
        snapshot_download(
            repo_id,
            local_dir=local_dir,
            ignore_patterns=["*.safetensors", "*.bin"],
        )
    else:
        print(f"Using existing tokenizer in {local_dir}")

    return AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)

def filter_by_token_count(df, column, tokenizer, min_tokens=None, max_tokens=None):
    """Filter DataFrame rows based on token count in specified column"""

    def count_tokens(text):
        if pd.isna(text):
            return 0
        return len(tokenizer.encode(str(text)))

    print("Counting tokens for each row...")
    token_counts = [count_tokens(text) for text in tqdm(df[column])]
    df['token_count'] = token_counts

    original_len = len(df)

    # Apply filters
    if min_tokens is not None:
        df = df[df['token_count'] >= min_tokens]
    if max_tokens is not None:
        df = df[df['token_count'] <= max_tokens]

    filtered_len = len(df)
    print(f"\nFiltered from {original_len} to {filtered_len} rows")

    return df

def main():
    # Step 1: Get input file
    input_files = [f for f in os.listdir('.') if f.endswith('.jsonl')]

    if not input_files:
        print("No JSONL files found in current directory")
        return

    print("Available JSONL files:")
    for i, file in enumerate(input_files):
        print(f"{i}: {file}")

    try:
        file_idx = int(input("Enter the number of the file to process: ").strip())
        input_file = input_files[file_idx]
    except (ValueError, IndexError):
        print("Invalid file selection")
        return

    # Step 2: Load the data
    print(f"\nLoading {input_file}...")
    df = pd.read_json(input_file, lines=True)

    # Step 3: Show columns and get user selection
    print("\nAvailable columns:")
    for i, col in enumerate(df.columns):
        print(f"{i}: {col}")

    try:
        col_idx = int(input("Enter the number of the column to tokenize: ").strip())
        column = df.columns[col_idx]
    except (ValueError, IndexError):
        print("Invalid column selection")
        return

    # Step 4: Get token count limits
    try:
        min_tokens = input("Enter minimum token count (or press Enter for no minimum): ").strip()
        min_tokens = int(min_tokens) if min_tokens else None

        max_tokens = input("Enter maximum token count (or press Enter for no maximum): ").strip()
        max_tokens = int(max_tokens) if max_tokens else None
    except ValueError:
        print("Invalid token count")
        return

    # Step 5: Load tokenizer and process
    try:
        tokenizer = load_tokenizer()

        # Filter the DataFrame
        filtered_df = filter_by_token_count(df, column, tokenizer, min_tokens, max_tokens)

        # Save the filtered DataFrame
        output_file = f"{os.path.splitext(input_file)[0]}_filtered.jsonl"
        filtered_df.to_json(output_file, orient='records', lines=True, force_ascii=False)
        print(f"\nSaved filtered data to {output_file}")

        # Print some statistics
        print("\nToken count statistics:")
        print(filtered_df['token_count'].describe())

        # Optional: Show a sample row with token boundaries
        if len(filtered_df) > 0:
            sample_text = filtered_df.iloc[0][column]
            sample_tokens = tokenizer.encode(str(sample_text))
            print("\nExample tokenization for first row:")
            print("Original text:")
            print(sample_text[:200] + "..." if len(str(sample_text)) > 200 else sample_text)
            print("\nToken boundaries:")
            print(add_token_boundaries(tokenizer, sample_tokens[:50]) + "..." if len(sample_tokens) > 50 else add_token_boundaries(tokenizer, sample_tokens))

    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main()
