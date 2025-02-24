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
import os
import pandas as pd
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from tqdm import tqdm
import json
import random
import math

def generate_single_token(final_letter):
    """Generate a single token (A or B) based on the final letter"""
    return " " + final_letter.upper()

def generate_convergence_sequence(final_letter, length=128):
    """Generate a convergence sequence using either exp or log pattern"""
    sequence = []
    convergence_type = random.choice(['exponential', 'logarithmic'])
    
    for i in range(length):
        progress = i / length
        
        if convergence_type == 'exponential':
            prob_shift = math.pow(progress, 3)
        else:  # logarithmic
            prob_shift = math.log(1 + 19*progress) / math.log(20)
            
        if final_letter.lower() == 'a':
            prob_a = 0.5 + (0.5 * prob_shift)
        else:
            prob_a = 0.5 - (0.5 * prob_shift)
        
        chosen_letter = 'A' if random.random() < prob_a else 'B'
        sequence.append(chosen_letter)
    
    return " " + " ".join(sequence)  # Add initial space and spaces between letters

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

def get_next_available_filename(base_filename):
    """Get the next available filename by adding version numbers"""
    if not os.path.exists(base_filename):
        return base_filename
    
    name, ext = os.path.splitext(base_filename)
    version = 2
    while True:
        new_filename = f"{name}_V{version}{ext}"
        if not os.path.exists(new_filename):
            return new_filename
        version += 1

def create_training_sample(row, tokenizer, use_convergence=True):
    """Create a training sample by tokenizing sections separately"""
    content = row['content']
    judgment = row['judgment'].strip().lower()  # Get just 'a' or 'b'
    
    # Ensure judgment is just 'a' or 'b'
    final_label = judgment[-1] if judgment and judgment[-1] in ['a', 'b'] else None
    if final_label is None:
        raise ValueError("Invalid judgment format - must end with 'a' or 'b'")
    
    # Generate sequence based on mode
    if use_convergence:
        sequence = generate_convergence_sequence(final_label)
    else:
        sequence = generate_single_token(final_label)
    
    # Tokenize sections separately
    content_tokens = tokenizer.encode(content + "\n", add_special_tokens=False)
    judgment_prefix = tokenizer.encode("ANSWER:", add_special_tokens=False)
    sequence_tokens = tokenizer.encode(sequence, add_special_tokens=False)
    
    # Combine tokens
    input_ids = (
        content_tokens +
        judgment_prefix +
        sequence_tokens
    )
    
    # Create labels with masking pattern
    labels = (
        [-100] * (len(content_tokens) + len(judgment_prefix)) +  # mask content and prefix
        sequence_tokens                                          # predict sequence
    )
    
    # Create attention mask (all 1s)
    attention_mask = [1] * len(input_ids)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def main():
    # Step 1: Select input file
    input_files = [f for f in os.listdir('.') if f.endswith('.jsonl')]

    if not input_files:
        print("No JSONL files found in current directory")
        return

    print("\nAvailable JSONL files:")
    for i, file in enumerate(input_files):
        print(f"{i}: {file}")

    try:
        file_idx = int(input("Enter the number of the file to process: ").strip())
        input_file = input_files[file_idx]
    except (ValueError, IndexError):
        print("Invalid file selection")
        return

    # Ask for mode selection
    while True:
        mode = input("\nSelect mode (1 for convergence sequence, 2 for single token): ").strip()
        if mode in ['1', '2']:
            use_convergence = (mode == '1')
            break
        print("Invalid selection. Please enter 1 or 2.")

    # Step 2: Load the data
    print(f"\nLoading {input_file}...")
    try:
        df = pd.read_json(input_file, lines=True)
    except Exception as e:
        print(f"Error loading JSONL file: {e}")
        return

    # Step 3: Load tokenizer
    print("\nLoading tokenizer...")
    try:
        tokenizer = load_tokenizer()
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # Step 4: Process each row
    print("\nProcessing rows...")
    output_data = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            sample = create_training_sample(row, tokenizer, use_convergence)
            output_data.append(sample)
        except Exception as e:
            print(f"Error processing row: {e}")
            continue

    # Step 5: Save the output
    mode_suffix = "convergence" if use_convergence else "single"
    base_output_file = f"{os.path.splitext(input_file)[0]}_{mode_suffix}_axolotl.jsonl"
    output_file = get_next_available_filename(base_output_file)

    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in output_data:
            json.dump(item, f)
            f.write('\n')

    # Print statistics
    print("\nProcessing Statistics:")
    print(f"Total rows processed: {len(df)}")
    print(f"Successfully converted: {len(output_data)}")
    print(f"Mode: {'Convergence Sequence' if use_convergence else 'Single Token'}")

    if output_data:
        print("\nSample conversion (first row):")
        sample = output_data[0]
        
        print("\nDetailed token analysis:")
        tokens = sample['input_ids']
        labels = sample['labels']

        print("\nFirst 50 tokens:")
        for i, (t, l) in enumerate(zip(tokens[:50], labels[:50])):
            text = tokenizer.decode([t])
            label_text = tokenizer.decode([l]) if l != -100 else "MASKED"
            print(f"{i:3d} | Token: {t:6d} | Label: {l:6d} | Text: {text:20s} | Label Text: {label_text}")

        print("\nLast tokens:")
        last_n = 128 if use_convergence else 10
        for i, (t, l) in enumerate(zip(tokens[-last_n:], labels[-last_n:]), start=len(tokens)-last_n):
            text = tokenizer.decode([t])
            label_text = tokenizer.decode([l]) if l != -100 else "MASKED"
            print(f"{i:3d} | Token: {t:6d} | Label: {l:6d} | Text: {text:20s} | Label Text: {label_text}")

        print("\nToken counts:")
        print(f"Total tokens: {len(tokens)}")
        print(f"Masked labels (-100): {labels.count(-100)}")
        print(f"Unmasked labels: {len(labels) - labels.count(-100)}")

if __name__ == "__main__":
    main()