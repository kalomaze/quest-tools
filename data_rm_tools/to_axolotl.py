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
from concurrent.futures import ProcessPoolExecutor
import math
import shutil

def load_variations(filepath="variations_expanded.json"):
    """Load the variations file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_next_available_filename(base_filename):
    """Get the next available filename by adding version numbers"""
    if not os.path.exists(base_filename):
        return base_filename
    
    # Split filename into name and extension
    name, ext = os.path.splitext(base_filename)
    
    # Try versions until we find an available filename
    version = 2
    while True:
        new_filename = f"{name}_V{version}{ext}"
        if not os.path.exists(new_filename):
            return new_filename
        version += 1

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

def format_to_xml(row):
    """Convert a row to pseudo-XML format with random objective placement"""
    variations = load_variations()
    objective = random.choice(variations)
    
    # 5% chance of no objective
    if random.random() < 0.05:
        return (
            "<corrupted>\n"
            f"{row['corrupted']}\n"
            "</corrupted>\n\n"
            "<original>\n"
            f"{row['original']}\n"
            "</original>"
        )
    
    # 80% chance of objective first (out of remaining 95%)
    if random.random() < (0.80/0.95):
        return (
            "<objective>\n"
            f"{objective}\n"
            "</objective>\n\n"
            "<corrupted>\n"
            f"{row['corrupted']}\n"
            "</corrupted>\n\n"
            "<original>\n"
            f"{row['original']}\n"
            "</original>"
        )
    
    # 20% chance of objective after corrupted
    return (
        "<corrupted>\n"
        f"{row['corrupted']}\n"
        "</corrupted>\n\n"
        "<objective>\n"
        f"{objective}\n"
        "</objective>\n\n"
        "<original>\n"
        f"{row['original']}\n"
        "</original>"
    )

def create_training_sample(row, tokenizer):
    """Create a training sample by tokenizing sections separately"""
    variations = load_variations()
    objective = random.choice(variations)
    
    # 5% chance of no objective
    has_objective = random.random() >= 0.05
    # If we have objective, 80% chance of it being first (out of remaining 95%)
    objective_first = random.random() < (0.80/0.95) if has_objective else False
    
    # Tokenize sections
    corrupt_start = tokenizer.encode("<corrupted>\n", add_special_tokens=False)
    corrupt_text = tokenizer.encode(f"{row['corrupted']}\n", add_special_tokens=False)
    corrupt_end = tokenizer.encode("</corrupted>\n\n", add_special_tokens=False)

    orig_start = tokenizer.encode("<original>\n", add_special_tokens=False)
    orig_text = tokenizer.encode(f"{row['original']}\n", add_special_tokens=False)
    orig_end = tokenizer.encode("</original>", add_special_tokens=False)
    
    if has_objective:
        obj_start = tokenizer.encode("<objective>\n", add_special_tokens=False)
        obj_text = tokenizer.encode(f"{objective}\n", add_special_tokens=False)
        obj_end = tokenizer.encode("</objective>\n\n", add_special_tokens=False)
    
    # Combine all token sequences based on placement
    if not has_objective:
        input_ids = (
            corrupt_start +
            corrupt_text +
            corrupt_end +
            orig_start +
            orig_text +
            orig_end
        )
        labels = (
            [-100] * len(corrupt_start) +   # <corrupted> tag
            [-100] * len(corrupt_text) +    # corrupted text
            [-100] * len(corrupt_end) +     # </corrupted> tag
            [-100] * len(orig_start) +      # <original> tag
            orig_text +                     # original text (actual labels)
            orig_end                        # </original> tag
        )
    elif objective_first:
        input_ids = (
            obj_start +
            obj_text +
            obj_end +
            corrupt_start +
            corrupt_text +
            corrupt_end +
            orig_start +
            orig_text +
            orig_end
        )
        labels = (
            [-100] * len(obj_start) +       # <objective> tag
            [-100] * len(obj_text) +        # objective text
            [-100] * len(obj_end) +         # </objective> tag
            [-100] * len(corrupt_start) +   # <corrupted> tag
            [-100] * len(corrupt_text) +    # corrupted text
            [-100] * len(corrupt_end) +     # </corrupted> tag
            [-100] * len(orig_start) +      # <original> tag
            orig_text +                     # original text (actual labels)
            orig_end                        # </original> tag
        )
    else:
        input_ids = (
            corrupt_start +
            corrupt_text +
            corrupt_end +
            obj_start +
            obj_text +
            obj_end +
            orig_start +
            orig_text +
            orig_end
        )
        labels = (
            [-100] * len(corrupt_start) +   # <corrupted> tag
            [-100] * len(corrupt_text) +    # corrupted text
            [-100] * len(corrupt_end) +     # </corrupted> tag
            [-100] * len(obj_start) +       # <objective> tag
            [-100] * len(obj_text) +        # objective text
            [-100] * len(obj_end) +         # </objective> tag
            [-100] * len(orig_start) +      # <original> tag
            orig_text +                     # original text (actual labels)
            orig_end                        # </original> tag
        )

    # Create attention mask (all 1s)
    attention_mask = [1] * len(input_ids)

    # Verify lengths match
    assert len(input_ids) == len(labels) == len(attention_mask), \
        "Mismatched lengths in token sequences"

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def process_chunk(chunk_data, chunk_id, tokenizer):
    """Process a chunk of data and save to temporary files"""
    chunk_output_axolotl = []
    chunk_output_xml = []
    
    for _, row in chunk_data.iterrows():
        # Create XML format
        xml_text = format_to_xml(row)
        chunk_output_xml.append({'text': xml_text})

        # Create training sample
        try:
            sample = create_training_sample(row, tokenizer)
            chunk_output_axolotl.append(sample)
        except Exception as e:
            print(f"Error processing row in chunk {chunk_id}: {e}")
            continue
    
    # Save chunk to temporary files
    temp_xml_file = f"temp_chunk_{chunk_id}_xml.jsonl"
    temp_axolotl_file = f"temp_chunk_{chunk_id}_axolotl.jsonl"
    
    with open(temp_xml_file, 'w', encoding='utf-8') as f:
        for item in chunk_output_xml:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
            
    with open(temp_axolotl_file, 'w', encoding='utf-8') as f:
        for item in chunk_output_axolotl:
            json.dump(item, f)
            f.write('\n')
    
    return temp_xml_file, temp_axolotl_file

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

    # Calculate chunks
    CHUNK_SIZE = 4000
    num_chunks = math.ceil(len(df) / CHUNK_SIZE)
    chunks = [df[i:i + CHUNK_SIZE] for i in range(0, len(df), CHUNK_SIZE)]
    
    # Process chunks in parallel
    temp_files = []
    print("\nProcessing chunks in parallel...")
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_chunk, chunk, i, tokenizer)
            for i, chunk in enumerate(chunks)
        ]
        
        for future in tqdm(futures, total=len(futures)):
            temp_files.append(future.result())
    
    # Get output filenames
    base_xml_output_file = f"{os.path.splitext(input_file)[0]}_xml.jsonl"
    base_axolotl_output_file = f"{os.path.splitext(input_file)[0]}_axolotl.jsonl"
    xml_output_file = get_next_available_filename(base_xml_output_file)
    axolotl_output_file = get_next_available_filename(base_axolotl_output_file)
    
    # Merge XML files
    print(f"\nMerging chunks into {xml_output_file}...")
    with open(xml_output_file, 'w', encoding='utf-8') as outfile:
        for temp_xml, _ in temp_files:
            with open(temp_xml, 'r', encoding='utf-8') as infile:
                shutil.copyfileobj(infile, outfile)
    
    # Merge Axolotl files
    print(f"Merging chunks into {axolotl_output_file}...")
    with open(axolotl_output_file, 'w', encoding='utf-8') as outfile:
        for _, temp_axolotl in temp_files:
            with open(temp_axolotl, 'r', encoding='utf-8') as infile:
                shutil.copyfileobj(infile, outfile)
    
    # Cleanup temporary files
    print("Cleaning up temporary files...")
    for temp_xml, temp_axolotl in temp_files:
        os.remove(temp_xml)
        os.remove(temp_axolotl)
    
    # Print statistics
    print("\nProcessing Statistics:")
    print(f"Total rows processed: {len(df)}")
    print(f"Number of chunks processed: {num_chunks}")

    # Show samples
    print("\nSample conversion (first chunk):")
    with open(xml_output_file, 'r', encoding='utf-8') as f:
        first_xml = json.loads(f.readline())
    with open(axolotl_output_file, 'r', encoding='utf-8') as f:
        first_axolotl = json.loads(f.readline())

    print("\nIntermediate XML format:")
    print(first_xml['text'])

    print("\nAxolotl format:")
    sample = first_axolotl

    print("\nDetailed token analysis:")
    tokens = sample['input_ids']
    labels = sample['labels']

    print("\nFirst 50 tokens:")
    for i, (t, l) in enumerate(zip(tokens[:50], labels[:50])):
        text = tokenizer.decode([t])
        label_text = tokenizer.decode([l]) if l != -100 else "MASKED"
        print(f"{i:3d} | Token: {t:6d} | Label: {l:6d} | Text: {text:20s} | Label Text: {label_text}")

    print("\nToken counts:")
    print(f"Total tokens: {len(tokens)}")
    print(f"Masked labels (-100): {labels.count(-100)}")
    print(f"Unmasked labels: {len(labels) - labels.count(-100)}")

if __name__ == "__main__":
    main()