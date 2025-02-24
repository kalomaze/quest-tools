#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas",
#     "numpy",
#     "tqdm",
# ]
# ///
import pandas as pd
import numpy as np
import random
import string
import os
from tqdm import tqdm

# Character sets
UTF8_CHARS = [chr(i) for i in range(0x0100, 0x0800)]
ALPHANUMERIC_CHARS = list(string.ascii_letters + string.digits)

def generate_pattern(length, pattern_type, range_min, range_max, alphanumeric):
    """Generate corruption pattern based on specified parameters"""
    x = np.linspace(0, 1, length)
    max_val = random.uniform(range_min, range_max)

    if pattern_type == 'constant':
        pattern = np.full(length, max_val)
    elif pattern_type == 'gaussian':
        mean = random.uniform(0.3, 0.7)
        std = random.uniform(0.1, 0.3)
        pattern = max_val * np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    elif pattern_type == 'gaussian_50':
        mid = length // 2
        pattern = np.zeros(length)
        # First gaussian
        x1 = np.linspace(0, 1, mid)
        mean1 = 0.5
        std1 = 0.3
        pattern[:mid] = max_val * np.exp(-((x1 - mean1) ** 2) / (2 * std1 ** 2))
        # Second gaussian
        x2 = np.linspace(0, 1, length - mid)
        mean2 = 0.5
        std2 = 0.3
        pattern[mid:] = max_val * np.exp(-((x2 - mean2) ** 2) / (2 * std2 ** 2))
    elif pattern_type == 'gaussian_25':
        quarter = length // 4
        pattern = np.zeros(length)
        # Four gaussians, one for each quarter
        for i in range(4):
            x_section = np.linspace(0, 1, quarter)
            mean = 0.5
            std = 0.3
            start_idx = i * quarter
            end_idx = start_idx + quarter
            pattern[start_idx:end_idx] = max_val * np.exp(-((x_section - mean) ** 2) / (2 * std ** 2))
    elif pattern_type == 'linear':
        pattern = max_val * x
    elif pattern_type == 'linear_reverse':
        pattern = max_val * (1-x)
    elif pattern_type == 'block_50':
        pattern = np.zeros(length)
        start = random.randint(0, length // 2)
        block_length = length // 2
        pattern[start:start+block_length] = max_val
    elif pattern_type == 'block_25':
        pattern = np.zeros(length)
        start = random.randint(0, int(length * 0.75))
        block_length = length // 4
        pattern[start:start+block_length] = max_val
    elif pattern_type == 'staircase_50':
        mid = length // 2
        pattern = np.zeros(length)
        pattern[:mid] = max_val * np.linspace(0, 1, mid)
        pattern[mid:] = max_val * np.linspace(0, 1, length - mid)
    elif pattern_type == 'staircase_50_reverse':
        mid = length // 2
        pattern = np.zeros(length)
        pattern[:mid] = max_val * np.linspace(1, 0, mid)
        pattern[mid:] = max_val * np.linspace(1, 0, length - mid)
    elif pattern_type == 'staircase_25':
        quarter = length // 4
        pattern = np.zeros(length)
        pattern[:quarter] = max_val * np.linspace(0, 1, quarter)
        pattern[quarter:2*quarter] = max_val * np.linspace(0, 1, quarter)
        pattern[2*quarter:3*quarter] = max_val * np.linspace(0, 1, quarter)
        pattern[3*quarter:] = max_val * np.linspace(0, 1, length - 3*quarter)
    elif pattern_type == 'staircase_25_reverse':
        quarter = length // 4
        pattern = np.zeros(length)
        pattern[:quarter] = max_val * np.linspace(1, 0, quarter)
        pattern[quarter:2*quarter] = max_val * np.linspace(1, 0, quarter)
        pattern[2*quarter:3*quarter] = max_val * np.linspace(1, 0, quarter)
        pattern[3*quarter:] = max_val * np.linspace(1, 0, length - 3*quarter)
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")

    return pattern

def corrupt_text(text, pattern, alphanumeric=False):
    """Corrupt text based on pattern"""
    if not isinstance(text, str):
        return text

    chars = ALPHANUMERIC_CHARS if alphanumeric else UTF8_CHARS
    result = ""

    for i, char in enumerate(text):
        if char == '\n':
            result += char
            continue

        if random.random() * 100 < pattern[min(i, len(pattern)-1)]:
            result += random.choice(chars)
        else:
            result += char

    return result

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

    # Step 3: Select column
    print("\nAvailable columns:")
    for i, col in enumerate(df.columns):
        print(f"{i}: {col}")

    try:
        col_idx = int(input("Enter the number of the column to corrupt: ").strip())
        column = df.columns[col_idx]
    except (ValueError, IndexError):
        print("Invalid column selection")
        return

    # Step 4: Select patterns to include
    patterns = [
        'constant',
        'gaussian',
        'gaussian_50',
        'gaussian_25', 
        'linear',
        'linear_reverse',
        'block_50',
        'block_25',
        'staircase_50',
        'staircase_50_reverse',
        'staircase_25',
        'staircase_25_reverse'
    ]
    
    enabled_patterns = []
    print("\nEnable/disable corruption patterns:")
    for pattern in patterns:
        while True:
            response = input(f"Include {pattern}? (y/n): ").lower().strip()
            if response in ['y', 'n']:
                if response == 'y':
                    enabled_patterns.append(pattern)
                break
            print("Please enter 'y' or 'n'")

    if not enabled_patterns:
        print("No patterns selected!")
        return

    # Step 5: Get corruption range
    try:
        print("\nEnter corruption range (0-100):")
        range_min = float(input("Minimum corruption percentage: ").strip())
        range_max = float(input("Maximum corruption percentage: ").strip())

        if not (0 <= range_min <= 100 and 0 <= range_max <= 100 and range_min <= range_max):
            raise ValueError
    except ValueError:
        print("Invalid range values")
        return

    # Step 6: Choose character mode
    try:
        print("\nCharacter mode:")
        print("0: UTF-8 (wide range of characters)")
        print("1: Alphanumeric (letters and numbers only)")
        mode_choice = int(input("Enter your choice (0/1): ").strip())
        alphanumeric = bool(mode_choice)
        if mode_choice not in [0, 1]:
            raise ValueError
    except ValueError:
        print("Invalid mode selection")
        return

    # Step 7: Optional random seed
    try:
        seed_input = input("\nEnter random seed (or press Enter to skip): ").strip()
        if seed_input:
            random.seed(int(seed_input))
            np.random.seed(int(seed_input))
    except ValueError:
        print("Invalid seed value, proceeding without seed")

    # Create corrupted version of the text
    print("\nCorrupting texts...")
    corrupted_texts = []
    used_patterns = []  # Track which pattern was used for each text

    for text in tqdm(df[column]):
        if pd.isna(text):
            corrupted_texts.append(text)
            used_patterns.append('none')
            continue

        # Randomly select pattern for this text
        pattern_type = random.choice(enabled_patterns)
        used_patterns.append(pattern_type)

        pattern = generate_pattern(
            length=len(str(text)),
            pattern_type=pattern_type,
            range_min=range_min,
            range_max=range_max,
            alphanumeric=alphanumeric
        )

        corrupted = corrupt_text(
            text=str(text),
            pattern=pattern,
            alphanumeric=alphanumeric
        )
        corrupted_texts.append(corrupted)

    # Create output DataFrame
    output_df = pd.DataFrame({
        'corrupted': corrupted_texts,
        'original': df[column],
        'pattern_used': used_patterns
    })

    # Save to output file
    output_file = f"{os.path.splitext(input_file)[0]}_corrupted.jsonl"
    print(f"\nSaving to {output_file}...")
    output_df.to_json(output_file, orient='records', lines=True, force_ascii=False)

    # Print statistics
    print("\nCorruption Statistics:")
    print(f"Total rows processed: {len(df)}")
    print(f"Enabled patterns: {', '.join(enabled_patterns)}")
    print(f"Corruption range: {range_min}% - {range_max}%")
    print(f"Character mode: {'Alphanumeric' if alphanumeric else 'UTF-8'}")
    
    # Pattern usage statistics
    print("\nPattern usage:")
    pattern_counts = pd.Series(used_patterns).value_counts()
    for pattern, count in pattern_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{pattern}: {count} times ({percentage:.1f}%)")

    # Show a sample
    if len(df) > 0:
        print("\nSample corruption (first row):")
        print("Pattern used:", output_df['pattern_used'].iloc[0])
        print("Original:", output_df['original'].iloc[0][:100], "..." if len(str(output_df['original'].iloc[0])) > 100 else "")
        print("Corrupted:", output_df['corrupted'].iloc[0][:100], "..." if len(str(output_df['corrupted'].iloc[0])) > 100 else "")

if __name__ == '__main__':
    main()
