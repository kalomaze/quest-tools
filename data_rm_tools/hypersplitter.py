import json
import argparse
from pathlib import Path
import math

def get_split_ratio():
    while True:
        try:
            split = input("Enter split ratio (e.g., '60/40' or '70/30'): ")
            first, second = map(int, split.split('/'))
            if first + second != 100:
                print("Split must add up to 100")
                continue
            return first, second
        except ValueError:
            print("Invalid format. Please use format like '60/40'")

def split_jsonl(file_path, first_ratio, second_ratio):
    # Read all lines
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    first_split = math.floor((first_ratio/100) * total_lines)
    
    # Create output file names
    original_path = Path(file_path)
    first_output = original_path.with_name(f"{original_path.stem}__{first_ratio}SPLIT{original_path.suffix}")
    second_output = original_path.with_name(f"{original_path.stem}__{second_ratio}SPLIT{original_path.suffix}")
    
    # Write first split
    with open(first_output, 'w') as f:
        for line in lines[:first_split]:
            f.write(line)
    
    # Write second split
    with open(second_output, 'w') as f:
        for line in lines[first_split:]:
            f.write(line)
    
    print(f"Split complete!")
    print(f"First file ({first_ratio}%): {first_output}")
    print(f"Second file ({second_ratio}%): {second_output}")
    print(f"Lines in first file: {first_split}")
    print(f"Lines in second file: {total_lines - first_split}")

def main():
    parser = argparse.ArgumentParser(description='Split a JSONL file into two parts')
    parser.add_argument('file_path', type=str, help='Path to the JSONL file')
    args = parser.parse_args()
    
    # Verify file exists
    if not Path(args.file_path).exists():
        print(f"Error: File '{args.file_path}' does not exist")
        return
    
    # Get split ratio from user
    first_ratio, second_ratio = get_split_ratio()
    
    # Perform the split
    split_jsonl(args.file_path, first_ratio, second_ratio)

if __name__ == "__main__":
    main()