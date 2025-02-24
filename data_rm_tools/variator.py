import json
import random

def add_noise(text):
    """Add controlled noise while preserving <original> tag integrity"""
    variations = []
    variations.append(text)  # Original clean version
    
    # Split text into parts preserving <original> tag
    parts = text.split('<original>')
    before = parts[0]
    after = parts[1] if len(parts) > 1 else ''
    
    # Common transformations that don't touch the tag
    transforms = [
        # Typos in common words
        lambda s: s.replace('restore', 'resore'),
        lambda s: s.replace('careful', 'carefull'),
        lambda s: s.replace('proper', 'propper'),
        lambda s: s.replace('repair', 'repare'),
        lambda s: s.replace('content', 'contnet'),
        lambda s: s.replace('complete', 'compleet'),
        lambda s: s.replace('reconstruct', 'reconstrukt'),
        lambda s: s.replace('attention', 'attension'),
        
        # Single/double letter typos
        lambda s: s.replace('ll', 'l'),
        lambda s: s.replace('ss', 's'),
        lambda s: s.replace('ff', 'f'),
        
        # Capitalization
        lambda s: s.lower(),
        lambda s: s.capitalize(),
        
        # Minor punctuation
        lambda s: s.replace('.', ''),
        lambda s: s.replace(':', ''),
        lambda s: s + '.',
    ]
    
    # Generate variations while keeping tag intact
    for _ in range(9):  # 9 variations + original = 10 total
        before_mod = before
        after_mod = after
        
        # Apply 1-2 random transformations to each part
        for _ in range(random.randint(1, 2)):
            transform = random.choice(transforms)
            before_mod = transform(before_mod)
        for _ in range(random.randint(1, 2)):
            transform = random.choice(transforms)
            after_mod = transform(after_mod)
            
        # Recombine while preserving tag
        new_variation = before_mod + '<original>' + after_mod
        new_variation = ' '.join(new_variation.split())  # Clean up spaces
        variations.append(new_variation)
    
    return variations

# Load base variations
with open('variations.json', 'r') as f:
    base_variations = json.load(f)

# Generate noisy variations
all_variations = []
for base in base_variations:
    all_variations.extend(add_noise(base))

# Remove duplicates while preserving order
seen = set()
final_variations = []
for var in all_variations:
    if var not in seen:
        final_variations.append(var)
        seen.add(var)

# Save expanded variations
with open('variations_expanded.json', 'w', encoding='utf-8') as f:
    json.dump(final_variations, f, indent=2, ensure_ascii=False)

print(f"Original variations: {len(base_variations)}")
print(f"Expanded variations: {len(final_variations)}")