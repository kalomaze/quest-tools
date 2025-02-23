import asyncio
import aiohttp
import json
import os
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import difflib
import random
import numpy as np
from text_corruptor_utils import (generate_corruption_pattern, 
                                corrupt_text_segment,
                                DEFAULT_API_KEY,
                                DEFAULT_OBJECTIVE)

def load_config(config_path='cli_config.txt'):
    """Load configuration from file."""
    config = {
        'ATTEMPTS_PER_SAMPLE': 16,
        'SAMPLE_BATCH_SIZE': 2,
        'DEBUG_LENGTH_CHECK': True,
        'DEBUG_CONSTANT_CORRUPTION': False,
        'NEW_CORRUPT_EACH_ATTEMPT': False,
        'API_URL': "https://expensive-editorial-objectives-blacks.trycloudflare.com",
        'API_KEY': "YOUR_API_KEY",
        'JSONL_PATH': "/Users/bepis/Downloads/0exqfd.jsonl"
    }
    
    try:
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Convert value to appropriate type
                    if value.lower() == 'true':
                        config[key] = True
                    elif value.lower() == 'false':
                        config[key] = False
                    elif value.isdigit():
                        config[key] = int(value)
                    else:
                        config[key] = value
                        
        print("[Config] Successfully loaded configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
            
    except FileNotFoundError:
        print(f"[Config] Warning: Config file not found at {config_path}. Using defaults.")
    except Exception as e:
        print(f"[Config] Error loading config: {str(e)}. Using defaults.")
    
    return config

def get_match_ratio(original, restored):
    if not original or not restored:
        return 0
    matcher = difflib.SequenceMatcher(None, original, restored)
    return matcher.ratio() * 100

async def get_model_name(session, api_key, api_url):
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        url = f"{api_url}/v1/models"
        print(f"[ModelLoader] Requesting models from: {url}")
        
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                models = [m['id'] for m in data.get('data', [])]
                print(f"[ModelLoader] Successfully loaded {len(models)} models.")
                return models[0] if models else None
            else:
                print(f"[ModelLoader] API error: {response.status}")
                return None
    except Exception as e:
        print(f"[ModelLoader] Connection error: {str(e)}")
        return None

async def generate_attempt(session, corrupted_text, model_name, config):
    try:
        prompt = f"{corrupted_text}\n\n{DEFAULT_OBJECTIVE}\n\n<original>"
        
        headers = {
            "Authorization": f"Bearer {config['API_KEY']}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": True,
            "max_tokens": 4096,
            "temperature": 0.9,
            "top_p": 0.999,
        }
        
        url = f"{config['API_URL']}/v1/completions"
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status != 200:
                print(f"[GenerationWorker] API error {response.status}: {await response.text()}")
                return None
            
            token_buffer = ""
            async for line in response.content:
                try:
                    line_text = line.decode('utf-8')
                    if line_text.startswith('data: '):
                        chunk = line_text[6:].strip()
                        if chunk == '[DONE]':
                            break
                        data = json.loads(chunk)
                        text = data.get('choices', [{}])[0].get('text', '')
                        token_buffer += text
                        
                        if any(tag in token_buffer for tag in ["<obj", "<orig", "</orig", "</object"]):
                            splits = []
                            if "<obj" in token_buffer:
                                splits.append(token_buffer.split("<obj")[0])
                            if "<orig" in token_buffer:
                                splits.append(token_buffer.split("<orig")[0])
                            if "</orig" in token_buffer:
                                splits.append(token_buffer.split("</orig")[0])
                            if "</object" in token_buffer:
                                splits.append(token_buffer.split("</object")[0])
                            
                            if splits:
                                clean_text = min(splits, key=len)
                                return f"{clean_text.strip()}"
                            
                        if len(token_buffer) > 100:
                            lines = token_buffer.split('\n')
                            for line in lines[1:]:
                                if len(line.strip()) > 50:
                                    if token_buffer.count(line) > 1:
                                        clean_text = token_buffer.split(line)[0] + line
                                        return clean_text.strip()
                                        
                except Exception as e:
                    print(f"[GenerationWorker] Error processing chunk: {str(e)}")
                    continue
            
            return token_buffer.strip()
            
    except Exception as e:
        print(f"[GenerationWorker] Error: {str(e)}")
        return None

def apply_constant_corruption(text):
    """Apply Gaussian corruption to a random position within the middle 50% region (25%-75%) of text."""
    text_length = len(text)
    current_text = text
    
    min_start = int(text_length * 0.25)
    max_start = int(text_length * 0.75)
    block_size = max(1, int(text_length * 0.50))
    start_pos = random.randint(min_start, max_start - block_size)
    pattern = np.zeros(text_length)
    
    for _ in range(3):
        gaussian_pattern = generate_corruption_pattern('Gaussian', block_size, 95, 95)
        pattern[start_pos:start_pos + block_size] = gaussian_pattern[:block_size]
        current_text = corrupt_text_segment(current_text, pattern, False)
    
    return current_text

async def process_entries_parallel(session, entries, model_name, config, output_file):
    # Sort entries by length
    entries_with_idx = [(idx, entry) for idx, entry in enumerate(entries)]
    entries_with_idx.sort(key=lambda x: len(x[1]['text']))
    
    # Track results for each entry
    results = {idx: {'valid_results': [], 'power_stats': {}} 
              for idx, _ in entries_with_idx}
    
    # Create all tasks upfront
    all_tasks = {}
    entries_queue = []
    
    for idx, entry in entries_with_idx:
        if config['DEBUG_LENGTH_CHECK']:
            text_length = len(entry['text'])
            if text_length < 64 or text_length > 9000:
                continue
                
        # Create all attempts for this entry
        for attempt_idx in range(config['ATTEMPTS_PER_SAMPLE']):
            entries_queue.append((idx, entry['text']))
    
    # Process in batches of SAMPLE_BATCH_SIZE
    with tqdm(total=len(entries)) as pbar:
        completed_entries = set()
        
        while entries_queue or all_tasks:
            # Start new tasks up to batch size
            while len(all_tasks) < config['SAMPLE_BATCH_SIZE'] and entries_queue:
                idx, text = entries_queue.pop(0)
                
                if config['NEW_CORRUPT_EACH_ATTEMPT']:
                    if config.get('DEBUG_CONSTANT_CORRUPTION', False):
                        corrupted = apply_constant_corruption(text)
                    else:
                        pattern = generate_corruption_pattern('Gaussian', len(text), 100, 100)
                        corrupted = corrupt_text_segment(text, pattern, False)
                else:
                    # Use consistent corruption for all attempts
                    if config.get('DEBUG_CONSTANT_CORRUPTION', False):
                        corrupted = apply_constant_corruption(text)
                    else:
                        pattern = generate_corruption_pattern('Gaussian', len(text), 100, 100)
                        corrupted = corrupt_text_segment(text, pattern, False)
                
                task = asyncio.create_task(generate_attempt(session, corrupted, model_name, config))
                all_tasks[task] = (idx, text)
            
            if not all_tasks:
                break
                
            # Wait for any task to complete
            done, _ = await asyncio.wait(
                all_tasks.keys(),
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Process completed tasks
            for task in done:
                idx, original_text = all_tasks[task]
                del all_tasks[task]
                
                try:
                    result = await task
                    if result:
                        score = get_match_ratio(original_text, result)
                        results[idx]['valid_results'].append((result, score))
                        results[idx]['valid_results'].sort(key=lambda x: x[1])
                        
                        # Update power statistics
                        num_results = len(results[idx]['valid_results'])
                        if num_results in [1, 2, 4, 8, 16, 32, 64]:
                            best_result, best_score = results[idx]['valid_results'][-1]
                            results[idx]['power_stats'][num_results] = {'score': best_score}
                            print(f"[Powers][{idx}] Best of {num_results}: {best_score:.2f}%")
                
                except Exception as e:
                    print(f"Error processing result for entry {idx}: {str(e)}")
                
                # Check if entry is complete and save immediately
                if idx not in completed_entries and \
                   len(results[idx]['valid_results']) >= config['ATTEMPTS_PER_SAMPLE']:
                    completed_entries.add(idx)
                    pbar.update(1)
                    
                    # Save result immediately when entry completes
                    best_result, best_score = max(results[idx]['valid_results'], key=lambda x: x[1])
                    entry_result = {
                        'original': entries[idx]['text'],
                        'restoration': best_result,
                        'match_score': best_score,
                        'power_stats': results[idx]['power_stats']
                    }
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(entry_result) + '\n')
    
    return list(completed_entries)

async def main():
    config = load_config()
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"restoration_results_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    
    connector = aiohttp.TCPConnector(limit=32)
    async with aiohttp.ClientSession(connector=connector) as session:
        print("[Main] Getting model name...")
        model_name = await get_model_name(session, config['API_KEY'], config['API_URL'])
        if not model_name:
            print("[Main] Failed to get model name")
            return
        print(f"[Main] Using model: {model_name}")
        
        print("[Main] Reading local JSONL file...")
        try:
            with open(config['JSONL_PATH'], 'r', encoding='utf-8') as f:
                content = f.read()
                entries = [json.loads(line) for line in content.strip().split('\n')]
            print(f"Successfully loaded {len(entries)} entries from local file")
            
        except Exception as e:
            print(f"[Main] Failed to read local JSONL: {str(e)}")
            return

        print(f"[Main] Processing {len(entries)} entries...")
        completed_entries = await process_entries_parallel(session, entries, model_name, config, output_file)
        
        successful = len(completed_entries)
        print("\n[Main] Final Statistics:")
        print(f"Total entries processed: {len(entries)}")
        print(f"Successful restorations: {successful}")
        print(f"Success rate: {(successful/len(entries))*100:.2f}%")
        print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    asyncio.run(main())