#!/usr/bin/env python3
import os
import glob
import json
import numpy as np
import subprocess
from collections import defaultdict

# First, find all potential NPZ directories using find command
print("Searching for directories containing NPZ files...")
find_cmd = "find /home/ubuntu -name '*.npz' -type f | head -n 5"
proc = subprocess.run(find_cmd, shell=True, capture_output=True, text=True)
sample_files = proc.stdout.strip().split('\n')

# Get the directories that contain NPZ files
npz_dirs = set()
for filepath in sample_files:
    if filepath:  # Skip empty lines
        npz_dirs.add(os.path.dirname(filepath))

print(f"\nFound {len(npz_dirs)} directories with NPZ files:")
for i, directory in enumerate(npz_dirs):
    cmd = f"ls {directory}/*.npz 2>/dev/null | wc -l"
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    file_count = proc.stdout.strip()
    print(f"{i+1}. {directory} - contains ~{file_count} NPZ files")

# Check a sample from each directory to determine if it's WAV2VEC data
print("\nAnalyzing sample files from each directory:")
wav2vec_dirs = {}

for directory in npz_dirs:
    # Get a sample file from this directory
    cmd = f"ls {directory}/*.npz 2>/dev/null | head -n 1"
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    sample_file = proc.stdout.strip()
    
    if not sample_file:
        continue
    
    try:
        data = np.load(sample_file, allow_pickle=True)
        keys = list(data.keys())
        
        # Check if this looks like WAV2VEC data
        is_wav2vec = False
        if 'wav2vec_features' in keys:
            is_wav2vec = True
            feature_key = 'wav2vec_features'
        elif 'embeddings' in keys:
            is_wav2vec = True
            feature_key = 'embeddings'
        
        if is_wav2vec:
            # Count files in this directory
            cmd = f"find {directory} -name '*.npz' | wc -l"
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            count = proc.stdout.strip()
            
            wav2vec_dirs[directory] = {
                'count': count,
                'feature_key': feature_key,
                'sample_keys': keys
            }
            
            print(f"✓ {directory} - WAV2VEC data with key '{feature_key}', ~{count} files")
        else:
            print(f"✗ {directory} - Not WAV2VEC data, keys: {keys}")
    except Exception as e:
        print(f"✗ {directory} - Error loading {sample_file}: {str(e)}")

# Summarize findings
print("\n===== WAV2VEC Feature Files Summary =====")
total_files = sum(int(info['count']) for info in wav2vec_dirs.values())
print(f"Found {len(wav2vec_dirs)} directories with WAV2VEC feature files, total: ~{total_files} files")

for directory, info in wav2vec_dirs.items():
    # Show a few example files
    cmd = f"ls {directory}/*.npz | head -n 3"
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    examples = proc.stdout.strip().split('\n')
    
    print(f"\nDirectory: {directory}")
    print(f"Count: ~{info['count']} files")
    print(f"Feature key: '{info['feature_key']}'")
    print(f"Sample keys: {info['sample_keys']}")
    print("Example files:")
    for ex in examples:
        if ex:
            print(f"  - {os.path.basename(ex)}")
