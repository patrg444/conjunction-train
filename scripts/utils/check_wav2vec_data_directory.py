#!/usr/bin/env python3
"""
Script to check the wav2vec data directory structure on EC2
to diagnose the 'No valid files found' error.
"""

import os
import sys
import glob
import argparse
from collections import Counter

def main():
    parser = argparse.ArgumentParser(description='Check wav2vec data directory structure')
    parser.add_argument('--data_dir', type=str, default='/home/ubuntu/audio_emotion/wav2vec_features',
                        help='Path to wav2vec features directory')
    args = parser.parse_args()
    
    data_dir = args.data_dir
    print(f"Checking data directory: {data_dir}")
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"ERROR: Directory does not exist: {data_dir}")
        return
    
    # List all directories in the data path
    print("\nDirectories in data path:")
    for item in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, item)):
            print(f"  {item}/")
    
    # Find all .npz files
    all_files = glob.glob(os.path.join(data_dir, "**/*.npz"), recursive=True)
    print(f"\nFound {len(all_files)} .npz files")
    
    if len(all_files) == 0:
        print("ERROR: No .npz files found in the data directory!")
        # List the top-level structure
        print("\nListing top-level structure:")
        os.system(f"find {data_dir} -maxdepth 2 -type d | sort")
        return
    
    # Check parent directory names
    parent_dirs = []
    for file in all_files[:min(100, len(all_files))]:
        parts = file.split(os.sep)
        parent_dir = parts[-2].lower() if len(parts) >= 2 else "unknown"
        parent_dirs.append(parent_dir)
    
    # Count occurrences of each parent directory
    parent_counts = Counter(parent_dirs)
    
    print("\nParent directory names (showing up to 100 files):")
    for parent, count in parent_counts.most_common():
        print(f"  {parent}: {count} files")
    
    # Check file structure of a few examples
    print("\nExample file paths:")
    for file in all_files[:5]:
        print(f"  {file}")
        try:
            import numpy as np
            data = np.load(file)
            print(f"    Shape: {data.shape}, Type: {data.dtype}")
        except Exception as e:
            print(f"    Error loading file: {e}")
    
    # Check for normalization files
    print("\nChecking for normalization statistics:")
    mean_file = os.path.join(data_dir, "wav2vec_mean.npy")
    std_file = os.path.join(data_dir, "wav2vec_std.npy")
    
    if os.path.exists(mean_file):
        print(f"  wav2vec_mean.npy: FOUND")
    else:
        print(f"  wav2vec_mean.npy: NOT FOUND")
    
    if os.path.exists(std_file):
        print(f"  wav2vec_std.npy: FOUND")
    else:
        print(f"  wav2vec_std.npy: NOT FOUND")
    
    print("\nExpected emotion labels in parent directories:")
    expected = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]
    found = [e for e in expected if e in parent_counts]
    missing = [e for e in expected if e not in parent_counts]
    
    print("  Found emotions:", ', '.join(found))
    print("  Missing emotions:", ', '.join(missing))
    
    # Suggest fix based on findings
    print("\nDiagnosis and suggestions:")
    if len(all_files) == 0:
        print("  ISSUE: No .npz files found in the data directory.")
        print("  SUGGESTION: Check that wav2vec features have been extracted properly.")
    elif not any(e in parent_counts for e in expected):
        print("  ISSUE: None of the expected emotion directories were found.")
        print("  SUGGESTION: Check the directory structure. The expected structure is:")
        print("  /home/ubuntu/audio_emotion/wav2vec_features/neutral/*.npz")
        print("  /home/ubuntu/audio_emotion/wav2vec_features/happy/*.npz")
        print("  etc.")
    elif missing:
        print(f"  ISSUE: Some emotion directories are missing: {', '.join(missing)}")
        print("  SUGGESTION: This may be expected if you're using a subset of emotions.")
        print("  Check if the script needs to be updated to match your dataset.")
    elif not os.path.exists(mean_file) or not os.path.exists(std_file):
        print("  ISSUE: Normalization statistics files (mean/std) are missing.")
        print("  SUGGESTION: Run a preprocessing script to generate these files.")
    else:
        print("  Directory structure appears to be correct.")
        print("  Check for other issues in the script that may be causing the error.")

if __name__ == "__main__":
    main()
