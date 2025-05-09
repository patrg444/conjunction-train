#!/usr/bin/env python3
import os
import sys
import numpy as np
import glob
from tqdm import tqdm

# Path to facenet features
CREMA_D_FACENET_DIR = "./crema_d_features_facenet"

def inspect_npz_file(file_path):
    print(f"Inspecting: {file_path}")
    try:
        with np.load(file_path) as data:
            # Print all keys in the npz file
            print(f"  Keys: {data.files}")
            
            # For each key, print the shape if it's an array
            for key in data.files:
                if isinstance(data[key], np.ndarray):
                    print(f"  {key}.shape = {data[key].shape}")
                else:
                    print(f"  {key} is not an array")
                    
            # Return True if inspection was successful    
            return True
    except Exception as e:
        print(f"  Error loading file: {e}")
        return False

def main():
    # Check if specific files were mentioned in error message
    problem_files = [
        "1078_IOM_SAD_XX.npz", 
        "1005_DFA_FEA_XX.npz", 
        "1080_IWW_ANG_XX.npz"
    ]
    
    print("Checking specific problem files:")
    for filename in problem_files:
        full_path = os.path.join(CREMA_D_FACENET_DIR, filename)
        if os.path.exists(full_path):
            inspect_npz_file(full_path)
        else:
            print(f"File not found: {full_path}")
    
    # Count all .npz files and inspect a sample
    all_files = glob.glob(os.path.join(CREMA_D_FACENET_DIR, "*.npz"))
    print(f"\nFound {len(all_files)} total .npz files")
    
    # Check a random sample of 10 files (or all if less than 10)
    sample_size = min(10, len(all_files))
    if sample_size > 0:
        print(f"\nInspecting random sample of {sample_size} files:")
        sample_files = np.random.choice(all_files, sample_size, replace=False)
        
        success_count = 0
        for file_path in sample_files:
            if inspect_npz_file(file_path):
                success_count += 1
        
        print(f"\nSuccessfully inspected {success_count} out of {sample_size} files")
    
    # Check if 'features' key is more common than 'video_features'
    if len(all_files) > 0:
        print("\nChecking key distribution in all files:")
        key_counts = {}
        for file_path in tqdm(all_files, desc="Scanning files"):
            try:
                with np.load(file_path) as data:
                    for key in data.files:
                        key_counts[key] = key_counts.get(key, 0) + 1
            except:
                pass
        
        print("\nKey distribution:")
        for key, count in sorted(key_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  '{key}' found in {count} files ({count/len(all_files)*100:.1f}%)")

if __name__ == "__main__":
    main()
