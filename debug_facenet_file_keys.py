#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script to inspect the actual content of Facenet NPZ files.
This will help identify the root cause of the loading issues in the training script.
"""

import os
import sys
import numpy as np
import glob
import argparse
from tqdm import tqdm

def inspect_npz_file(file_path):
    """Inspect the contents of an NPZ file, showing keys and shapes."""
    try:
        with np.load(file_path, allow_pickle=True) as data:
            print(f"\n=== Inspecting: {os.path.basename(file_path)} ===")
            print(f"Available keys: {list(data.keys())}")
            
            # Check each key for shape information
            for key in data.keys():
                try:
                    array = data[key]
                    if isinstance(array, np.ndarray):
                        print(f"- {key}: shape={array.shape}, dtype={array.dtype}, min={np.min(array):.4f}, max={np.max(array):.4f}")
                    else:
                        print(f"- {key}: type={type(array)}, not a numpy array")
                except Exception as e:
                    print(f"- {key}: Error examining array: {e}")
            
            # Check expected key specifically
            if 'video_features' in data:
                features = data['video_features']
                if features.shape[0] > 0:
                    print(f"✓ 'video_features' key found with valid shape: {features.shape}")
                else:
                    print(f"✗ 'video_features' key found but has empty shape: {features.shape}")
            else:
                # If 'video_features' not found, look for alternative keys that might contain the features
                possible_feature_keys = ['features', 'facenet_features', 'embeddings']
                found_alternative = False
                for key in possible_feature_keys:
                    if key in data:
                        features = data[key]
                        print(f"⚠ Alternative key '{key}' found with shape: {features.shape}")
                        found_alternative = True
                
                if not found_alternative:
                    print(f"✗ 'video_features' key not found and no common alternatives detected")
                    
            return data.keys()
    except Exception as e:
        print(f"\n=== Error inspecting {os.path.basename(file_path)}: {e} ===")
        return []

def rename_key_in_npz(file_path, source_key, target_key):
    """Rename a key in an NPZ file by creating a new file with the target key."""
    try:
        # Load the original data
        with np.load(file_path, allow_pickle=True) as data:
            if source_key not in data:
                print(f"Source key '{source_key}' not found in {file_path}")
                return False
            
            # Create a dictionary with all existing data
            data_dict = {k: data[k] for k in data.keys()}
            
            # Replace the key
            data_dict[target_key] = data_dict.pop(source_key)
            
        # Save to a temporary file
        temp_path = file_path + '.temp'
        np.savez_compressed(temp_path, **data_dict)
        
        # Replace the original file
        os.replace(temp_path, file_path)
        
        print(f"Successfully renamed key '{source_key}' to '{target_key}' in {file_path}")
        return True
    except Exception as e:
        print(f"Error renaming key in {file_path}: {e}")
        return False

def summarize_keys_across_files(found_keys, file_count):
    """Summarize the keys found across all files."""
    print("\n=== Key Statistics Across All Files ===")
    
    key_counts = {}
    for keys in found_keys:
        for key in keys:
            key_counts[key] = key_counts.get(key, 0) + 1
    
    # Display statistics
    for key, count in sorted(key_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / file_count) * 100
        print(f"Key: '{key}' - Found in {count}/{file_count} files ({percentage:.1f}%)")
    
    # Check for consistency
    if 'video_features' in key_counts:
        vid_feat_count = key_counts['video_features']
        if vid_feat_count == file_count:
            print("\n✓ 'video_features' key consistently found in all files")
        else:
            print(f"\n⚠ 'video_features' key found in only {vid_feat_count}/{file_count} files ({(vid_feat_count/file_count)*100:.1f}%)")

def fix_npz_files(directory, source_key, target_key='video_features', dry_run=True):
    """Fix the NPZ files by renaming a key to 'video_features'."""
    files = glob.glob(os.path.join(directory, "**", "*.npz"), recursive=True)
    print(f"Found {len(files)} NPZ files in {directory}")
    
    files_to_fix = []
    
    # First, scan all files to find those that need fixing
    for file_path in tqdm(files, desc="Scanning files"):
        try:
            with np.load(file_path, allow_pickle=True) as data:
                has_source = source_key in data
                has_target = target_key in data
                
                if has_source and not has_target:
                    files_to_fix.append(file_path)
        except Exception:
            continue
    
    print(f"Found {len(files_to_fix)} files that need to be fixed")
    
    # Fix the files if not a dry run
    if not dry_run and files_to_fix:
        fixed_count = 0
        for file_path in tqdm(files_to_fix, desc="Fixing files"):
            if rename_key_in_npz(file_path, source_key, target_key):
                fixed_count += 1
        
        print(f"Successfully fixed {fixed_count}/{len(files_to_fix)} files")
    elif dry_run and files_to_fix:
        print("Dry run completed. Use --fix to actually rename the keys")
    
    return len(files_to_fix)

def main():
    parser = argparse.ArgumentParser(description='Debug and fix Facenet NPZ files')
    parser.add_argument('--ravdess_dir', type=str, default='/home/ubuntu/emotion-recognition/ravdess_features_facenet',
                      help='Directory containing RAVDESS Facenet features')
    parser.add_argument('--cremad_dir', type=str, default='/home/ubuntu/emotion-recognition/crema_d_features_facenet',
                      help='Directory containing CREMA-D Facenet features')
    parser.add_argument('--samples', type=int, default=5, 
                      help='Number of sample files to inspect')
    parser.add_argument('--fix', action='store_true',
                      help='Fix the NPZ files by renaming keys')
    parser.add_argument('--source_key', type=str, default='features',
                      help='Source key to rename to "video_features"')
    
    args = parser.parse_args()
    
    # Use local paths if the directories don't exist
    if not os.path.exists(args.ravdess_dir):
        args.ravdess_dir = './ravdess_features_facenet'
        print(f"Using local path for RAVDESS: {args.ravdess_dir}")
    
    if not os.path.exists(args.cremad_dir):
        args.cremad_dir = './crema_d_features_facenet'
        print(f"Using local path for CREMA-D: {args.cremad_dir}")
    
    # Find the NPZ files
    ravdess_files = glob.glob(os.path.join(args.ravdess_dir, "Actor_*", "*.npz"))
    cremad_files = glob.glob(os.path.join(args.cremad_dir, "*.npz"))
    
    print(f"Found {len(ravdess_files)} RAVDESS NPZ files")
    print(f"Found {len(cremad_files)} CREMA-D NPZ files")
    
    all_files = ravdess_files + cremad_files
    if not all_files:
        print("No NPZ files found. Check the directories.")
        return
    
    # If we need to fix the files, do it first
    if args.fix:
        print("\n=== Fixing RAVDESS files ===")
        fix_npz_files(args.ravdess_dir, args.source_key, dry_run=False)
        
        print("\n=== Fixing CREMA-D files ===")
        fix_npz_files(args.cremad_dir, args.source_key, dry_run=False)
    
    # Sample files from both datasets
    sample_count = min(args.samples, len(all_files))
    ravdess_samples = np.random.choice(ravdess_files, min(sample_count, len(ravdess_files)), replace=False) if ravdess_files else []
    cremad_samples = np.random.choice(cremad_files, min(sample_count, len(cremad_files)), replace=False) if cremad_files else []
    
    print(f"\n=== Inspecting {len(ravdess_samples)} RAVDESS samples ===")
    ravdess_keys = []
    for file_path in ravdess_samples:
        keys = inspect_npz_file(file_path)
        ravdess_keys.append(keys)
    
    print(f"\n=== Inspecting {len(cremad_samples)} CREMA-D samples ===")
    cremad_keys = []
    for file_path in cremad_samples:
        keys = inspect_npz_file(file_path)
        cremad_keys.append(keys)
    
    # Summarize findings
    if ravdess_samples:
        print("\n--- RAVDESS Key Summary ---")
        summarize_keys_across_files(ravdess_keys, len(ravdess_samples))
    
    if cremad_samples:
        print("\n--- CREMA-D Key Summary ---")
        summarize_keys_across_files(cremad_keys, len(cremad_samples))
    
    print("\n=== Overall Diagnosis ===")
    
    all_keys = ravdess_keys + cremad_keys
    all_keys_flat = [key for keys in all_keys for key in keys]
    has_video_features = 'video_features' in all_keys_flat
    has_features = 'features' in all_keys_flat
    
    if not has_video_features and has_features:
        print("✗ Problem Detected: Files contain 'features' key instead of 'video_features'")
        print("  ↳ Fix: Run this script with --fix to rename 'features' to 'video_features'")
    elif not has_video_features and not has_features:
        print("✗ Problem Detected: Neither 'video_features' nor 'features' keys found")
        print("  ↳ Check: The NPZ files might be corrupted or have a different structure")
    elif has_video_features:
        print("✓ 'video_features' key found in sample files")
        
    # Provide recommendation based on findings
    if not has_video_features:
        print("\nRecommended Action:")
        print(f"Run: python debug_facenet_file_keys.py --fix --source_key='{args.source_key}'")
    
if __name__ == "__main__":
    main()
