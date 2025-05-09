#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check wav2vec feature files for NaN or Inf values that could be causing training issues.
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import glob
import random

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Check wav2vec feature files for NaN/Inf values')
    
    parser.add_argument('--features_dir', type=str, required=True,
                        help='Directory containing wav2vec feature files')
    parser.add_argument('--sample_size', type=int, default=100,
                        help='Number of files to randomly sample (use 0 for all files)')
    parser.add_argument('--mean_path', type=str, default=None,
                        help='Path to pre-computed mean values for stats comparison')
    parser.add_argument('--std_path', type=str, default=None,
                        help='Path to pre-computed std values for stats comparison')
    
    return parser.parse_args()

def find_wav2vec_files(features_dir):
    """Find all .npy files in the given directory structure."""
    # Try with both possible directory structures
    ravdess_dir = os.path.join(features_dir, 'ravdess_features_wav2vec2')
    cremad_dir = os.path.join(features_dir, 'crema_d_features_wav2vec2')
    
    # If not found, try with data/ subdirectory
    if not os.path.exists(ravdess_dir) or not os.path.exists(cremad_dir):
        data_dir = os.path.join(features_dir, 'data')
        if os.path.exists(data_dir):
            ravdess_dir = os.path.join(data_dir, 'ravdess_features_wav2vec2')
            cremad_dir = os.path.join(data_dir, 'crema_d_features_wav2vec2')
            print(f"Looking in data subdirectory: {ravdess_dir} and {cremad_dir}")
    
    # Collect all wav2vec feature files
    all_files = []
    
    # RAVDESS files
    if os.path.exists(ravdess_dir):
        for actor_dir in glob.glob(os.path.join(ravdess_dir, 'Actor_*')):
            all_files.extend(glob.glob(os.path.join(actor_dir, '*.npy')))
        print(f"Found {len(all_files)} RAVDESS files")
    else:
        print(f"Warning: RAVDESS directory not found at {ravdess_dir}")
    
    # CREMA-D files
    cremad_files = []
    if os.path.exists(cremad_dir):
        cremad_files = glob.glob(os.path.join(cremad_dir, '*.npy'))
        all_files.extend(cremad_files)
        print(f"Found {len(cremad_files)} CREMA-D files")
    else:
        print(f"Warning: CREMA-D directory not found at {cremad_dir}")
    
    print(f"Total files found: {len(all_files)}")
    return all_files

def check_file_for_issues(file_path):
    """Check a single .npy file for NaN or Inf values."""
    try:
        data = np.load(file_path)
        
        # Check for NaN
        if np.isnan(data).any():
            return {'file': file_path, 'has_nan': True, 'has_inf': np.isinf(data).any(), 
                    'shape': data.shape, 'min': np.nanmin(data), 'max': np.nanmax(data)}
        
        # Check for Inf
        if np.isinf(data).any():
            return {'file': file_path, 'has_nan': False, 'has_inf': True, 
                    'shape': data.shape, 'min': np.min(data), 'max': np.inf}
        
        # No issues, return basic stats
        return {'file': file_path, 'has_nan': False, 'has_inf': False, 
                'shape': data.shape, 'min': np.min(data), 'max': np.max(data)}
    
    except Exception as e:
        return {'file': file_path, 'error': str(e)}

def compute_sample_stats(files, sample_size=100):
    """Compute statistics on a random sample of files."""
    if sample_size > 0 and sample_size < len(files):
        sample_files = random.sample(files, sample_size)
    else:
        sample_files = files
    
    print(f"Computing statistics on {len(sample_files)} sample files...")
    
    # Count total frames for pre-allocation
    total_frames = 0
    embedding_dim = None
    
    for file_path in tqdm(sample_files, desc="Counting frames"):
        try:
            features = np.load(file_path)
            total_frames += features.shape[0]
            if embedding_dim is None:
                embedding_dim = features.shape[1]
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if total_frames == 0 or embedding_dim is None:
        raise ValueError("No valid frames found in the sample")
    
    # Pre-allocate for all embeddings
    all_features = np.zeros((total_frames, embedding_dim), dtype=np.float32)
    
    # Fill the array
    idx = 0
    for file_path in tqdm(sample_files, desc="Loading embeddings"):
        try:
            features = np.load(file_path)
            if features.shape[0] > 0:
                frames = features.shape[0]
                all_features[idx:idx+frames] = features
                idx += frames
        except Exception as e:
            print(f"Error loading {file_path} for stats: {e}")
    
    # Compute stats
    mean = np.mean(all_features[:idx], axis=0)
    std = np.std(all_features[:idx], axis=0)
    
    # Overall stats (scalar values)
    stats = {
        'global_mean': float(np.mean(mean)),
        'global_std': float(np.mean(std)),
        'min_value': float(np.min(all_features[:idx])),
        'max_value': float(np.max(all_features[:idx])),
        'embedding_dim': embedding_dim,
        'total_frames': idx
    }
    
    return stats, mean, std

def compare_with_saved_stats(computed_mean, computed_std, saved_mean_path, saved_std_path):
    """Compare computed stats with saved normalization values."""
    if not saved_mean_path or not saved_std_path:
        return "No saved stats provided for comparison"
    
    if not os.path.exists(saved_mean_path) or not os.path.exists(saved_std_path):
        return "Saved stats files do not exist"
    
    saved_mean = np.load(saved_mean_path)
    saved_std = np.load(saved_std_path)
    
    if saved_mean.shape != computed_mean.shape or saved_std.shape != computed_std.shape:
        return f"Shape mismatch: Saved mean {saved_mean.shape}, computed mean {computed_mean.shape}"
    
    mean_diff = np.abs(saved_mean - computed_mean)
    std_diff = np.abs(saved_std - computed_std)
    
    mean_max_diff = np.max(mean_diff)
    mean_avg_diff = np.mean(mean_diff)
    std_max_diff = np.max(std_diff)
    std_avg_diff = np.mean(std_diff)
    
    return {
        'mean_max_diff': float(mean_max_diff),
        'mean_avg_diff': float(mean_avg_diff),
        'std_max_diff': float(std_max_diff),
        'std_avg_diff': float(std_avg_diff),
        'should_regenerate': mean_avg_diff > 0.01 or std_avg_diff > 0.01
    }

def main():
    """Main function."""
    args = parse_arguments()
    print(f"Checking wav2vec feature files in {args.features_dir}")
    
    # Find all wav2vec feature files
    all_files = find_wav2vec_files(args.features_dir)
    
    if not all_files:
        print("No files found. Exiting.")
        sys.exit(1)
    
    # Check for NaN/Inf values
    print("\nChecking files for NaN/Inf values...")
    
    if args.sample_size > 0 and args.sample_size < len(all_files):
        files_to_check = random.sample(all_files, args.sample_size)
        print(f"Randomly sampling {args.sample_size} files")
    else:
        files_to_check = all_files
        print(f"Checking all {len(all_files)} files")
    
    issues = []
    file_stats = []
    
    for file_path in tqdm(files_to_check):
        result = check_file_for_issues(file_path)
        file_stats.append(result)
        if result.get('has_nan', False) or result.get('has_inf', False) or 'error' in result:
            issues.append(result)
    
    # Report issues
    if issues:
        print(f"\nFound {len(issues)} files with issues:")
        for issue in issues:
            if 'error' in issue:
                print(f"  Error in {os.path.basename(issue['file'])}: {issue['error']}")
            else:
                print(f"  {os.path.basename(issue['file'])}: NaN={issue['has_nan']}, Inf={issue['has_inf']}, Shape={issue['shape']}")
    else:
        print("\nNo files with NaN/Inf values found.")
    
    # Compute and compare statistics
    try:
        print("\nComputing statistics on sample...")
        stats, computed_mean, computed_std = compute_sample_stats(all_files, args.sample_size)
        
        print("\nSample Statistics:")
        print(f"  Embedding Dimension: {stats['embedding_dim']}")
        print(f"  Total Frames: {stats['total_frames']}")
        print(f"  Global Mean: {stats['global_mean']:.6f}")
        print(f"  Global Std: {stats['global_std']:.6f}")
        print(f"  Min Value: {stats['min_value']:.6f}")
        print(f"  Max Value: {stats['max_value']:.6f}")
        
        # Compare with saved stats if provided
        if args.mean_path and args.std_path:
            print("\nComparing with saved normalization stats...")
            comparison = compare_with_saved_stats(computed_mean, computed_std, args.mean_path, args.std_path)
            
            if isinstance(comparison, dict):
                print(f"  Mean Maximum Difference: {comparison['mean_max_diff']:.6f}")
                print(f"  Mean Average Difference: {comparison['mean_avg_diff']:.6f}")
                print(f"  Std Maximum Difference: {comparison['std_max_diff']:.6f}")
                print(f"  Std Average Difference: {comparison['std_avg_diff']:.6f}")
                print(f"  Should Regenerate Stats: {comparison['should_regenerate']}")
            else:
                print(f"  {comparison}")
    
    except Exception as e:
        print(f"Error computing statistics: {e}")
    
    # Save a summary of file stats
    print("\nSaving file statistics summary...")
    with open("wav2vec_files_check_summary.txt", "w") as f:
        f.write(f"Total files checked: {len(files_to_check)}\n")
        f.write(f"Files with issues: {len(issues)}\n\n")
        
        if issues:
            f.write("Issues found:\n")
            for issue in issues:
                if 'error' in issue:
                    f.write(f"Error in {os.path.basename(issue['file'])}: {issue['error']}\n")
                else:
                    f.write(f"{os.path.basename(issue['file'])}: NaN={issue['has_nan']}, "
                            f"Inf={issue['has_inf']}, Shape={issue['shape']}\n")
        
        if stats:
            f.write("\nSample Statistics:\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
    
    print("Check completed. Results saved to wav2vec_files_check_summary.txt")

if __name__ == "__main__":
    main()
