#!/usr/bin/env python
"""
Extract CNN Audio Feature Normalization Statistics

This script extracts normalization statistics (mean and standard deviation) from
pre-computed CNN audio feature files (.npy). These statistics are crucial for
normalizing features during training or inference.
"""

import os
import sys
import numpy as np
import glob
import argparse
from tqdm import tqdm
# Assuming feature_normalizer.py is in the same directory or accessible via PYTHONPATH
from feature_normalizer import save_normalization_stats, NORMALIZATION_PATH_TEMPLATE

def process_features_from_npy_dirs(dir_paths, max_files_per_dir=None):
    """Load and process .npy feature files from multiple directories."""
    all_features_list = []
    total_files_processed = 0

    for dir_path in dir_paths:
        if not os.path.isdir(dir_path):
            print(f"Warning: Directory not found: {dir_path}. Skipping.")
            continue

        # Find all .npy files recursively
        search_pattern = os.path.join(dir_path, "**", "*.npy")
        files = glob.glob(search_pattern, recursive=True)

        if not files:
            print(f"Warning: No .npy files found in {dir_path}. Skipping.")
            continue

        if max_files_per_dir and len(files) > max_files_per_dir:
            print(f"Sampling {max_files_per_dir} files from {dir_path} (found {len(files)}).")
            # Consider using random sampling for better representation if needed
            files = files[:max_files_per_dir]
        else:
            print(f"Processing {len(files)} files from {dir_path}")

        dir_features = []
        for file_path in tqdm(files, desc=f"Loading {os.path.basename(dir_path)}"):
            try:
                # Load the .npy file
                # Features are expected to be (time_steps, feature_dim)
                features = np.load(file_path)

                # Basic validation (ensure it's 2D and not empty)
                if features.ndim == 2 and features.shape[0] > 0 and features.shape[1] > 0:
                    dir_features.append(features)
                else:
                    print(f"Warning: Skipping invalid or empty feature file: {file_path} with shape {features.shape}")

            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")

        if dir_features:
            all_features_list.extend(dir_features)
            total_files_processed += len(dir_features)

    if not all_features_list:
        print("Error: No valid features found across all specified directories.")
        return None

    print(f"\nConcatenating features from {total_files_processed} files...")
    # Concatenate along the time axis (axis 0)
    concatenated_features = np.vstack(all_features_list)

    print("Calculating mean and std...")
    # Compute mean and std across all frames (axis 0)
    mean = np.mean(concatenated_features, axis=0, keepdims=True)
    std = np.std(concatenated_features, axis=0, keepdims=True)

    # Avoid division by zero (replace std=0 with 1.0)
    std = np.where(std == 0, 1.0, std)

    print(f"\nComputed normalization statistics from {total_files_processed} files.")
    print(f"Feature dimension: {concatenated_features.shape[1]}")
    print(f"Total frames considered: {concatenated_features.shape[0]}")

    return mean, std

def main():
    parser = argparse.ArgumentParser(description='Extract CNN audio feature normalization statistics from .npy files')
    parser.add_argument('--feature_dirs', nargs='+', required=True,
                        help='List of directories containing the .npy feature files (e.g., data/crema_d_features_cnn_fixed data/ravdess_features_cnn_fixed)')
    parser.add_argument('--max_files_per_dir', type=int, default=None,
                        help='Maximum number of files to process per directory (for faster testing, processes all if None)')
    parser.add_argument('--stats_name', type=str, default="cnn_audio",
                        help='Name used for saving the stats file (e.g., "cnn_audio" -> cnn_audio_normalization_stats.pkl)')
    parser.add_argument('--output_dir', type=str, default="models/dynamic_padding_no_leakage",
                        help='Directory to save the normalization statistics file')

    args = parser.parse_args()

    # Compute stats from the provided directories
    stats = process_features_from_npy_dirs(args.feature_dirs, args.max_files_per_dir)

    if stats is None:
        print("Error: Could not compute normalization statistics.")
        return 1

    mean, std = stats

    # Determine the output path for the .pkl file
    output_stats_path = os.path.join(args.output_dir, f"{args.stats_name}_normalization_stats.pkl")

    # Save the statistics using the function from feature_normalizer
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    save_normalization_stats(mean, std, name=args.stats_name) # This will save to the path defined in feature_normalizer based on name
    print(f"\nSaved {args.stats_name} normalization statistics to {output_stats_path} (via feature_normalizer.py)")

    # Optionally, save raw numpy arrays as well if needed elsewhere
    # np.save(os.path.join(args.output_dir, f"{args.stats_name}_mean.npy"), mean)
    # np.save(os.path.join(args.output_dir, f"{args.stats_name}_std.npy"), std)
    # print(f"Saved raw NumPy arrays: {args.stats_name}_mean.npy, {args.stats_name}_std.npy")
    
    # Print some stats for verification
    print(f"Mean shape: {mean.shape}")
    print(f"Std shape: {std.shape}")
    print(f"Mean range: [{np.min(mean)}, {np.max(mean)}]")
    print(f"Std range: [{np.min(std)}, {np.max(std)}]")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
