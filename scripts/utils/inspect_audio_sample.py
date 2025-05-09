#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility script to inspect wav2vec embedding files (.npy).
Prints tensor shape, min/max values, and other statistics.
"""

import os
import sys
import numpy as np
import argparse
import glob
from tqdm import tqdm

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Inspect wav2vec embedding files')
    
    parser.add_argument('--file', type=str, default=None,
                        help='Path to specific .npy file to inspect')
    parser.add_argument('--dir', type=str, default=None,
                        help='Directory containing .npy files to summarize (e.g., data/ravdess_features_wav2vec2)')
    parser.add_argument('--sample_count', type=int, default=5,
                        help='Number of random samples to inspect when using --dir')
    
    return parser.parse_args()

def inspect_file(file_path):
    """Inspect a single .npy file and print its properties."""
    try:
        # Load the file
        data = np.load(file_path)
        
        # Get basic properties
        shape = data.shape
        dtype = data.dtype
        size_mb = data.nbytes / (1024 * 1024)
        
        # Calculate statistics
        min_val = np.min(data)
        max_val = np.max(data)
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        # Print information
        print("\nFile: {0}".format(os.path.basename(file_path)))
        print("Shape: {0}".format(shape))
        print("Data type: {0}".format(dtype))
        print("Size: {0:.2f} MB".format(size_mb))
        print("Min value: {0:.6f}".format(min_val))
        print("Max value: {0:.6f}".format(max_val))
        print("Mean value: {0:.6f}".format(mean_val))
        print("Std deviation: {0:.6f}".format(std_val))
        
        # Check for NaNs or infinities
        has_nan = np.isnan(data).any()
        has_inf = np.isinf(data).any()
        if has_nan:
            print("WARNING: NaN values detected!")
        if has_inf:
            print("WARNING: Infinite values detected!")
            
        # Show a small sample of the data
        print("\nSample values (first row):")
        if shape[0] > 0 and len(shape) > 1:
            print(data[0, :min(10, shape[1])])
        elif shape[0] > 0:
            print(data[:min(10, shape[0])])
            
        return True
    except Exception as e:
        print(f"Error inspecting file {file_path}: {e}")
        return False

def inspect_directory(directory, sample_count=5):
    """Inspect files in a directory and provide a summary."""
    # Find all .npy files in the directory
    if os.path.exists(os.path.join(directory, 'Actor_*')):
        # For RAVDESS with Actor_* subdirectories
        files = []
        for actor_dir in glob.glob(os.path.join(directory, 'Actor_*')):
            files.extend(glob.glob(os.path.join(actor_dir, '*.npy')))
    else:
        # For flat directory structure
        files = glob.glob(os.path.join(directory, '*.npy'))
    
    if not files:
        print(f"No .npy files found in {directory}")
        return
    
    print(f"Found {len(files)} .npy files in {directory}")
    
    # Collect statistics on all files
    shapes = []
    sizes = []
    
    # Process a subset of files to get statistics
    max_files = min(100, len(files))  # Limit to 100 files for statistics
    for file_path in tqdm(files[:max_files], desc="Analyzing files"):
        try:
            data = np.load(file_path)
            shapes.append(data.shape)
            sizes.append(data.nbytes / (1024 * 1024))
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Print summary statistics
    if shapes:
        print("\nSummary Statistics:")
        print(f"Total files: {len(files)}")
        
        # Extract sequence lengths (first dimension)
        seq_lengths = [s[0] for s in shapes]
        print(f"Sequence length - min: {min(seq_lengths)}, max: {max(seq_lengths)}, mean: {np.mean(seq_lengths):.1f}")
        print(f"File size (MB) - min: {min(sizes):.2f}, max: {max(sizes):.2f}, mean: {np.mean(sizes):.2f}")
        
        if len(shapes) > 0 and len(shapes[0]) > 1:
            # Extract feature dimensions (second dimension)
            feat_dims = [s[1] for s in shapes]
            if all(d == feat_dims[0] for d in feat_dims):
                print(f"Feature dimension: {feat_dims[0]} (consistent across files)")
            else:
                print(f"Feature dimension varies: min={min(feat_dims)}, max={max(feat_dims)}")
        
        # Print 95th percentile sequence length
        p95_length = int(np.percentile(seq_lengths, 95))
        print(f"95th percentile sequence length: {p95_length}")
        
        # Sample a few random files for detailed inspection
        print(f"\nInspecting {sample_count} random samples:")
        np.random.seed(42)  # For reproducibility
        sample_indices = np.random.choice(len(files), min(sample_count, len(files)), replace=False)
        for idx in sample_indices:
            inspect_file(files[idx])
    else:
        print("No valid files found for analysis")

def main():
    args = parse_arguments()
    
    if args.file:
        # Inspect a single file
        if not os.path.exists(args.file):
            print(f"Error: File {args.file} not found")
            return
        
        print(f"Inspecting file: {args.file}")
        inspect_file(args.file)
    
    elif args.dir:
        # Inspect a directory of files
        if not os.path.exists(args.dir):
            print(f"Error: Directory {args.dir} not found")
            return
        
        print(f"Inspecting directory: {args.dir}")
        inspect_directory(args.dir, args.sample_count)
    
    else:
        print("Error: Either --file or --dir must be specified")
        print("Example usage:")
        print("  python inspect_audio_sample.py --file path/to/embedding.npy")
        print("  python inspect_audio_sample.py --dir data/ravdess_features_wav2vec2 --sample_count 3")

if __name__ == "__main__":
    main()
