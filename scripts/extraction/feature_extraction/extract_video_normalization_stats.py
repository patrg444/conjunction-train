#!/usr/bin/env python3
"""
Extract Video Feature Normalization Statistics

This script computes mean and standard deviation over FaceNet video embeddings
used during training, saving them for real-time normalization.
"""

import os
import sys
import glob
import numpy as np
import argparse
from feature_normalizer import save_normalization_stats

def process_video_features(pattern, max_files=500):
    """Load video embedding .npz files and compute normalization stats."""
    files = glob.glob(pattern)
    if not files:
        print(f"Error: No files found matching pattern: {pattern}")
        return None
    if max_files and len(files) > max_files:
        files = files[:max_files]

    print(f"Processing {len(files)} files for video stats: {pattern}")
    all_feats = []
    for path in files:
        try:
            data = np.load(path)
            if 'video_features' not in data:
                continue
            feats = data['video_features']
            if feats.ndim != 2 or feats.shape[0] < 1:
                continue
            all_feats.append(feats)
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
    if not all_feats:
        print("Error: No valid video feature arrays found")
        return None

    concatenated = np.vstack(all_feats)
    mean = np.mean(concatenated, axis=0, keepdims=True)
    std = np.std(concatenated, axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    print(f"Computed video stats: samples={concatenated.shape[0]}, dim={concatenated.shape[1]}")
    return mean, std

def main():
    parser = argparse.ArgumentParser(description="Extract video normalization statistics")
    parser.add_argument("--ravdess", type=str,
                        default="ravdess_features_facenet/*/*.npz",
                        help="Glob pattern for RAVDESS video npz files")
    parser.add_argument("--cremad", type=str,
                        default="crema_d_features_facenet/*.npz",
                        help="Glob pattern for CREMA-D video npz files")
    parser.add_argument("--max-files", type=int, default=500,
                        help="Max number of files per dataset")
    args = parser.parse_args()

    rav_stats = process_video_features(args.ravdess, args.max_files)
    cre_stats = process_video_features(args.cremad, args.max_files)

    if rav_stats and cre_stats:
        mean = (rav_stats[0] + cre_stats[0]) / 2
        std = (rav_stats[1] + cre_stats[1]) / 2
        print("Using combined RAVDESS+CREMA-D stats")
    elif rav_stats:
        mean, std = rav_stats
        print("Using RAVDESS stats")
    elif cre_stats:
        mean, std = cre_stats
        print("Using CREMA-D stats")
    else:
        sys.exit(1)

    # Save the statistics using our unified normalizer
    save_normalization_stats(mean, std, name="video")
    print("Saved video normalization statistics for use with feature_normalizer")
    
    # For backward compatibility, also save as .npy files
    np.save("video_mean.npy", mean)
    np.save("video_std.npy", std)
    print("Also saved video normalization NumPy arrays: video_mean.npy, video_std.npy")

if __name__ == "__main__":
    main()
