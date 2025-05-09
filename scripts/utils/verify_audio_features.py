#!/usr/bin/env python3
"""Verify the audio features in RAVDESS processed npz files."""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def verify_audio_features(processed_dir, num_files=5, plot_histograms=False):
    """Check the audio features in the processed NPZ files to ensure they contain meaningful data."""
    files = os.listdir(processed_dir)
    npz_files = [f for f in files if f.endswith('.npz')]
    
    if len(npz_files) == 0:
        print(f"No NPZ files found in {processed_dir}")
        return
    
    print(f"Verifying audio features in {min(num_files, len(npz_files))} files from {processed_dir}")
    
    # Track statistics across files
    all_stats = defaultdict(list)
    
    for i, filename in enumerate(npz_files[:num_files]):
        npz_path = os.path.join(processed_dir, filename)
        data = np.load(npz_path)
        
        # Basic info
        print(f"\nFile {i+1}: {filename}")
        
        # Analyze audio features
        if 'audio_features' in data:
            audio_features = data['audio_features']
            
            # Check shape
            print(f"  Audio feature shape: {audio_features.shape}")
            
            # Check for all zeros
            zero_frames = np.all(audio_features == 0, axis=1).sum()
            if zero_frames > 0:
                print(f"  WARNING: {zero_frames} frames ({zero_frames/audio_features.shape[0]:.2%}) have all zero features!")
            else:
                print(f"  No frames with all zero features (good!)")
            
            # Basic stats
            feature_mean = np.mean(audio_features)
            feature_std = np.std(audio_features)
            feature_min = np.min(audio_features)
            feature_max = np.max(audio_features)
            
            print(f"  Feature statistics:")
            print(f"    Mean: {feature_mean:.4f}")
            print(f"    Std:  {feature_std:.4f}")
            print(f"    Min:  {feature_min:.4f}")
            print(f"    Max:  {feature_max:.4f}")
            
            # Track statistics for later comparison
            all_stats['mean'].append(feature_mean)
            all_stats['std'].append(feature_std)
            all_stats['min'].append(feature_min)
            all_stats['max'].append(feature_max)
            
            # Histogram of feature values if requested
            if plot_histograms and i == 0:  # Only for the first file
                plt.figure(figsize=(10, 6))
                plt.hist(audio_features.flatten(), bins=50, alpha=0.75)
                plt.title(f"Audio Feature Distribution: {filename}")
                plt.xlabel("Feature Value")
                plt.ylabel("Count")
                plt.savefig("audio_feature_histogram.png")
                print(f"  Saved histogram to audio_feature_histogram.png")
        else:
            print(f"  ERROR: No audio features found in {filename}")
    
    # Report on consistency across files
    if len(all_stats['mean']) > 1:
        print("\nConsistency across files:")
        print(f"  Mean values range: {min(all_stats['mean']):.4f} to {max(all_stats['mean']):.4f}")
        print(f"  Std values range:  {min(all_stats['std']):.4f} to {max(all_stats['std']):.4f}")
        print(f"  Min values range:  {min(all_stats['min']):.4f} to {max(all_stats['min']):.4f}")
        print(f"  Max values range:  {min(all_stats['max']):.4f} to {max(all_stats['max']):.4f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        processed_dir = sys.argv[1]
    else:
        processed_dir = "ravdess_features"
    
    num_files = 5
    if len(sys.argv) > 2:
        try:
            num_files = int(sys.argv[2])
        except ValueError:
            pass
    
    plot_histograms = False
    if len(sys.argv) > 3 and sys.argv[3].lower() in ['true', 't', '1', 'yes', 'y']:
        plot_histograms = True
    
    verify_audio_features(processed_dir, num_files, plot_histograms)
