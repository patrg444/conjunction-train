#!/usr/bin/env python3
"""
Fix the normalization statistics for emotion recognition training.

This script:
1. Checks if audio_normalization_stats.pkl and video_normalization_stats.pkl exist
2. If not, it computes them from available feature files
3. Ensures they're in the correct location for the training script
"""

import os
import pickle
import numpy as np
import glob
import argparse
from pathlib import Path

def check_normalization_stats(model_dir='models/dynamic_padding_no_leakage'):
    """Check if normalization statistics files exist."""
    audio_stats_path = os.path.join(model_dir, 'audio_normalization_stats.pkl')
    video_stats_path = os.path.join(model_dir, 'video_normalization_stats.pkl')
    
    audio_exists = os.path.exists(audio_stats_path)
    video_exists = os.path.exists(video_stats_path)
    
    print(f"Audio normalization stats: {'EXISTS' if audio_exists else 'MISSING'}")
    print(f"Video normalization stats: {'EXISTS' if video_exists else 'MISSING'}")
    
    return audio_exists, video_exists

def compute_audio_stats(ravdess_pattern='ravdess_features_facenet/*/*.npz', 
                       cremad_pattern='crema_d_features_facenet/*.npz',
                       model_dir='models/dynamic_padding_no_leakage'):
    """Compute audio normalization statistics from feature files."""
    print(f"Computing audio normalization statistics...")
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Find all feature files
    ravdess_files = glob.glob(ravdess_pattern)
    cremad_files = glob.glob(cremad_pattern)
    
    all_files = ravdess_files + cremad_files
    if not all_files:
        print("No feature files found. Cannot compute statistics.")
        return False
    
    print(f"Found {len(all_files)} feature files.")
    
    # Extract audio features
    audio_features = []
    for i, file in enumerate(all_files):
        if i % 100 == 0:
            print(f"Processing file {i+1}/{len(all_files)}...")
        
        try:
            data = np.load(file, allow_pickle=True)
            if 'audio_features' in data:
                audio_feat = data['audio_features']
                if audio_feat.shape[0] > 0 and audio_feat.shape[1] > 0:
                    audio_features.append(audio_feat)
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    if not audio_features:
        print("No valid audio features found.")
        return False
    
    # Compute statistics
    audio_features_flat = np.vstack(audio_features)
    audio_mean = np.mean(audio_features_flat, axis=0)
    audio_std = np.std(audio_features_flat, axis=0)
    
    # Replace zeros in std to avoid division by zero
    audio_std[audio_std < 1e-8] = 1.0
    
    # Save statistics
    stats = {
        'mean': audio_mean,
        'std': audio_std
    }
    
    output_path = os.path.join(model_dir, 'audio_normalization_stats.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(stats, f)
    
    print(f"Audio normalization stats saved to {output_path}")
    return True

def compute_video_stats(ravdess_pattern='ravdess_features_facenet/*/*.npz', 
                       cremad_pattern='crema_d_features_facenet/*.npz',
                       model_dir='models/dynamic_padding_no_leakage'):
    """Compute video normalization statistics from feature files."""
    print(f"Computing video normalization statistics...")
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Find all feature files
    ravdess_files = glob.glob(ravdess_pattern)
    cremad_files = glob.glob(cremad_pattern)
    
    all_files = ravdess_files + cremad_files
    if not all_files:
        print("No feature files found. Cannot compute statistics.")
        return False
    
    print(f"Found {len(all_files)} feature files.")
    
    # Extract video features
    video_features = []
    for i, file in enumerate(all_files):
        if i % 100 == 0:
            print(f"Processing file {i+1}/{len(all_files)}...")
        
        try:
            data = np.load(file, allow_pickle=True)
            if 'video_features' in data:
                video_feat = data['video_features']
                if video_feat.shape[0] > 0 and video_feat.shape[1] > 0:
                    video_features.append(video_feat)
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    if not video_features:
        print("No valid video features found.")
        return False
    
    # Compute statistics
    video_features_flat = np.vstack(video_features)
    video_mean = np.mean(video_features_flat, axis=0)
    video_std = np.std(video_features_flat, axis=0)
    
    # Replace zeros in std to avoid division by zero
    video_std[video_std < 1e-8] = 1.0
    
    # Save statistics
    stats = {
        'mean': video_mean,
        'std': video_std
    }
    
    output_path = os.path.join(model_dir, 'video_normalization_stats.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(stats, f)
    
    print(f"Video normalization stats saved to {output_path}")
    return True

def fix_normalization_stats(model_dir='models/dynamic_padding_no_leakage'):
    """Fix normalization statistics if they're missing."""
    audio_exists, video_exists = check_normalization_stats(model_dir)
    
    # Compute audio stats if missing
    if not audio_exists:
        compute_audio_stats(model_dir=model_dir)
    
    # Compute video stats if missing
    if not video_exists:
        compute_video_stats(model_dir=model_dir)
    
    # Check again to see if we fixed the issues
    audio_exists, video_exists = check_normalization_stats(model_dir)
    
    if audio_exists and video_exists:
        print("Normalization statistics are now all present.")
        return True
    else:
        print("Failed to create all normalization statistics.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fix normalization statistics for emotion recognition training")
    parser.add_argument("--model-dir", type=str, default="models/dynamic_padding_no_leakage",
                        help="Directory for model and normalization files")
    
    args = parser.parse_args()
    fix_normalization_stats(args.model_dir)

if __name__ == "__main__":
    main()
