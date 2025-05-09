#!/usr/bin/env python3
"""
Generate a manifest file for emotion recognition video datasets.

This script creates a CSV manifest file from the RAVDESS and CREMA-D datasets, 
with video file paths, emotion labels, and train/val/test splits.

Usage:
    python generate_video_manifest.py --ravdess_dir /path/to/ravdess_videos 
                                     --crema_dir /path/to/crema_d_videos
                                     --output manifest.csv
                                     --split_ratio 0.7,0.15,0.15
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import random
import re
from tqdm import tqdm

# Emotion labels mapping
RAVDESS_EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# CREMA-D has the emotion in the filename
CREMA_EMOTIONS = {
    'ANG': 'angry',
    'DIS': 'disgust',
    'FEA': 'fearful',
    'HAP': 'happy',
    'NEU': 'neutral',
    'SAD': 'sad'
}

# Common emotions between datasets
COMMON_EMOTIONS = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad']


def extract_emotion_ravdess(filename):
    """Extract emotion from RAVDESS filename."""
    # RAVDESS format: 03-01-06-01-02-01-12.mp4
    # Emotion is the 3rd component (index 2 when split by -)
    try:
        parts = filename.split('-')
        if len(parts) >= 3:
            emotion_code = parts[2]
            return RAVDESS_EMOTIONS.get(emotion_code, None)
    except Exception as e:
        print(f"Error parsing RAVDESS filename {filename}: {e}")
    return None


def extract_emotion_crema(filename):
    """Extract emotion from CREMA-D filename."""
    # CREMA-D format: 1076_MTI_SAD_XX.mp4
    try:
        parts = filename.split('_')
        if len(parts) >= 3:
            emotion_code = parts[2]
            return CREMA_EMOTIONS.get(emotion_code, None)
    except Exception as e:
        print(f"Error parsing CREMA-D filename {filename}: {e}")
    return None


def create_manifest(ravdess_dir, crema_dir, output_file, split_ratio):
    """Create manifest file from video datasets."""
    ravdess_dir = Path(ravdess_dir)
    crema_dir = Path(crema_dir)
    
    data = {
        'path': [],
        'dataset': [],
        'label': [],
        'split': []
    }
    
    # Process RAVDESS
    if ravdess_dir.exists():
        print(f"Processing RAVDESS dataset in {ravdess_dir}...")
        for actor_dir in sorted(ravdess_dir.glob("Actor_*")):
            for video_file in tqdm(sorted(actor_dir.glob("*.mp4")), desc=f"Processing {actor_dir.name}"):
                emotion = extract_emotion_ravdess(video_file.name)
                if emotion and emotion in COMMON_EMOTIONS:
                    data['path'].append(str(video_file))
                    data['dataset'].append('ravdess')
                    data['label'].append(emotion)
                    data['split'].append('')  # Will set later
    else:
        print(f"Warning: RAVDESS directory {ravdess_dir} not found")
    
    # Process CREMA-D
    if crema_dir.exists():
        print(f"Processing CREMA-D dataset in {crema_dir}...")
        for video_file in tqdm(sorted(crema_dir.glob("*.mp4")), desc="Processing CREMA-D"):
            emotion = extract_emotion_crema(video_file.name)
            if emotion and emotion in COMMON_EMOTIONS:
                data['path'].append(str(video_file))
                data['dataset'].append('crema')
                data['label'].append(emotion)
                data['split'].append('')  # Will set later
    else:
        print(f"Warning: CREMA-D directory {crema_dir} not found")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Check if we have data
    if len(df) == 0:
        print("Error: No valid video files found!")
        return None
    
    print(f"Found {len(df)} valid video files")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    # Create stratified splits
    train_ratio, val_ratio, test_ratio = split_ratio
    
    # For reproducibility
    random.seed(42)
    
    # Group by emotion to ensure stratified sampling
    for label in df['label'].unique():
        indices = df[df['label'] == label].index.tolist()  # Convert to list before shuffling
        random.shuffle(indices)
        
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        df.loc[train_indices, 'split'] = 'train'
        df.loc[val_indices, 'split'] = 'val'
        df.loc[test_indices, 'split'] = 'test'
    
    # Verify split distribution
    print("\nSplit distribution:")
    print(df['split'].value_counts())
    
    print("\nClass distribution by split:")
    print(pd.crosstab(df['label'], df['split']))
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"\nManifest saved to {output_file}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate video manifest file")
    parser.add_argument("--ravdess_dir", type=str, default="/home/ubuntu/datasets/ravdess_videos",
                        help="Path to RAVDESS dataset directory")
    parser.add_argument("--crema_dir", type=str, default="/home/ubuntu/datasets/crema_d_videos",
                        help="Path to CREMA-D dataset directory")
    parser.add_argument("--output", type=str, default="/home/ubuntu/datasets/video_manifest.csv",
                        help="Output manifest file path")
    parser.add_argument("--split_ratio", type=str, default="0.7,0.15,0.15",
                        help="Train,val,test split ratio")
    
    args = parser.parse_args()
    
    # Parse split ratio
    split_ratio = list(map(float, args.split_ratio.split(',')))
    assert len(split_ratio) == 3, "Split ratio must have 3 values"
    assert sum(split_ratio) == 1.0, "Split ratio must sum to 1.0"
    
    create_manifest(args.ravdess_dir, args.crema_dir, args.output, split_ratio)


if __name__ == "__main__":
    main()
