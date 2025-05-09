#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify dataset splitting for Facenet video-only emotion recognition.
This script focuses on validating the correct splitting of data into
training and validation sets to prevent zero-length dataset issues.
"""

import os
import sys
import numpy as np
import glob
import argparse
from tqdm import tqdm
import random
import re

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from scripts.fixed_video_facenet_generator import FixedVideoFacenetGenerator
except ImportError:
    from fixed_video_facenet_generator import FixedVideoFacenetGenerator

# Emotion mappings
emotion_map = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
emotion_names = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad'}
ravdess_emotion_map = {
    '01': 'NEU', '02': 'NEU', '03': 'HAP', '04': 'SAD', 
    '05': 'ANG', '06': 'FEA', '07': 'DIS', '08': 'FEA'  # Map surprise to fear
}

def extract_emotion_from_filename(file_path):
    """Extract emotion label from filename for both RAVDESS and CREMA-D."""
    base_name = os.path.basename(file_path)
    file_name = os.path.splitext(base_name)[0]
    
    # Remove '_facenet_features' suffix if it exists
    if file_name.endswith('_facenet_features'):
        file_name = file_name[:-len('_facenet_features')]
    
    emotion_code = None
    
    # RAVDESS format: 01-01-01-01-01-01-01.npz (emotion is 3rd segment)
    if "Actor_" in file_path:
        parts = file_name.split('-')
        if len(parts) >= 3:
            emotion_code = ravdess_emotion_map.get(parts[2], None)
    
    # CREMA-D format: 1001_DFA_ANG_XX.npz (emotion is 3rd segment)
    else:
        parts = file_name.split('_')
        if len(parts) >= 3:
            emotion_code = parts[2]
    
    if emotion_code in emotion_map:
        return emotion_map[emotion_code]
    else:
        return None

def get_all_feature_files(ravdess_dir, cremad_dir):
    """Get all feature files from RAVDESS and CREMA-D datasets."""
    all_files = []
    
    # Get RAVDESS files
    if os.path.exists(ravdess_dir):
        for actor_dir in glob.glob(os.path.join(ravdess_dir, "Actor_*")):
            all_files.extend(glob.glob(os.path.join(actor_dir, "*.npz")))
    
    # Get CREMA-D files
    if os.path.exists(cremad_dir):
        all_files.extend(glob.glob(os.path.join(cremad_dir, "*.npz")))
    
    return all_files

def analyze_dataset_split(ravdess_dir, cremad_dir, train_ratio=0.8, seed=42):
    """Analyze dataset splitting to validate training and validation sets."""
    print("=== Dataset Split Validation ===")
    
    # Get all feature files
    all_files = get_all_feature_files(ravdess_dir, cremad_dir)
    print(f"Total files found: {len(all_files)}")
    
    # Extract emotion labels
    valid_files = []
    all_labels = []
    
    for file_path in tqdm(all_files, desc="Processing files"):
        emotion_idx = extract_emotion_from_filename(file_path)
        if emotion_idx is not None:
            try:
                # Just verify that the file can be loaded
                with np.load(file_path, allow_pickle=True) as data:
                    if 'video_features' in data:
                        # File is valid with features and label
                        valid_files.append(file_path)
                        all_labels.append(emotion_idx)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    all_labels = np.array(all_labels)
    print(f"Valid files: {len(valid_files)}")
    
    # Count per-emotion files
    label_counts = {}
    for label in all_labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\nEmotion Distribution in Full Dataset:")
    for emotion_idx, count in sorted(label_counts.items()):
        percentage = (count / len(all_labels)) * 100
        print(f"- {emotion_names[emotion_idx]}: {count} ({percentage:.1f}%)")
    
    # Perform the split like we would in the generator
    np.random.seed(seed)
    indices = np.arange(len(valid_files))
    np.random.shuffle(indices)
    
    train_size = int(len(indices) * train_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_files = [valid_files[i] for i in train_indices]
    val_files = [valid_files[i] for i in val_indices]
    
    train_labels = all_labels[train_indices]
    val_labels = all_labels[val_indices]
    
    print(f"\nDataset Split (Train Ratio: {train_ratio}):")
    print(f"- Training samples: {len(train_files)}")
    print(f"- Validation samples: {len(val_files)}")
    
    # Analyze training set distribution
    train_label_counts = {}
    for label in train_labels:
        train_label_counts[label] = train_label_counts.get(label, 0) + 1
    
    print("\nEmotion Distribution in Training Set:")
    for emotion_idx, count in sorted(train_label_counts.items()):
        percentage = (count / len(train_labels)) * 100
        print(f"- {emotion_names[emotion_idx]}: {count} ({percentage:.1f}%)")
    
    # Analyze validation set distribution
    val_label_counts = {}
    for label in val_labels:
        val_label_counts[label] = val_label_counts.get(label, 0) + 1
    
    print("\nEmotion Distribution in Validation Set:")
    for emotion_idx, count in sorted(val_label_counts.items()):
        percentage = (count / len(val_labels)) * 100
        print(f"- {emotion_names[emotion_idx]}: {count} ({percentage:.1f}%)")
    
    # Initialize generator with the files directly
    print("\nInitializing generator with training files...")
    train_gen = FixedVideoFacenetGenerator(
        video_feature_files=train_files,
        labels=train_labels,
        batch_size=32,
        shuffle=True,
        normalize_features=True
    )
    
    print("Initializing generator with validation files...")
    val_gen = FixedVideoFacenetGenerator(
        video_feature_files=val_files,
        labels=val_labels,
        batch_size=32,
        shuffle=False,
        normalize_features=True
    )
    
    print(f"\nTraining generator batches: {len(train_gen)}")
    print(f"Validation generator batches: {len(val_gen)}")
    
    # Fetch sample batches to verify
    print("\nFetching a sample batch from training generator...")
    train_features, train_batch_labels = next(iter(train_gen))
    
    print("Fetching a sample batch from validation generator...")
    val_features, val_batch_labels = next(iter(val_gen))
    
    print(f"\nTraining batch shape: {train_features.shape}")
    print(f"Validation batch shape: {val_features.shape}")
    
    print("\n=== Dataset Split Validation Successful ===")
    print("Both training and validation generators are properly initialized and producing data.")
    print("No zero-length validation dataset issues detected.")

def main():
    parser = argparse.ArgumentParser(description="Verify dataset splitting for Facenet emotion recognition")
    parser.add_argument("--ravdess_dir", type=str, default="./ravdess_features_facenet", 
                        help="Directory with RAVDESS features")
    parser.add_argument("--cremad_dir", type=str, default="./crema_d_features_facenet", 
                        help="Directory with CREMA-D features")
    parser.add_argument("--train_ratio", type=float, default=0.8, 
                        help="Ratio of data to use for training (default: 0.8)")
    args = parser.parse_args()
    
    analyze_dataset_split(args.ravdess_dir, args.cremad_dir, args.train_ratio)

if __name__ == "__main__":
    main()
