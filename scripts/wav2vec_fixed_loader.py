#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed loader for wav2vec features with numerical stability improvements.
This module implements robust loading and normalization of wav2vec embeddings
to prevent NaN and Inf values in training.
"""

import os
import numpy as np
from tqdm import tqdm
import glob
import tensorflow as tf

def improved_load_features(file_path, mean, std, clip_value=5.0):
    """
    Load and normalize wav2vec features with numerical safeguards.
    
    Args:
        file_path: Path to the .npy file containing wav2vec features
        mean: Mean values for normalization (array of shape [feature_dim])
        std: Standard deviation values for normalization (array of shape [feature_dim])
        clip_value: Value to clip features to before normalization
        
    Returns:
        Normalized features (array of shape [time_steps, feature_dim])
    """
    # Load the features and ensure float32 precision
    features = np.load(file_path).astype(np.float32)
    
    # Safety checks and fixes
    if np.isnan(features).any() or np.isinf(features).any():
        # Replace NaN/Inf with zeros
        features = np.nan_to_num(features)
    
    # Clip extreme values that could cause numerical issues
    features = np.clip(features, -clip_value, clip_value)
    
    # Normalize with epsilon for stability
    epsilon = 1e-7
    std_safe = np.maximum(std, epsilon)
    normalized = (features - mean) / std_safe
    
    return normalized

def load_dataset_files(features_dir):
    """
    Load paths to all wav2vec feature files in the dataset.
    
    Args:
        features_dir: Base directory containing the dataset
        
    Returns:
        List of file paths and corresponding emotion labels
    """
    ravdess_dir = os.path.join(features_dir, 'ravdess_features_wav2vec2')
    cremad_dir = os.path.join(features_dir, 'crema_d_features_wav2vec2')
    
    # If not found, try with data/ subdirectory
    if not os.path.exists(ravdess_dir) or not os.path.exists(cremad_dir):
        data_dir = os.path.join(features_dir, 'data')
        if os.path.exists(data_dir):
            ravdess_dir = os.path.join(data_dir, 'ravdess_features_wav2vec2')
            cremad_dir = os.path.join(data_dir, 'crema_d_features_wav2vec2')
            print(f"Looking in data subdirectory: {ravdess_dir} and {cremad_dir}")
    
    # Define emotion mappings
    ravdess_emotion_map = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    
    # Map to 8 emotions
    emotion_to_index = {
        'neutral': 0,
        'calm': 1,
        'happy': 2,
        'sad': 3,
        'angry': 4,
        'fearful': 5,
        'disgust': 6,
        'surprised': 7
    }
    
    # CREMA-D emotion mapping
    cremad_emotion_map = {
        'NEU': 'neutral',
        'HAP': 'happy',
        'SAD': 'sad',
        'ANG': 'angry',
        'FEA': 'fearful',
        'DIS': 'disgust'
    }
    
    # Initialize lists for files and labels
    audio_files = []
    labels = []
    skipped = 0
    
    # Collect RAVDESS files
    if os.path.exists(ravdess_dir):
        ravdess_files = []
        for actor_dir in glob.glob(os.path.join(ravdess_dir, 'Actor_*')):
            ravdess_files.extend(glob.glob(os.path.join(actor_dir, '*.npy')))
            
        print(f"Found {len(ravdess_files)} RAVDESS files")
        
        for file_path in tqdm(ravdess_files, desc="Processing RAVDESS files"):
            try:
                # Extract emotion code from filename (e.g., 03-01-01-01-01-01-01.npy)
                filename = os.path.basename(file_path)
                parts = filename.split('-')
                emotion_code = parts[2]
                
                # Map to emotion label
                emotion = ravdess_emotion_map.get(emotion_code)
                
                if emotion:
                    # Get emotion index
                    emotion_idx = emotion_to_index.get(emotion)
                    
                    if emotion_idx is not None:
                        audio_files.append(file_path)
                        labels.append(emotion_idx)
                    else:
                        skipped += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f"Error processing RAVDESS file {file_path}: {e}")
                skipped += 1
    
    # Collect CREMA-D files
    if os.path.exists(cremad_dir):
        cremad_files = glob.glob(os.path.join(cremad_dir, '*.npy'))
        print(f"Found {len(cremad_files)} CREMA-D files")
        
        for file_path in tqdm(cremad_files, desc="Processing CREMA-D files"):
            try:
                # Extract emotion code from filename (e.g., 1023_DFA_ANG_XX.npy)
                filename = os.path.basename(file_path)
                parts = filename.split('_')
                emotion_code = parts[2]
                
                # Map to common emotion set
                cremad_emotion = cremad_emotion_map.get(emotion_code)
                
                if cremad_emotion:
                    # Get emotion index
                    emotion_idx = emotion_to_index.get(cremad_emotion)
                    
                    if emotion_idx is not None:
                        audio_files.append(file_path)
                        labels.append(emotion_idx)
                    else:
                        skipped += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f"Error processing CREMA-D file {file_path}: {e}")
                skipped += 1
    
    print(f"Loaded {len(audio_files)} files total")
    print(f"Skipped {skipped} files due to parsing errors or excluded emotions")
    
    return audio_files, np.array(labels)

def create_tf_dataset(files, labels, mean, std, batch_size=32, val_split=0.1, max_len=None):
    """
    Create TensorFlow datasets for training and validation.
    
    Args:
        files: List of paths to wav2vec embedding .npy files
        labels: List of corresponding emotion labels
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        batch_size: Batch size for training
        val_split: Validation split ratio
        max_len: Maximum sequence length (None for dynamic)
        
    Returns:
        Training dataset, validation dataset, number of classes, and class weights
    """
    # Determine maximum sequence length if not specified
    if max_len is None:
        # Find the 95th percentile length to avoid outliers
        lengths = []
        for file_path in tqdm(files[:min(1000, len(files))], desc="Calculating sequence lengths"):
            try:
                features = np.load(file_path)
                lengths.append(features.shape[0])
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if lengths:
            max_len = int(np.percentile(lengths, 95))
            print(f"Using max sequence length of {max_len} (95th percentile)")
        else:
            max_len = 150  # Default if no files could be loaded
            print(f"Using default max sequence length of {max_len}")
    
    # Count class occurrences for weighting
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    class_counts = np.bincount(labels, minlength=num_classes)
    
    # Calculate class weights (inverse frequency)
    class_weights = 1.0 / (class_counts + 1e-6)  # Add epsilon to avoid division by zero
    class_weights = class_weights / np.sum(class_weights) * num_classes
    
    # Create stratified train/val split
    indices = np.arange(len(files))
    train_indices, val_indices = [], []
    
    for label in unique_labels:
        label_indices = indices[labels == label]
        np.random.shuffle(label_indices)
        
        split_idx = int(len(label_indices) * (1 - val_split))
        train_indices.extend(label_indices[:split_idx])
        val_indices.extend(label_indices[split_idx:])
    
    # Shuffle the indices
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    
    # Print class distribution
    print("\nClass distribution:")
    for label in unique_labels:
        print(f"  Class {label}: {class_counts[label]} samples, weight: {class_weights[label]:.4f}")
    
    print(f"\nTrain samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    
    # Create robust feature loading function
    def load_and_preprocess(file_idx):
        file_path = files[file_idx]
        label = labels[file_idx]
        
        # Load and normalize features with safeguards
        features = improved_load_features(file_path, mean, std)
        
        # Pad or truncate
        if features.shape[0] > max_len:
            features = features[:max_len]
        
        # One-hot encode label
        one_hot = tf.one_hot(label, num_classes)
        
        return features, one_hot
    
    # Create a function to load for the tf.data pipeline
    def generator_fn(indices):
        for idx in indices:
            try:
                yield load_and_preprocess(idx)
            except Exception as e:
                print(f"Error loading file at index {idx}: {e}")
                continue
    
    # Create TensorFlow datasets with specified output shapes
    output_signature = (
        tf.TensorSpec(shape=(None, mean.shape[0]), dtype=tf.float32),
        tf.TensorSpec(shape=(num_classes,), dtype=tf.float32)
    )
    
    train_ds = tf.data.Dataset.from_generator(
        lambda: generator_fn(train_indices),
        output_signature=output_signature
    )
    
    val_ds = tf.data.Dataset.from_generator(
        lambda: generator_fn(val_indices),
        output_signature=output_signature
    )
    
    # Pad sequences to handle variable lengths
    train_ds = train_ds.padded_batch(
        batch_size,
        padded_shapes=([None, mean.shape[0]], [num_classes]),
        padding_values=(0.0, 0.0)
    )
    val_ds = val_ds.padded_batch(
        batch_size,
        padded_shapes=([None, mean.shape[0]], [num_classes]),
        padding_values=(0.0, 0.0)
    )
    
    # Prefetch for better performance
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, num_classes, class_weights

if __name__ == "__main__":
    # This can be used to test the module standalone
    import argparse
    
    parser = argparse.ArgumentParser(description='Test wav2vec features loading')
    parser.add_argument('--features_dir', type=str, required=True,
                        help='Directory containing wav2vec feature files')
    parser.add_argument('--mean_path', type=str, required=True,
                        help='Path to mean.npy file')
    parser.add_argument('--std_path', type=str, required=True,
                        help='Path to std.npy file')
    
    args = parser.parse_args()
    
    # Load stats
    mean = np.load(args.mean_path)
    std = np.load(args.std_path)
    
    # Test loading
    audio_files, labels = load_dataset_files(args.features_dir)
    
    # Test a few files
    for i in range(min(5, len(audio_files))):
        file_path = audio_files[i]
        print(f"\nLoading {file_path}")
        features = improved_load_features(file_path, mean, std)
        print(f"  Shape: {features.shape}")
        print(f"  Has NaN: {np.isnan(features).any()}")
        print(f"  Has Inf: {np.isinf(features).any()}")
        print(f"  Min/Max: {np.min(features):.4f} / {np.max(features):.4f}")
