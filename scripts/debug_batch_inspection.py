#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch inspection tool for FaceNet features and labels.
This script loads a batch from the fixed generator and performs detailed analysis
to identify potential issues with feature distributions, labels, etc.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import argparse
import glob
from tqdm import tqdm
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

def load_feature_files(ravdess_dir, cremad_dir):
    """
    Load video feature files from RAVDESS and CREMA-D datasets.
    
    This replicates the data loading logic from the training script.
    """
    print("\n=== Loading Feature Files ===")
    
    # Find all feature files
    ravdess_files = []
    if os.path.exists(ravdess_dir):
        for actor_dir in glob.glob(os.path.join(ravdess_dir, "Actor_*")):
            ravdess_files.extend(glob.glob(os.path.join(actor_dir, "*.npz")))
    
    cremad_files = []
    if os.path.exists(cremad_dir):
        cremad_files = glob.glob(os.path.join(cremad_dir, "*.npz"))
    
    print(f"Found {len(ravdess_files)} RAVDESS files")
    print(f"Found {len(cremad_files)} CREMA-D files")
    
    all_files = ravdess_files + cremad_files
    print(f"Total files: {len(all_files)}")
    
    if not all_files:
        raise ValueError("No feature files found! Check the paths.")
    
    return all_files

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

def process_feature_files(feature_files):
    """Process feature files to extract features and labels."""
    valid_files = []
    all_labels = []
    
    print("\nProcessing files:")
    for file_path in tqdm(feature_files):
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
    
    print(f"\nData Loading Summary:")
    print(f"- Valid files: {len(valid_files)}")
    print(f"- Skipped files: {len(feature_files) - len(valid_files)}")
    
    # Print emotion distribution
    label_counts = {}
    for label in all_labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\nEmotion Distribution:")
    for emotion_idx, count in sorted(label_counts.items()):
        percentage = (count / len(all_labels)) * 100
        print(f"- {emotion_names[emotion_idx]}: {count} ({percentage:.1f}%)")
    
    return valid_files, np.array(all_labels)

def analyze_batch_statistics(batch_features, batch_labels):
    """Analyze statistics of a batch of features and labels."""
    print("\n== Batch Feature Statistics ==")
    
    # Feature shape analysis
    print(f"Feature batch shape: {batch_features.shape}")
    
    # Feature value distribution
    print(f"Mean value: {np.mean(batch_features):.6f}")
    print(f"Std deviation: {np.std(batch_features):.6f}")
    print(f"Min value: {np.min(batch_features):.6f}")
    print(f"Max value: {np.max(batch_features):.6f}")
    
    # Check for NaN/Inf values
    nan_count = np.isnan(batch_features).sum()
    inf_count = np.isinf(batch_features).sum()
    print(f"NaN values: {nan_count}")
    print(f"Inf values: {inf_count}")
    
    # Per-sequence analysis
    seq_lengths = [len(seq) for seq in batch_features]
    print(f"Sequence lengths - Min: {min(seq_lengths)}, Max: {max(seq_lengths)}, Mean: {np.mean(seq_lengths):.1f}")
    
    # Label distribution
    print("\n== Batch Label Statistics ==")
    print(f"Label batch shape: {batch_labels.shape}")
    
    unique_labels, counts = np.unique(batch_labels, return_counts=True)
    label_dist = {emotion_names[int(label)]: count for label, count in zip(unique_labels, counts)}
    print("Label distribution:", label_dist)
    
    # Calculate per-emotion feature statistics
    print("\n== Per-Emotion Feature Statistics ==")
    for emotion_idx in range(6):
        emotion_mask = batch_labels == emotion_idx
        if np.any(emotion_mask):
            emotion_features = np.concatenate([feat for feat, label in zip(batch_features, batch_labels) 
                                              if label == emotion_idx])
            print(f"{emotion_names[emotion_idx]}:")
            print(f"  Mean: {np.mean(emotion_features):.6f}")
            print(f"  Std: {np.std(emotion_features):.6f}")
            print(f"  Min: {np.min(emotion_features):.6f}")
            print(f"  Max: {np.max(emotion_features):.6f}")
    
    return seq_lengths

def plot_batch_distributions(batch_features, batch_labels, seq_lengths):
    """Generate plots for batch feature distributions."""
    # Create output directory
    os.makedirs("batch_analysis", exist_ok=True)
    
    # Plot feature histograms
    plt.figure(figsize=(10, 6))
    # Convert TensorFlow tensors to numpy arrays if needed
    flattened_features = []
    for f in batch_features:
        if hasattr(f, 'numpy'):
            flattened_features.append(f.numpy().flatten())
        else:
            flattened_features.append(f.flatten())
    
    flattened_features = np.concatenate(flattened_features)
    plt.hist(flattened_features, bins=50, alpha=0.7)
    plt.title('Distribution of Feature Values')
    plt.xlabel('Feature Value')
    plt.ylabel('Count')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("batch_analysis/feature_distribution.png")
    
    # Plot sequence length distribution
    plt.figure(figsize=(10, 6))
    plt.hist(seq_lengths, bins=20, alpha=0.7)
    plt.title('Distribution of Sequence Lengths')
    plt.xlabel('Sequence Length (Frames)')
    plt.ylabel('Count')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("batch_analysis/sequence_lengths.png")
    
    # Plot label distribution
    plt.figure(figsize=(10, 6))
    unique_labels, counts = np.unique(batch_labels, return_counts=True)
    sorted_indices = np.argsort(unique_labels)
    plt.bar([emotion_names[int(label)] for label in unique_labels[sorted_indices]], 
            counts[sorted_indices], alpha=0.7)
    plt.title('Distribution of Emotion Labels in Batch')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("batch_analysis/label_distribution.png")
    
    # Plot feature dimensions
    feature_dim = batch_features[0].shape[1]
    if feature_dim <= 512:  # Only for reasonably-sized feature vectors
        # Sample a few frames from each sequence for mean/std calculation
        sampled_frames = []
        for seq in batch_features:
            if len(seq) > 0:
                # Sample up to 5 frames from each sequence
                indices = np.linspace(0, len(seq)-1, min(5, len(seq))).astype(int)
                sampled_frames.extend([seq[i] for i in indices])
        
        if sampled_frames:
            sampled_frames = np.array(sampled_frames)
            
            # Calculate mean and std per dimension
            dim_means = np.mean(sampled_frames, axis=0)
            dim_stds = np.std(sampled_frames, axis=0)
            
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.plot(dim_means)
            plt.title('Mean Value per Feature Dimension')
            plt.xlabel('Feature Dimension')
            plt.ylabel('Mean Value')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.subplot(1, 2, 2)
            plt.plot(dim_stds)
            plt.title('Std Deviation per Feature Dimension')
            plt.xlabel('Feature Dimension')
            plt.ylabel('Std Deviation')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig("batch_analysis/per_dimension_stats.png")
    
    # Plot per-frame features for a random sequence in the batch
    if len(batch_features) > 0:
        random_idx = random.randint(0, len(batch_features) - 1)
        seq_features = batch_features[random_idx]
        
        # Convert to numpy if it's a tensor
        if hasattr(seq_features, 'numpy'):
            seq_features = seq_features.numpy()
            
        if hasattr(batch_labels, 'numpy'):
            labels_np = batch_labels.numpy()
            emotion = emotion_names[int(labels_np[random_idx])]
        else:
            emotion = emotion_names[int(batch_labels[random_idx])]
        
        plt.figure(figsize=(12, 6))
        
        # Choose a subset of dimensions to visualize
        dims_to_plot = min(10, seq_features.shape[1])
        random_dims = random.sample(range(seq_features.shape[1]), dims_to_plot)
        
        for i, dim in enumerate(random_dims):
            plt.plot(seq_features[:, dim], label=f'Dim {dim}')
        
        plt.title(f'Feature Values Over Time - {emotion} Sample')
        plt.xlabel('Frame')
        plt.ylabel('Feature Value')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("batch_analysis/features_over_time.png")
    
    print(f"\nPlots saved to batch_analysis/ directory")

def check_batch_gradient_flow(model, batch_features, batch_labels):
    """Test gradient flow through the model for a batch."""
    # Prepare inputs for the model (might need padding)
    max_len = max(len(seq) for seq in batch_features)
    padded_features = np.zeros((len(batch_features), max_len, batch_features[0].shape[1]))
    
    for i, seq in enumerate(batch_features):
        padded_features[i, :len(seq), :] = seq
    
    # Convert to tensors
    features_tensor = tf.convert_to_tensor(padded_features, dtype=tf.float32)
    labels_tensor = tf.convert_to_tensor(batch_labels, dtype=tf.int32)
    
    # Define a loss function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    # Define a gradient tape to watch the gradients
    with tf.GradientTape() as tape:
        # Get model predictions
        predictions = model(features_tensor, training=True)
        
        # Calculate loss
        loss = loss_fn(labels_tensor, predictions)
    
    # Get the gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    print("\n== Gradient Flow Check ==")
    
    # Check for zero, NaN, or inf gradients
    has_nans = False
    has_zeros = False
    for i, (var, grad) in enumerate(zip(model.trainable_variables, gradients)):
        if grad is None:
            print(f"Layer {i}, variable {var.name}: No gradient")
            has_zeros = True
        else:
            if np.isnan(grad.numpy()).any():
                print(f"Layer {i}, variable {var.name}: Has NaN gradients")
                has_nans = True
            if np.isinf(grad.numpy()).any():
                print(f"Layer {i}, variable {var.name}: Has Inf gradients")
                has_nans = True
            
            grad_norm = tf.norm(grad).numpy()
            if grad_norm == 0:
                print(f"Layer {i}, variable {var.name}: All zero gradients (norm = 0)")
                has_zeros = True
            elif grad_norm < 1e-7:
                print(f"Layer {i}, variable {var.name}: Very small gradient norm = {grad_norm:.10f}")
    
    if not has_nans and not has_zeros:
        print("Gradient flow check: PASSED - All gradients are flowing properly")
    else:
        if has_nans:
            print("Gradient flow check: FAILED - Found NaN or Inf gradients")
        if has_zeros:
            print("Gradient flow check: WARNING - Found some zero gradients")
    
    # Compute gradient statistics
    grad_norms = [tf.norm(grad).numpy() for grad in gradients if grad is not None]
    if grad_norms:
        print(f"Mean gradient norm: {np.mean(grad_norms):.6f}")
        print(f"Min gradient norm: {min(grad_norms):.6f}")
        print(f"Max gradient norm: {max(grad_norms):.6f}")

def main():
    """Main function to inspect batches."""
    parser = argparse.ArgumentParser(description="Debug batch inspection for FaceNet video features")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size to inspect")
    parser.add_argument("--ravdess_dir", type=str, default="./ravdess_features_facenet", 
                        help="Directory with RAVDESS features")
    parser.add_argument("--cremad_dir", type=str, default="./crema_d_features_facenet", 
                        help="Directory with CREMA-D features")
    parser.add_argument("--check_gradients", action="store_true", 
                        help="Check gradient flow through a model (requires trained model)")
    parser.add_argument("--model_path", type=str, 
                        help="Path to trained model to check gradients (required if --check_gradients is set)")
    args = parser.parse_args()
    
    print("=== FaceNet Video Batch Inspector ===")
    
    # Load and process files (like in training script)
    all_files = load_feature_files(args.ravdess_dir, args.cremad_dir)
    valid_files, all_labels = process_feature_files(all_files)
    
    print("\nInitializing video data generator...")
    # Use a subset of files for analysis
    if len(valid_files) > 100:
        sample_indices = random.sample(range(len(valid_files)), 100)
        sample_files = [valid_files[i] for i in sample_indices]
        sample_labels = all_labels[sample_indices]
    else:
        sample_files = valid_files
        sample_labels = all_labels
    
    # Initialize generator with the actual parameters it expects
    train_gen = FixedVideoFacenetGenerator(
        video_feature_files=sample_files,
        labels=sample_labels,
        batch_size=args.batch_size,
        shuffle=True,
        normalize_features=True
    )
    
    # Get a sample batch
    print("Fetching a random batch...")
    batch_features, batch_labels = next(iter(train_gen))
    
    # Analyze batch statistics
    seq_lengths = analyze_batch_statistics(batch_features, batch_labels)
    
    # Generate plots
    plot_batch_distributions(batch_features, batch_labels, seq_lengths)
    
    # Check gradient flow if requested
    if args.check_gradients:
        if not args.model_path:
            print("Error: --model_path is required when --check_gradients is set")
            return
        
        print(f"\nLoading model from {args.model_path} to check gradient flow...")
        try:
            model = tf.keras.models.load_model(args.model_path)
            check_batch_gradient_flow(model, batch_features, batch_labels)
        except Exception as e:
            print(f"Error loading or evaluating model: {e}")
    
    print("\n=== Batch Inspection Complete ===")

if __name__ == "__main__":
    main()
