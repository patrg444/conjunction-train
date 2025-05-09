#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnose numerical stability issues with wav2vec features.
This tool loads features similar to the training script but applies various fixes
to track down when/where NaN values are being introduced.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import glob
from tqdm import tqdm
import argparse
import random

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Diagnose wav2vec feature numerical issues')
    
    parser.add_argument('--features_dir', type=str, required=True,
                        help='Directory containing wav2vec feature files')
    parser.add_argument('--mean_path', type=str, default=None,
                        help='Path to pre-computed mean values for normalization')
    parser.add_argument('--std_path', type=str, default=None,
                        help='Path to pre-computed std values for normalization')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for testing')
    parser.add_argument('--fix_mode', type=str, default='clip',
                        choices=['none', 'clip', 'replace', 'remove'],
                        help='Method to fix potential numerical issues')
    parser.add_argument('--clip_value', type=float, default=10.0,
                        help='Value to clip features to (applies only if fix_mode is clip)')
    
    return parser.parse_args()

def find_wav2vec_files(features_dir):
    """Find all wav2vec feature files."""
    ravdess_dir = os.path.join(features_dir, 'ravdess_features_wav2vec2')
    cremad_dir = os.path.join(features_dir, 'crema_d_features_wav2vec2')
    
    # If not found, try with data/ subdirectory
    if not os.path.exists(ravdess_dir) or not os.path.exists(cremad_dir):
        data_dir = os.path.join(features_dir, 'data')
        if os.path.exists(data_dir):
            ravdess_dir = os.path.join(data_dir, 'ravdess_features_wav2vec2')
            cremad_dir = os.path.join(data_dir, 'crema_d_features_wav2vec2')
            print(f"Looking in data subdirectory: {ravdess_dir} and {cremad_dir}")
    
    all_files = []
    
    # RAVDESS files
    if os.path.exists(ravdess_dir):
        for actor_dir in glob.glob(os.path.join(ravdess_dir, 'Actor_*')):
            all_files.extend(glob.glob(os.path.join(actor_dir, '*.npy')))
        print(f"Found {len(all_files)} RAVDESS files")
    
    # CREMA-D files
    cremad_files = []
    if os.path.exists(cremad_dir):
        cremad_files = glob.glob(os.path.join(cremad_dir, '*.npy'))
        all_files.extend(cremad_files)
        print(f"Found {len(cremad_files)} CREMA-D files")
    
    print(f"Total files found: {len(all_files)}")
    return all_files

def load_and_analyze_features(file_path, fix_mode='none', clip_value=10.0):
    """Load features and analyze/fix potential numerical issues."""
    try:
        features = np.load(file_path).astype(np.float32)
        
        # Original stats
        stats = {
            'min': float(np.min(features)),
            'max': float(np.max(features)),
            'mean': float(np.mean(features)),
            'std': float(np.std(features)),
            'has_nan': bool(np.isnan(features).any()),
            'has_inf': bool(np.isinf(features).any()),
            'zeros': int(np.sum(features == 0)),
            'extreme_values': int(np.sum(np.abs(features) > 10.0)),
            'shape': features.shape
        }
        
        # Apply fixes based on mode
        if fix_mode == 'clip':
            # Clip values to a safe range
            features = np.clip(features, -clip_value, clip_value)
        elif fix_mode == 'replace':
            # Replace NaN/Inf values with zeros
            features = np.nan_to_num(features)
            # Also clip extreme values
            features = np.clip(features, -clip_value, clip_value)
        elif fix_mode == 'remove':
            # Find NaN/Inf rows and create a mask
            bad_rows = np.isnan(features).any(axis=1) | np.isinf(features).any(axis=1)
            # Also identify rows with extreme values
            extreme_rows = np.abs(features).max(axis=1) > clip_value
            # Combine masks
            bad_rows = bad_rows | extreme_rows
            # Keep only good rows
            if bad_rows.any():
                features = features[~bad_rows]
                stats['removed_rows'] = int(np.sum(bad_rows))
            else:
                stats['removed_rows'] = 0
        
        # Post-fix stats
        stats['fixed_min'] = float(np.min(features))
        stats['fixed_max'] = float(np.max(features))
        stats['fixed_mean'] = float(np.mean(features))
        stats['fixed_std'] = float(np.std(features))
        stats['fixed_has_nan'] = bool(np.isnan(features).any())
        stats['fixed_has_inf'] = bool(np.isinf(features).any())
        stats['fixed_zeros'] = int(np.sum(features == 0))
        stats['fixed_extreme_values'] = int(np.sum(np.abs(features) > 10.0))
        stats['fixed_shape'] = features.shape
        
        return features, stats
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, {'error': str(e)}

def simulate_preprocessing(files, mean_path, std_path, batch_size=32, fix_mode='clip', clip_value=10.0):
    """Simulate the preprocessing pipeline to detect potential issues."""
    # Load or compute normalization stats
    if mean_path and std_path and os.path.exists(mean_path) and os.path.exists(std_path):
        print(f"Loading pre-computed normalization stats from {mean_path} and {std_path}")
        mean = np.load(mean_path)
        std = np.load(std_path)
    else:
        print("Error: Normalization stats are required. Please provide mean_path and std_path.")
        return
    
    # Shuffle files
    random.shuffle(files)
    
    # Take a subset for testing
    test_files = files[:min(100, len(files))]
    
    # Process files and collect stats
    all_stats = []
    fixed_features = []
    
    print(f"Processing {len(test_files)} files with fix_mode={fix_mode}, clip_value={clip_value}...")
    for file_path in tqdm(test_files):
        features, stats = load_and_analyze_features(file_path, fix_mode, clip_value)
        if features is not None:
            all_stats.append(stats)
            fixed_features.append(features)
    
    # Analyze the issues
    problematic_files = [s for s in all_stats if 'error' in s or s.get('has_nan', False) or s.get('has_inf', False)]
    extreme_value_files = [s for s in all_stats if 'error' not in s and s.get('extreme_values', 0) > 0]
    
    print("\nSummary:")
    print(f"Files with errors or NaN/Inf values: {len(problematic_files)}")
    print(f"Files with extreme values (>10.0): {len(extreme_value_files)}")
    
    # Simulate batch normalization
    if fixed_features:
        print("\nTesting batch processing...")
        batch_start = 0
        max_batches = 5
        for batch_idx in range(min(max_batches, len(fixed_features) // batch_size)):
            batch_features = fixed_features[batch_start:batch_start+batch_size]
            
            # Pad sequences to the same length for batching
            max_length = max(f.shape[0] for f in batch_features)
            padded_batch = np.zeros((len(batch_features), max_length, mean.shape[0]), dtype=np.float32)
            
            for i, feature in enumerate(batch_features):
                padded_batch[i, :feature.shape[0], :] = feature
            
            # Normalize using mean/std
            normalized_batch = (padded_batch - mean) / std
            
            # Check for NaN/Inf after normalization
            has_nan = np.isnan(normalized_batch).any()
            has_inf = np.isinf(normalized_batch).any()
            
            print(f"  Batch {batch_idx+1}: Shape={normalized_batch.shape}, "
                 f"NaN={has_nan}, Inf={has_inf}, "
                 f"Min={normalized_batch.min():.6f}, Max={normalized_batch.max():.6f}")
            
            batch_start += batch_size
        
        # Test forward pass with a simple model
        print("\nSimulating forward pass with a simple model...")
        # Create a very small LSTM model
        input_shape = (None, mean.shape[0])
        inputs = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(7, activation='softmax')(x)
        
        model = tf.keras.models.Model(inputs=inputs, outputs=x)
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy']
        )
        
        # Test with a single batch
        batch_features = fixed_features[:batch_size]
        max_length = max(f.shape[0] for f in batch_features)
        padded_batch = np.zeros((len(batch_features), max_length, mean.shape[0]), dtype=np.float32)
        
        for i, feature in enumerate(batch_features):
            padded_batch[i, :feature.shape[0], :] = feature
        
        # Normalize
        normalized_batch = (padded_batch - mean) / std
        
        # Create fake labels
        fake_labels = np.random.randint(0, 7, size=len(batch_features))
        one_hot_labels = tf.keras.utils.to_categorical(fake_labels, num_classes=7)
        
        # Run prediction
        try:
            with tf.GradientTape() as tape:
                predictions = model(normalized_batch, training=True)
                loss = model.loss(one_hot_labels, predictions)
                
            print(f"  Forward pass successful: Loss={loss.numpy():.6f}")
            print(f"  Predictions shape: {predictions.shape}")
            print(f"  Predictions sample: {predictions[0, :3].numpy()}")
            
            # Check gradients
            grads = tape.gradient(loss, model.trainable_variables)
            grads_ok = True
            for g in grads:
                if g is not None and (tf.math.reduce_any(tf.math.is_nan(g)) or tf.math.reduce_any(tf.math.is_inf(g))):
                    grads_ok = False
                    break
            
            print(f"  Gradients OK: {grads_ok}")
            
        except Exception as e:
            print(f"  Forward pass failed: {e}")
    
    print("\nRecommendations:")
    if problematic_files:
        print(f"- Use fix_mode='replace' or fix_mode='remove' to handle the {len(problematic_files)} problematic files")
    if extreme_value_files:
        print(f"- Use fix_mode='clip' with clip_value={clip_value} to handle the {len(extreme_value_files)} files with extreme values")
    
    # Generate improved preprocessing code
    print("\nImproved feature loading code:")
    print("""
def improved_load_features(file_path, mean, std, clip_value=10.0):
    features = np.load(file_path).astype(np.float32)
    
    # Safety checks and fixes
    if np.isnan(features).any() or np.isinf(features).any():
        # Option 1: Replace NaN/Inf with zeros
        features = np.nan_to_num(features)
        
        # Option 2: Remove problematic rows (uncomment if needed)
        # bad_rows = np.isnan(features).any(axis=1) | np.isinf(features).any(axis=1)
        # if bad_rows.any():
        #     features = features[~bad_rows]
    
    # Clip extreme values that could cause numerical issues
    features = np.clip(features, -clip_value, clip_value)
    
    # Normalize with epsilon for stability
    epsilon = 1e-7
    std_safe = np.maximum(std, epsilon)
    normalized = (features - mean) / std_safe
    
    return normalized
""")

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Find wav2vec files
    files = find_wav2vec_files(args.features_dir)
    
    if not files:
        print("No files found. Exiting.")
        sys.exit(1)
    
    # Simulate preprocessing
    simulate_preprocessing(
        files,
        args.mean_path,
        args.std_path,
        args.batch_size,
        args.fix_mode,
        args.clip_value
    )

if __name__ == "__main__":
    main()
