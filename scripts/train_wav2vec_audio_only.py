#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio-only emotion recognition model using wav2vec2 embeddings.
This script removes any video feature dependencies and operates purely on wav2vec embeddings.
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf

# Explicitly set to float32 for numerical stability (NOT mixed_float16)
tf.keras.mixed_precision.set_global_policy("float32")

from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Bidirectional, BatchNormalization
from tensorflow.keras.layers import Masking, GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau, Callback
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
import time
import glob
import random
import json
import math
from tqdm import tqdm
import sklearn.metrics as metrics
from datetime import datetime

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

class F1ScoreCallback(Callback):
    """
    Callback to calculate F1 score after each epoch
    """
    def __init__(self, validation_data):
        super(F1ScoreCallback, self).__init__()
        self.validation_data = validation_data
        self.val_f1s = []
        
    def on_epoch_end(self, epoch, logs={}):
        # Get predictions
        val_x, val_y = self.validation_data
        val_predict = np.argmax(self.model.predict(val_x), axis=1)
        val_true = np.argmax(val_y, axis=1)
        
        # Calculate F1 score (macro avg)
        _val_f1 = metrics.f1_score(val_true, val_predict, average='macro')
        self.val_f1s.append(_val_f1)
        logs['val_f1'] = _val_f1
        print(f" — val_f1: {_val_f1:.4f}")

class WarmUpCosineDecayScheduler(Callback):
    """ 
    Implements warm-up with cosine decay learning rate scheduling 
    """
    def __init__(self, learning_rate_base, total_epochs, warmup_epochs=5, min_learning_rate=5e-6):
        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_learning_rate = min_learning_rate
        self.learning_rates = []
    
    def _set_lr(self, learning_rate):
        """Safely set learning rate for different optimizer types"""
        opt = self.model.optimizer
        # Handle both tf.Variable and plain Python types (float/string)
        if isinstance(opt.learning_rate, tf.Variable):
            tf.keras.backend.set_value(opt.learning_rate, learning_rate)
        else:
            # For plain Python types, direct assignment
            opt.learning_rate = learning_rate

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'learning_rate'):
            raise ValueError('Optimizer must have a "learning_rate" attribute.')
            
        if epoch < self.warmup_epochs:
            learning_rate = self.learning_rate_base * (epoch + 1) / self.warmup_epochs
        else:
            decay_epochs = self.total_epochs - self.warmup_epochs
            epoch_decay = epoch - self.warmup_epochs
            cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch_decay / decay_epochs))
            learning_rate = self.min_learning_rate + (self.learning_rate_base - self.min_learning_rate) * cosine_decay
        
        # Set learning rate using the safe method
        self._set_lr(learning_rate)
        self.learning_rates.append(learning_rate)
        print(f"\nEpoch {epoch+1}: Learning rate set to {learning_rate:.6f}")

# SpecAugment-style augmentation for wav2vec embeddings
def apply_spec_augment(x, time_mask_param=70, freq_mask_param=10, num_masks=2):
    """
    Apply SpecAugment-style masking to wav2vec embeddings
    
    Args:
        x: Input tensor of shape [batch_size, time_steps, embedding_dim]
        time_mask_param: Maximum possible length of time mask
        freq_mask_param: Maximum possible length of frequency mask 
        num_masks: Number of masks to apply
    
    Returns:
        Augmented tensor of same shape as input
    """
    shape = tf.shape(x)
    batch_size, time_steps, embedding_dim = shape[0], shape[1], shape[2]
    
    # Use tf.minimum to safely adjust mask params without conditionals
    time_mask_param = tf.minimum(time_mask_param, time_steps // 2)
    freq_mask_param = tf.minimum(freq_mask_param, embedding_dim // 2)
    
    x_aug = x
    
    # Apply time masking
    for i in range(num_masks):
        # Generate random mask length
        t = tf.random.uniform([], 0, time_mask_param, dtype=tf.int32)
        
        # Generate random start point (ensure we don't go out of bounds)
        max_start = tf.maximum(1, time_steps - t)
        t0 = tf.random.uniform([], 0, max_start, dtype=tf.int32)
        
        # Create mask
        mask = tf.concat([
            tf.ones([batch_size, t0, embedding_dim]),
            tf.zeros([batch_size, t, embedding_dim]),
            tf.ones([batch_size, time_steps - t0 - t, embedding_dim])
        ], axis=1)
        
        # Apply mask
        x_aug = x_aug * mask
    
    # Apply frequency masking
    for i in range(num_masks):
        # Generate random mask length
        f = tf.random.uniform([], 0, freq_mask_param, dtype=tf.int32)
        
        # Generate random start point (ensure we don't go out of bounds)
        max_start = tf.maximum(1, embedding_dim - f)
        f0 = tf.random.uniform([], 0, max_start, dtype=tf.int32)
        
        # Create mask
        mask = tf.concat([
            tf.ones([batch_size, time_steps, f0]),
            tf.zeros([batch_size, time_steps, f]),
            tf.ones([batch_size, time_steps, embedding_dim - f0 - f])
        ], axis=2)
        
        # Apply mask
        x_aug = x_aug * mask
    
    return x_aug

# Mask-aware attention-based pooling layer
class AttentionPooling(tf.keras.layers.Layer):
    """
    Attention-based pooling layer that respects masking.
    """
    def __init__(self, units=128):
        super(AttentionPooling, self).__init__()
        self.units = units
        self.attention_weights = None  # Store for visualization if needed
    
    def build(self, input_shape):
        self.w = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )
        self.u = self.add_weight(
            name="context_vector",
            shape=(self.units, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        super(AttentionPooling, self).build(input_shape)
    
    def compute_mask(self, inputs, mask=None):
        # Output does not need a mask
        return None
    
    def compute_output_shape(self, input_shape):
        # Output shape is (batch_size, features)
        return (input_shape[0], input_shape[2])
        
    def call(self, inputs, mask=None):
        # inputs shape: (batch_size, time_steps, features)
        # Get input dtype for mixed precision compatibility
        dtype = inputs.dtype
        
        # Linear projection
        uit = tf.tensordot(inputs, self.w, axes=1) + self.b  # (batch_size, time_steps, units)
        uit = tf.nn.tanh(uit)
        
        # Compute attention scores
        scores = tf.tensordot(uit, self.u, axes=1)  # (batch_size, time_steps, 1)
        scores = tf.squeeze(scores, axis=-1)  # (batch_size, time_steps)
        
        # Apply mask if provided
        if mask is not None:
            # Convert mask from boolean to float with matching dtype
            mask = tf.cast(mask, dtype=dtype)
            # Set attention scores for masked timesteps to large negative value
            # Use constant with matching dtype
            scores = scores + (1.0 - mask) * tf.constant(-1e9, dtype=dtype)
        
        # Compute attention weights with softmax
        attention_weights = tf.nn.softmax(scores, axis=1)  # (batch_size, time_steps)
        self.attention_weights = attention_weights  # Store for later inspection
        
        # Apply attention weights to input sequence
        context = tf.reduce_sum(inputs * tf.expand_dims(attention_weights, -1), axis=1)  # (batch_size, features)
        
        return context

def create_audio_only_model(input_shape, num_classes, dropout_rate=0.5, use_augmentation=False):  # Increase dropout for regularization
    """
    Create an audio-only LSTM model for emotion recognition using wav2vec embeddings.

    Args:
        input_shape: Shape of the audio embedding input (time_steps, embedding_dim)
        num_classes: Number of emotion classes to predict
        dropout_rate: Dropout rate for regularization
        use_augmentation: Whether to use SpecAugment-style augmentation during training

    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = Input(shape=input_shape, name='audio_input')
    
    # Create custom augmentation layer instead of Lambda
    if use_augmentation:
        # Use a dedicated layer for augmentation in graph mode
        class SpecAugLayer(tf.keras.layers.Layer):
            def __init__(self, time_mask_param=70, freq_mask_param=10, num_masks=2):
                super(SpecAugLayer, self).__init__()
                self.time_mask_param = time_mask_param
                self.freq_mask_param = freq_mask_param
                self.num_masks = num_masks
                
            def call(self, x, training=None):
                if training:
                    return self._augment(x)
                return x
                
            def _augment(self, x):
                shape = tf.shape(x)
                batch_size, time_steps, embedding_dim = shape[0], shape[1], shape[2]
                
                # Adjust params safely
                t_param = tf.minimum(self.time_mask_param, time_steps // 2)
                f_param = tf.minimum(self.freq_mask_param, embedding_dim // 2)
                
                x_aug = x
                
                # Apply time masking with tensor operations only
                for i in range(self.num_masks):
                    t = tf.random.uniform([], 0, t_param, dtype=tf.int32)
                    t0 = tf.random.uniform([], 0, tf.maximum(1, time_steps - t), dtype=tf.int32)
                    
                    # Create mask tensors directly
                    mask = tf.concat([
                        tf.ones([batch_size, t0, embedding_dim]),
                        tf.zeros([batch_size, t, embedding_dim]),
                        tf.ones([batch_size, time_steps - t0 - t, embedding_dim])
                    ], axis=1)
                    
                    x_aug = x_aug * mask
                
                # Apply frequency masking
                for i in range(self.num_masks):
                    f = tf.random.uniform([], 0, f_param, dtype=tf.int32)
                    f0 = tf.random.uniform([], 0, tf.maximum(1, embedding_dim - f), dtype=tf.int32)
                    
                    mask = tf.concat([
                        tf.ones([batch_size, time_steps, f0]),
                        tf.zeros([batch_size, time_steps, f]),
                        tf.ones([batch_size, time_steps, embedding_dim - f0 - f])
                    ], axis=2)
                    
                    x_aug = x_aug * mask
                
                return x_aug
        
        # Apply the custom layer
        x = SpecAugLayer()(inputs)
    else:
        x = inputs
    
    # Masking layer to handle variable-length sequences
    x = Masking(mask_value=0.0)(x)
    
    # Add batch normalization before LSTM layers
    x = BatchNormalization()(x)
    
    # First Bi-LSTM layer with return sequences (smaller size, less memory usage)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=dropout_rate, 
                         recurrent_dropout=0.0))(x)  # Remove recurrent dropout for stability
    x = Dropout(dropout_rate)(x)
    
    # Second Bi-LSTM layer with return sequences (smaller size)
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=dropout_rate, 
                         recurrent_dropout=0.0))(x)  # Remove recurrent dropout for stability
    x = Dropout(dropout_rate)(x)
    
    # Attention-based pooling (mask-aware)
    x = AttentionPooling(units=128)(x)
    
    # Dense layer with batch normalization (smaller size)
    x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train audio-only emotion recognition model using wav2vec embeddings')
    
    parser.add_argument('--features_dir', type=str, required=True,
                        help='Directory containing wav2vec feature files')
    parser.add_argument('--mean_path', type=str, default=None,
                        help='Path to pre-computed mean values for normalization')
    parser.add_argument('--std_path', type=str, default=None,
                        help='Path to pre-computed std values for normalization')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=6e-4,
                        help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='TensorBoard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Model checkpoint directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def load_wav2vec_files(features_dir):
    """
    Load wav2vec feature files and extract labels.
    
    Args:
        features_dir: Base directory containing ravdess_features_wav2vec2/ and crema_d_features_wav2vec2/
           OR containing data/ravdess_features_wav2vec2/ and data/crema_d_features_wav2vec2/
        
    Returns:
        List of file paths and corresponding labels
    """
    # Try with both possible directory structures
    ravdess_dir = os.path.join(features_dir, 'ravdess_features_wav2vec2')
    cremad_dir = os.path.join(features_dir, 'crema_d_features_wav2vec2')
    
    # If not found, try with data/ subdirectory
    if not os.path.exists(ravdess_dir) or not os.path.exists(cremad_dir):
        data_dir = os.path.join(features_dir, 'data')
        if os.path.exists(data_dir):
            ravdess_dir = os.path.join(data_dir, 'ravdess_features_wav2vec2')
            cremad_dir = os.path.join(data_dir, 'crema_d_features_wav2vec2')
            print(f"Looking in data subdirectory: {ravdess_dir} and {cremad_dir}")
    
    # Verify directories exist
    if not os.path.exists(ravdess_dir):
        print(f"Warning: RAVDESS directory not found at {ravdess_dir}")
    if not os.path.exists(cremad_dir):
        print(f"Warning: CREMA-D directory not found at {cremad_dir}")
    
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
    
    # Map to 7 common emotions (or optionally 6 if dropping 'calm')
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
    
    # Alternative mapping that drops 'calm' (use if needed)
    cremad_emotion_map = {
        'NEU': 'neutral',
        'HAP': 'happy',
        'SAD': 'sad',
        'ANG': 'angry',
        'FEA': 'fearful',
        'DIS': 'disgust'
    }
    
    # Collect all wav2vec feature files
    ravdess_files = []
    for actor_dir in glob.glob(os.path.join(ravdess_dir, 'Actor_*')):
        ravdess_files.extend(glob.glob(os.path.join(actor_dir, '*.npy')))
    
    cremad_files = glob.glob(os.path.join(cremad_dir, '*.npy'))
    
    # Initialize lists for files and labels
    audio_files = []
    labels = []
    skipped = 0
    
    # Process RAVDESS files
    for file_path in tqdm(ravdess_files, desc="Processing RAVDESS files"):
        try:
            # Extract emotion code from filename (e.g., 03-01-01-01-01-01-01.npy)
            filename = os.path.basename(file_path)
            parts = filename.split('-')
            emotion_code = parts[2]
            
            # Map to emotion label
            emotion = ravdess_emotion_map.get(emotion_code)
            
            if emotion:
                # Check if we want to exclude 'calm' to match CREMA-D
                # if emotion == 'calm':
                #     skipped += 1
                #     continue
                
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
    
    # Process CREMA-D files
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
    
    # Convert to array and return
    return audio_files, np.array(labels)

def compute_normalization_stats(files):
    """
    Compute mean and standard deviation over all wav2vec embeddings.
    
    Args:
        files: List of paths to wav2vec embedding .npy files
        
    Returns:
        mean and std arrays
    """
    print("Computing normalization statistics...")
    # First pass - get total number of frames to pre-allocate memory
    total_frames = 0
    
    for file_path in tqdm(files, desc="Counting frames"):
        try:
            features = np.load(file_path)
            total_frames += features.shape[0]
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if total_frames == 0:
        raise ValueError("No valid frames found in the dataset")
    
    # Get embedding dimension from first valid file
    for file_path in files:
        try:
            sample = np.load(file_path)
            embedding_dim = sample.shape[1]
            break
        except:
            continue
    
    # Pre-allocate a large array for all embeddings
    all_features = np.zeros((total_frames, embedding_dim), dtype=np.float32)
    
    # Second pass - fill the array
    idx = 0
    for file_path in tqdm(files, desc="Loading embeddings"):
        try:
            features = np.load(file_path)
            if features.shape[0] > 0:  # Check if non-empty
                frames = features.shape[0]
                all_features[idx:idx+frames] = features
                idx += frames
        except Exception as e:
            print(f"Error loading {file_path} for stats: {e}")
    
    # Compute statistics on collected embeddings
    mean = np.mean(all_features[:idx], axis=0)
    std = np.std(all_features[:idx], axis=0)
    
    # Replace zeros in std to avoid division by zero
    std = np.where(std < 1e-10, 1.0, std)
    
    return mean, std

def normalize_features(features, mean, std):
    """Normalize features using pre-computed mean and std."""
    return (features - mean) / std

def pad_sequence(sequence, max_len=None):
    """Pad sequence to fixed length or return as is if max_len is None."""
    if max_len is None or sequence.shape[0] <= max_len:
        return sequence
    else:
        return sequence[:max_len]

def create_datasets(files, labels, mean, std, max_len=None, batch_size=64, val_split=0.1):
    """
    Create TensorFlow datasets for training and validation.
    
    Args:
        files: List of paths to wav2vec embedding .npy files
        labels: List of corresponding emotion labels
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        max_len: Maximum sequence length (None for dynamic)
        batch_size: Batch size for training
        val_split: Validation split ratio
        
    Returns:
        Training dataset, validation dataset, and number of classes
    """
    # Determine maximum sequence length if not specified
    if max_len is None:
        # Find the 95th percentile length to avoid outliers
        lengths = []
        for file_path in tqdm(files, desc="Calculating sequence lengths"):
            try:
                features = np.load(file_path)
                lengths.append(features.shape[0])
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        max_len = int(np.percentile(lengths, 95))
        print(f"Using max sequence length of {max_len} (95th percentile)")
    
    # Count class occurrences for weighting
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    class_counts = np.bincount(labels, minlength=num_classes)
    class_weights = 1.0 / class_counts
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
    
    # Create data loading function
    def load_and_preprocess(file_idx):
        file_path = files[file_idx]
        label = labels[file_idx]
        
        # Load features
        features = np.load(file_path).astype(np.float32)
        
        # Normalize
        normalized = normalize_features(features, mean, std)
        
        # Pad or truncate
        padded = pad_sequence(normalized, max_len)
        
        # One-hot encode label
        one_hot = tf.one_hot(label, num_classes)
        
        return padded, one_hot
    
    # Create a function to load for the tf.data pipeline
    def generator_fn(indices):
        for idx in indices:
            try:
                yield load_and_preprocess(idx)
            except Exception as e:
                print(f"Error loading file at index {idx}: {e}")
                continue
    
    # Create TensorFlow datasets
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
    
    # Always use padded_batch to handle different sequence lengths within batches
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

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set random seed for reproducibility
    global RANDOM_SEED
    RANDOM_SEED = args.seed
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    print("Audio-only Wav2Vec Emotion Recognition Training")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Random seed: {RANDOM_SEED}")
    
    # Load audio files and labels
    audio_files, labels = load_wav2vec_files(args.features_dir)
    
    if len(audio_files) == 0:
        print("Error: No audio files found. Check the features_dir path.")
        sys.exit(1)
    
    # Create output directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Compute or load normalization statistics
    if args.mean_path and args.std_path and os.path.exists(args.mean_path) and os.path.exists(args.std_path):
        print(f"Loading pre-computed normalization stats from {args.mean_path} and {args.std_path}")
        mean = np.load(args.mean_path)
        std = np.load(args.std_path)
    else:
        print("Computing normalization statistics...")
        mean, std = compute_normalization_stats(audio_files)
        
        # Save statistics if paths provided
        if args.mean_path and args.std_path:
            print(f"Saving normalization stats to {args.mean_path} and {args.std_path}")
            np.save(args.mean_path, mean)
            np.save(args.std_path, std)
    
    print(f"Embedding dimension: {mean.shape[0]}")
    
    # Create datasets for training and validation
    train_ds, val_ds, num_classes, class_weights = create_datasets(
        audio_files, labels, mean, std, 
        max_len=None,  # Use None for dynamic sequence handling
        batch_size=args.batch_size,
        val_split=args.val_split
    )
    
    # Create model with augmentation disabled to avoid graph mode issues
    model = create_audio_only_model(
        input_shape=(None, mean.shape[0]),
        num_classes=num_classes,
        dropout_rate=0.5,  # Increased dropout rate matches the model definition
        use_augmentation=False  # Disable augmentation until TF graph mode issues are resolved
    )
    
    # Use SGD with momentum - most stable optimizer for numerical issues
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=args.lr/100,  # Much lower learning rate (100x reduction) for initial stability
        momentum=0.9,
        nesterov=True
    )
    
    # Apply label smoothing to avoid log(0) issues with hard one-hot targets
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    # Test run a single forward pass on fake data to check for NaN issues
    print("Testing forward pass with random data...")
    batch_shape = (args.batch_size, 100, mean.shape[0])  # (batch, time, features)
    test_input = tf.random.normal(batch_shape, mean=0.0, stddev=0.1)
    test_labels = tf.one_hot(tf.random.uniform(
        shape=(args.batch_size,), 
        minval=0, 
        maxval=num_classes-1, 
        dtype=tf.int32
    ), depth=num_classes)
    
    # Run test forward pass with gradient recording
    with tf.GradientTape() as tape:
        test_output = model(test_input, training=True)
        test_loss = model.loss(test_labels, test_output)
    
    # Check outputs for NaN
    if tf.math.reduce_any(tf.math.is_nan(test_output)):
        print("WARNING: NaN detected in forward pass output!")
    else:
        print("Forward pass output looks good (no NaNs)")
    
    # Check loss for NaN
    if tf.math.is_nan(test_loss):
        print("WARNING: NaN detected in loss calculation!")
    else:
        print(f"Test loss calculation looks good: {test_loss:.4f}")
    
    # Check gradients for NaN
    grads = tape.gradient(test_loss, model.trainable_variables)
    has_nan_grads = False
    for g in grads:
        if g is not None and tf.math.reduce_any(tf.math.is_nan(g)):
            has_nan_grads = True
            break
    
    if has_nan_grads:
        print("WARNING: NaN detected in gradients!")
    else:
        print("Gradients look good (no NaNs)")
    
    # Model summary
    model.summary()
    
    # Set up callbacks with improved debugging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(args.checkpoint_dir, f"wav2vec_audio_only_{timestamp}_best.weights.h5")
    
    # Custom callback to detect NaN losses
    class NaNLossDetector(Callback):
        def on_batch_end(self, batch, logs=None):
            logs = logs or {}
            loss = logs.get('loss')
            if loss is not None and (math.isnan(loss) or math.isinf(loss)):
                print(f'NaN/Inf loss detected: {loss} at batch {batch}, stopping training.')
                self.model.stop_training = True
                
        def on_epoch_begin(self, epoch, logs=None):
            print(f"Epoch {epoch+1}: Current learning rate: {float(self.model.optimizer.learning_rate.numpy()):.8f}")
    
    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,  # Increased patience
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(args.log_dir, f"wav2vec_audio_only_{timestamp}"),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,  # More aggressive reduction
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        # Warm-up LR schedule is critical for stability
        WarmUpCosineDecayScheduler(
            learning_rate_base=args.lr/100,  # Using the reduced LR
            total_epochs=args.epochs,
            warmup_epochs=10,  # Extended warm-up period
            min_learning_rate=1e-7
        ),
        NaNLossDetector(),
    ]
    
    # Train model
    print("\nStarting training...")
    start_time = time.time()
    
    # Get a single batch for the F1 callback
    for x_val, y_val in val_ds.take(1):
        val_data = (x_val, y_val)
        callbacks.append(F1ScoreCallback(val_data))
        break
    
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        class_weight=dict(enumerate(class_weights)),
        verbose=1
    )
    
    end_time = time.time()
    print(f"\nTraining completed in {(end_time - start_time) / 60:.2f} minutes.")
    
    # Save final model
    final_model_path = os.path.join(args.checkpoint_dir, f"wav2vec_audio_only_{timestamp}_final.weights.h5")
    model.save_weights(final_model_path)
    print(f"Final model weights saved to {final_model_path}")
    
    # Save model architecture
    model_json = model.to_json()
    with open(os.path.join(args.checkpoint_dir, f"wav2vec_audio_only_{timestamp}_architecture.json"), 'w') as f:
        f.write(model_json)
    
    # Save training history
    with open(os.path.join(args.checkpoint_dir, f"wav2vec_audio_only_{timestamp}_history.json"), 'w') as f:
        history_dict = {key: [float(x) for x in values] for key, values in history.history.items()}
        json.dump(history_dict, f, indent=4)
    
    print("\nTraining history saved.")
    print("\nRun TensorBoard with:")
    print(f"tensorboard --logdir={args.log_dir}")
    
    # Final evaluation on validation set
    print("\nEvaluating best model on validation set...")
    model.load_weights(checkpoint_path)
    val_loss, val_acc = model.evaluate(val_ds, verbose=1)
    print(f"Validation accuracy: {val_acc:.4f}")

if __name__ == "__main__":
    main()
