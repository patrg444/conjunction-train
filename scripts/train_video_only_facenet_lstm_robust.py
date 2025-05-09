#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust version of video-only LSTM model training for Facenet features.
This version adds key fallbacks, better error handling, and class weighting.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Bidirectional, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
import time
import glob
import random
from tqdm import tqdm
from collections import Counter

# Try to import the specialized video-only generator
try:
    # Try relative import first (if run from scripts directory)
    from video_only_facenet_generator import VideoOnlyFacenetGenerator 
except ImportError:
    # Fall back to absolute import (if run from project root)
    from scripts.video_only_facenet_generator import VideoOnlyFacenetGenerator

# --- Model Configuration Parameters ---
BATCH_SIZE = 24 
EPOCHS = 150
NUM_CLASSES = 6
PATIENCE = 20
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
L2_REGULARIZATION = 0.001
MAX_NORM_CONSTRAINT = 3.0
LEARNING_RATE = 0.0005

# Model specific parameters (increased size slightly)
LSTM_UNITS = [256, 128]
DENSE_UNITS = [256, 128]
FACENET_FEATURE_DIM = 512

# New parameters for robustness
POSSIBLE_FEATURE_KEYS = ['video_features', 'features', 'facenet_features', 'embeddings']
APPLY_CLASS_WEIGHTING = True
USE_LAYER_NORM = True
USE_FIXED_MAX_SEQ_LEN = None  # Set to an integer value to use a fixed max sequence length

print("ROBUST VIDEO ONLY FACENET LSTM")
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
print("Python version:", sys.version)

# --- Learning Rate Scheduler ---
class WarmUpCosineDecayScheduler(Callback):
    """ Implements warm-up with cosine decay learning rate scheduling. """
    def __init__(self, learning_rate_base, total_epochs, warmup_epochs=10, min_learning_rate=1e-6):
        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_learning_rate = min_learning_rate
        self.learning_rates = []

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs: 
            learning_rate = self.learning_rate_base * (epoch + 1) / self.warmup_epochs
        else:
            decay_epochs = self.total_epochs - self.warmup_epochs 
            epoch_decay = epoch - self.warmup_epochs
            cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch_decay / decay_epochs))
            learning_rate = self.min_learning_rate + (self.learning_rate_base - self.min_learning_rate) * cosine_decay
        
        self.model.optimizer.learning_rate.assign(learning_rate)
        self.learning_rates.append(learning_rate)
        print(f"\nEpoch {epoch+1}: Learning rate set to {learning_rate:.6f}")

# --- Model Definition (Video-Only) ---
def create_video_lstm_model(video_feature_dim, class_weights=None):
    """Creates a robust video-only LSTM model with options for class weighting"""
    
    print("Creating video-only LSTM model (Robust):")
    print(f"- Video feature dimension: {video_feature_dim}")
    print(f"- L2 regularization: {L2_REGULARIZATION}")
    print(f"- Max norm constraint: {MAX_NORM_CONSTRAINT}")
    print(f"- LSTM Units: {LSTM_UNITS}")
    print(f"- Dense Units: {DENSE_UNITS}")
    print(f"- Layer normalization: {'Enabled' if USE_LAYER_NORM else 'Disabled'}")
    print(f"- Class weighting: {'Enabled' if class_weights is not None else 'Disabled'}")
    if class_weights:
        print("- Class weights:", class_weights)
    print(f"- Initial learning rate: {LEARNING_RATE}")

    # Input layer
    video_input = Input(shape=(None, video_feature_dim), name='video_input')
    
    # Optional masking layer to handle variable length sequences
    x = tf.keras.layers.Masking(mask_value=0.0)(video_input)
    
    # First LSTM layer
    x = Bidirectional(LSTM(
        LSTM_UNITS[0], return_sequences=True, 
        dropout=0.3, recurrent_dropout=0.2,
        kernel_regularizer=l2(L2_REGULARIZATION), 
        recurrent_regularizer=l2(L2_REGULARIZATION/2),
        kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT), 
        recurrent_constraint=MaxNorm(MAX_NORM_CONSTRAINT)
    ))(x)
    if USE_LAYER_NORM:
        x = LayerNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Second LSTM layer
    x = Bidirectional(LSTM(
        LSTM_UNITS[1], return_sequences=False, 
        dropout=0.3, recurrent_dropout=0.2,
        kernel_regularizer=l2(L2_REGULARIZATION), 
        recurrent_regularizer=l2(L2_REGULARIZATION/2),
        kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT), 
        recurrent_constraint=MaxNorm(MAX_NORM_CONSTRAINT)
    ))(x)
    if USE_LAYER_NORM:
        x = LayerNormalization()(x)
    x = Dropout(0.4)(x)
    
    # First Dense layer
    x = Dense(
        DENSE_UNITS[0], activation='relu',
        kernel_regularizer=l2(L2_REGULARIZATION), 
        kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT)
    )(x)
    if USE_LAYER_NORM:
        x = LayerNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Second Dense layer
    x = Dense(
        DENSE_UNITS[1], activation='relu',
        kernel_regularizer=l2(L2_REGULARIZATION), 
        kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT)
    )(x)
    if USE_LAYER_NORM:
        x = LayerNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Output layer
    output = Dense(NUM_CLASSES, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=video_input, outputs=output)
    
    # Compile model
    optimizer = Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# --- Robust Data Loading (Video-Only) ---
def load_data_robust(facenet_dir_ravdess, facenet_dir_cremad):
    """
    Loads data with robust error handling and support for different key formats.
    Returns valid files, labels, and optional class weights.
    """
    # Find all .npz files in the given directories
    ravdess_files = glob.glob(os.path.join(facenet_dir_ravdess, "Actor_*", "*.npz"))
    cremad_files = glob.glob(os.path.join(facenet_dir_cremad, "*.npz"))
    all_files = ravdess_files + cremad_files
    
    print(f"Found {len(ravdess_files)} potential RAVDESS files")
    print(f"Found {len(cremad_files)} potential CREMA-D files")
    print(f"Total files: {len(all_files)}")
    
    # Emotion mapping dictionaries
    emotion_map = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
    ravdess_emotion_map = {'01': 'NEU', '02': 'NEU', '03': 'HAP', '04': 'SAD', 
                          '05': 'ANG', '06': 'FEA', '07': 'DIS', '08': 'FEA'}
    
    valid_files = []
    labels = []
    emotion_counts = Counter()
    skipped_files = 0
    missing_video_features = 0
    label_parsing_errors = 0
    empty_features = 0
    
    # Process all files
    for file_path in tqdm(all_files, desc="Processing files"):
        try:
            # Extract base name for parsing the label
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            if base_name.endswith('_facenet_features'):
                base_name = base_name[:-len('_facenet_features')]
            
            # Default label to None (will be skipped if not updated)
            label = None
            
            # Parse RAVDESS filenames
            if "Actor_" in file_path:
                parts = base_name.split('-')
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    emotion_name = ravdess_emotion_map.get(emotion_code)
                    if emotion_name in emotion_map:
                        label_idx = emotion_map[emotion_name]
                        label = np.zeros(NUM_CLASSES)
                        label[label_idx] = 1.0
                        emotion_counts[emotion_name] += 1
                    else:
                        print(f"Warning: RAVDESS emotion code '{emotion_code}' not recognized in {file_path}")
                        label_parsing_errors += 1
                else:
                    print(f"Warning: RAVDESS filename format error in {file_path}")
                    label_parsing_errors += 1
            
            # Parse CREMA-D filenames
            else:
                parts = base_name.split('_')
                if len(parts) >= 3:
                    emotion_name = parts[2]
                    if emotion_name in emotion_map:
                        label_idx = emotion_map[emotion_name]
                        label = np.zeros(NUM_CLASSES)
                        label[label_idx] = 1.0
                        emotion_counts[emotion_name] += 1
                    else:
                        print(f"Warning: CREMA-D emotion name '{emotion_name}' not recognized in {file_path}")
                        label_parsing_errors += 1
                else:
                    print(f"Warning: CREMA-D filename format error in {file_path}")
                    label_parsing_errors += 1
            
            # If we successfully generated a label, check the feature file
            if label is not None:
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    try:
                        with np.load(file_path, allow_pickle=True) as data:
                            # Try multiple possible keys for the video features
                            features = None
                            feature_key_used = None
                            
                            for key in POSSIBLE_FEATURE_KEYS:
                                if key in data:
                                    features = data[key]
                                    feature_key_used = key
                                    break
                            
                            if features is not None and features.shape[0] > 0:
                                valid_files.append(file_path)
                                labels.append(label)
                            else:
                                if feature_key_used:
                                    empty_features += 1
                                    print(f"Warning: Empty features using key '{feature_key_used}' in {file_path}")
                                else:
                                    missing_video_features += 1
                                    print(f"Warning: No valid feature key found in {file_path}. Available keys: {list(data.keys())}")
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                        skipped_files += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            skipped_files += 1
    
    # Print statistics
    print("\nData Loading Statistics:")
    print(f"- Valid files found: {len(valid_files)}")
    print(f"- Files with label parsing errors: {label_parsing_errors}")
    print(f"- Files missing video features: {missing_video_features}")
    print(f"- Files with empty features: {empty_features}")
    print(f"- Other skipped files: {skipped_files}")
    
    print("\nEmotion Distribution:")
    total_emotions = sum(emotion_counts.values())
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: emotion_map[x[0]]):
        print(f"- {emotion}: {count} ({count/total_emotions*100:.1f}%)")
    
    # Calculate class weights if enabled
    class_weights = None
    if APPLY_CLASS_WEIGHTING and len(valid_files) > 0:
        class_counts = np.zeros(NUM_CLASSES)
        for emotion, count in emotion_counts.items():
            if emotion in emotion_map:
                class_counts[emotion_map[emotion]] = count
        
        # Avoid division by zero
        class_counts = np.maximum(class_counts, 1)
        
        # Weight calculation: n_samples / (n_classes * class_count)
        n_samples = np.sum(class_counts)
        class_weights = {}
        for class_idx in range(NUM_CLASSES):
            class_weights[class_idx] = n_samples / (NUM_CLASSES * class_counts[class_idx])
        
        print("\nCalculated Class Weights:")
        for class_idx, weight in class_weights.items():
            for emotion, idx in emotion_map.items():
                if idx == class_idx:
                    print(f"- {emotion} (class {class_idx}): {weight:.4f}")
                    break
    
    if len(valid_files) == 0:
        # Emergency measures - look inside npz files to see what keys are available
        print("\n*** EMERGENCY KEY INSPECTION ***")
        for i, file_path in enumerate(all_files[:10]):  # Only check first 10 files
            try:
                with np.load(file_path, allow_pickle=True) as data:
                    print(f"{os.path.basename(file_path)}: Keys = {list(data.keys())}")
            except Exception as e:
                print(f"Error inspecting {file_path}: {e}")
    
    return valid_files, np.array(labels), class_weights

# --- Training Function ---
def train_model():
    """Main function to train the video-only LSTM model."""
    print("\n=== Starting Robust Video-Only Facenet LSTM Training ===")
    
    # Define feature directories (paths can be adjusted as needed)
    RAVDESS_FACENET_DIR = "/home/ubuntu/emotion-recognition/ravdess_features_facenet"
    CREMA_D_FACENET_DIR = "/home/ubuntu/emotion-recognition/crema_d_features_facenet"
    
    # Check if the directories exist, use local paths if they don't
    if not os.path.exists(RAVDESS_FACENET_DIR):
        RAVDESS_FACENET_DIR = "./ravdess_features_facenet"
        print(f"Using local path for RAVDESS: {RAVDESS_FACENET_DIR}")
    
    if not os.path.exists(CREMA_D_FACENET_DIR):
        CREMA_D_FACENET_DIR = "./crema_d_features_facenet"
        print(f"Using local path for CREMA-D: {CREMA_D_FACENET_DIR}")
    
    # Robustly load data with possible class weights
    valid_files, all_labels, class_weights = load_data_robust(
        RAVDESS_FACENET_DIR, CREMA_D_FACENET_DIR
    )
    
    if len(valid_files) == 0:
        print("No valid files found. Exiting.")
        sys.exit(1)
    
    # Split data into train/validation
    indices = np.arange(len(valid_files))
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices)
    
    split_idx = int(len(indices) * TRAIN_RATIO)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_files = [valid_files[i] for i in train_indices]
    train_labels = all_labels[train_indices]
    val_files = [valid_files[i] for i in val_indices]
    val_labels = all_labels[val_indices]
    
    print(f"\nTrain/Val split:")
    print(f"- Train samples: {len(train_files)}")
    print(f"- Validation samples: {len(val_files)}")
    
    # Calculate or use provided max sequence length
    max_seq_len = USE_FIXED_MAX_SEQ_LEN  # Can be None, in which case calculated below
    
    if max_seq_len is None:
        print("\nCalculating maximum sequence length from training features...")
        max_seq_len = 0
        for f_path in tqdm(train_files[:100], desc="Checking lengths"):  # Only check 100 files for efficiency
            try:
                with np.load(f_path, allow_pickle=True) as data:
                    # Try multiple possible keys
                    for key in POSSIBLE_FEATURE_KEYS:
                        if key in data:
                            shape = data[key].shape
                            if len(shape) >= 1:
                                max_seq_len = max(max_seq_len, shape[0])
                                break
            except Exception as e:
                print(f"Warning: Could not read shape from {f_path}: {e}")
                continue
        
        # Give a sensible default if calculation failed
        if max_seq_len == 0:
            max_seq_len = 100
            print(f"Warning: Could not determine max_seq_len. Using default: {max_seq_len}")
        else:
            print(f"Calculated max_seq_len: {max_seq_len}")
    else:
        print(f"Using fixed max_seq_len: {max_seq_len}")
    
    # Create the data generators
    train_generator = VideoOnlyFacenetGenerator(
        video_feature_files=train_files,
        labels=train_labels,
        batch_size=BATCH_SIZE,
        shuffle=True,
        max_seq_len=max_seq_len,
        audio_mode='video_only'
    )
    
    val_generator = VideoOnlyFacenetGenerator(
        video_feature_files=val_files,
        labels=val_labels,
        batch_size=BATCH_SIZE,
        shuffle=False,
        max_seq_len=max_seq_len,
        audio_mode='video_only'
    )
    
    # Get feature dimension from generator
    video_feature_dim = train_generator.video_feature_dim
    if video_feature_dim == 0:
        video_feature_dim = FACENET_FEATURE_DIM
        print(f"Warning: Using default feature dimension: {video_feature_dim}")
    
    # Create model
    model = create_video_lstm_model(video_feature_dim, class_weights)
    model.summary()
    
    # Define callbacks
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"models/robust_video_only_facenet_lstm_{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_filename = os.path.join(checkpoint_dir, 'best_model.keras')
    print(f"Saving checkpoints to: {checkpoint_dir}")
    
    checkpoint = ModelCheckpoint(
        checkpoint_filename, monitor='val_accuracy', 
        save_best_only=True, mode='max', verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy', patience=PATIENCE, 
        mode='max', verbose=1, restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.6, patience=PATIENCE//2, 
        min_lr=5e-6, verbose=1, mode='min'
    )
    
    lr_scheduler = WarmUpCosineDecayScheduler(
        learning_rate_base=LEARNING_RATE, 
        total_epochs=EPOCHS, 
        warmup_epochs=10, 
        min_learning_rate=5e-6
    )
    
    callbacks_list = [checkpoint, early_stopping, reduce_lr, lr_scheduler]
    
    # Train the model
    print("\nStarting robust training...")
    start_time = time.time()
    
    # Include class_weight if available
    fit_params = {
        'x': train_generator,
        'epochs': EPOCHS,
        'validation_data': val_generator,
        'callbacks': callbacks_list,
        'verbose': 1
    }
    
    if APPLY_CLASS_WEIGHTING and class_weights:
        fit_params['class_weight'] = class_weights
        print("Using class weights for training")
    
    history = model.fit(**fit_params)
    
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    
    # Evaluate the best model
    print("\nEvaluating the best model...")
    if os.path.exists(checkpoint_filename):
        print(f"Loading best weights from {checkpoint_filename}")
        model.load_weights(checkpoint_filename)
    
    loss, accuracy = model.evaluate(val_generator, verbose=1)
    print(f"Best model validation accuracy: {accuracy:.4f}")
    
    # Save the final model
    final_model_path = os.path.join(checkpoint_dir, "final_model.keras")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Save training history
    history_path = os.path.join(checkpoint_dir, "training_history.npy")
    np.save(history_path, history.history)
    print(f"Training history saved to {history_path}")
    
    # Print success message with directory
    print(f"\nTraining successful! Model saved in: {checkpoint_dir}")
    print("You can monitor training progress with TensorBoard or check the best model.")

if __name__ == '__main__':
    train_model()
