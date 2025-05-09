#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video-only emotion recognition model using LSTM on Facenet features.
This fixed version ensures proper data loading, feature normalization,
and label consistency to avoid random accuracy.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Bidirectional, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback, TensorBoard
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
import time
import glob
import random
from tqdm import tqdm
from collections import Counter

# Import the fixed generator
try:
    from fixed_video_facenet_generator import FixedVideoFacenetGenerator
except ImportError:
    from scripts.fixed_video_facenet_generator import FixedVideoFacenetGenerator

# Configuration parameters
BATCH_SIZE = 24
EPOCHS = 150
NUM_CLASSES = 6
PATIENCE = 20
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
L2_REGULARIZATION = 0.001
MAX_NORM_CONSTRAINT = 3.0
LEARNING_RATE = 0.0005
LSTM_UNITS = [256, 128]
DENSE_UNITS = [256, 128]
FACENET_FEATURE_DIM = 512
# Enable these features for better model performance
USE_LAYER_NORM = True
APPLY_CLASS_WEIGHTING = True
NORMALIZE_FEATURES = True
DEBUG_MODE = True  # Set to True for verbose output

print("===== VIDEO-ONLY FACENET LSTM (FIXED VERSION) =====")
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
print("Python version:", sys.version)

class WarmUpCosineDecayScheduler(Callback):
    """Implements warm-up with cosine decay learning rate scheduling."""
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

def create_video_lstm_model(video_feature_dim, class_weights=None):
    """Creates an optimized video-only LSTM model for emotion recognition."""
    
    print("Creating video-only LSTM model (Fixed Version):")
    print(f"- Video feature dimension: {video_feature_dim}")
    print(f"- L2 regularization: {L2_REGULARIZATION}")
    print(f"- Max norm constraint: {MAX_NORM_CONSTRAINT}")
    print(f"- LSTM Units: {LSTM_UNITS}")
    print(f"- Dense Units: {DENSE_UNITS}")
    print(f"- Layer normalization: {'Enabled' if USE_LAYER_NORM else 'Disabled'}")
    print(f"- Class weighting: {'Enabled' if class_weights is not None else 'Disabled'}")
    if class_weights:
        print("- Class weights:", class_weights)
    
    # Input layer
    video_input = Input(shape=(None, video_feature_dim), name='video_input')
    
    # Masking layer to handle variable length sequences
    x = tf.keras.layers.Masking(mask_value=0.0)(video_input)
    
    # First LSTM layer
    x = Bidirectional(LSTM(
        LSTM_UNITS[0], return_sequences=True, 
        dropout=0.25, recurrent_dropout=0.1,
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
        dropout=0.25, recurrent_dropout=0.1,
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
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer, 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    return model

def load_data_and_labels(facenet_dir_ravdess, facenet_dir_cremad):
    """
    Load video feature files and create corresponding emotion labels.
    Labels are derived from filename patterns rather than from inside the files.
    """
    # Find all NPZ files
    ravdess_files = glob.glob(os.path.join(facenet_dir_ravdess, "Actor_*", "*.npz"))
    cremad_files = glob.glob(os.path.join(facenet_dir_cremad, "*.npz"))
    all_files = ravdess_files + cremad_files
    
    print(f"Found {len(ravdess_files)} RAVDESS files")
    print(f"Found {len(cremad_files)} CREMA-D files")
    print(f"Total files: {len(all_files)}")
    
    # Standard emotion mapping
    emotion_map = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
    ravdess_emotion_map = {
        '01': 'NEU', '02': 'NEU', '03': 'HAP', '04': 'SAD', 
        '05': 'ANG', '06': 'FEA', '07': 'DIS', '08': 'FEA'  # Map surprise to fear
    }
    
    labels = []
    valid_files = []
    emotion_counts = Counter()
    skipped_files = 0
    
    # Process files and extract labels from filenames
    for file_path in tqdm(all_files, desc="Processing files"):
        try:
            base_name = os.path.basename(file_path)
            file_name = os.path.splitext(base_name)[0]
            
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
                # One-hot encode the label
                label = np.zeros(NUM_CLASSES, dtype=np.float32)
                label[emotion_map[emotion_code]] = 1.0
                
                # Verify the file exists and has video_features
                try:
                    with np.load(file_path, allow_pickle=True) as data:
                        if 'video_features' in data and data['video_features'].shape[0] > 0:
                            valid_files.append(file_path)
                            labels.append(label)
                            emotion_counts[emotion_code] += 1
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"Error loading {file_path}: {e}")
                    skipped_files += 1
            else:
                if DEBUG_MODE:
                    print(f"Skipping {file_path}: Unknown emotion code: {emotion_code}")
                skipped_files += 1
        except Exception as e:
            if DEBUG_MODE:
                print(f"Error processing {file_path}: {e}")
            skipped_files += 1
    
    print(f"\nData Loading Summary:")
    print(f"- Valid files: {len(valid_files)}")
    print(f"- Skipped files: {skipped_files}")
    
    # Print emotion distribution
    print("\nEmotion Distribution:")
    total_count = sum(emotion_counts.values())
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: emotion_map[x[0]]):
        print(f"- {emotion}: {count} ({count/total_count*100:.1f}%)")
    
    # Calculate class weights if enabled
    class_weights = None
    if APPLY_CLASS_WEIGHTING and total_count > 0:
        class_counts = np.zeros(NUM_CLASSES)
        for emotion, count in emotion_counts.items():
            if emotion in emotion_map:
                class_counts[emotion_map[emotion]] = count
        
        # Avoid division by zero
        class_counts = np.maximum(class_counts, 1)
        
        # Weight calculation: n_samples / (n_classes * class_count)
        class_weights = {}
        for class_idx in range(NUM_CLASSES):
            class_weights[class_idx] = total_count / (NUM_CLASSES * class_counts[class_idx])
    
    return valid_files, np.array(labels), class_weights

def train_model():
    """Main function to train the video-only LSTM model."""
    print("\n=== Starting Fixed Video-Only Facenet LSTM Training ===")
    
    # Define feature directories
    RAVDESS_FACENET_DIR = "/home/ubuntu/emotion-recognition/ravdess_features_facenet"
    CREMA_D_FACENET_DIR = "/home/ubuntu/emotion-recognition/crema_d_features_facenet"
    
    # Use local paths if EC2 paths don't exist
    if not os.path.exists(RAVDESS_FACENET_DIR):
        RAVDESS_FACENET_DIR = "./ravdess_features_facenet"
        print(f"Using local path for RAVDESS: {RAVDESS_FACENET_DIR}")
    
    if not os.path.exists(CREMA_D_FACENET_DIR):
        CREMA_D_FACENET_DIR = "./crema_d_features_facenet"
        print(f"Using local path for CREMA-D: {CREMA_D_FACENET_DIR}")
    
    # Load data and create labels
    valid_files, labels, class_weights = load_data_and_labels(
        RAVDESS_FACENET_DIR, CREMA_D_FACENET_DIR
    )
    
    if len(valid_files) == 0:
        print("Error: No valid files found. Exiting.")
        sys.exit(1)
    
    # Split data into train/val sets
    np.random.seed(RANDOM_SEED)
    indices = np.arange(len(valid_files))
    np.random.shuffle(indices)
    
    split_idx = int(len(indices) * TRAIN_RATIO)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_files = [valid_files[i] for i in train_indices]
    train_labels = labels[train_indices]
    val_files = [valid_files[i] for i in val_indices]
    val_labels = labels[val_indices]
    
    print(f"\nTrain/Val split:")
    print(f"- Train samples: {len(train_files)}")
    print(f"- Validation samples: {len(val_files)}")
    
    # Calculate maximum sequence length from subset of training files
    max_seq_len = None  # Will be calculated by the generator
    
    # Create data generators with the fixed generator
    train_generator = FixedVideoFacenetGenerator(
        video_feature_files=train_files,
        labels=train_labels,
        batch_size=BATCH_SIZE,
        shuffle=True,
        max_seq_len=max_seq_len,
        normalize_features=NORMALIZE_FEATURES,
        debug_mode=DEBUG_MODE
    )
    
    val_generator = FixedVideoFacenetGenerator(
        video_feature_files=val_files,
        labels=val_labels,
        batch_size=BATCH_SIZE,
        shuffle=False,
        max_seq_len=max_seq_len,
        normalize_features=NORMALIZE_FEATURES,
        debug_mode=DEBUG_MODE
    )
    
    # Create model
    model = create_video_lstm_model(FACENET_FEATURE_DIM, class_weights)
    model.summary()
    
    # Set up checkpoint directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"models/facenet_lstm_fixed_{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(checkpoint_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=PATIENCE,
            mode='max',
            verbose=1,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=PATIENCE // 2,
            min_lr=1e-6,
            verbose=1
        ),
        WarmUpCosineDecayScheduler(
            learning_rate_base=LEARNING_RATE,
            total_epochs=EPOCHS,
            warmup_epochs=10,
            min_learning_rate=1e-6
        ),
        TensorBoard(
            log_dir=os.path.join(checkpoint_dir, 'logs'),
            update_freq='epoch'
        )
    ]
    
    # Train the model
    print("\nStarting training...")
    start_time = time.time()
    
    # Setup fit parameters
    fit_params = {
        'x': train_generator,
        'epochs': EPOCHS,
        'validation_data': val_generator,
        'callbacks': callbacks,
        'verbose': 1
    }
    
    # Add class weights if enabled
    if APPLY_CLASS_WEIGHTING and class_weights:
        fit_params['class_weight'] = class_weights
        print("Using class weights for training")
    
    # Execute training
    history = model.fit(**fit_params)
    
    # Report training time
    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Evaluate the best model
    print("\nEvaluating the best model...")
    best_model_path = os.path.join(checkpoint_dir, 'best_model.h5')
    if os.path.exists(best_model_path):
        model.load_weights(best_model_path)
    
    # Final evaluation
    loss, accuracy = model.evaluate(val_generator, verbose=1)
    print(f"\nFinal validation accuracy: {accuracy:.4f}")
    
    # Save evaluation metrics
    metrics_path = os.path.join(checkpoint_dir, "metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"Final validation accuracy: {accuracy:.4f}\n")
        f.write(f"Final validation loss: {loss:.4f}\n")
        f.write(f"Training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s\n")
    
    print(f"\nTraining successful! Model saved in: {checkpoint_dir}")

if __name__ == '__main__':
    train_model()
