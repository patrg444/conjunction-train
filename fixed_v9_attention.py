#!/usr/bin/env python3
"""
WAV2VEC audio emotion recognition model with attention mechanism and improved regularization.
Key improvements:
- Temporal attention mechanism to focus on salient frames
- Layer normalization and recurrent dropout for better regularization
- Improved learning rate schedule with cosine decay
- Train/val split before normalization to avoid data leakage
- Increased sequence padding with masking to preserve temporal information
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Bidirectional, LSTM, LayerNormalization,
    Attention, GlobalAveragePooling1D, Masking
)
import glob
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
import math
from tensorflow.keras.regularizers import l2

# Set random seed
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class CosineDecayWithWarmup(Callback):
    """
    Combined learning rate scheduler with:
    1. Linear warmup for initial epochs
    2. ReduceLROnPlateau for middle epochs
    3. Cosine decay for final epochs
    """
    def __init__(
        self,
        min_lr=5e-6,
        max_lr=3e-4,
        warmup_epochs=8,
        cosine_decay_start=20,
        total_epochs=100,
        verbose=1
    ):
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_epochs = warmup_epochs
        self.cosine_decay_start = cosine_decay_start
        self.total_epochs = total_epochs
        self.verbose = verbose
        self.reduce_lr = None
        
    def on_train_begin(self, logs=None):
        # Start with minimum learning rate
        self.model.optimizer.learning_rate.assign(self.min_lr)
        
        # Initialize ReduceLROnPlateau
        self.reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=5,
            min_lr=self.min_lr,
            verbose=self.verbose
        )
        self.reduce_lr.set_model(self.model)
        self.reduce_lr.on_train_begin(logs)
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # Linear warmup
            warmup_lr = self.min_lr + (self.max_lr - self.min_lr) * (epoch / self.warmup_epochs)
            self.model.optimizer.learning_rate.assign(warmup_lr)
            if self.verbose > 0:
                print(f"\nEpoch {epoch+1}: CosineDecayWithWarmup setting learning rate to {warmup_lr:.6f} (warmup phase).")
        elif epoch >= self.cosine_decay_start:
            # Cosine decay from current LR to min_lr
            progress = (epoch - self.cosine_decay_start) / (self.total_epochs - self.cosine_decay_start)
            progress = min(1.0, progress)  # Cap at 1.0
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            current_lr = self.min_lr + cosine_factor * (float(self.model.optimizer.learning_rate) - self.min_lr)
            self.model.optimizer.learning_rate.assign(current_lr)
            if self.verbose > 0:
                print(f"\nEpoch {epoch+1}: CosineDecayWithWarmup setting learning rate to {current_lr:.6f} (cosine decay phase).")
        
    def on_epoch_end(self, epoch, logs=None):
        # Let ReduceLROnPlateau handle mid-training between warmup and cosine decay
        if self.warmup_epochs <= epoch < self.cosine_decay_start:
            self.reduce_lr.on_epoch_end(epoch, logs)

def load_wav2vec_data(dataset_path, max_samples=None):
    print(f"Loading data from {dataset_path}")
    npz_files = glob.glob(os.path.join(dataset_path, "*.npz"))
    if not npz_files:
        print(f"No npz files found in {dataset_path}")
        return [], []
    
    print(f"Found {len(npz_files)} files")
    if max_samples and max_samples < len(npz_files):
        npz_files = npz_files[:max_samples]
    
    features = []
    labels = []
    sequence_lengths = []
    
    # Count instances of each emotion
    emotion_counts = {}
    
    for i, npz_file in enumerate(npz_files):
        if i % 500 == 0:
            print(f"Processing file {i+1}/{len(npz_files)}")
            
        try:
            data = np.load(npz_file, allow_pickle=True)
            
            # Fixed key name: using wav2vec_features
            feature = data['wav2vec_features']
            sequence_lengths.append(len(feature))
            
            label = data['emotion'].item() if isinstance(data['emotion'], np.ndarray) else data['emotion']
            
            if label in emotion_counts:
                emotion_counts[label] += 1
            else:
                emotion_counts[label] = 1
            
            features.append(feature)
            labels.append(label)
        except Exception as e:
            print(f"Error loading {npz_file}: {e}")
    
    # Print emotion distribution
    for emotion, count in sorted(emotion_counts.items()):
        print(f"  {emotion}: {count} samples")
    
    # Print sequence length statistics
    sequence_lengths = np.array(sequence_lengths)
    print(f"Sequence length statistics:")
    print(f"  Min: {np.min(sequence_lengths)}")
    print(f"  Max: {np.max(sequence_lengths)}")
    print(f"  Mean: {np.mean(sequence_lengths):.2f}")
    print(f"  Median: {np.median(sequence_lengths)}")
    print(f"  95th percentile: {np.percentile(sequence_lengths, 95)}")
    print(f"  98th percentile: {np.percentile(sequence_lengths, 98)}")
    
    return features, labels

def normalize_features(features, mean_path=None, std_path=None, save_stats=False):
    """
    Normalize features using mean and std.
    Optionally save or load statistics from disk.
    """
    # Calculate stats on flattened features
    all_features = np.vstack([f for f in features])
    
    if mean_path and os.path.exists(mean_path) and std_path and os.path.exists(std_path):
        print("Loading existing normalization statistics")
        mean = np.load(mean_path)
        std = np.load(std_path)
    else:
        print("Computing normalization statistics")
        mean = np.mean(all_features, axis=0)
        std = np.std(all_features, axis=0)
        std[std < 1e-5] = 1e-5  # Avoid division by zero or very small values
        
        if save_stats and mean_path and std_path:
            np.save(mean_path, mean)
            np.save(std_path, std)
    
    # Normalize each feature sequence
    normalized_features = []
    for feature in features:
        normalized_features.append((feature - mean) / std)
    
    return normalized_features, mean, std

def pad_sequences(features, max_length=None):
    if max_length is None:
        # Use the 98th percentile length to include more sequence information
        # while still avoiding outliers
        lengths = [len(f) for f in features]
        max_length = int(np.percentile(lengths, 98))
    
    print(f"Padding sequences to length {max_length}")
    
    # Get feature dimension
    feature_dim = features[0].shape[1]
    
    # Initialize output array
    padded_features = np.zeros((len(features), max_length, feature_dim))
    
    # Fill with actual data (truncate if needed)
    for i, feature in enumerate(features):
        seq_length = min(len(feature), max_length)
        padded_features[i, :seq_length, :] = feature[:seq_length]
    
    return padded_features, max_length

def build_model(input_shape, num_classes):
    print(f"Building model with {num_classes} output classes...")
    
    # Input layer with masking to handle variable-length sequences
    input_layer = Input(shape=input_shape, name="input_layer")
    masked_input = Masking(mask_value=0.0)(input_layer)
    
    # Bidirectional LSTM with recurrent dropout for regularization
    x = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.25))(masked_input)
    
    # Add Layer Normalization for more stable training
    x = LayerNormalization()(x)
    
    # Self-attention mechanism to focus on important frames
    context_vector = Attention(use_scale=True)([x, x])
    
    # Global pooling to convert attention output to fixed-length representation
    x = GlobalAveragePooling1D()(context_vector)
    
    # Dense layers with stronger regularization
    x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    output_layer = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),  # Starting with min LR, scheduler will adjust
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    
    return model

def main():
    # Set paths
    dataset_path = "/home/ubuntu/audio_emotion/models/wav2vec"
    checkpoint_dir = "/home/ubuntu/audio_emotion/checkpoints"
    mean_path = "/home/ubuntu/audio_emotion/audio_mean_v9.npy"
    std_path = "/home/ubuntu/audio_emotion/audio_std_v9.npy"
    
    # Enable smoke test mode if environment variable is set
    smoke_test = os.getenv("SMOKE_TEST") == "1"
    if smoke_test:
        print("RUNNING IN SMOKE TEST MODE - using minimal dataset")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load data
    features, labels = load_wav2vec_data(dataset_path, max_samples=10 if smoke_test else None)
    if not features:
        print("No data loaded. Exiting.")
        return
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    print(f"Original unique labels: {label_encoder.classes_}")
    print(f"Number of classes after encoding: {len(label_encoder.classes_)}")
    
    # Split data BEFORE normalization to prevent data leakage
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        features, encoded_labels, test_size=0.1, random_state=42, stratify=encoded_labels
    )
    
    print(f"Train samples: {len(X_train_raw)}")
    print(f"Validation samples: {len(X_val_raw)}")
    
    # Normalize features using only training data stats to avoid data leakage
    X_train_norm, mean, std = normalize_features(
        X_train_raw, 
        mean_path=mean_path if not smoke_test else None, 
        std_path=std_path if not smoke_test else None,
        save_stats=not smoke_test
    )
    
    # Apply the same normalization to validation data
    X_val_norm = [(x - mean) / std for x in X_val_raw]
    
    # Calculate max length based on training data percentiles
    train_lengths = [len(f) for f in X_train_norm]
    max_length = int(np.percentile(train_lengths, 98))  # Using 98th percentile instead of 95th
    
    # Pad sequences
    X_train, _ = pad_sequences(X_train_norm, max_length)
    X_val, _ = pad_sequences(X_val_norm, max_length)
    
    print(f"Padded train shape: {X_train.shape}")
    print(f"Padded validation shape: {X_val.shape}")
    
    # Calculate class weights for imbalanced data
    class_weights = {}
    max_count = max(Counter(y_train).values())
    for class_id, count in Counter(y_train).items():
        weight = max_count / count
        class_weights[class_id] = weight
        print(f"  Class {class_id}: {count} samples (weight: {weight:.4f})")
    
    # Build and train the model
    input_shape = (max_length, X_train.shape[2])
    model = build_model(input_shape, len(label_encoder.classes_))
    
    # Set up callbacks
    callbacks = [
        CosineDecayWithWarmup(
            min_lr=5e-6,
            max_lr=3e-4,
            warmup_epochs=8,
            cosine_decay_start=20,
            total_epochs=100,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_dir, "best_model_v9.h5"), 
            save_best_only=True, 
            monitor='val_accuracy', 
            mode='max', 
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', 
            patience=12, 
            restore_best_weights=True, 
            verbose=1
        ),
        # TensorBoard for optional visualization
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(checkpoint_dir, "logs"),
            histogram_freq=1
        )
    ]
    
    # In smoke test mode, only run a single batch to verify the model works
    if smoke_test:
        print("SMOKE TEST: Running single batch to verify model...")
        model.fit(
            X_train[:2], 
            y_train[:2],
            epochs=1,
            batch_size=2,
            verbose=1
        )
        print("SMOKE TEST PASSED: Model can train on data.")
        return
    
    # Train the model with full data
    history = model.fit(
        X_train, 
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Evaluate the model
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=1)
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    
    # Save model and label encoder
    model.save(os.path.join(checkpoint_dir, "final_model_v9.h5"))
    np.save(os.path.join(checkpoint_dir, "label_classes_v9.npy"), label_encoder.classes_)
    
    print("Training completed.")

if __name__ == "__main__":
    # Run smoke test if environment variable is set
    if os.getenv("SMOKE_TEST") == "1":
        # Use very small subset for testing
        main()
    else:
        # Run full training
        main()
