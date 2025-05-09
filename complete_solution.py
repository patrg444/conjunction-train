#!/usr/bin/env python3
"""
Complete WAV2VEC solution that:
1. Fixes the comma syntax error in set_value calls
2. Uses correct key names in NPZ files (wav2vec_features)
3. Adds padding to handle variable-length sequences
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Bidirectional, LSTM
import glob
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class WarmUpReduceLROnPlateau(Callback):
    """Learning rate scheduler with warm-up and plateau reduction."""
    
    def __init__(
        self,
        min_lr=1e-6,
        max_lr=1e-3,
        warmup_epochs=5,
        patience=5,
        factor=0.5,
        monitor='val_loss',
        mode='min',
        verbose=1
    ):
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.factor = factor
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best = float('inf') if mode == 'min' else -float('inf')
        self.wait = 0
        self.best_weights = None
        self.restore_best = True
        
    def on_train_begin(self, logs=None):
        # Initialize the learning rate at min_lr
        self.model.optimizer.learning_rate = self.min_lr
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            warmup_lr = self.min_lr + (self.max_lr - self.min_lr) * (epoch / self.warmup_epochs)
            # Fixed line: added comma between learning_rate and warmup_lr
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, warmup_lr)
            if self.verbose > 0:
                print(f"\nEpoch {epoch+1}: WarmUpReduceLROnPlateau setting learning rate to {warmup_lr:.6f}.")
        
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if epoch >= self.warmup_epochs:
            if self.mode == 'min':
                if current < self.best:
                    self.best = current
                    self.wait = 0
                    if self.restore_best:
                        self.best_weights = self.model.get_weights()
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        old_lr = float(self.model.optimizer.learning_rate)
                        if old_lr > self.min_lr:
                            new_lr = old_lr * self.factor
                            new_lr = max(new_lr, self.min_lr)
                            # Fixed line: added comma between learning_rate and new_lr
                            tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
                            self.wait = 0
                            if self.verbose > 0:
                                print(f"\nEpoch {epoch+1}: ReduceLROnPlateau reducing learning rate to {new_lr:.6f}.")

def load_wav2vec_data(dataset_path, max_samples=None):
    """Load WAV2VEC features and labels from the dataset directory."""
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
            
            # Fixed key name: using wav2vec_features instead of embedding
            feature = data['wav2vec_features']
            sequence_lengths.append(len(feature))
            
            # Make sure to use the emotion key correctly
            label = data['emotion'].item() if isinstance(data['emotion'], np.ndarray) else data['emotion']
            
            # Update emotion counts
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
    
    return features, labels

def normalize_features(features, mean_path=None, std_path=None):
    """Normalize features using mean and standard deviation."""
    # First, calculate stats on flattened features
    all_features = np.vstack([f for f in features])
    
    if mean_path and os.path.exists(mean_path) and std_path and os.path.exists(std_path):
        print("Loading existing normalization statistics")
        mean = np.load(mean_path)
        std = np.load(std_path)
    else:
        print("Computing normalization statistics")
        mean = np.mean(all_features, axis=0)
        std = np.std(all_features, axis=0)
        std[std == 0] = 1e-5  # Avoid division by zero
        
        if mean_path and std_path:
            np.save(mean_path, mean)
            np.save(std_path, std)
    
    # Normalize each feature sequence
    normalized_features = []
    for feature in features:
        normalized_features.append((feature - mean) / std)
    
    return normalized_features

def pad_sequences(features, max_length=None):
    """Pad sequences to the same length."""
    if max_length is None:
        # Use the 95th percentile length to avoid outliers
        lengths = [len(f) for f in features]
        max_length = int(np.percentile(lengths, 95))
    
    print(f"Padding sequences to length {max_length}")
    
    # Get feature dimension
    feature_dim = features[0].shape[1]
    
    # Initialize output array
    padded_features = np.zeros((len(features), max_length, feature_dim))
    
    # Fill with actual data (truncate if needed)
    for i, feature in enumerate(features):
        seq_length = min(len(feature), max_length)
        padded_features[i, :seq_length, :] = feature[:seq_length]
    
    return padded_features

def build_model(input_shape, num_classes):
    """Build Wav2Vec emotional recognition model."""
    print(f"Building model with {num_classes} output classes...")
    input_layer = Input(shape=input_shape, name="input_layer")
    
    # Bidirectional LSTM layers
    x = Bidirectional(LSTM(128, return_sequences=True))(input_layer)
    x = Bidirectional(LSTM(128))(x)
    
    # Dense layers with dropout
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Output layer
    output_layer = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    
    return model

def main():
    dataset_path = "/home/ubuntu/audio_emotion/models/wav2vec"
    checkpoint_dir = "/home/ubuntu/audio_emotion/checkpoints"
    mean_path = "/home/ubuntu/audio_emotion/audio_mean.npy"
    std_path = "/home/ubuntu/audio_emotion/audio_std.npy"
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load data
    features, labels = load_wav2vec_data(dataset_path)
    if not features:
        print("No data loaded. Exiting.")
        return
    
    # Normalize features
    features = normalize_features(features, mean_path, std_path)
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    print(f"Original unique label values: {np.unique(encoded_labels)}")
    print(f"Number of classes after encoding: {len(label_encoder.classes_)}")
    
    # Split data before padding to prevent data leakage
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        features, encoded_labels, test_size=0.1, random_state=42, stratify=encoded_labels
    )
    
    print(f"Train samples: {len(X_train_raw)}")
    print(f"Validation samples: {len(X_val_raw)}")
    
    # Pad sequences
    # Calculate max length based on training data only to prevent data leakage
    train_lengths = [len(f) for f in X_train_raw]
    max_length = int(np.percentile(train_lengths, 95))
    
    # Pad training and validation data
    X_train = pad_sequences(X_train_raw, max_length)
    X_val = pad_sequences(X_val_raw, max_length)
    
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
    input_shape = (max_length, X_train.shape[2])  # Fixed sequence length
    model = build_model(input_shape, len(label_encoder.classes_))
    
    # Set up callbacks
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.h5")
    callbacks = [
        WarmUpReduceLROnPlateau(
            min_lr=1e-6, 
            max_lr=1e-3, 
            warmup_epochs=5, 
            patience=7, 
            factor=0.5, 
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, 
            save_best_only=True, 
            monitor='val_accuracy', 
            mode='max', 
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=15, 
            restore_best_weights=True, 
            verbose=1
        )
    ]
    
    # Train the model
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
    
    # Save model
    model.save(os.path.join(checkpoint_dir, "final_model.h5"))
    
    # Save label encoder classes for inference
    np.save(os.path.join(checkpoint_dir, "label_classes.npy"), label_encoder.classes_)
    
    print("Training completed.")

if __name__ == "__main__":
    main()
