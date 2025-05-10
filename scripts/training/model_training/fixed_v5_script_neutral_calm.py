#!/usr/bin/env python3
# coding: utf-8
"""
Wav2Vec Audio Emotion Recognition Training Script
Version 5: Maps 'calm' emotion to 'neutral' emotion
"""

import os
import datetime
import argparse
import glob
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, Callback
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, LSTM, TimeDistributed, Bidirectional, Attention, Input, Layer, GlobalAveragePooling1D, BatchNormalization

# Utility function for feature loading to handle various file formats
def improved_load_features(file_path):
    """Improved function to load features from various file formats"""
    if file_path.endswith('.npz'):
        # Load NPZ file
        try:
            data = np.load(file_path)
            # Check for various keys
            for key in ['features', 'embedding', 'embeddings', 'wav2vec']:
                if key in data:
                    return data[key]
            
            # If no recognized keys, return the first array
            return data[list(data.keys())[0]]
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    elif file_path.endswith('.npy'):
        # Load NPY file
        try:
            return np.load(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    else:
        print(f"Unsupported file format: {file_path}")
        return None

# Custom callback to detect NaN values
class NaNLossDetector(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None and (np.isnan(loss) or np.isinf(loss)):
            print(f"\nEpoch {epoch}: Invalid loss, terminating training")
            self.model.stop_training = True

# Fixed learning rate scheduler with .value() issue corrected
class WarmUpReduceLROnPlateau(Callback):
    def __init__(
        self,
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1,
        mode='auto',
        min_delta=1e-4,
        cooldown=0,
        min_lr=1e-8,
        warmup_epochs=5,
        learning_rate_base=0.001,
        **kwargs
    ):
        super().__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.min_delta = min_delta
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.learning_rate_base = learning_rate_base

        self.monitor_op = None
        self.best = None
        self.cooldown_counter = 0
        self.wait = 0

        if mode == 'min' or (mode == 'auto' and 'loss' in monitor):
            self.monitor_op = lambda a, b: a < b - min_delta
            self.best = float('inf')
        else:
            self.monitor_op = lambda a, b: a > b + min_delta
            self.best = -float('inf')

    def on_epoch_begin(self, epoch, logs=None):
        # Apply learning rate warm-up
        if epoch < self.warmup_epochs:
            # Ensure learning_rate is float32 for consistency
            learning_rate = tf.cast(self.learning_rate_base * (epoch + 1) / self.warmup_epochs, dtype=tf.float32)
            if self.verbose > 0:
                print(f'\nEpoch {epoch+1}: WarmUpReduceLROnPlateau setting learning rate to {learning_rate.numpy()}.')
            # Directly assign the learning rate value
            self.model.optimizer.learning_rate.assign(learning_rate)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)

        if epoch < self.warmup_epochs:
            return  # Skip LR reduction during warm-up

        if current is None:
            if self.verbose > 0:
                print(f'\nWarmUpReduceLROnPlateau conditioned on metric `{self.monitor}` which is not available. Available metrics are: {", ".join(list(logs.keys()))}')
            return

        if self.in_cooldown():
            self.cooldown_counter -= 1
            self.wait = 0

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        elif not self.in_cooldown():
            self.wait += 1
            if self.wait >= self.patience:
                # Get current LR value correctly - FIXED for TF compatibility
                # Using float() instead of value() for ResourceVariable access
                old_lr = float(self.model.optimizer.learning_rate)
                if old_lr > self.min_lr:
                    new_lr = old_lr * self.factor
                    # Ensure new_lr is float32
                    new_lr = tf.cast(max(new_lr, self.min_lr), dtype=tf.float32)
                    # Assign the new learning rate value
                    self.model.optimizer.learning_rate.assign(new_lr)
                    if self.verbose > 0:
                        print(f'\nEpoch {epoch+1}: WarmUpReduceLROnPlateau reducing learning rate to {new_lr.numpy()}.')
                    self.cooldown_counter = self.cooldown
                    self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0

class SelfAttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(SelfAttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(SelfAttentionLayer, self).build(input_shape)
    
    def call(self, x):
        # Linear transformation
        e = tf.keras.backend.squeeze(tf.keras.backend.dot(x, self.W) + self.b, axis=-1)
        # Attention weights
        a = tf.keras.backend.softmax(e)
        # Weighted sum (context vector)
        output = x * tf.keras.backend.expand_dims(a, axis=-1)
        return output
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        return super(SelfAttentionLayer, self).get_config()

def load_emotion_data(data_dir, test_size=0.1, use_cache=False, cache_dir="cache"):
    """
    Load and prepare wav2vec feature data for emotion recognition.
    Supports new data structure with flat directory and NPZ files.
    Maps 'calm' to 'neutral' emotion.
    """
    cache_file = os.path.join(cache_dir, "wav2vec_data_cache_neutral_calm.npz")
    os.makedirs(cache_dir, exist_ok=True)

    if use_cache and os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        cache = np.load(cache_file, allow_pickle=True)
        X_train = cache['X_train']
        X_val = cache['X_val']
        y_train = cache['y_train']
        y_val = cache['y_val']
        class_weights = cache['class_weights'].item()
        label_encoder = cache['label_encoder'].item()
        max_length = cache['max_length'].item()
        return X_train, X_val, y_train, y_val, class_weights, label_encoder, max_length

    print(f"Loading data from {data_dir}")

    # Check if we need to look in models/wav2vec directory
    if not os.path.exists(data_dir):
        alt_dir = os.path.join(os.path.dirname(data_dir), "models", "wav2vec")
        if os.path.exists(alt_dir):
            print(f"Directory {data_dir} not found. Using {alt_dir} instead.")
            data_dir = alt_dir

    # Find all .npz or .npy files
    feature_files = []
    for ext in ['.npz', '.npy']:
        feature_files.extend(glob.glob(os.path.join(data_dir, f"*{ext}")))

    print(f"Found {len(feature_files)} feature files")

    if len(feature_files) == 0:
        raise ValueError(f"No valid files found in {data_dir}!")

    # Extract labels from filenames
    labels = []
    valid_files = []

    # Map emotion labels to indices for consistent encoding
    # MODIFICATION: Map 'calm' (1) to the same index as 'neutral' (0)
    emotion_to_index = {
        'neutral': 0,
        'calm': 0,  # MODIFIED: Map 'calm' to index 0 (same as 'neutral')
        'happy': 2,
        'sad': 3,
        'angry': 4,
        'fear': 5,
        'disgust': 6,
        'surprise': 7
    }

    skipped_files = 0
    for file_path in feature_files:
        base_name = os.path.basename(file_path)
        
        # Try to extract label from filename
        try:
            # Try different file naming conventions
            if '_' in base_name:
                parts = base_name.split('_')
                
                # Handle various file formats
                if len(parts) >= 3:
                    # Try to parse emotion from different positions
                    emotion = None
                    
                    # Check common positions for emotion
                    for i in [2, 3]:
                        if i < len(parts):
                            potential_emotion = parts[i].lower()
                            if potential_emotion in emotion_to_index:
                                emotion = potential_emotion
                                break
                    
                    if emotion is None:
                        # Try looking for known emotion names anywhere in the filename
                        for part in parts:
                            part_lower = part.lower()
                            if part_lower in emotion_to_index:
                                emotion = part_lower
                                break
                    
                    if emotion:
                        emotion_idx = emotion_to_index.get(emotion)
                        if emotion_idx is not None:
                            # Append to valid lists only if we have a valid emotion
                            labels.append(emotion_idx)
                            valid_files.append(file_path)
                        else:
                            skipped_files += 1
                    else:
                        skipped_files += 1
                else:
                    skipped_files += 1
            else:
                skipped_files += 1
                
        except Exception as e:
            print(f"Error parsing filename {base_name}: {e}")
            skipped_files += 1
            continue

    print(f"Skipped {skipped_files} files due to parsing errors or excluded emotions")
    print(f"Using {len(valid_files)} valid files")

    if len(valid_files) == 0:
        raise ValueError("No valid files found after parsing!")

    # Load normalization statistics if available
    mean_path = os.path.join(data_dir, "audio_mean.npy")
    std_path = os.path.join(data_dir, "audio_std.npy")
    
    if os.path.exists(mean_path) and os.path.exists(std_path):
        print("Loading existing normalization statistics")
        audio_mean = np.load(mean_path)
        audio_std = np.load(std_path)
    else:
        print("Computing normalization statistics...")
        # Load a sample to get feature dimensions
        sample_features = improved_load_features(valid_files[0])
        if sample_features is None:
            raise ValueError(f"Failed to load the first file: {valid_files[0]}")
        
        feature_dim = sample_features.shape[-1]
        # Initialize sum, sum of squares, and count for mean/std calculation
        feature_sum = np.zeros(feature_dim)
        feature_squared_sum = np.zeros(feature_dim)
        total_frames = 0
        
        # Compute statistics (mean and std) from data
        for i, file_path in enumerate(valid_files):
            if i % 500 == 0:
                print(f"Processed {i}/{len(valid_files)} files")
            
            features = improved_load_features(file_path)
            if features is None:
                continue
                
            feature_sum += np.sum(features, axis=0)
            feature_squared_sum += np.sum(features**2, axis=0)
            total_frames += features.shape[0]
        
        # Calculate mean and std
        audio_mean = feature_sum / total_frames
        audio_std = np.sqrt(feature_squared_sum / total_frames - audio_mean**2)
        
        # Save for future use
        np.save(mean_path, audio_mean)
        np.save(std_path, audio_std)

    # Find the 95th percentile of sequence length
    # Load all features to determine sequence length distribution
    print("Determining sequence length distribution...")
    lengths = []
    processed_files = 0
    failed_files = 0
    
    for file_path in valid_files:
        try:
            features = improved_load_features(file_path)
            if features is not None:
                lengths.append(features.shape[0])
                processed_files += 1
            else:
                failed_files += 1
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            failed_files += 1
    
    print(f"Processed {processed_files}/{len(valid_files)} files")
    print(f"Failed to process {failed_files} files")
    
    max_length = int(np.percentile(lengths, 95))
    print(f"Using max sequence length of {max_length} (95th percentile)")

    # Prepare data with fixed-length sequences
    X = []
    y = []
    
    for i, file_path in enumerate(valid_files):
        if i % 500 == 0:
            print(f"Processed {i}/{len(valid_files)} files")
            
        # Load and normalize features
        features = improved_load_features(file_path)
        if features is None:
            continue
            
        # Normalize features
        features = (features - audio_mean) / (audio_std + 1e-8)
        
        # Handle sequence length
        if features.shape[0] > max_length:
            # Truncate longer sequences
            features = features[:max_length]
        elif features.shape[0] < max_length:
            # Pad shorter sequences with zeros
            padding = np.zeros((max_length - features.shape[0], features.shape[1]))
            features = np.vstack([features, padding])
        
        X.append(features)
        y.append(labels[i])

    X = np.array(X)
    y = np.array(y)

    # Create label encoder and one-hot encode the labels
    label_encoder = LabelEncoder()
    label_encoder.fit(np.unique(y))
    y_encoded = label_encoder.transform(y)
    
    # Calculate class weights for imbalanced classes
    class_counts = np.bincount(y_encoded)
    total_samples = len(y_encoded)
    
    # Create a dictionary of class weights
    class_weights = {}
    for i in range(len(class_counts)):
        if class_counts[i] > 0:  # Avoid division by zero
            # Weight = total_samples / (n_classes * samples_in_class)
            class_weights[i] = total_samples / (len(class_counts) * class_counts[i])
        else:
            # Default weight for missing classes
            class_weights[i] = 1.0
    
    # Print class distribution
    print("\nClass distribution:")
    for i in range(len(class_counts)):
        if class_counts[i] > 0:
            class_name = label_encoder.inverse_transform([i])[0]
            print(f"  Class {i} ({class_name}): {class_counts[i]} samples (weight: {class_weights[i]:.4f})")
    
    # Print any classes missing from the training set
    missing_classes = [i for i in range(len(label_encoder.classes_)) if i not in y_encoded]
    if missing_classes:
        print(f"  Classes missing from training set (assigned weight 1.0): {missing_classes}")
    
    # One-hot encode labels
    num_classes = len(label_encoder.classes_)
    y_onehot = np.zeros((len(y_encoded), num_classes))
    for i, label_idx in enumerate(y_encoded):
        y_onehot[i, label_idx] = 1
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_onehot, test_size=test_size, random_state=42, stratify=y_encoded
    )
    
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Cache processed data
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(
            cache_file,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            class_weights=class_weights,
            label_encoder=label_encoder,
            max_length=max_length
        )
    
    return X_train, X_val, y_train, y_val, class_weights, label_encoder, max_length

def build_model(input_shape, num_classes, dropout_rate=0.5):
    """Build wav2vec-based emotion recognition model"""
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Bidirectional LSTM layers
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = Dropout(dropout_rate)(x)
    
    # Self-attention
    x = SelfAttentionLayer()(x)
    
    # Global pooling
    x = GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def train_model(data_dir, batch_size=64, epochs=100, learning_rate=0.001, dropout_rate=0.5, 
                model_name=None, use_cache=False, checkpoint_dir="checkpoints", early_stop=None):
    """Train wav2vec-based emotion recognition model"""
    
    # Create checkpoints directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load and prepare data
    X_train, X_val, y_train, y_val, class_weights, label_encoder, max_length = load_emotion_data(
        data_dir, test_size=0.1, use_cache=use_cache
    )
    
    # Build model
    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]
    model = build_model(input_shape, num_classes, dropout_rate)
    
    # Setup optimizer with gradient clipping
    # Use only clipnorm or clipvalue, not both
    # For stability we'll use clipnorm
    optimizer = SGD(
        learning_rate=learning_rate,
        momentum=0.9,
        nesterov=True,
        clipnorm=1.0  # Only use clipnorm, not both
    )
    
    # Compile model with label smoothing for stability
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    # Generate model name with timestamp if not provided
    if model_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"wav2vec_audio_only_neutral_calm_{timestamp}"
    
    # Setup callbacks with custom LR scheduler and NaN detection
    callbacks = [
        ModelCheckpoint(
            os.path.join(checkpoint_dir, f"{model_name}_best.weights.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            mode='max',
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(checkpoint_dir, f"{model_name}_final.weights.h5"),
            save_weights_only=True,
            save_best_only=False,
            save_freq='epoch'
        ),
        TensorBoard(
            log_dir=os.path.join("logs", model_name),
            histogram_freq=1,
            update_freq='epoch'
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=early_stop if early_stop else 20,
            restore_best_weights=True,
            verbose=1
        ),
        WarmUpReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
            warmup_epochs=5,
            learning_rate_base=learning_rate
        ),
        NaNLossDetector()
    ]
    
    # Train model
    print(f"Training model: {model_name}")
    print(f"Input shape: {input_shape}, Classes: {num_classes}")
    print(f"Training for {epochs} epochs with batch size {batch_size}")
    
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Save history for later analysis
    history_file = os.path.join(checkpoint_dir, f"{model_name}_history.json")
    with open(history_file, 'w') as f:
        history_dict = {key: [float(x) for x in values] for key, values in history.history.items()}
        json.dump(history_dict, f)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, f"{model_name}_training_curves.png"))
    
    # Final evaluation
    print("\nEvaluating model on validation set:")
    results = model.evaluate(X_val, y_val, batch_size=batch_size, verbose=1)
    print(f"Validation loss: {results[0]:.4f}, Validation accuracy: {results[1]:.4f}")
    
    # Save validation results in a summary file
    with open(os.path.join(checkpoint_dir, f"{model_name}_validation_summary.csv"), 'w') as f:
        f.write("metric,value\n")
        f.write(f"val_loss,{results[0]}\n")
        f.write(f"val_accuracy,{results[1]:.4f}\n")
    
    # Also save as a json file
    with open(os.path.join(checkpoint_dir, f"{model_name}_validation_accuracy.json"), 'w') as f:
        json.dump({"val_accuracy": float(results[1])}, f)

    print(f"Training complete. Best weights saved to {checkpoint_dir}/{model_name}_best.weights.h5")
    
    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train wav2vec-based emotion recognition model")
    parser.add_argument("--data_dir", type=str, default="data/wav2vec", 
                        help="Directory containing wav2vec features")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=100, 
                        help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, 
                        help="Initial learning rate")
    parser.add_argument("--dropout", type=float, default=0.5, 
                        help="Dropout rate")
    parser.add_argument("--model_name", type=str, default=None, 
                        help="Custom model name prefix")
    parser.add_argument("--use_cache", action="store_true", 
                        help="Use cached pre-processed data")
    parser.add_argument("--early_stop", type=int, default=20, 
                        help="Early stopping patience")
    
    args = parser.parse_args()
    
    # Train model with parsed arguments
    train_model(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        dropout_rate=args.dropout,
        model_name=args.model_name,
        use_cache=args.use_cache,
        early_stop=args.early_stop
    )
