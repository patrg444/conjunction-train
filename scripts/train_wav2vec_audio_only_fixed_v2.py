#!/usr/bin/env python3
# Improved wav2vec feature-based audio emotion recognition training
# This version includes numerical stability improvements

import os
import argparse
import math
import json
import numpy as np
import tensorflow as tf
import random
import glob
import datetime
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Dropout, Bidirectional,
    LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D,
    Attention, TimeDistributed, Conv1D, MaxPooling1D, Lambda, Reshape
)
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, Callback
)
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Include the fixed loader code directly for numerical stability features
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

# Set precision explicitly to float32 for numerical stability
tf.keras.mixed_precision.set_global_policy("float32")

# Enable operation determinism for reproducibility
tf.config.experimental.enable_op_determinism()

# Seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Callback to detect NaN in loss
class NaNLossDetector(Callback):
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None and (math.isnan(loss) or math.isinf(loss)):
            print(f'NaN/Inf loss detected: {loss} at batch {batch}, stopping training.')
            self.model.stop_training = True

# Custom learning rate scheduler with warm-up
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
        warmup_epochs=10,
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
            learning_rate = self.learning_rate_base * (epoch + 1) / self.warmup_epochs
            if self.verbose > 0:
                print(f'\nEpoch {epoch+1}: WarmUpReduceLROnPlateau setting learning rate to {learning_rate}.')
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, learning_rate)
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if epoch < self.warmup_epochs:
            return  # Skip LR reduction during warm-up
            
        if current is None:
            if self.verbose > 0:
                print(f'\nWarmUpReduceLROnPlateau conditioned on metric `{self.monitor}` which is not available. Available metrics are: {",".join(list(logs.keys()))}')
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
                old_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                if old_lr > self.min_lr:
                    new_lr = old_lr * self.factor
                    new_lr = max(new_lr, self.min_lr)
                    tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
                    if self.verbose > 0:
                        print(f'\nEpoch {epoch+1}: WarmUpReduceLROnPlateau reducing learning rate to {new_lr}.')
                    self.cooldown_counter = self.cooldown
                    self.wait = 0
    
    def in_cooldown(self):
        return self.cooldown_counter > 0

def load_emotion_data(data_dir, test_size=0.1, use_cache=False, cache_dir="cache"):
    """
    Load and prepare wav2vec feature data for emotion recognition.
    Includes numerical stability enhancements.
    """
    cache_file = os.path.join(cache_dir, "wav2vec_data_cache.npz")
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
    
    # Find all .npz files
    feature_files = glob.glob(os.path.join(data_dir, "**/*.npz"), recursive=True)
    print(f"Found {len(feature_files)} feature files")
    
    # Extract labels from filenames - they're in the parent directory name
    labels = []
    valid_files = []
    
    allowed_emotions = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]
    emotion_map = {e: i for i, e in enumerate(allowed_emotions)}
    
    skipped = 0
    for file in feature_files:
        try:
            # Extract emotion from parent directory name
            parts = file.split(os.sep)
            emotion = parts[-2].lower()  # Parent directory is the emotion
            
            # Skip files that don't match our emotion set
            if emotion not in emotion_map:
                skipped += 1
                continue
                
            labels.append(emotion)
            valid_files.append(file)
        except Exception as e:
            print(f"Error parsing {file}: {e}")
            skipped += 1
    
    print(f"Skipped {skipped} files due to parsing errors or excluded emotions")
    print(f"Using {len(valid_files)} valid files")
    
    if len(valid_files) == 0:
        raise ValueError("No valid files found! Check your data directory and file structure.")
    
    # Create and fit label encoder
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform([emotion_map[l] for l in labels])
    
    # Calculate sequence lengths to find the optimal max length
    # Using 95th percentile to avoid padding too much due to outliers
    sequence_lengths = []
    for file in valid_files:
        try:
            features = np.load(file)
            sequence_lengths.append(len(features))
        except:
            pass
    
    max_length = int(np.percentile(sequence_lengths, 95))
    print(f"Using max sequence length of {max_length} (95th percentile)")
    
    # Load normalization statistics
    mean_file = os.path.join(data_dir, "wav2vec_mean.npy")
    std_file = os.path.join(data_dir, "wav2vec_std.npy")
    
    if os.path.exists(mean_file) and os.path.exists(std_file):
        mean = np.load(mean_file)
        std = np.load(std_file)
    else:
        print("Normalization statistics not found. Computing from data...")
        # Compute statistics from the data
        all_features = []
        for file in valid_files[:min(1000, len(valid_files))]:  # Use a subset to save memory
            try:
                features = np.load(file)
                all_features.append(features)
            except:
                continue
        
        all_features = np.vstack(all_features)
        mean = np.mean(all_features, axis=0)
        std = np.std(all_features, axis=0)
        
        # Save statistics for future use
        np.save(os.path.join(data_dir, "wav2vec_mean.npy"), mean)
        np.save(os.path.join(data_dir, "wav2vec_std.npy"), std)
    
    # Load and preprocess features
    X = []
    y = []
    errors = 0
    
    for i, (file, label) in enumerate(zip(valid_files, encoded_labels)):
        try:
            # Load with improved loader for numerical stability
            features = improved_load_features(file, mean, std, clip_value=5.0)
            
            # Pad or truncate sequence
            if len(features) > max_length:
                features = features[:max_length]
            elif len(features) < max_length:
                pad_width = ((0, max_length - len(features)), (0, 0))
                features = np.pad(features, pad_width, mode='constant')
            
            X.append(features)
            y.append(label)
            
            if (i+1) % 500 == 0:
                print(f"Processed {i+1}/{len(valid_files)} files")
        except Exception as e:
            print(f"Error processing {file}: {e}")
            errors += 1
    
    print(f"Failed to process {errors} files")
    
    # Convert to numpy arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    
    # Create train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y
    )
    
    # Print class distribution and calculate class weights
    classes = np.unique(y)
    class_weights = {}
    print("\nClass distribution:")
    for c in classes:
        count = np.sum(y_train == c)
        weight = len(y_train) / (len(classes) * count)
        class_weights[c] = weight
        print(f"  Class {c}: {count} samples (weight: {weight:.4f})")
    
    print(f"\nTrain samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, num_classes=len(classes))
    y_val = to_categorical(y_val, num_classes=len(classes))
    
    if use_cache:
        print(f"Saving data cache to {cache_file}")
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
    """
    Build a wav2vec-based emotion recognition model 
    with numerical stability enhancements.
    """
    inputs = Input(shape=input_shape)
    
    # Initial dimension reduction and normalization
    x = Conv1D(128, 3, padding='same', activation='relu')(inputs)
    x = MaxPooling1D(2)(x)
    
    # LayerNormalization instead of BatchNormalization for stability
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # First BiLSTM layer with constrained weights for stability
    x = Bidirectional(LSTM(
        128, 
        return_sequences=True,
        dropout=0.2,
        recurrent_dropout=0,
        kernel_constraint=MaxNorm(3.0),
        recurrent_constraint=MaxNorm(3.0)
    ))(x)
    
    # LayerNormalization for numerical stability
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Self-attention mechanism - numerical stability implementation
    query_value_dim = 64
    
    # Create query, key, value projections
    query = Dense(query_value_dim)(x)
    key = Dense(query_value_dim)(x)
    value = Dense(query_value_dim)(x)
    
    # Compute attention scores
    scores = tf.matmul(query, key, transpose_b=True)
    
    # Scale scores
    scores = scores / tf.math.sqrt(tf.cast(tf.shape(key)[-1], tf.float32))
    
    # Numeric stability in softmax calculation
    scores_max = tf.reduce_max(scores, axis=2, keepdims=True)
    scores_shifted = scores - scores_max  # Subtract max for stability
    attention_weights = tf.nn.softmax(scores_shifted, axis=2)
    
    # Apply attention weights to values
    context_vector = tf.matmul(attention_weights, value)
    
    # Combine with original input via residual connection
    x = tf.concat([x, context_vector], axis=2)
    
    # Second BiLSTM layer
    x = Bidirectional(LSTM(
        64, 
        return_sequences=True,
        dropout=0.2,
        recurrent_dropout=0,
        kernel_constraint=MaxNorm(2.0),
        recurrent_constraint=MaxNorm(2.0)
    ))(x)
    
    # Global pooling for sequence reduction
    x = GlobalAveragePooling1D()(x)
    
    # Dropout for regularization
    x = Dropout(dropout_rate)(x)
    
    # Output layer
    x = Dense(64, activation='relu')(x)
    
    # Logit clipping for numerical stability in final layer
    x = Dense(num_classes)(x)
    x = Lambda(lambda z: tf.clip_by_value(z, -15.0, 15.0), name='logit_clipping')(x)
    
    outputs = tf.nn.softmax(x, name='predictions')
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def train_model(args):
    """Train the wav2vec emotion recognition model with stability improvements"""
    data_dir = args.data_dir
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.lr
    dropout_rate = args.dropout
    model_name = args.model_name
    
    # Create checkpoints directory
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load and prepare data
    X_train, X_val, y_train, y_val, class_weights, label_encoder, max_length = load_emotion_data(
        data_dir, test_size=0.1, use_cache=args.use_cache
    )
    
    # Build model
    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]
    model = build_model(input_shape, num_classes, dropout_rate)
    
    # Setup optimizer with gradient clipping
    # Use only clipnorm or clipvalue, not both
    # For stability, we'll use clipnorm
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
        model_name = f"wav2vec_audio_only_fixed_v2_{timestamp}"
    
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
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        WarmUpReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
            warmup_epochs=10,
            learning_rate_base=learning_rate
        ),
        NaNLossDetector()
    ]
    
    # Train model
    print(f"Starting training: {model_name}")
    print(f"Input shape: {input_shape}, Classes: {num_classes}")
    print(model.summary())
    
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
    if not args.no_plots:
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
        f.write(f"val_accuracy,{results[1]}\n")
    
    # Also save as a json file
    with open(os.path.join(checkpoint_dir, f"{model_name}_validation_accuracy.json"), 'w') as f:
        json.dump({"val_accuracy": float(results[1])}, f)
    
    print(f"Training complete. Model saved as {model_name}")
    return model, history

def main():
    parser = argparse.ArgumentParser(description="Train wav2vec audio emotion recognition model with stability improvements")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing wav2vec features")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--model_name", type=str, default=None, help="Name for saving the model")
    parser.add_argument("--use_cache", action="store_true", help="Use cached data if available")
    parser.add_argument("--no_plots", action="store_true", help="Skip generating training plots")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with numeric checking")
    
    args = parser.parse_args()
    
    # Enable numeric checking in debug mode
    if args.debug:
        print("Numeric checking enabled - will detect NaN/Inf during execution")
        tf.debugging.enable_check_numerics()
    
    # Start training
    train_model(args)

if __name__ == "__main__":
    main()
