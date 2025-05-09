#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for audio-only humor detection on TED-Humor dataset
using pre-extracted Covarep features.
Based on train_wav2vec_audio_only_fixed_v4.py but adapted for TED-Humor data loading.
"""

import os
import argparse
import math
import json
import numpy as np
import tensorflow as tf
import random
import glob
import re
import datetime
import time
import pickle # Needed for loading TED-Humor data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Dropout, Bidirectional,
    LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D,
    Attention, TimeDistributed, Conv1D, MaxPooling1D, Lambda, Reshape,
    Layer
)
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, Callback
)
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import yaml # Needed to read the config file

# Import the custom dataset class
from dataloaders.ted_humor_dataset import TedHumorDataset

# Set precision explicitly to float32 for numerical stability
tf.keras.mixed_precision.set_global_policy("float32")

# Enable operation determinism for reproducibility
tf.config.experimental.enable_op_determinism()

# Seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Custom attention layer that uses Keras ops instead of raw TF ops
class SelfAttentionLayer(Layer):
    def __init__(self, dim, **kwargs):
        super(SelfAttentionLayer, self).__init__(**kwargs)
        self.dim = dim
        self.query_dense = Dense(dim)
        self.key_dense = Dense(dim)
        self.value_dense = Dense(dim)

    def call(self, inputs):
        # Project inputs to query, key, value
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # Attention mechanism - Using Keras ops for compatibility
        # Scale dot-product attention
        scale = tf.math.sqrt(tf.cast(self.dim, tf.float32))

        # Use MultiHeadAttention which correctly handles Keras tensors
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=1,
            key_dim=self.dim,
            dropout=0.0
        )(query=query, key=key, value=value)

        # Concatenate with original input for residual-like connection
        return tf.concat([inputs, attention_output], axis=2)

    def get_config(self):
        config = super(SelfAttentionLayer, self).get_config()
        config.update({"dim": self.dim})
        return config

# --- Modified Data Loading Function for Keras ---
def load_ted_humor_data_for_keras(dataset_path, test_size=0.1, use_cache=False, cache_dir="cache"):
    """
    Load and prepare TED-Humor data for Keras training.
    Uses TedHumorDataset (PyTorch) internally and converts to NumPy.
    """
    cache_file = os.path.join(cache_dir, "ted_humor_data_cache_keras_v1.npz")
    os.makedirs(cache_dir, exist_ok=True)

    if use_cache and os.path.exists(cache_file):
        print(f"Loading cached Keras data from {cache_file}")
        cache = np.load(cache_file, allow_pickle=True)
        X_train = cache['X_train']
        X_val = cache['X_val']
        y_train = cache['y_train']
        y_val = cache['y_val']
        class_weights = cache['class_weights'].item()
        label_encoder = cache['label_encoder'].item()
        max_length = cache['max_length'].item()
        feature_dim = cache['feature_dim'].item()
        return X_train, X_val, y_train, y_val, class_weights, label_encoder, max_length, feature_dim

    print(f"Loading TED-Humor data for Keras from {dataset_path}")

    # Instantiate the TedHumorDataset for train and dev splits
    try:
        train_dataset = TedHumorDataset(dataset_path, split='train')
        val_dataset = TedHumorDataset(dataset_path, split='dev') # Assuming 'dev' is the validation split
    except Exception as e:
        print(f"Error initializing TedHumorDataset from {dataset_path}: {e}")
        raise e

    # Collect all features and labels from both splits to determine max_length and feature_dim
    all_features_list = []
    all_labels_list = []

    print("Collecting data from train split...")
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        if sample is not None:
            all_features_list.append(sample['audio_input_values'].numpy()) # Convert PyTorch tensor to NumPy
            all_labels_list.append(sample['label'].item()) # Convert PyTorch tensor to scalar

    print("Collecting data from dev split...")
    for i in range(len(val_dataset)):
        sample = val_dataset[i]
        if sample is not None:
            all_features_list.append(sample['audio_input_values'].numpy()) # Convert PyTorch tensor to NumPy
            all_labels_list.append(sample['label'].item()) # Convert PyTorch tensor to scalar

    if not all_features_list or not all_labels_list:
        raise ValueError(f"No valid data loaded from {dataset_path}.")

    # Determine feature dimension and max length
    feature_dim = all_features_list[0].shape[-1] # Assuming features are [time_steps, feature_dim]
    sequence_lengths = [len(f) for f in all_features_list]
    max_length = int(np.percentile(sequence_lengths, 95)) # Use 95th percentile

    print(f"Feature dimension: {feature_dim}")
    print(f"Using max sequence length of {max_length} (95th percentile)")

    # Pad or truncate sequences and convert to final NumPy arrays
    X = []
    y = []
    for features, label in zip(all_features_list, all_labels_list):
        if len(features) > max_length:
            features = features[:max_length]
        elif len(features) < max_length:
            pad_width = ((0, max_length - len(features)), (0, 0))
            features = np.pad(features, pad_width, mode='constant', constant_values=0) # Pad with 0s

        X.append(features)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    # Create train/val split from the combined data
    # This ensures a consistent split regardless of how TedHumorDataset splits internally
    label_encoder = LabelEncoder()
    # Fit with all possible classes (0 and 1 for binary)
    label_encoder.fit(np.unique(y)) # Fit on actual unique labels

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y
    )

    # Calculate class weights
    classes = np.unique(y_train)
    class_weights = {}
    print("\nClass distribution (Training set):")
    for c in classes:
        count = np.sum(y_train == c)
        if count > 0:
            weight = len(y_train) / (len(classes) * count)
            class_weights[int(c)] = float(weight) # Ensure keys are int and values are float
            print(f"  Class {c}: {count} samples (weight: {weight:.4f})")
        else:
             class_weights[int(c)] = 0.0
             print(f"  Class {c}: {count} samples (weight: {class_weights[int(c)]:.4f}) - No samples in training set!")

    # Ensure all classes expected by the model (based on num_labels in config) have a weight
    # This is important if stratify didn't include all classes in the train split
    for c in range(len(label_encoder.classes_)):
        if c not in class_weights:
             class_weights[c] = 0.0
             print(f"  Class {c}: 0 samples (weight: {class_weights[c]:.4f}) - Added missing class to weights.")


    print(f"\nTrain samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, num_classes=len(label_encoder.classes_))
    y_val = to_categorical(y_val, num_classes=len(label_encoder.classes_))

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
            max_length=max_length,
            feature_dim=feature_dim # Save feature dimension
        )

    return X_train, X_val, y_train, y_val, class_weights, label_encoder, max_length, feature_dim


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

def build_model(input_shape, num_classes, dropout_rate=0.5):
    """
    Build an emotion recognition model for sequential features.
    Adapted from the wav2vec model but designed to take pre-extracted features.
    """
    inputs = Input(shape=input_shape)

    # Initial dimension reduction and normalization
    # Adjust Conv1D filter size or remove if input dim is small
    # Assuming input_shape is (max_length, feature_dim)
    feature_dim = input_shape[-1]

    # If feature_dim is small, maybe skip Conv1D or use smaller filters
    if feature_dim > 64: # Arbitrary threshold, adjust as needed
        x = Conv1D(128, 3, padding='same', activation='relu')(inputs)
        x = MaxPooling1D(2)(x)
    else:
        x = inputs # Directly use input if feature_dim is small

    # LayerNormalization for stability
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

    # Self-attention mechanism - using Keras-compatible approach via custom layer
    # Adjust dim based on the output dimension of the previous layer (128*2 for BiLSTM)
    attention_dim = 128 * 2 # BiLSTM output dim
    x = SelfAttentionLayer(dim=attention_dim)(x)


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

def train_model(config):
    """Train the audio emotion recognition model using the provided config."""
    data_path = config['dataset_path']
    batch_size = config['batch_size']
    epochs = config['max_epochs']
    learning_rate = config['learning_rate']
    dropout_rate = config.get('dropout', 0.5) # Use default if not in config
    model_name = config.get('model_name', 'ted_humor_audio_model') # Use default if not in config
    num_labels = config['num_labels'] # Get num_labels from config

    # Create checkpoints directory
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load and prepare data using the TED-Humor specific loader for Keras
    X_train, X_val, y_train, y_val, class_weights, label_encoder, max_length, feature_dim = load_ted_humor_data_for_keras(
        data_path, test_size=config.get('test_size', 0.1), use_cache=config.get('use_cache', False) # Use test_size and use_cache from config
    )

    # Build model
    # Input shape is (max_length, feature_dim)
    input_shape = (max_length, feature_dim)
    # Ensure num_classes matches the actual number of unique labels loaded
    actual_num_classes = len(label_encoder.classes_)
    if num_labels != actual_num_classes:
        print(f"Warning: Config num_labels ({num_labels}) does not match actual number of classes loaded ({actual_num_classes}). Using {actual_num_classes}.")
        num_classes = actual_num_classes
    else:
        num_classes = num_labels

    model = build_model(input_shape, num_classes, dropout_rate)

    # Setup optimizer with gradient clipping
    # Use Adam optimizer as defined in the original script's train_model function
    optimizer = Adam(
        learning_rate=learning_rate,
        clipnorm=config.get('gradient_clip_val', 1.0) # Use clipnorm from config
    )

    # Compile model with label smoothing for stability
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=config.get('label_smoothing', 0.1)), # Use label_smoothing from config if available
        metrics=['accuracy'] # Keep accuracy as a metric
    )

    # Generate model name with timestamp if not provided in config
    if model_name == 'ted_humor_audio_model': # Check if using the default name
         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
         model_name = f"ted_humor_audio_{timestamp}"


    # Setup callbacks with custom LR scheduler and NaN detection
    callbacks = [
        ModelCheckpoint(
            os.path.join(checkpoint_dir, f"{model_name}_best.weights.h5"),
            monitor=config['model_checkpoint']['monitor'], # Use monitor from config
            save_best_only=config['model_checkpoint']['save_top_k'] > 0, # Save best if save_top_k > 0
            save_weights_only=True,
            mode=config['model_checkpoint']['mode'], # Use mode from config
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(checkpoint_dir, f"{model_name}_final.weights.h5"),
            save_weights_only=True,
            save_best_only=False,
            save_freq='epoch'
        ),
        TensorBoard(
            log_dir=os.path.join(config['logger']['init_args']['save_dir'], model_name), # Use log_dir from config
            histogram_freq=1,
            update_freq='epoch'
        ),
        EarlyStopping(
            monitor=config['early_stopping']['monitor'], # Use monitor from config
            patience=config['early_stopping']['patience'], # Use patience from config
            restore_best_weights=True,
            verbose=1,
            mode=config['early_stopping']['mode'] # Use mode from config
        ),
        WarmUpReduceLROnPlateau(
            monitor=config['early_stopping']['monitor'], # Use monitor from config for LR reduction
            factor=config.get('lr_reduce_factor', 0.5), # Add lr_reduce_factor to config if needed
            patience=config.get('lr_reduce_patience', 5), # Add lr_reduce_patience to config if needed
            min_lr=config.get('min_lr', 1e-6), # Add min_lr to config if needed
            verbose=1,
            warmup_epochs=config.get('warmup_epochs', 10), # Add warmup_epochs to config if needed
            learning_rate_base=learning_rate # Use initial learning rate from config
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
    if not config.get('no_plots', False): # Use no_plots from config if available
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
        plt.savefig(os.path.join(config['logger']['init_args']['save_dir'], model_name, f"{model_name}_training_curves.png")) # Save plots in log dir

    # Final evaluation
    print("\nEvaluating model on validation set:")
    results = model.evaluate(X_val, y_val, batch_size=batch_size, verbose=1)
    print(f"Validation loss: {results[0]:.4f}, Validation accuracy: {results[1]:.4f}")

    # Save validation results in a summary file
    with open(os.path.join(config['logger']['init_args']['save_dir'], model_name, f"{model_name}_validation_summary.csv"), 'w') as f: # Save summary in log dir
        f.write("metric,value\n")
        f.write(f"val_loss,{results[0]}\n")
        f.write(f"val_accuracy,{results[1]}\n")

    # Also save as a json file
    with open(os.path.join(config['logger']['init_args']['save_dir'], model_name, f"{model_name}_validation_accuracy.json"), 'w') as f: # Save accuracy in log dir
        json.dump({"val_accuracy": float(results[1])}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train audio-only humor detection model on TED-Humor.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    # Remove individual arguments that are now in the config file
    # parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the TED-Humor pickle files.")
    # parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    # parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train.")
    # parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    # parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate.")
    # parser.add_argument("--model_name", type=str, default=None, help="Optional name for the model and log directory.")
    parser.add_argument("--use_cache", action="store_true", help="Use cached data if available.")
    parser.add_argument("--no_plots", action="store_true", help="Do not generate training plots.")

    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Add command line arguments to config, overriding if present
    config['use_cache'] = args.use_cache
    config['no_plots'] = args.no_plots
    # The config file should now contain all necessary training parameters

    train_model(config)
