#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Non-branched hybrid model using Conv1D for audio and LSTM/BiLSTM for video.
- No TCN.
- No Attention.
- Simplified fusion.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Bidirectional, Concatenate
from tensorflow.keras.layers import TimeDistributed, Conv1D, MaxPooling1D, Flatten, BatchNormalization, Masking
from tensorflow.keras.layers import Activation, Add, SpatialDropout1D, LayerNormalization # Removed MultiHeadAttention
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
import time
import glob
import random
from synchronized_data_generator import SynchronizedAugmentationDataGenerator
from sequence_data_generator import ValidationDataGenerator

# Global variables (kept similar for comparison, adjust if needed)
BATCH_SIZE = 24
EPOCHS = 125
NUM_CLASSES = 6
PATIENCE = 15
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
AUGMENTATION_FACTOR = 2.5
L2_REGULARIZATION = 0.002
MAX_NORM_CONSTRAINT = 3.0
LEARNING_RATE = 0.0006
WEIGHT_DECAY = 0.0005 # Note: Adam optimizer doesn't directly use weight_decay like AdamW

# Model specific parameters
AUDIO_CONV_FILTERS = [64, 128]
AUDIO_DENSE_UNITS = 64 # Added dense layer after audio conv
VIDEO_LSTM_UNITS = [128, 64] # LSTM units for video branch
MERGED_DENSE_UNITS = [256, 128]

print("NON-BRANCHED HYBRID MODEL WITH CONV1D (AUDIO) AND LSTM (VIDEO)")
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
print("Python version:", sys.version)

class WarmUpCosineDecayScheduler(Callback):
    """
    Implements warm-up with cosine decay learning rate scheduling.
    (Keeping this scheduler as it's generally useful)
    """
    def __init__(self, learning_rate_base, total_epochs, warmup_epochs=10, min_learning_rate=1e-6):
        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_learning_rate = min_learning_rate
        self.learning_rates = []

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        if epoch < self.warmup_epochs:
            learning_rate = self.learning_rate_base * (epoch + 1) / self.warmup_epochs
        else:
            decay_epochs = self.total_epochs - self.warmup_epochs
            epoch_decay = epoch - self.warmup_epochs
            cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch_decay / decay_epochs))
            learning_rate = self.min_learning_rate + (self.learning_rate_base - self.min_learning_rate) * cosine_decay

        tf.keras.backend.set_value(self.model.optimizer.learning_rate, learning_rate)
        self.learning_rates.append(learning_rate)
        print(f"\nEpoch {epoch+1}: Learning rate set to {learning_rate:.6f}")

# Removed attention_block and residual_tcn_block functions as they are not used

def create_lstm_conv1d_model(audio_feature_dim, video_feature_dim):
    """
    Create a non-branched hybrid model using Conv1D for audio and LSTM/BiLSTM for video.

    Args:
        audio_feature_dim: Dimensionality of audio features
        video_feature_dim: Dimensionality of video features

    Returns:
        Compiled Keras model
    """
    print("Creating non-branched hybrid model with Conv1D (Audio) and LSTM (Video):")
    print("- Audio feature dimension:", audio_feature_dim)
    print("- Video feature dimension:", video_feature_dim)
    print(f"- L2 regularization strength: {L2_REGULARIZATION}")
    print(f"- Weight constraint: {MAX_NORM_CONSTRAINT}")
    print(f"- Audio Conv Filters: {AUDIO_CONV_FILTERS}")
    print(f"- Video LSTM Units: {VIDEO_LSTM_UNITS}")
    print(f"- Merged Dense Units: {MERGED_DENSE_UNITS}")
    print(f"- Learning rate: {LEARNING_RATE} with warm-up and cosine decay")

    # --- Audio Branch (Conv1D) ---
    audio_input = Input(shape=(None, audio_feature_dim), name='audio_input')
    audio_masked = Masking(mask_value=0.0)(audio_input)

    # Conv layers
    audio_x = Conv1D(
        AUDIO_CONV_FILTERS[0], kernel_size=3, activation='relu', padding='same',
        kernel_regularizer=l2(L2_REGULARIZATION), kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT)
    )(audio_masked)
    audio_x = BatchNormalization()(audio_x)
    audio_x = SpatialDropout1D(0.2)(audio_x)
    audio_x = MaxPooling1D(pool_size=2)(audio_x)

    audio_x = Conv1D(
        AUDIO_CONV_FILTERS[1], kernel_size=3, activation='relu', padding='same',
        kernel_regularizer=l2(L2_REGULARIZATION), kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT)
    )(audio_x)
    audio_x = BatchNormalization()(audio_x)
    audio_x = SpatialDropout1D(0.2)(audio_x)
    audio_x = MaxPooling1D(pool_size=2)(audio_x)

    # Flatten or Pool before Dense
    # Using GlobalMaxPooling to get a fixed-size output vector
    audio_x = GlobalMaxPooling1D()(audio_x)

    # Dense layer for audio features
    audio_x = Dense(
        AUDIO_DENSE_UNITS, activation='relu',
        kernel_regularizer=l2(L2_REGULARIZATION), kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT)
    )(audio_x)
    audio_x = LayerNormalization()(audio_x)
    audio_x = Dropout(0.4)(audio_x)

    # --- Video Branch (LSTM/BiLSTM) ---
    video_input = Input(shape=(None, video_feature_dim), name='video_input')
    video_masked = Masking(mask_value=0.0)(video_input)

    # Apply bidirectional LSTM layers
    video_x = Bidirectional(LSTM(
        VIDEO_LSTM_UNITS[0], return_sequences=True, dropout=0.3, recurrent_dropout=0.2,
        kernel_regularizer=l2(L2_REGULARIZATION), recurrent_regularizer=l2(L2_REGULARIZATION/2),
        kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT), recurrent_constraint=MaxNorm(MAX_NORM_CONSTRAINT)
    ))(video_masked)
    video_x = Dropout(0.3)(video_x) # Apply dropout after BiLSTM

    video_x = Bidirectional(LSTM(
        VIDEO_LSTM_UNITS[1], dropout=0.3, recurrent_dropout=0.2, # Removed return_sequences=True for the last LSTM layer
        kernel_regularizer=l2(L2_REGULARIZATION), recurrent_regularizer=l2(L2_REGULARIZATION/2),
        kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT), recurrent_constraint=MaxNorm(MAX_NORM_CONSTRAINT)
    ))(video_x)
    # Output of last BiLSTM is already (batch_size, units*2)

    # Dense layer for video features (optional, could directly concatenate)
    video_x = Dense(
        VIDEO_LSTM_UNITS[0], activation='relu', # Using first LSTM size for consistency
        kernel_regularizer=l2(L2_REGULARIZATION), kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT)
    )(video_x)
    video_x = LayerNormalization()(video_x)
    video_x = Dropout(0.4)(video_x)

    # --- Fusion (Non-Branched) ---
    merged = Concatenate()([audio_x, video_x])

    # Dense layers for combined features
    merged = Dense(
        MERGED_DENSE_UNITS[0], activation='relu',
        kernel_regularizer=l2(L2_REGULARIZATION), kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT)
    )(merged)
    merged = LayerNormalization()(merged)
    merged = Dropout(0.5)(merged)

    merged = Dense(
        MERGED_DENSE_UNITS[1], activation='relu',
        kernel_regularizer=l2(L2_REGULARIZATION), kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT)
    )(merged)
    merged = LayerNormalization()(merged)
    merged = Dropout(0.4)(merged)

    # Output layer
    output = Dense(
        NUM_CLASSES, activation='softmax',
        kernel_regularizer=l2(L2_REGULARIZATION), kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT)
    )(merged)

    # Create model
    model = Model(inputs={'video_input': video_input, 'audio_input': audio_input}, outputs=output)

    # Optimizer
    optimizer = Adam(
        learning_rate=LEARNING_RATE,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# --- Data Processing (Identical to previous script) ---
def process_data(path_pattern, max_files=None):
    """Load and process NPZ files containing features."""
    files = glob.glob(path_pattern)
    if max_files:
        files = files[:max_files]

    if not files:
        print("Error: No files found matching pattern: %s" % path_pattern)
        return None, None, None

    print("Processing %d files from %s" % (len(files), path_pattern))

    audio_data = []
    video_data = []
    labels = []

    emotion_map = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
    ravdess_emotion_map = {'01': 'NEU', '02': 'NEU', '03': 'HAP', '04': 'SAD', '05': 'ANG', '06': 'FEA', '07': 'DIS'}

    skipped = 0
    for file_path in files:
        try:
            filename = os.path.basename(file_path)
            if '-' in filename:
                parts = filename.split('-')
                emotion_code = ravdess_emotion_map.get(parts[2], None) if len(parts) >= 3 else None
            else:
                parts = filename.split('_')
                emotion_code = parts[2] if len(parts) >= 3 else None

            if emotion_code not in emotion_map:
                skipped += 1
                continue

            data = np.load(file_path)
            if 'audio_features' not in data or 'video_features' not in data:
                skipped += 1
                continue

            audio_features = data['audio_features']
            video_features = data['video_features']

            if len(audio_features) < 5 or len(video_features) < 5:
                skipped += 1
                continue

            audio_data.append(audio_features)
            video_data.append(video_features)
            label = np.zeros(NUM_CLASSES)
            label[emotion_map[emotion_code]] = 1
            labels.append(label)

        except Exception as e:
            print("Error processing file %s: %s" % (file_path, str(e)))
            skipped += 1

    print("Processed %d files, skipped %d files" % (len(audio_data), skipped))
    return audio_data, video_data, np.array(labels)

def normalize_features(features_list, mean=None, std=None):
    """Normalize a list of variable-length feature arrays."""
    if mean is None or std is None:
        all_features = np.vstack([feat for feat in features_list])
        mean = np.mean(all_features, axis=0, keepdims=True)
        std = np.std(all_features, axis=0, keepdims=True)

    std = np.where(std == 0, 1.0, std)
    normalized_list = [(features - mean) / std for features in features_list]
    return normalized_list, mean, std

# --- Training Function (Mostly identical, updated model creation call and checkpoint name) ---
def train_model():
    """Main function to train the LSTM/Conv1D model."""
    print("Starting LSTM/Conv1D model training...")
    print(f"- Using learning rate: {LEARNING_RATE}")

    ravdess_pattern = "ravdess_features_facenet/*/*.npz"
    ravdess_audio, ravdess_video, ravdess_labels = process_data(ravdess_pattern)
    cremad_pattern = "crema_d_features_facenet/*.npz"
    cremad_audio, cremad_video, cremad_labels = process_data(cremad_pattern)

    if ravdess_audio is None and cremad_audio is None:
        print("Error: Failed to load any data")
        return

    all_audio = []
    all_video = []
    all_labels = None
    if ravdess_audio is not None:
        all_audio.extend(ravdess_audio)
        all_video.extend(ravdess_video)
        all_labels = ravdess_labels
    if cremad_audio is not None:
        all_audio.extend(cremad_audio)
        all_video.extend(cremad_video)
        if all_labels is None: all_labels = cremad_labels
        else: all_labels = np.vstack([all_labels, cremad_labels])

    print(f"Combined: {len(all_audio)} total samples")

    class_indices = [np.where(all_labels[:, i] == 1)[0] for i in range(NUM_CLASSES)]
    train_idx, val_idx = [], []
    np.random.seed(RANDOM_SEED)
    for indices in class_indices:
        np.random.shuffle(indices)
        split_idx = int(len(indices) * TRAIN_RATIO)
        train_idx.extend(indices[:split_idx])
        val_idx.extend(indices[split_idx:])
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)

    train_audio = [all_audio[i] for i in train_idx]
    train_video = [all_video[i] for i in train_idx]
    train_labels = all_labels[train_idx]
    val_audio = [all_audio[i] for i in val_idx]
    val_video = [all_video[i] for i in val_idx]
    val_labels = all_labels[val_idx]

    print("Train/Val split:")
    print("- Train samples:", len(train_audio))
    print("- Validation samples:", len(val_audio))

    print("Normalizing features (training statistics only)...")
    train_audio, audio_mean, audio_std = normalize_features(train_audio)
    val_audio, _, _ = normalize_features(val_audio, mean=audio_mean, std=audio_std)
    print("Normalization complete.")

    audio_feature_dim = train_audio[0].shape[1]
    video_feature_dim = train_video[0].shape[1]

    print("Creating data generators...")
    train_generator = SynchronizedAugmentationDataGenerator(
        train_video, train_audio, train_labels, batch_size=BATCH_SIZE, shuffle=True,
        augmentation_factor=AUGMENTATION_FACTOR, augmentation_probability=0.8
    )
    val_generator = ValidationDataGenerator(
        val_video, val_audio, val_labels, batch_size=BATCH_SIZE
    )

    # Create the NEW model
    model = create_lstm_conv1d_model(audio_feature_dim, video_feature_dim) # Updated function call
    model.summary()

    # Define callbacks (updated checkpoint name)
    checkpoint_filename = 'best_model_lstm_conv1d.keras' # New checkpoint name
    checkpoint = ModelCheckpoint(
        checkpoint_filename, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='val_accuracy', patience=PATIENCE * 2, mode='max', verbose=1, restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.6, patience=PATIENCE, min_lr=5e-6, verbose=1, mode='min'
    )
    lr_scheduler = WarmUpCosineDecayScheduler(
        learning_rate_base=LEARNING_RATE, total_epochs=EPOCHS, warmup_epochs=10, min_learning_rate=5e-6
    )

    print("Starting training...")
    start_time = time.time()
    history = model.fit(
        train_generator, epochs=EPOCHS, validation_data=val_generator,
        callbacks=[checkpoint, early_stopping, reduce_lr, lr_scheduler], verbose=1
    )
    end_time = time.time()
    print("Training finished in %.2f seconds" % (end_time - start_time))

    print("Evaluating the best model...")
    model.load_weights(checkpoint_filename) # Load from the new checkpoint name
    loss, accuracy = model.evaluate(val_generator, verbose=1)
    print(f"Best model validation accuracy ({checkpoint_filename}): {accuracy:.4f}")

if __name__ == '__main__':
    train_model()
