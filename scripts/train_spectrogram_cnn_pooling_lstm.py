#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trains an audio-only model using LSTM/BiLSTM on precomputed CNN audio features.
(Modified from early-fusion hybrid model).
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Bidirectional, Concatenate, Masking, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
import time
import glob
import random
from tqdm import tqdm
# from spectrogram_cnn_pooling_generator import SpectrogramCnnPoolingGenerator # OLD generator
from precomputed_cnn_audio_generator import PrecomputedCnnAudioGenerator # Import the audio-only generator

# Global variables (match train_audio_pooling_lstm.py where applicable)
BATCH_SIZE = 24 # Keep consistent, adjust if memory issues arise
EPOCHS = 150 # Increased slightly as CNN features might take longer to converge
NUM_CLASSES = 6
PATIENCE = 20 # Increased patience slightly
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
L2_REGULARIZATION = 0.002
MAX_NORM_CONSTRAINT = 3.0
LEARNING_RATE = 0.0005 # Slightly reduced LR might be beneficial

# Model specific parameters (match train_audio_pooling_lstm.py)
LSTM_UNITS = [128, 64]
DENSE_UNITS = [256, 128]

print("EARLY FUSION (SPECTROGRAM CNN POOLING) HYBRID MODEL WITH LSTM")
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
print("Python version:", sys.version)

# --- Learning Rate Scheduler (copied from train_audio_pooling_lstm.py) ---
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
        if not hasattr(self.model.optimizer, 'lr'): raise ValueError('Optimizer must have a "lr" attribute.') # Corrected: Added comma
        if epoch < self.warmup_epochs: learning_rate = self.learning_rate_base * (epoch + 1) / self.warmup_epochs
        else:
            decay_epochs = self.total_epochs - self.warmup_epochs; epoch_decay = epoch - self.warmup_epochs
            cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch_decay / decay_epochs))
            learning_rate = self.min_learning_rate + (self.learning_rate_base - self.min_learning_rate) * cosine_decay
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, learning_rate)
        self.learning_rates.append(learning_rate); print(f"\nEpoch {epoch+1}: Learning rate set to {learning_rate:.6f}")

# --- Model Definition (Audio-Only) ---
def create_combined_lstm_model(audio_feature_dim): # Use a clear parameter name for audio dim
    """
    Creates an audio-only LSTM model using precomputed CNN features.
    """
    print("Creating audio-only LSTM model (Precomputed CNN Audio):") # Updated print
    print("- Audio feature dimension:", audio_feature_dim) # Use the new parameter name
    print(f"- L2 regularization strength: {L2_REGULARIZATION}")
    print(f"- Weight constraint: {MAX_NORM_CONSTRAINT}")
    print(f"- LSTM Units: {LSTM_UNITS}")
    print(f"- Dense Units: {DENSE_UNITS}")
    print(f"- Learning rate: {LEARNING_RATE} with warm-up and cosine decay")

    audio_input = Input(shape=(None, audio_feature_dim), name='audio_input') # Use function parameter
    masked_input = Masking(mask_value=0.0)(audio_input) # Use audio_input

    x = Bidirectional(LSTM(
        LSTM_UNITS[0], return_sequences=True, dropout=0.3, recurrent_dropout=0.2, # Keep dropout moderate
        kernel_regularizer=l2(L2_REGULARIZATION), recurrent_regularizer=l2(L2_REGULARIZATION/2),
        kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT), recurrent_constraint=MaxNorm(MAX_NORM_CONSTRAINT)
    ))(masked_input)
    x = Dropout(0.4)(x) # Slightly increased dropout

    x = Bidirectional(LSTM(
        LSTM_UNITS[1], return_sequences=False, dropout=0.3, recurrent_dropout=0.2, # Keep dropout moderate
        kernel_regularizer=l2(L2_REGULARIZATION), recurrent_regularizer=l2(L2_REGULARIZATION/2),
        kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT), recurrent_constraint=MaxNorm(MAX_NORM_CONSTRAINT)
    ))(x)
    x = Dropout(0.5)(x) # Slightly increased dropout

    # Classification Head
    x = Dense(
        DENSE_UNITS[0], activation='relu',
        kernel_regularizer=l2(L2_REGULARIZATION), kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT)
    )(x)
    x = LayerNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(
        DENSE_UNITS[1], activation='relu',
        kernel_regularizer=l2(L2_REGULARIZATION), kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT)
    )(x)
    x = LayerNormalization()(x)
    x = Dropout(0.4)(x)

    output = Dense(
        NUM_CLASSES, activation='softmax',
        kernel_regularizer=l2(L2_REGULARIZATION), kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT)
    )(x)

    model = Model(inputs=audio_input, outputs=output) # Use audio_input

    optimizer = Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Data Loading (Audio-Only) ---
def load_data_paths_and_labels_audio_only(cnn_audio_dir_ravdess, cnn_audio_dir_cremad):
    """Finds precomputed CNN audio feature files and extracts labels."""
    # Find precomputed CNN audio feature files
    cnn_audio_files = glob.glob(os.path.join(cnn_audio_dir_ravdess, "Actor_*", "*.npy")) + \
                      glob.glob(os.path.join(cnn_audio_dir_cremad, "*.npy"))

    labels = []
    emotion_map = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
    ravdess_emotion_map = {'01': 'NEU', '02': 'NEU', '03': 'HAP', '04': 'SAD', '05': 'ANG', '06': 'FEA', '07': 'DIS'}

    print(f"Found {len(cnn_audio_files)} precomputed CNN audio files. Extracting labels...")
    skipped_label = 0
    valid_cnn_audio_files = []

    for cnn_audio_file in tqdm(cnn_audio_files, desc="Extracting labels"):
        base_name = os.path.splitext(os.path.basename(cnn_audio_file))[0]
        label = None

        # Extract label
        try:
            if "Actor_" in cnn_audio_file: # RAVDESS
                parts = base_name.split('-')
                emotion_code = ravdess_emotion_map.get(parts[2], None)
                if emotion_code in emotion_map:
                    label = np.zeros(len(emotion_map))
                    label[emotion_map[emotion_code]] = 1
            else: # CREMA-D
                parts = base_name.split('_')
                emotion_code = parts[2]
                if emotion_code in emotion_map:
                    label = np.zeros(len(emotion_map))
                    label[emotion_map[emotion_code]] = 1
        except Exception as e:
            print(f"Label parsing error for {cnn_audio_file}: {e}")
            label = None # Ensure label is None on error

        if label is not None:
            valid_cnn_audio_files.append(cnn_audio_file)
            labels.append(label)
        else:
            skipped_label += 1

    print(f"Found {len(valid_cnn_audio_files)} CNN audio files with valid labels.")
    print(f"Skipped {skipped_label} due to label parsing issues.")

    if not valid_cnn_audio_files:
        raise FileNotFoundError("No CNN audio files with valid labels found. Ensure CNN audio preprocessing ran and paths are correct.")

    return valid_cnn_audio_files, np.array(labels)


# --- Training Function (Audio-Only) ---
def train_model():
    """Main function to train the audio-only LSTM model using precomputed CNN audio features."""
    print("Starting Precomputed CNN Audio LSTM model training (Audio-Only)...")
    print(f"- Using learning rate: {LEARNING_RATE}")

    # Define feature directories (POINT TO FIXED CNN AUDIO FEATURES)
    RAVDESS_CNN_AUDIO_DIR = "data/ravdess_features_cnn_fixed" # Use FIXED features
    CREMA_D_CNN_AUDIO_DIR = "data/crema_d_features_cnn_fixed" # Use FIXED features
    # Video directories removed

    # Load file paths and labels (Audio-Only)
    try:
        cnn_audio_files, all_labels = load_data_paths_and_labels_audio_only(
            RAVDESS_CNN_AUDIO_DIR, CREMA_D_CNN_AUDIO_DIR
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Split data into train/validation sets (indices for file lists)
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

    # Create file lists for train/val (Audio-Only)
    train_cnn_audio_files = [cnn_audio_files[i] for i in train_idx]
    train_labels = all_labels[train_idx]
    val_cnn_audio_files = [cnn_audio_files[i] for i in val_idx]
    val_labels = all_labels[val_idx]

    print("\nTrain/Val split:")
    print(f"- Train samples: {len(train_cnn_audio_files)}")
    print(f"- Validation samples: {len(val_cnn_audio_files)}")

    # Create data generators (Audio-Only)
    print("\nCreating data generators (Precomputed CNN Audio - Audio Only)...")
    train_generator = PrecomputedCnnAudioGenerator(
        train_cnn_audio_files, train_labels, batch_size=BATCH_SIZE, shuffle=True
    )
    val_generator = PrecomputedCnnAudioGenerator(
        val_cnn_audio_files, val_labels, batch_size=BATCH_SIZE, shuffle=False
    )

    # Get audio dimension from generator
    cnn_audio_dim = train_generator.cnn_audio_dim # Use cnn_audio_dim from generator
    if cnn_audio_dim == 0:
        print("Error: Could not determine CNN audio feature dimension from generator.")
        sys.exit(1)

    # Create the audio-only model
    model = create_combined_lstm_model(cnn_audio_dim) # Pass audio dim
    model.summary()

    # Define callbacks
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"models/precomputed_cnn_lstm_audio_only_{timestamp}" # Updated dir name
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_filename = os.path.join(checkpoint_dir, 'best_model_precomputed_cnn_audio_lstm_audio_only.h5') # Updated checkpoint name
    print(f"Saving checkpoints to: {checkpoint_dir}")
    checkpoint = ModelCheckpoint(checkpoint_filename, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1, save_format='h5')
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=PATIENCE * 2, mode='max', verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=PATIENCE, min_lr=5e-6, verbose=1, mode='min')
    lr_scheduler = WarmUpCosineDecayScheduler(learning_rate_base=LEARNING_RATE, total_epochs=EPOCHS, warmup_epochs=10, min_learning_rate=5e-6)

    print("\nStarting training (Audio-Only)...")
    start_time = time.time()
    history = model.fit(
        train_generator, epochs=EPOCHS, validation_data=val_generator,
        callbacks=[checkpoint, early_stopping, reduce_lr, lr_scheduler], verbose=1
    )
    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds")

    print("\nEvaluating the best model...")
    if os.path.exists(checkpoint_filename):
        print(f"Loading best weights from {checkpoint_filename}")
        model.load_weights(checkpoint_filename)
    else:
        print("Warning: Checkpoint file not found. Evaluating model with final weights.")

    loss, accuracy = model.evaluate(val_generator, verbose=1)
    print(f"Best model validation accuracy ({checkpoint_filename}): {accuracy:.4f}")

if __name__ == '__main__':
    train_model()
