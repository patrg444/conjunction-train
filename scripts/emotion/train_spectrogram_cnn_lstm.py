#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid model using CNN for audio (Mel-spectrograms) and LSTM/BiLSTM for video.
Uses precomputed spectrograms and video features.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Bidirectional, Concatenate, Flatten, GlobalAveragePooling2D, Reshape # Added Reshape
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, BatchNormalization, Masking # Changed Conv1D->Conv2D, MaxPooling1D->MaxPooling2D
from tensorflow.keras.layers import Activation, Add, SpatialDropout1D, LayerNormalization # Removed MultiHeadAttention for now
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
import time
import glob
import random
from tqdm import tqdm # Import tqdm
from spectrogram_sync_generator import SpectrogramSyncGenerator # Import the new generator

# Global variables
BATCH_SIZE = 24 # Might need adjustment based on memory usage with CNNs
EPOCHS = 175
NUM_CLASSES = 6
PATIENCE = 15
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
# AUGMENTATION_FACTOR = 2.5 # Augmentation not implemented in SpectrogramSyncGenerator yet
L2_REGULARIZATION = 0.002
MAX_NORM_CONSTRAINT = 3.0
LEARNING_RATE = 0.0006

# Model specific parameters
# Audio CNN parameters (Example - adjust as needed)
AUDIO_CNN_FILTERS = [32, 64, 128]
AUDIO_CNN_KERNEL_SIZE = (3, 3)
AUDIO_POOL_SIZE = (2, 2)
AUDIO_DENSE_UNITS = 128 # Dense layer after CNN flatten

# Video LSTM parameters (Similar to previous models)
VIDEO_LSTM_UNITS = [128, 64]
MERGED_DENSE_UNITS = [256, 128]

print("HYBRID MODEL WITH CNN (AUDIO SPECTROGRAM) + LSTM (VIDEO)")
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
print("Python version:", sys.version)

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
        if not hasattr(self.model.optimizer, 'lr'): raise ValueError('Optimizer must have a "lr" attribute.')
        if epoch < self.warmup_epochs: learning_rate = self.learning_rate_base * (epoch + 1) / self.warmup_epochs
        else:
            decay_epochs = self.total_epochs - self.warmup_epochs; epoch_decay = epoch - self.warmup_epochs
            cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch_decay / decay_epochs))
            learning_rate = self.min_learning_rate + (self.learning_rate_base - self.min_learning_rate) * cosine_decay
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, learning_rate)
        self.learning_rates.append(learning_rate); print(f"\nEpoch {epoch+1}: Learning rate set to {learning_rate:.6f}")

def create_spectrogram_cnn_lstm_model(spec_input_shape, video_feature_dim):
    """
    Create a hybrid model using CNN for audio spectrograms and LSTM/BiLSTM for video.

    Args:
        spec_input_shape: Shape of the spectrogram input (time_steps, n_mels, 1)
        video_feature_dim: Dimensionality of video features

    Returns:
        Compiled Keras model
    """
    print("Creating hybrid model with CNN (Audio Spectrogram) and LSTM (Video):")
    print("- Spectrogram Input Shape:", spec_input_shape)
    print("- Video Feature Dimension:", video_feature_dim)
    print(f"- L2 Regularization Strength: {L2_REGULARIZATION}")
    print(f"- Weight Constraint: {MAX_NORM_CONSTRAINT}")
    print(f"- Audio CNN Filters: {AUDIO_CNN_FILTERS}")
    print(f"- Video LSTM Units: {VIDEO_LSTM_UNITS}")
    print(f"- Merged Dense Units: {MERGED_DENSE_UNITS}")
    print(f"- Learning Rate: {LEARNING_RATE} with warm-up and cosine decay")

    # --- Audio Branch (CNN) ---
    # Input shape: (batch, time_steps, n_mels, 1)
    audio_input = Input(shape=spec_input_shape, name='audio_input')

    # CNN layers
    audio_x = BatchNormalization()(audio_input) # Normalize input first
    audio_x = Conv2D(AUDIO_CNN_FILTERS[0], AUDIO_CNN_KERNEL_SIZE, activation='relu', padding='same', kernel_regularizer=l2(L2_REGULARIZATION), kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT))(audio_x)
    audio_x = MaxPooling2D(pool_size=AUDIO_POOL_SIZE)(audio_x)
    audio_x = Dropout(0.2)(audio_x)

    audio_x = BatchNormalization()(audio_x)
    audio_x = Conv2D(AUDIO_CNN_FILTERS[1], AUDIO_CNN_KERNEL_SIZE, activation='relu', padding='same', kernel_regularizer=l2(L2_REGULARIZATION), kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT))(audio_x)
    audio_x = MaxPooling2D(pool_size=AUDIO_POOL_SIZE)(audio_x)
    audio_x = Dropout(0.2)(audio_x)

    audio_x = BatchNormalization()(audio_x)
    audio_x = Conv2D(AUDIO_CNN_FILTERS[2], AUDIO_CNN_KERNEL_SIZE, activation='relu', padding='same', kernel_regularizer=l2(L2_REGULARIZATION), kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT))(audio_x)
    audio_x = MaxPooling2D(pool_size=AUDIO_POOL_SIZE)(audio_x)
    audio_x = Dropout(0.3)(audio_x)

    # Reshape CNN output for LSTM: (batch, time, freq*filters)
    # The shape after the last MaxPooling2D is (None, None, spec_input_shape[1] // 8, AUDIO_CNN_FILTERS[-1])
    # We need to know the frequency dimension after pooling to reshape correctly.
    # Assuming spec_input_shape[1] is n_mels (e.g., 128), after 3 pools of (2,2), it becomes 128 / (2*2*2) = 16.
    # The number of filters is AUDIO_CNN_FILTERS[-1] = 128.
    # Target shape for Reshape: (-1, features_per_timestep) where features = reduced_freq * filters
    # We need the time dimension, which is None. Let Reshape infer it.
    # The target shape should be (batch_size, time_steps, features)
    # Keras infers batch_size. time_steps is the dimension that's None after CNN.
    # features = reduced_freq * filters = 16 * 128 = 2048 (assuming n_mels=128)
    # Let's calculate reduced_freq dynamically if possible, otherwise assume 16 for n_mels=128
    reduced_freq_dim = spec_input_shape[1] // (AUDIO_POOL_SIZE[0] ** 3) # Calculate based on pooling
    reshape_target_dim = reduced_freq_dim * AUDIO_CNN_FILTERS[-1]
    audio_x = Reshape((-1, reshape_target_dim))(audio_x) # Reshape to (batch, time_steps, features)

    # Apply LSTM layers to the reshaped audio features
    audio_lstm_units = [64, 32] # Smaller LSTM for audio branch
    audio_x = Bidirectional(LSTM(audio_lstm_units[0], return_sequences=True, dropout=0.3, recurrent_dropout=0.2, kernel_regularizer=l2(L2_REGULARIZATION), recurrent_regularizer=l2(L2_REGULARIZATION/2), kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT), recurrent_constraint=MaxNorm(MAX_NORM_CONSTRAINT)))(audio_x)
    audio_x = Dropout(0.3)(audio_x)

    audio_x = Bidirectional(LSTM(audio_lstm_units[1], return_sequences=False, dropout=0.3, recurrent_dropout=0.2, kernel_regularizer=l2(L2_REGULARIZATION), recurrent_regularizer=l2(L2_REGULARIZATION/2), kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT), recurrent_constraint=MaxNorm(MAX_NORM_CONSTRAINT)))(audio_x)
    audio_output = Dropout(0.4)(audio_x) # Final audio representation after LSTM

    # --- Video Branch (LSTM/BiLSTM) ---
    video_input = Input(shape=(None, video_feature_dim), name='video_input')
    video_masked = Masking(mask_value=0.0)(video_input)

    # Apply bidirectional LSTM layers
    video_x = Bidirectional(LSTM(VIDEO_LSTM_UNITS[0], return_sequences=True, dropout=0.3, recurrent_dropout=0.2, kernel_regularizer=l2(L2_REGULARIZATION), recurrent_regularizer=l2(L2_REGULARIZATION/2), kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT), recurrent_constraint=MaxNorm(MAX_NORM_CONSTRAINT)))(video_masked)
    video_x = Dropout(0.3)(video_x)

    video_x = Bidirectional(LSTM(VIDEO_LSTM_UNITS[1], return_sequences=False, dropout=0.3, recurrent_dropout=0.2, kernel_regularizer=l2(L2_REGULARIZATION), recurrent_regularizer=l2(L2_REGULARIZATION/2), kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT), recurrent_constraint=MaxNorm(MAX_NORM_CONSTRAINT)))(video_x)
    video_output = Dropout(0.4)(video_x) # Final video representation

    # --- Fusion ---
    merged = Concatenate()([audio_output, video_output])

    # --- Classification Head ---
    merged = Dense(MERGED_DENSE_UNITS[0], activation='relu', kernel_regularizer=l2(L2_REGULARIZATION), kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT))(merged)
    merged = LayerNormalization()(merged)
    merged = Dropout(0.5)(merged)

    merged = Dense(MERGED_DENSE_UNITS[1], activation='relu', kernel_regularizer=l2(L2_REGULARIZATION), kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT))(merged)
    merged = LayerNormalization()(merged)
    merged = Dropout(0.4)(merged)

    output = Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=l2(L2_REGULARIZATION), kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT))(merged)

    # Create model
    model = Model(inputs={'video_input': video_input, 'audio_input': audio_input}, outputs=output)

    # Optimizer
    optimizer = Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-7)

    # Compile model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# --- Data Loading and Splitting ---
def load_data_paths_and_labels(spec_base_dir_ravdess, spec_base_dir_cremad, video_base_dir_ravdess, video_base_dir_cremad):
    """Finds corresponding spectrogram and video feature files and extracts labels."""
    spec_files = glob.glob(os.path.join(spec_base_dir_ravdess, "Actor_*", "*.npy")) + \
                 glob.glob(os.path.join(spec_base_dir_cremad, "*.npy"))

    video_files = []
    matched_spec_files = []
    labels = []
    emotion_map = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
    ravdess_emotion_map = {'01': 'NEU', '02': 'NEU', '03': 'HAP', '04': 'SAD', '05': 'ANG', '06': 'FEA', '07': 'DIS'}

    print(f"Found {len(spec_files)} spectrogram files. Matching video features...")
    skipped_video = 0
    skipped_label = 0

    for spec_file in tqdm(spec_files, desc="Matching files"):
        base_name = os.path.splitext(os.path.basename(spec_file))[0]
        video_file = None
        label = None

        # Find corresponding video file and label
        if "Actor_" in spec_file: # RAVDESS
            actor_folder = os.path.basename(os.path.dirname(spec_file))
            potential_video_file = os.path.join(video_base_dir_ravdess, actor_folder, base_name + ".npz")
            if os.path.exists(potential_video_file):
                video_file = potential_video_file
                try:
                    parts = base_name.split('-')
                    emotion_code = ravdess_emotion_map.get(parts[2], None)
                    if emotion_code in emotion_map:
                        label = np.zeros(len(emotion_map))
                        label[emotion_map[emotion_code]] = 1
                except: pass
            else: skipped_video += 1
        else: # CREMA-D
            potential_video_file = os.path.join(video_base_dir_cremad, base_name + ".npz")
            if os.path.exists(potential_video_file):
                video_file = potential_video_file
                try:
                    parts = base_name.split('_')
                    emotion_code = parts[2]
                    if emotion_code in emotion_map:
                        label = np.zeros(len(emotion_map))
                        label[emotion_map[emotion_code]] = 1
                except: pass
            else: skipped_video += 1

        if video_file and label is not None:
            video_files.append(video_file)
            matched_spec_files.append(spec_file)
            labels.append(label)
        elif video_file and label is None:
            skipped_label += 1

    print(f"Matched {len(video_files)} video/spectrogram pairs with labels.")
    print(f"Skipped {skipped_video} due to missing video NPZ.")
    print(f"Skipped {skipped_label} due to label parsing issues.")

    if not video_files:
        raise FileNotFoundError("No matching video/spectrogram pairs found. Ensure preprocessing ran and paths are correct.")

    return video_files, matched_spec_files, np.array(labels)

def normalize_video_features(features_list, mean=None, std=None):
    """Normalize a list of variable-length video feature arrays."""
    # This function is specific to video features loaded from NPZ
    if mean is None or std is None:
        # Need to load features to calculate stats
        all_features_list = []
        print("Calculating normalization stats for video features...")
        for file_path in tqdm(features_list, desc="Loading video for norm stats"):
             try:
                 with np.load(file_path) as data:
                     if 'video_features' in data:
                         all_features_list.append(data['video_features'])
             except Exception as e:
                 print(f"Warning: Could not load {file_path} for norm stats: {e}")
        if not all_features_list:
             raise ValueError("No valid video features found to calculate normalization statistics.")
        all_features = np.vstack(all_features_list)
        mean = np.mean(all_features, axis=0, keepdims=True)
        std = np.std(all_features, axis=0, keepdims=True)
        print("Video normalization stats calculated.")

    std = np.where(std == 0, 1.0, std) # Avoid division by zero
    # Note: Normalization now happens inside the generator for video
    return mean, std


# --- Training Function ---
def train_model():
    """Main function to train the Spectrogram CNN + LSTM model."""
    print("Starting Spectrogram CNN + LSTM model training...")
    print(f"- Using learning rate: {LEARNING_RATE}")

    # Define feature directories (relative to current script location or project root)
    # Assuming the script is run from the project root where 'data' is located
    RAVDESS_SPEC_DIR = "data/ravdess_features_spectrogram"
    CREMA_D_SPEC_DIR = "data/crema_d_features_spectrogram"
    RAVDESS_VIDEO_FEAT_DIR = "data/ravdess_features_facenet"
    CREMA_D_VIDEO_FEAT_DIR = "data/crema_d_features_facenet"

    # Load file paths and labels
    try:
        video_files, spec_files, all_labels = load_data_paths_and_labels(
            RAVDESS_SPEC_DIR, CREMA_D_SPEC_DIR, RAVDESS_VIDEO_FEAT_DIR, CREMA_D_VIDEO_FEAT_DIR
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the spectrogram preprocessing script has been run successfully.")
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

    # Create file lists for train/val
    train_video_files = [video_files[i] for i in train_idx]
    train_spec_files = [spec_files[i] for i in train_idx]
    train_labels = all_labels[train_idx]
    val_video_files = [video_files[i] for i in val_idx]
    val_spec_files = [spec_files[i] for i in val_idx]
    val_labels = all_labels[val_idx]

    print("\nTrain/Val split:")
    print(f"- Train samples: {len(train_video_files)}")
    print(f"- Validation samples: {len(val_video_files)}")

    # Calculate normalization stats for VIDEO features based ONLY on training data
    print("\nCalculating video normalization statistics (from training set)...")
    video_mean, video_std = normalize_video_features(train_video_files)
    # Spectrograms are typically NOT normalized in the same way (e.g., using instance normalization or batch norm in the model)
    print("Video normalization stats obtained.")

    # Determine input shapes for the model (load one sample)
    # Need to estimate max sequence lengths or use dynamic padding in generator
    # For simplicity here, let's assume dynamic padding within the generator for now
    # The generator will determine the actual shape needed per batch
    # However, the CNN needs a defined input shape. Let's load one spectrogram to get n_mels.
    try:
        temp_spec = np.load(train_spec_files[0])
        n_mels = temp_spec.shape[0] # Get number of mel bins
        # We need to decide on a fixed time dimension for the CNN input, or use Global Pooling earlier
        # Let's use None for time and add Global Pooling in the model before Flatten
        spec_input_shape = (None, n_mels, 1) # (time, mels, channels)
    except Exception as e:
        print(f"Error loading sample spectrogram to determine shape: {e}")
        sys.exit(1)

    # video_feature_dim = train_generator.video_dim # Removed: Get from generator after init

    print("\nCreating data generators...")
    # Pass normalization stats to the generator for video features
    # The generator itself doesn't handle normalization in this version,
    # so we'll apply it within the model or modify the generator later if needed.
    # For now, we rely on BatchNormalization in the model.
    train_generator = SpectrogramSyncGenerator(
        train_video_files, train_spec_files, train_labels, batch_size=BATCH_SIZE, shuffle=True
    )
    val_generator = SpectrogramSyncGenerator(
        val_video_files, val_spec_files, val_labels, batch_size=BATCH_SIZE, shuffle=False
    )

    # Get dimensions from initialized generator
    video_feature_dim = train_generator.video_dim
    n_mels = train_generator.spec_n_mels
    spec_input_shape = (None, n_mels, 1) # Update shape based on generator

    if video_feature_dim == 0 or n_mels == 0:
        print("Error: Could not determine feature dimensions from generator.")
        sys.exit(1)

    # Create the model
    model = create_spectrogram_cnn_lstm_model(spec_input_shape, video_feature_dim)
    model.summary()

    # Define callbacks
    checkpoint_filename = 'best_model_spectrogram_cnn_lstm.keras' # New checkpoint name
    checkpoint = ModelCheckpoint(checkpoint_filename, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=PATIENCE * 2, mode='max', verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=PATIENCE, min_lr=5e-6, verbose=1, mode='min')
    lr_scheduler = WarmUpCosineDecayScheduler(learning_rate_base=LEARNING_RATE, total_epochs=EPOCHS, warmup_epochs=10, min_learning_rate=5e-6)

    print("\nStarting training...")
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
