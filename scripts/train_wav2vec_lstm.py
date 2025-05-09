#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid model using Wav2Vec2 embeddings for audio and LSTM/BiLSTM for video.
Uses precomputed Wav2Vec2 embeddings and video features.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Bidirectional, Concatenate, Flatten, GlobalAveragePooling2D, Reshape # Added Reshape
from tensorflow.keras.layers import TimeDistributed, BatchNormalization, Masking # Removed CNN layers
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
# Import the new generator
from wav2vec_sync_generator import Wav2VecSyncGenerator

# Global variables
BATCH_SIZE = 32 # Can likely increase batch size without CNNs
EPOCHS = 125
NUM_CLASSES = 6
PATIENCE = 15
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
# AUGMENTATION_FACTOR = 2.5 # Augmentation not implemented in SpectrogramSyncGenerator yet
L2_REGULARIZATION = 0.002
MAX_NORM_CONSTRAINT = 3.0
LEARNING_RATE = 0.0006

# Model specific parameters
# Audio LSTM parameters (Wav2Vec2 embeddings)
# Wav2Vec2 base embedding dim is 768
AUDIO_LSTM_UNITS = [128, 64] # Example, adjust as needed

# Video LSTM parameters (Similar to previous models)
VIDEO_LSTM_UNITS = [128, 64]
MERGED_DENSE_UNITS = [256, 128]

print("HYBRID MODEL WITH WAV2VEC2 (AUDIO) + LSTM (VIDEO)")
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

def create_wav2vec_lstm_model(audio_input_shape, video_feature_dim):
    """
    Create a hybrid model using LSTM for audio (Wav2Vec2 embeddings) and LSTM/BiLSTM for video.

    Args:
        audio_input_shape: Shape of the audio embedding input (time_steps, embedding_dim)
        video_feature_dim: Dimensionality of video features

    Returns:
        Compiled Keras model
    """
    print("Creating hybrid model with Wav2Vec2 (Audio) and LSTM (Video):")
    print("- Audio Input Shape:", audio_input_shape)
    print("- Video Feature Dimension:", video_feature_dim)
    print(f"- L2 Regularization Strength: {L2_REGULARIZATION}")
    print(f"- Weight Constraint: {MAX_NORM_CONSTRAINT}")
    print(f"- Audio LSTM Units: {AUDIO_LSTM_UNITS}")
    print(f"- Video LSTM Units: {VIDEO_LSTM_UNITS}")
    print(f"- Merged Dense Units: {MERGED_DENSE_UNITS}")
    print(f"- Learning Rate: {LEARNING_RATE} with warm-up and cosine decay")

    # --- Audio Branch (LSTM on Wav2Vec2 Embeddings) ---
    # Input shape: (batch, time_steps, embedding_dim)
    audio_input = Input(shape=audio_input_shape, name='audio_input')
    audio_masked = Masking(mask_value=0.0)(audio_input) # Assuming 0.0 padding if needed

    # Apply LSTM layers to the audio embeddings
    audio_x = Bidirectional(LSTM(AUDIO_LSTM_UNITS[0], return_sequences=True, dropout=0.3, recurrent_dropout=0.2, kernel_regularizer=l2(L2_REGULARIZATION), recurrent_regularizer=l2(L2_REGULARIZATION/2), kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT), recurrent_constraint=MaxNorm(MAX_NORM_CONSTRAINT)))(audio_masked)
    audio_x = Dropout(0.3)(audio_x)

    audio_x = Bidirectional(LSTM(AUDIO_LSTM_UNITS[1], return_sequences=False, dropout=0.3, recurrent_dropout=0.2, kernel_regularizer=l2(L2_REGULARIZATION), recurrent_regularizer=l2(L2_REGULARIZATION/2), kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT), recurrent_constraint=MaxNorm(MAX_NORM_CONSTRAINT)))(audio_x)
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
def load_data_paths_and_labels(audio_base_dir_ravdess, audio_base_dir_cremad, video_base_dir_ravdess, video_base_dir_cremad):
    """Finds corresponding audio embedding and video feature files and extracts labels."""
    # Update to look for audio embedding files (.npy)
    audio_files = glob.glob(os.path.join(audio_base_dir_ravdess, "Actor_*", "*.npy")) + \
                  glob.glob(os.path.join(audio_base_dir_cremad, "*.npy"))

    video_files = []
    matched_audio_files = []
    labels = []
    emotion_map = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
    ravdess_emotion_map = {'01': 'NEU', '02': 'NEU', '03': 'HAP', '04': 'SAD', '05': 'ANG', '06': 'FEA', '07': 'DIS'}

    print(f"Found {len(audio_files)} audio embedding files. Matching video features...")
    skipped_video = 0
    skipped_label = 0

    for audio_file_path in tqdm(audio_files, desc="Matching files"):
        base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
        video_file = None
        label = None

        # Find corresponding video file and label
        if "Actor_" in audio_file_path: # RAVDESS
            # Assumes audio files are in Actor_* subdirs matching video features
            actor_folder = os.path.basename(os.path.dirname(audio_file_path))
            potential_video_file = os.path.join(video_base_dir_ravdess, actor_folder, base_name + ".npz")
            if os.path.exists(potential_video_file):
                video_file = potential_video_file
                try:
                    parts = base_name.split('-')
                    emotion_code = ravdess_emotion_map.get(parts[2], None)
                    if emotion_code in emotion_map:
                        label = np.zeros(len(emotion_map))
                        label[emotion_map[emotion_code]] = 1
                except Exception: pass # Ignore parsing errors
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
                except Exception: pass # Ignore parsing errors
            else: skipped_video += 1

        if video_file and label is not None:
            video_files.append(video_file)
            matched_audio_files.append(audio_file_path) # Store matched audio path
            labels.append(label)
        elif video_file and label is None:
            skipped_label += 1

    print(f"Matched {len(video_files)} video/audio embedding pairs with labels.")
    print(f"Skipped {skipped_video} due to missing video NPZ.")
    print(f"Skipped {skipped_label} due to label parsing issues.")

    if not video_files:
        raise FileNotFoundError("No matching video/audio embedding pairs found. Ensure embedding extraction ran and paths are correct.")

    return video_files, matched_audio_files, np.array(labels) # Return audio files list

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
    """Main function to train the Wav2Vec2 + LSTM model."""
    print("Starting Wav2Vec2 + LSTM model training...")
    print(f"- Using learning rate: {LEARNING_RATE}")

    # Define feature directories (relative to current script location or project root)
    # Assuming the script is run from the project root where 'data' is located
    RAVDESS_AUDIO_DIR = "data/ravdess_features_wav2vec2" # Updated path
    CREMA_D_AUDIO_DIR = "data/crema_d_features_wav2vec2" # Updated path
    RAVDESS_VIDEO_FEAT_DIR = "data/ravdess_features_facenet"
    CREMA_D_VIDEO_FEAT_DIR = "data/crema_d_features_facenet"

    # Load file paths and labels
    try:
        # Update function call to use audio dirs
        video_files, audio_files, all_labels = load_data_paths_and_labels(
            RAVDESS_AUDIO_DIR, CREMA_D_AUDIO_DIR, RAVDESS_VIDEO_FEAT_DIR, CREMA_D_VIDEO_FEAT_DIR
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the Wav2Vec2 embedding extraction script has been run successfully.")
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
    train_audio_files = [audio_files[i] for i in train_idx] # Use audio files list
    train_labels = all_labels[train_idx]
    val_video_files = [video_files[i] for i in val_idx]
    val_audio_files = [audio_files[i] for i in val_idx] # Use audio files list
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
    # Wav2Vec2 embeddings have shape (time_steps, embedding_dim)
    try:
        # Load one audio embedding file to get the embedding dimension
        temp_audio_embedding = np.load(train_audio_files[0])
        audio_embedding_dim = temp_audio_embedding.shape[1]
        audio_input_shape = (None, audio_embedding_dim) # Time steps is variable
    except Exception as e:
        print(f"Error loading sample Wav2Vec2 embedding to determine shape: {e}")
        sys.exit(1)

    # Load one video file to get video dimension (if not already known)
    # This part remains the same, but we need a generator instance first
    # Let's assume video dim is determined after generator init

    print("\nCreating data generators...")
    # TODO: Need a Wav2VecSyncGenerator or modify SpectrogramSyncGenerator
    # Use the new Wav2VecSyncGenerator
    train_generator = Wav2VecSyncGenerator(
        train_video_files, train_audio_files, train_labels, batch_size=BATCH_SIZE, shuffle=True
    )
    val_generator = Wav2VecSyncGenerator(
        val_video_files, val_audio_files, val_labels, batch_size=BATCH_SIZE, shuffle=False
    )

    # Get dimensions from initialized generator
    video_feature_dim = train_generator.video_dim
    audio_embedding_dim = train_generator.audio_dim # Get actual dim from generator
    audio_input_shape = (None, audio_embedding_dim)

    # Remove placeholder dimensions
    # video_feature_dim = 512 # Assuming FaceNet default
    # audio_input_shape = (None, 768) # Assuming Wav2Vec2 base default

    if video_feature_dim == 0 or audio_embedding_dim == 0: # Check audio_embedding_dim
        print("Error: Could not determine feature dimensions from generator.")
        sys.exit(1)

    # Create the model
    model = create_wav2vec_lstm_model(audio_input_shape, video_feature_dim) # Use new model function
    model.summary()

    # Define callbacks
    checkpoint_filename = 'best_model_wav2vec_lstm.keras' # New checkpoint name
    checkpoint = ModelCheckpoint(checkpoint_filename, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=PATIENCE * 2, mode='max', verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=PATIENCE, min_lr=5e-6, verbose=1, mode='min')
    lr_scheduler = WarmUpCosineDecayScheduler(learning_rate_base=LEARNING_RATE, total_epochs=EPOCHS, warmup_epochs=10, min_learning_rate=5e-6)

    print("\nStarting training...") # Uncommented
    start_time = time.time() # Uncommented
    history = model.fit( # Uncommented
        train_generator, epochs=EPOCHS, validation_data=val_generator,
        callbacks=[checkpoint, early_stopping, reduce_lr, lr_scheduler], verbose=1
    )
    end_time = time.time() # Uncommented
    print(f"Training finished in {end_time - start_time:.2f} seconds") # Uncommented

    print("\nEvaluating the best model...") # Uncommented
    if os.path.exists(checkpoint_filename): # Uncommented
        print(f"Loading best weights from {checkpoint_filename}") # Uncommented
        model.load_weights(checkpoint_filename) # Uncommented
    else: # Uncommented
        print("Warning: Checkpoint file not found. Evaluating model with final weights.") # Uncommented

    loss, accuracy = model.evaluate(val_generator, verbose=1) # Uncommented
    print(f"Best model validation accuracy ({checkpoint_filename}): {accuracy:.4f}") # Uncommented

if __name__ == '__main__':
    # Ensure generator is imported and used
    train_model() # Uncommented
