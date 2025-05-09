#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data generator that loads precomputed CNN audio features for input to an LSTM model.
(Modified for audio-only operation).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import glob
import math
import random
from tqdm import tqdm
import traceback

# --- Timing Constants ---
# Spectrogram step used *before* CNN processing (from preprocess_spectrograms.py)
ORIGINAL_SPECTROGRAM_STEP_SECONDS = 0.032 # 512 / 16000
# CNN downsampling factor (3 MaxPooling layers with pool_size=2)
CNN_DOWNSAMPLING_FACTOR = 2 * 2 * 2 # = 8
# Effective time step of the precomputed CNN features
CNN_FEATURE_STEP_SECONDS = ORIGINAL_SPECTROGRAM_STEP_SECONDS * CNN_DOWNSAMPLING_FACTOR # 0.032 * 8 = 0.256

VIDEO_FPS = 15.0
VIDEO_STEP_SECONDS = 1.0 / VIDEO_FPS

# --- Feature Dimensions (Need to be determined or passed) ---
# Dimension of the output from the CNN layer
# Should be determined from the files, but set a default just in case
PRECOMPUTED_CNN_AUDIO_DIM = 2048 # Default based on previous checks

class PrecomputedCnnAudioGenerator(Sequence):
    """
    Generates batches of precomputed CNN audio features. (Audio-Only)
    """
    def __init__(self, cnn_audio_feature_files, labels, batch_size=32, shuffle=True, max_seq_len=None):
        if not (len(cnn_audio_feature_files) == len(labels)):
            raise ValueError("CNN audio features and labels lists must have the same length.")

        self.cnn_audio_files = cnn_audio_feature_files # Use precomputed features
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_seq_len = max_seq_len
        self.indices = np.arange(len(self.cnn_audio_files))

        # Determine dimensions
        self.cnn_audio_dim = 0 # Will determine from file
        self.num_classes = labels[0].shape[0] if len(labels) > 0 else 0

        print("Determining CNN audio feature dimension from first valid sample...")
        for i in range(len(self.cnn_audio_files)):
            try:
                # Load the shape of the precomputed CNN features
                cnn_feat_shape = np.load(self.cnn_audio_files[i]).shape
                # Expecting (time_steps_cnn, features)
                if len(cnn_feat_shape) == 2:
                     self.cnn_audio_dim = cnn_feat_shape[1]
                     print(f"  Detected CNN feature shape: {cnn_feat_shape} -> Dim: {self.cnn_audio_dim}")
                     break # Found dimension
                else:
                     # Try flattening if it's 3D (time, freq, filters) - less likely for precomputed
                     if len(cnn_feat_shape) == 3:
                         self.cnn_audio_dim = cnn_feat_shape[1] * cnn_feat_shape[2]
                         print(f"  Detected 3D CNN feature shape: {cnn_feat_shape} -> Flattened Dim: {self.cnn_audio_dim}")
                         break
                     else:
                         print(f"  Warning: Unexpected CNN feature shape {cnn_feat_shape} for file {self.cnn_audio_files[i]}. Skipping.")
                         continue
            except Exception as e:
                print(f"  Warning: Could not load dimensions from sample {i} ({self.cnn_audio_files[i]}): {e}")
                continue

        if self.cnn_audio_dim == 0:
             print("Error: Could not determine CNN audio feature dimension from files.")
             self.cnn_audio_dim = PRECOMPUTED_CNN_AUDIO_DIM
             print(f"Warning: Using default CNN Audio dimension: {self.cnn_audio_dim}")

        # Combined dim is just the audio dim now
        self.combined_dim = self.cnn_audio_dim

        print(f"\nCreated PrecomputedCnnAudioGenerator (Audio-Only):")
        print(f"- Samples: {len(self.indices)}")
        print(f"- Precomputed CNN Audio Dim: {self.cnn_audio_dim}")
        print(f"- Max Sequence Len: {self.max_seq_len if self.max_seq_len else 'Dynamic'}")
        print(f"- CNN Feature Step (Informational): {CNN_FEATURE_STEP_SECONDS*1000:.1f} ms")

        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        batch_audio_features = [] # Renamed from batch_combined_features
        batch_labels_list = []

        for i in batch_indices:
            try:
                # Load precomputed CNN audio features
                cnn_audio_feat = np.load(self.cnn_audio_files[i]).astype(np.float32)

                # Ensure it's 2D: (time_steps_cnn, features)
                if len(cnn_audio_feat.shape) == 3:
                    # Flatten if 3D (e.g., time, freq, filters)
                    cnn_audio_feat = cnn_audio_feat.reshape(cnn_audio_feat.shape[0], -1)
                elif len(cnn_audio_feat.shape) != 2:
                     raise ValueError(f"Unexpected loaded CNN feature shape {cnn_audio_feat.shape} for file {self.cnn_audio_files[i]}")

                # Check if feature dimension matches expected
                if cnn_audio_feat.shape[1] != self.cnn_audio_dim:
                     print(f"Warning: Feature dim mismatch for {self.cnn_audio_files[i]}. Expected {self.cnn_audio_dim}, got {cnn_audio_feat.shape[1]}. Skipping.")
                     continue

                if cnn_audio_feat.shape[0] == 0: # Skip empty features
                    print(f"Warning: Empty feature file {self.cnn_audio_files[i]}. Skipping.")
                    continue

                # No video features or alignment needed
                batch_audio_features.append(cnn_audio_feat)
                batch_labels_list.append(self.labels[i])

            except Exception as e:
                print(f"ERROR in generator for index {i} ({self.cnn_audio_files[i]}): {e}")
                # print(traceback.format_exc()) # Optional detailed traceback
                continue

        if not batch_audio_features:
             # Return empty tensors matching expected output shapes
             dummy_audio = np.zeros((0, self.max_seq_len if self.max_seq_len else 1, self.cnn_audio_dim), dtype=np.float32)
             dummy_labels = np.zeros((0, self.num_classes), dtype=np.float32)
             return tf.convert_to_tensor(dummy_audio), tf.convert_to_tensor(dummy_labels)

        batch_labels = np.array(batch_labels_list, dtype=np.float32)

        # Pad the audio sequences
        if self.max_seq_len is not None:
            batch_padded = pad_sequences(batch_audio_features, maxlen=self.max_seq_len, dtype='float32', padding='post', truncating='post')
        else:
            # Pad to the longest sequence in the current batch
            max_len_in_batch = max(len(seq) for seq in batch_audio_features)
            batch_padded = pad_sequences(batch_audio_features, maxlen=max_len_in_batch, dtype='float32', padding='post', truncating='post')

        batch_padded_tensor = tf.convert_to_tensor(batch_padded, dtype=tf.float32)
        batch_labels_tensor = tf.convert_to_tensor(batch_labels, dtype=tf.float32)

        # Return only audio features and labels
        return batch_padded_tensor, batch_labels_tensor

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# Example Usage (Optional - Audio Only)
if __name__ == '__main__':
    # This requires precomputed CNN audio features
    print("PrecomputedCnnAudioGenerator (Audio-Only) - Example Usage")
    # Define dummy paths (replace with actual paths if testing)
    # POINT THESE TO YOUR *FIXED* CNN FEATURE DIRECTORIES
    RAVDESS_CNN_DIR = "../data/ravdess_features_cnn_fixed"
    CREMA_D_CNN_DIR = "../data/crema_d_features_cnn_fixed"

    try:
        # Find CNN audio files
        cnn_files_all = glob.glob(os.path.join(RAVDESS_CNN_DIR, "Actor_*", "*.npy")) + \
                        glob.glob(os.path.join(CREMA_D_CNN_DIR, "*.npy"))

        if not cnn_files_all:
            print("No CNN audio feature files found in example paths.")
            print(f"Checked: {RAVDESS_CNN_DIR} and {CREMA_D_CNN_DIR}")
        else:
            print(f"Found {len(cnn_files_all)} CNN audio files for example.")
            # Create dummy labels
            dummy_labels_list = [np.random.randint(0, 6) for _ in cnn_files_all]
            dummy_labels = tf.keras.utils.to_categorical(dummy_labels_list, num_classes=6)

            # Create generator instance (Audio-Only)
            generator = PrecomputedCnnAudioGenerator(
                cnn_files_all, # Only pass audio files
                dummy_labels,
                batch_size=4
            )
            print(f"\nGenerator length (batches): {len(generator)}")

            # Get first batch
            print("\nAttempting to get first batch...")
            if len(generator) > 0:
                try:
                     batch_x, batch_y = generator[0]
                     print(f"Batch X (Audio Features) shape: {batch_x.shape}") # (batch, seq_len, cnn_audio_dim)
                     print(f"Batch Y shape: {batch_y.shape}") # (batch, num_classes)
                except Exception as e:
                     print(f"Failed to get batch: {e}")
                     # print(traceback.format_exc()) # Uncomment for detailed error
            else:
                print("Generator is empty, cannot get batch.")

            print("\nGenerator structure test completed.")

    except Exception as e:
        print(f"Error during example setup: {e}")
        # print(traceback.format_exc()) # Uncomment for detailed error
