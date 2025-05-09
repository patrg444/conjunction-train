#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data generator that loads spectrograms, extracts features using a CNN for time slices
corresponding to video frames, pools these features, and combines them with video features.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, GlobalAveragePooling2D, Dropout, Reshape
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
import os
import glob
import math
import random
from tqdm import tqdm

# Define timing constants (assuming spectrogram time step matches audio feature step)
# Calculated from preprocess_spectrograms.py: HOP_LENGTH / SAMPLE_RATE = 512 / 16000
SPECTROGRAM_STEP_SECONDS = 0.032 # Corrected value based on preprocessing parameters
VIDEO_FPS = 15.0
VIDEO_STEP_SECONDS = 1.0 / VIDEO_FPS

# Define CNN parameters (matching train_spectrogram_cnn_lstm.py)
AUDIO_CNN_FILTERS = [32, 64, 128]
AUDIO_CNN_KERNEL_SIZE = (3, 3)
AUDIO_POOL_SIZE = (2, 2)
L2_REGULARIZATION = 0.002 # Match training script
MAX_NORM_CONSTRAINT = 3.0 # Match training script

def build_cnn_feature_extractor(input_shape):
    """
    Builds the CNN part of the model for feature extraction from spectrogram slices.
    Args:
        input_shape: Shape of the input spectrogram slice (time_steps_slice, n_mels, 1)
                     Note: time_steps_slice will vary. Using GlobalAveragePooling2D handles this.
    Returns:
        A Keras Model for feature extraction.
    """
    spec_slice_input = Input(shape=input_shape, name='spec_slice_input')

    # CNN layers (match train_spectrogram_cnn_lstm.py)
    x = BatchNormalization()(spec_slice_input)
    x = Conv2D(AUDIO_CNN_FILTERS[0], AUDIO_CNN_KERNEL_SIZE, activation='relu', padding='same', kernel_regularizer=l2(L2_REGULARIZATION), kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT))(x)
    x = MaxPooling2D(pool_size=AUDIO_POOL_SIZE, padding='same')(x) # Explicitly set padding to 'same'
    x = Dropout(0.2)(x)

    x = BatchNormalization()(x)
    x = Conv2D(AUDIO_CNN_FILTERS[1], AUDIO_CNN_KERNEL_SIZE, activation='relu', padding='same', kernel_regularizer=l2(L2_REGULARIZATION), kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT))(x)
    x = MaxPooling2D(pool_size=AUDIO_POOL_SIZE, padding='same')(x) # Explicitly set padding to 'same'
    x = Dropout(0.2)(x)

    x = BatchNormalization()(x)
    x = Conv2D(AUDIO_CNN_FILTERS[2], AUDIO_CNN_KERNEL_SIZE, activation='relu', padding='same', kernel_regularizer=l2(L2_REGULARIZATION), kernel_constraint=MaxNorm(MAX_NORM_CONSTRAINT))(x)
    x = MaxPooling2D(pool_size=AUDIO_POOL_SIZE, padding='same')(x) # Explicitly set padding to 'same'
    x = Dropout(0.3)(x) # Shape: (batch, T/8, 16, 128)

    # Reshape to (batch, T/8, features) where features = 16 * 128
    # The -1 infers the time dimension (T/8)
    cnn_output = Reshape((-1, 16 * AUDIO_CNN_FILTERS[-1]), name='reshape_cnn_output')(x) # Output shape: (batch, T/8, 2048)

    model = Model(inputs=spec_slice_input, outputs=cnn_output, name="cnn_feature_extractor")
    # No compilation needed for feature extraction model used within generator
    return model

class SpectrogramCnnPoolingGenerator(Sequence):
    """
    Generates batches of combined data by:
    1. Loading video features and full spectrograms.
    2. For each video frame interval:
        a. Extracting the corresponding spectrogram time slice.
        b. Passing the slice through a pre-built CNN feature extractor.
        c. Pooling (averaging) the CNN features for that interval.
    3. Concatenating pooled CNN audio features with video features.
    4. Padding the resulting combined sequence.
    """
    def __init__(self, video_feature_files, spectrogram_files, labels, batch_size=32, shuffle=True, max_seq_len=None):
        if not (len(video_feature_files) == len(spectrogram_files) == len(labels)):
            raise ValueError("Video features, spectrograms, and labels lists must have the same length.")

        self.video_files = video_feature_files
        self.spec_files = spectrogram_files
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_seq_len = max_seq_len
        self.indices = np.arange(len(self.video_files))

        # Determine dimensions and build CNN extractor
        self.video_dim = 0
        self.spec_n_mels = 0
        # Output dim is now the reshaped dimension: 16 * last_filter_size
        self.cnn_output_dim = 16 * AUDIO_CNN_FILTERS[-1] # 16 * 128 = 2048
        self.num_classes = labels[0].shape[0] if len(labels) > 0 else 0

        for i in range(len(self.video_files)):
            try:
                with np.load(self.video_files[i]) as data:
                    if 'video_features' in data:
                        self.video_dim = data['video_features'].shape[1]
                spec = np.load(self.spec_files[i])
                self.spec_n_mels = spec.shape[0]
                if self.video_dim > 0 and self.spec_n_mels > 0:
                    break
            except Exception as e:
                print(f"Warning: Could not load dimensions from sample {i}: {e}")
                continue

        if self.video_dim == 0 or self.spec_n_mels == 0:
             print("Warning: Could not determine feature dimensions. Using defaults.")
             # Provide default dimensions if needed, or raise error
             self.video_dim = 512 # Example default
             self.spec_n_mels = 128 # Example default

        self.combined_dim = self.cnn_output_dim + self.video_dim

        # Build the CNN feature extractor model
        # Input shape for CNN: (time_steps_slice, n_mels, 1) - time can be None
        cnn_input_shape = (None, self.spec_n_mels, 1)
        self.cnn_extractor = build_cnn_feature_extractor(cnn_input_shape)
        print("CNN Feature Extractor Summary:")
        self.cnn_extractor.summary(line_length=100)


        print(f"\nCreated SpectrogramCnnPoolingGenerator:")
        print(f"- Samples: {len(self.indices)}")
        print(f"- Video Dim: {self.video_dim}")
        print(f"- Spectrogram Mel Bins: {self.spec_n_mels}")
        print(f"- CNN Output Dim (Pooled Audio): {self.cnn_output_dim}")
        print(f"- Combined Dim: {self.combined_dim}")
        print(f"- Max Sequence Len: {self.max_seq_len if self.max_seq_len else 'Dynamic'}")
        print(f"- Spectrogram Step assumed: {SPECTROGRAM_STEP_SECONDS*1000:.1f} ms")

        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        batch_combined_features = []
        batch_labels_list = []

        for i in batch_indices:
            try:
                # Load video features
                with np.load(self.video_files[i]) as data:
                    if 'video_features' not in data: continue
                    video_feat = data['video_features'].astype(np.float32)

                # Load spectrogram and transpose: (time_frames, n_mels)
                spec_feat = np.load(self.spec_files[i]).astype(np.float32)
                spec_feat = np.transpose(spec_feat, (1, 0))

                num_video_frames = video_feat.shape[0]
                num_spec_frames = spec_feat.shape[0]

                if num_video_frames == 0 or num_spec_frames == 0: continue

                combined_sequence = np.zeros((num_video_frames, self.combined_dim), dtype=np.float32)

                # --- Core Logic: Slice Spectrogram, Extract CNN features, Pool ---
                for t in range(num_video_frames):
                    video_start_time = t * VIDEO_STEP_SECONDS
                    video_end_time = (t + 1) * VIDEO_STEP_SECONDS

                    # Find corresponding spectrogram frame indices
                    start_spec_idx = math.floor(video_start_time / SPECTROGRAM_STEP_SECONDS)
                    end_spec_idx = math.ceil(video_end_time / SPECTROGRAM_STEP_SECONDS)
                    start_spec_idx = max(0, start_spec_idx)
                    end_spec_idx = min(num_spec_frames, end_spec_idx)

                    spec_slice = spec_feat[start_spec_idx:end_spec_idx] # Shape: (slice_len, n_mels)

                    if spec_slice.shape[0] > 0:
                        # Add channel dimension and batch dimension for CNN input
                        spec_slice_cnn_input = np.expand_dims(spec_slice, axis=-1) # (slice_len, n_mels, 1)
                        spec_slice_cnn_input = np.expand_dims(spec_slice_cnn_input, axis=0) # (1, slice_len, n_mels, 1)

                        # Extract features using the CNN model
                        # WARNING: This runs inference inside the generator loop - potentially slow!
                        # Output shape from Reshape is (1, T_slice/8, cnn_output_dim)
                        cnn_features_sequence = self.cnn_extractor.predict_on_batch(tf.convert_to_tensor(spec_slice_cnn_input, dtype=tf.float32))

                        # Average the features over the reduced time dimension (T_slice/8) for this video frame interval
                        # Output shape: (1, cnn_output_dim)
                        pooled_audio_features = tf.reduce_mean(cnn_features_sequence, axis=1)
                        pooled_audio = pooled_audio_features[0] # Remove batch dim -> (cnn_output_dim,)
                    else:
                        pooled_audio = np.zeros(self.cnn_output_dim, dtype=np.float32) # Shape (2048,)

                    # Concatenate pooled audio CNN features with video features
                    combined_sequence[t, :self.cnn_output_dim] = pooled_audio # Assign (2048,) features
                    combined_sequence[t, self.cnn_output_dim:] = video_feat[t] # Assign (512,) features
                # --- End Core Logic ---

                batch_combined_features.append(combined_sequence)
                batch_labels_list.append(self.labels[i])

            except Exception as e:
                print(f"ERROR in generator for index {i}: {e}")
                # Optionally skip or handle differently
                continue

        if not batch_combined_features:
             # Return empty tensors matching expected output shapes
             dummy_combined = np.zeros((0, self.max_seq_len if self.max_seq_len else 1, self.combined_dim), dtype=np.float32)
             dummy_labels = np.zeros((0, self.num_classes), dtype=np.float32)
             return tf.convert_to_tensor(dummy_combined), tf.convert_to_tensor(dummy_labels)


        batch_labels = np.array(batch_labels_list, dtype=np.float32)

        # Pad the combined sequences
        if self.max_seq_len is not None:
            batch_padded = pad_sequences(batch_combined_features, maxlen=self.max_seq_len, dtype='float32', padding='post', truncating='post')
        else:
            max_len_in_batch = max(len(seq) for seq in batch_combined_features)
            batch_padded = pad_sequences(batch_combined_features, maxlen=max_len_in_batch, dtype='float32', padding='post', truncating='post')

        batch_padded_tensor = tf.convert_to_tensor(batch_padded, dtype=tf.float32)
        batch_labels_tensor = tf.convert_to_tensor(batch_labels, dtype=tf.float32)

        return batch_padded_tensor, batch_labels_tensor # Single combined input tensor

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# Example Usage (Optional)
if __name__ == '__main__':
    # This requires precomputed spectrograms and video features
    print("SpectrogramCnnPoolingGenerator - Example Usage (requires precomputed data)")
    # Define dummy paths (replace with actual paths if testing)
    RAVDESS_SPEC_DIR = "../data/ravdess_features_spectrogram"
    CREMA_D_SPEC_DIR = "../data/crema_d_features_spectrogram"
    RAVDESS_VIDEO_FEAT_DIR = "../data/ravdess_features_facenet"
    CREMA_D_VIDEO_FEAT_DIR = "../data/crema_d_features_facenet"

    try:
        # Reuse loading logic from train script (or simplify for dummy test)
        spec_files_all = glob.glob(os.path.join(RAVDESS_SPEC_DIR, "Actor_*", "*.npy")) + \
                         glob.glob(os.path.join(CREMA_D_SPEC_DIR, "*.npy"))
        # Create dummy labels and video file list for testing structure
        dummy_labels = tf.keras.utils.to_categorical(np.random.randint(0, 6, len(spec_files_all)), num_classes=6)
        dummy_video_files = ["dummy.npz"] * len(spec_files_all) # Placeholder

        if not spec_files_all:
            print("No spectrogram files found in example paths.")
        else:
            # Create generator instance
            generator = SpectrogramCnnPoolingGenerator(
                dummy_video_files, # Use dummy video paths for structure test
                spec_files_all,
                dummy_labels,
                batch_size=4
            )
            print(f"\nGenerator length (batches): {len(generator)}")

            # Get first batch (will likely fail if dummy video files don't exist or data loading fails)
            print("\nAttempting to get first batch (might fail without real data)...")
            try:
                 batch_x, batch_y = generator[0]
                 print(f"Batch X shape: {batch_x.shape}") # (batch, seq_len, combined_dim)
                 print(f"Batch Y shape: {batch_y.shape}") # (batch, num_classes)
            except Exception as e:
                 print(f"Failed to get batch (expected if using dummy paths/data): {e}")

            print("\nGenerator structure test completed.")

    except Exception as e:
        print(f"Error during example setup: {e}")
