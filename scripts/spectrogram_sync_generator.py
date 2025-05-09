#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data generator for loading precomputed Mel-spectrograms and video features,
handling synchronization and padding.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import glob
import math
import random

class SpectrogramSyncGenerator(Sequence):
    """
    Generates batches of synchronized Mel-spectrograms and video features.
    Assumes features are precomputed and stored in corresponding directories.
    """
    def __init__(self, video_feature_files, spectrogram_files, labels, batch_size=32, shuffle=True, max_video_len=None, max_spec_len=None):
        """
        Initialize the generator.

        Args:
            video_feature_files: List of paths to video feature .npz files.
            spectrogram_files: List of paths to spectrogram .npy files (must correspond to video files).
            labels: Array of one-hot encoded labels corresponding to the files.
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle the data after each epoch.
            max_video_len: If provided, pad video sequences to this fixed length.
            max_spec_len: If provided, pad spectrogram sequences (time axis) to this fixed length.
        """
        if not (len(video_feature_files) == len(spectrogram_files) == len(labels)):
            raise ValueError("Video features, spectrograms, and labels lists must have the same length.")

        self.video_files = video_feature_files
        self.spec_files = spectrogram_files
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_video_len = max_video_len
        self.max_spec_len = max_spec_len
        self.indices = np.arange(len(self.video_files))

        # Determine dimensions from first valid sample
        self.video_dim = 0
        self.spec_n_mels = 0 # Spectrogram height (Mel bins)
        self.num_classes = labels[0].shape[0] if len(labels) > 0 else 0
        for i in range(len(self.video_files)):
            try:
                with np.load(self.video_files[i]) as data:
                    if 'video_features' in data:
                        self.video_dim = data['video_features'].shape[1]
                spec = np.load(self.spec_files[i])
                self.spec_n_mels = spec.shape[0] # Spectrogram height
                if self.video_dim > 0 and self.spec_n_mels > 0:
                    break # Found valid dimensions
            except Exception as e:
                print(f"Warning: Could not load dimensions from sample {i}: {e}")
                continue
        if self.video_dim == 0 or self.spec_n_mels == 0:
             print("Warning: Could not determine feature dimensions from initial samples.")


        print(f"Created SpectrogramSyncGenerator:")
        print(f"- Samples: {len(self.indices)}")
        print(f"- Video Dim: {self.video_dim}")
        print(f"- Spectrogram Mel Bins: {self.spec_n_mels}")
        print(f"- Max Video Len: {self.max_video_len if self.max_video_len else 'Dynamic'}")
        print(f"- Max Spectrogram Len: {self.max_spec_len if self.max_spec_len else 'Dynamic'}")

        self.on_epoch_end()

    def __len__(self):
        """Return the number of batches per epoch."""
        return math.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, index):
        """Generate one batch of data."""
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        batch_video = []
        batch_spec = []
        batch_labels_list = []

        for i in batch_indices:
            try:
                # Load video features
                with np.load(self.video_files[i]) as data:
                    if 'video_features' not in data: continue
                    video_feat = data['video_features'].astype(np.float32)

                # Load spectrogram
                spec_feat = np.load(self.spec_files[i]).astype(np.float32) # Shape: (n_mels, time_frames)
                # Transpose spectrogram to (time_frames, n_mels) for sequence processing
                spec_feat = np.transpose(spec_feat, (1, 0))

                # Basic check for empty features
                if video_feat.shape[0] == 0 or spec_feat.shape[0] == 0:
                    continue

                batch_video.append(video_feat)
                batch_spec.append(spec_feat)
                batch_labels_list.append(self.labels[i])

            except Exception as e:
                print(f"Warning: Error loading data for index {i} (Video: {self.video_files[i]}, Spec: {self.spec_files[i]}): {e}")
                continue # Skip problematic sample

        # Handle empty batch case
        if not batch_video:
            # Return empty tensors with expected rank but zero size in batch dim
             dummy_video = np.zeros((0, self.max_video_len if self.max_video_len else 1, self.video_dim), dtype=np.float32)
             dummy_spec = np.zeros((0, self.max_spec_len if self.max_spec_len else 1, self.spec_n_mels), dtype=np.float32)
             dummy_labels = np.zeros((0, self.num_classes), dtype=np.float32)
             # Add channel dimension for spectrogram (for CNNs)
             dummy_spec = np.expand_dims(dummy_spec, axis=-1)
             return {'video_input': tf.convert_to_tensor(dummy_video), 'audio_input': tf.convert_to_tensor(dummy_spec)}, tf.convert_to_tensor(dummy_labels)


        # Pad sequences
        video_maxlen = self.max_video_len if self.max_video_len else max(len(seq) for seq in batch_video)
        spec_maxlen = self.max_spec_len if self.max_spec_len else max(len(seq) for seq in batch_spec)

        batch_video_padded = pad_sequences(batch_video, maxlen=video_maxlen, dtype='float32', padding='post', truncating='post')
        batch_spec_padded = pad_sequences(batch_spec, maxlen=spec_maxlen, dtype='float32', padding='post', truncating='post')

        # Add channel dimension for spectrogram (for CNNs) -> (batch, time, mels, 1)
        batch_spec_padded = np.expand_dims(batch_spec_padded, axis=-1)

        batch_labels = np.array(batch_labels_list, dtype=np.float32)

        # Return dictionary of inputs
        batch_inputs = {
            'video_input': tf.convert_to_tensor(batch_video_padded, dtype=tf.float32),
            'audio_input': tf.convert_to_tensor(batch_spec_padded, dtype=tf.float32)
        }
        batch_labels_tensor = tf.convert_to_tensor(batch_labels, dtype=tf.float32)

        return batch_inputs, batch_labels_tensor

    def on_epoch_end(self):
        """Shuffle indices after each epoch if shuffle is True."""
        if self.shuffle:
            np.random.shuffle(self.indices)

# Example Usage (Optional - for testing)
if __name__ == '__main__':
    # This example assumes you have run preprocess_spectrograms.py
    # and have corresponding video feature files (.npz) and spectrogram files (.npy)

    RAVDESS_SPEC_DIR = "../data/ravdess_features_spectrogram"
    CREMA_D_SPEC_DIR = "../data/crema_d_features_spectrogram"
    RAVDESS_VIDEO_FEAT_DIR = "../data/ravdess_features_facenet" # Assuming FaceNet features exist
    CREMA_D_VIDEO_FEAT_DIR = "../data/crema_d_features_facenet"

    print("Searching for precomputed features...")
    # Find corresponding files (adjust paths if needed)
    spec_files = glob.glob(os.path.join(RAVDESS_SPEC_DIR, "Actor_*", "*.npy")) + \
                 glob.glob(os.path.join(CREMA_D_SPEC_DIR, "*.npy"))

    video_files = []
    corresponding_labels = []
    emotion_map = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
    ravdess_emotion_map = {'01': 'NEU', '02': 'NEU', '03': 'HAP', '04': 'SAD', '05': 'ANG', '06': 'FEA', '07': 'DIS'}

    print(f"Found {len(spec_files)} spectrogram files. Matching video features...")

    skipped_video = 0
    skipped_label = 0
    for spec_file in spec_files:
        base_name = os.path.splitext(os.path.basename(spec_file))[0]
        video_file = None
        label = None

        # Find corresponding video file and label
        if "Actor_" in spec_file: # RAVDESS
            actor_folder = os.path.basename(os.path.dirname(spec_file))
            potential_video_file = os.path.join(RAVDESS_VIDEO_FEAT_DIR, actor_folder, base_name + ".npz")
            if os.path.exists(potential_video_file):
                video_file = potential_video_file
                try:
                    parts = base_name.split('-')
                    emotion_code = ravdess_emotion_map.get(parts[2], None)
                    if emotion_code in emotion_map:
                        label = np.zeros(len(emotion_map))
                        label[emotion_map[emotion_code]] = 1
                except: pass # Ignore label errors for now
            else: skipped_video += 1
        else: # CREMA-D
            potential_video_file = os.path.join(CREMA_D_VIDEO_FEAT_DIR, base_name + ".npz")
            if os.path.exists(potential_video_file):
                video_file = potential_video_file
                try:
                    parts = base_name.split('_')
                    emotion_code = parts[2]
                    if emotion_code in emotion_map:
                        label = np.zeros(len(emotion_map))
                        label[emotion_map[emotion_code]] = 1
                except: pass # Ignore label errors for now
            else: skipped_video += 1

        if video_file and label is not None:
            video_files.append(video_file)
            corresponding_labels.append(label)
        elif video_file and label is None:
            skipped_label += 1


    print(f"Matched {len(video_files)} video files.")
    print(f"Skipped {skipped_video} due to missing video NPZ.")
    print(f"Skipped {skipped_label} due to label parsing issues.")

    if not video_files:
        print("Error: No matching video/spectrogram pairs found. Ensure preprocessing ran and paths are correct.")
    else:
        corresponding_labels = np.array(corresponding_labels)
        # Create generator
        generator = SpectrogramSyncGenerator(video_files, spec_files[:len(video_files)], corresponding_labels, batch_size=4) # Match lengths

        print(f"\nGenerator length (batches): {len(generator)}")

        # Get first batch
        print("\nGetting first batch...")
        batch_x_dict, batch_y = generator[0]

        print(f"Batch X keys: {list(batch_x_dict.keys())}")
        print(f"Batch Video Input Shape: {batch_x_dict['video_input'].shape}") # (batch, video_len, video_dim)
        print(f"Batch Audio Input Shape: {batch_x_dict['audio_input'].shape}") # (batch, spec_len, n_mels, 1)
        print(f"Batch Y shape: {batch_y.shape}") # (batch, num_classes)

        print("\nGenerator test completed.")
