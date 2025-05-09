#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data generator for loading precomputed Wav2Vec2 embeddings and video features,
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

class Wav2VecSyncGenerator(Sequence): # Renamed class
    """
    Generates batches of synchronized Wav2Vec2 embeddings and video features.
    Assumes features are precomputed and stored in corresponding directories.
    """
    def __init__(self, video_feature_files, wav2vec_embedding_files, labels, batch_size=32, shuffle=True, max_video_len=None, max_audio_len=None): # Changed spectrogram_files to wav2vec_embedding_files, max_spec_len to max_audio_len
        """
        Initialize the generator.

        Args:
            video_feature_files: List of paths to video feature .npz files.
            wav2vec_embedding_files: List of paths to Wav2Vec2 embedding .npy files (must correspond to video files).
            labels: Array of one-hot encoded labels corresponding to the files.
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle the data after each epoch.
            max_video_len: If provided, pad video sequences to this fixed length.
            max_audio_len: If provided, pad audio embedding sequences (time axis) to this fixed length.
        """
        if not (len(video_feature_files) == len(wav2vec_embedding_files) == len(labels)):
            raise ValueError("Video features, Wav2Vec2 embeddings, and labels lists must have the same length.")

        self.video_files = video_feature_files
        self.audio_files = wav2vec_embedding_files # Renamed attribute
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_video_len = max_video_len
        self.max_audio_len = max_audio_len # Renamed attribute
        self.indices = np.arange(len(self.video_files))

        # Determine dimensions from first valid sample
        self.video_dim = 0
        self.audio_dim = 0 # Embedding dimension
        self.num_classes = labels[0].shape[0] if len(labels) > 0 else 0
        for i in range(len(self.video_files)):
            try:
                with np.load(self.video_files[i]) as data:
                    if 'video_features' in data:
                        self.video_dim = data['video_features'].shape[1]
                # Load audio embedding to get its dimension
                audio_embedding = np.load(self.audio_files[i])
                if audio_embedding.ndim == 2: # Expecting (time_steps, embedding_dim)
                    self.audio_dim = audio_embedding.shape[1]
                else:
                    print(f"Warning: Unexpected audio embedding shape {audio_embedding.shape} for file {self.audio_files[i]}")

                if self.video_dim > 0 and self.audio_dim > 0:
                    break # Found valid dimensions
            except Exception as e:
                print(f"Warning: Could not load dimensions from sample {i}: {e}")
                continue
        if self.video_dim == 0 or self.audio_dim == 0:
             print("Warning: Could not determine feature dimensions from initial samples.")


        print(f"Created Wav2VecSyncGenerator:") # Updated class name
        print(f"- Samples: {len(self.indices)}")
        print(f"- Video Dim: {self.video_dim}")
        print(f"- Audio Embedding Dim: {self.audio_dim}") # Updated attribute name
        print(f"- Max Video Len: {self.max_video_len if self.max_video_len else 'Dynamic'}")
        print(f"- Max Audio Len: {self.max_audio_len if self.max_audio_len else 'Dynamic'}") # Updated attribute name

        self.on_epoch_end()

    def __len__(self):
        """Return the number of batches per epoch."""
        return math.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, index):
        """Generate one batch of data."""
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        batch_video = []
        batch_audio = [] # Renamed list
        batch_labels_list = []

        for i in batch_indices:
            try:
                # Load video features
                with np.load(self.video_files[i]) as data:
                    if 'video_features' not in data: continue
                    video_feat = data['video_features'].astype(np.float32)

                # Load Wav2Vec2 embedding
                audio_feat = np.load(self.audio_files[i]).astype(np.float32) # Shape: (time_steps, embedding_dim)

                # Basic check for empty features
                if video_feat.shape[0] == 0 or audio_feat.shape[0] == 0:
                    continue

                batch_video.append(video_feat)
                batch_audio.append(audio_feat) # Append to renamed list
                batch_labels_list.append(self.labels[i])

            except Exception as e:
                print(f"Warning: Error loading data for index {i} (Video: {self.video_files[i]}, Audio: {self.audio_files[i]}): {e}") # Updated file reference
                continue # Skip problematic sample

        # Handle empty batch case
        if not batch_video:
            # Return empty tensors with expected rank but zero size in batch dim
             dummy_video = np.zeros((0, self.max_video_len if self.max_video_len else 1, self.video_dim), dtype=np.float32)
             dummy_audio = np.zeros((0, self.max_audio_len if self.max_audio_len else 1, self.audio_dim), dtype=np.float32) # Use audio_dim
             dummy_labels = np.zeros((0, self.num_classes), dtype=np.float32)
             # No need to add channel dimension for embeddings
             return {'video_input': tf.convert_to_tensor(dummy_video), 'audio_input': tf.convert_to_tensor(dummy_audio)}, tf.convert_to_tensor(dummy_labels)


        # Pad sequences
        video_maxlen = self.max_video_len if self.max_video_len else max(len(seq) for seq in batch_video)
        audio_maxlen = self.max_audio_len if self.max_audio_len else max(len(seq) for seq in batch_audio) # Use audio list

        batch_video_padded = pad_sequences(batch_video, maxlen=video_maxlen, dtype='float32', padding='post', truncating='post')
        batch_audio_padded = pad_sequences(batch_audio, maxlen=audio_maxlen, dtype='float32', padding='post', truncating='post') # Use audio list

        # No need to add channel dimension for embeddings

        batch_labels = np.array(batch_labels_list, dtype=np.float32)

        # Return dictionary of inputs
        batch_inputs = {
            'video_input': tf.convert_to_tensor(batch_video_padded, dtype=tf.float32),
            'audio_input': tf.convert_to_tensor(batch_audio_padded, dtype=tf.float32) # Use padded audio embeddings
        }
        batch_labels_tensor = tf.convert_to_tensor(batch_labels, dtype=tf.float32)

        return batch_inputs, batch_labels_tensor

    def on_epoch_end(self):
        """Shuffle indices after each epoch if shuffle is True."""
        if self.shuffle:
            np.random.shuffle(self.indices)

# Example Usage (Optional - for testing with dummy data)
if __name__ == '__main__':
    # This example uses dummy data since real embeddings might not exist yet

    print("Testing Wav2VecSyncGenerator with dummy data...")
    num_samples = 100
    video_dim = 512
    audio_embedding_dim = 768 # Example for wav2vec2-base
    num_classes = 6
    batch_size = 4

    # Create dummy file paths and labels
    dummy_video_files = [f"dummy_vid_{i}.npz" for i in range(num_samples)]
    dummy_audio_files = [f"dummy_audio_{i}.npy" for i in range(num_samples)]
    dummy_labels_list = [tf.keras.utils.to_categorical(random.randint(0, num_classes-1), num_classes=num_classes) for _ in range(num_samples)]
    dummy_labels = np.array(dummy_labels_list)

    # Create dummy feature files (optional, generator checks dimensions from first file)
    # For the generator to initialize correctly, we need at least one pair of files
    # Let's create one dummy pair
    dummy_data_dir = "dummy_data_wav2vec_gen" # Use a unique name
    if not os.path.exists(dummy_data_dir): os.makedirs(dummy_data_dir)
    dummy_vid_path = os.path.join(dummy_data_dir, dummy_video_files[0])
    dummy_audio_path = os.path.join(dummy_data_dir, dummy_audio_files[0])

    try: # Wrap file creation in try/finally for cleanup
        # Create dummy video features (variable length)
        vid_len = random.randint(50, 150)
        dummy_vid_feat = np.random.rand(vid_len, video_dim).astype(np.float32)
        np.savez(dummy_vid_path, video_features=dummy_vid_feat)

        # Create dummy audio embeddings (variable length)
        audio_len = random.randint(100, 300)
        dummy_audio_feat = np.random.rand(audio_len, audio_embedding_dim).astype(np.float32)
        np.save(dummy_audio_path, dummy_audio_feat)

        # Replace file lists with dummy paths for testing init
        test_video_files = [dummy_vid_path] * num_samples # Use dummy path for all
        test_audio_files = [dummy_audio_path] * num_samples # Use dummy path for all

        # Create generator
        generator = Wav2VecSyncGenerator(test_video_files, test_audio_files, dummy_labels, batch_size=batch_size) # Use dummy lists

        print(f"\nGenerator length (batches): {len(generator)}")

        # Get first batch (will load the same dummy files repeatedly in this test)
        print("\nGetting first batch...")
        batch_x_dict, batch_y = generator[0]

        print(f"Batch X keys: {list(batch_x_dict.keys())}")
        print(f"Batch Video Input Shape: {batch_x_dict['video_input'].shape}") # (batch, video_len, video_dim)
        print(f"Batch Audio Input Shape: {batch_x_dict['audio_input'].shape}") # (batch, audio_len, audio_dim)
        print(f"Batch Y shape: {batch_y.shape}") # (batch, num_classes)

        print("\nGenerator test completed.")

    finally: # Ensure cleanup happens
        # Clean up dummy files
        try:
            if os.path.exists(dummy_vid_path): os.remove(dummy_vid_path)
            if os.path.exists(dummy_audio_path): os.remove(dummy_audio_path)
            if os.path.exists(dummy_data_dir): os.rmdir(dummy_data_dir)
        except OSError as e:
            print(f"Error cleaning up dummy data: {e}")

    # --- Original Example Code (Commented Out) ---
    # (The rest of the original example code remains commented out)
    # ...
